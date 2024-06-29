#include "runtime.h"

#include <queue>
#include <string>
#include <unordered_map>

#ifdef _MSC_VER
#include <process.h>
#else
#include <unistd.h>
#endif

#include <mpi.h>

#if CELERITY_USE_MIMALLOC
// override default new/delete operators to use the mimalloc memory allocator
#include <mimalloc-new-delete.h>
#endif

#include "affinity.h"
#include "buffer.h"
#include "buffer_manager.h"
#include "cgf_diagnostics.h"
#include "command_graph.h"
#include "distributed_graph_generator.h"
#include "host_object.h"
#include "legacy_executor.h"
#include "log.h"
#include "mpi_support.h"
#include "named_threads.h"
#include "print_graph.h"
#include "scheduler.h"
#include "task_manager.h"
#include "user_bench.h"
#include "utils.h"
#include "version.h"

namespace celerity {
namespace detail {

	std::unique_ptr<runtime> runtime::instance = nullptr;

	void runtime::mpi_initialize_once(int* argc, char*** argv) {
		assert(!m_mpi_initialized);
		int provided;
		MPI_Init_thread(argc, argv, MPI_THREAD_MULTIPLE, &provided);
		assert(provided == MPI_THREAD_MULTIPLE);
		m_mpi_initialized = true;
	}

	void runtime::mpi_finalize_once() {
		assert(m_mpi_initialized && !m_mpi_finalized && (!m_test_mode || !instance));
		MPI_Finalize();
		m_mpi_finalized = true;
	}

	void runtime::init(int* argc, char** argv[], device_or_selector user_device_or_selector) {
		assert(!instance);
		instance = std::unique_ptr<runtime>(new runtime(argc, argv, user_device_or_selector));
	}

	runtime& runtime::get_instance() {
		if(instance == nullptr) { throw std::runtime_error("Runtime has not been initialized"); }
		return *instance;
	}

	static auto get_pid() {
#ifdef _MSC_VER
		return _getpid();
#else
		return getpid();
#endif
	}

	static std::string get_version_string() {
		using namespace celerity::version;
		return fmt::format("{}.{}.{} {}{}", major, minor, patch, git_revision, git_dirty ? "-dirty" : "");
	}

	static const char* get_build_type() {
#if defined(CELERITY_DETAIL_ENABLE_DEBUG)
		return "debug";
#else
		return "release";
#endif
	}

	static const char* get_mimalloc_string() {
#if CELERITY_USE_MIMALLOC
		return "using mimalloc";
#else
		return "using the default allocator";
#endif
	}

	static std::string get_sycl_version() {
#if defined(__HIPSYCL__) || defined(__HIPSYCL_TRANSFORM__)
		return fmt::format("hipSYCL {}.{}.{}", HIPSYCL_VERSION_MAJOR, HIPSYCL_VERSION_MINOR, HIPSYCL_VERSION_PATCH);
#elif CELERITY_DPCPP
		return "DPC++ / Clang " __clang_version__;
#elif CELERITY_SIMSYCL
		return "SimSYCL " SIMSYCL_VERSION;
#else
#error "unknown SYCL implementation"
#endif
	}

	runtime::runtime(int* argc, char** argv[], device_or_selector user_device_or_selector) {
		if(m_test_mode) {
			assert(m_test_active && "initializing the runtime from a test without a runtime_fixture");
		} else {
			mpi_initialize_once(argc, argv);
		}

		int world_size;
		MPI_Comm_size(MPI_COMM_WORLD, &world_size);
		m_num_nodes = world_size;

		m_cfg = std::make_unique<config>(argc, argv);
		if(m_cfg->is_dry_run()) { m_num_nodes = m_cfg->get_dry_run_nodes(); }

		int world_rank;
		MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
		m_local_nid = world_rank;

		if(!m_test_mode) { // do not touch logger settings in tests, where the full (trace) logs are captured
			spdlog::set_level(m_cfg->get_log_level());
			spdlog::set_pattern(fmt::format("[%Y-%m-%d %H:%M:%S.%e] [{:0{}}] [%^%l%$] %v", world_rank, int(ceil(log10(world_size)))));
		}

#ifndef __APPLE__
		if(const uint32_t cores = affinity_cores_available(); cores < min_cores_needed) {
			CELERITY_WARN("Celerity has detected that only {} logical cores are available to this process. It is recommended to assign at least {} "
			              "logical cores. Performance may be negatively impacted.",
			    cores, min_cores_needed);
		}
#endif
		m_user_bench = std::make_unique<experimental::bench::detail::user_benchmarker>(*m_cfg, static_cast<node_id>(world_rank));

		cgf_diagnostics::make_available();

		m_h_queue = std::make_unique<host_queue>();
		m_d_queue = std::make_unique<device_queue>();

		// Initialize worker classes (but don't start them up yet)
		m_buffer_mngr = std::make_unique<buffer_manager>(*m_d_queue);

		m_reduction_mngr = std::make_unique<reduction_manager>();
		m_host_object_mngr = std::make_unique<host_object_manager>();

		if(m_cfg->should_record()) m_task_recorder = std::make_unique<task_recorder>();

		task_manager::policy_set task_mngr_policy;
		// Merely _declaring_ an uninitialized read is legitimate as long as the kernel does not actually perform the read at runtime - this might happen in the
		// first iteration of a submit-loop. We could get rid of this case by making access-modes a runtime property of accessors (cf
		// https://github.com/celerity/meta/issues/74).
		task_mngr_policy.uninitialized_read_error = CELERITY_ACCESS_PATTERN_DIAGNOSTICS ? error_policy::log_warning : error_policy::ignore;

		m_task_mngr = std::make_unique<task_manager>(m_num_nodes, m_h_queue.get(), m_task_recorder.get(), task_mngr_policy);
		if(m_cfg->get_horizon_step()) m_task_mngr->set_horizon_step(m_cfg->get_horizon_step().value());
		if(m_cfg->get_horizon_max_parallelism()) m_task_mngr->set_horizon_max_parallelism(m_cfg->get_horizon_max_parallelism().value());

		m_exec = std::make_unique<legacy_executor>(m_num_nodes, m_local_nid, *m_h_queue, *m_d_queue, *m_task_mngr, *m_buffer_mngr, *m_reduction_mngr);

		m_cdag = std::make_unique<command_graph>();
		if(m_cfg->should_record()) m_command_recorder = std::make_unique<command_recorder>();

		distributed_graph_generator::policy_set dggen_policy;
		// Any uninitialized read that is observed on CDAG generation was already logged on task generation, unless we have a bug.
		dggen_policy.uninitialized_read_error = error_policy::ignore;
		dggen_policy.overlapping_write_error = CELERITY_ACCESS_PATTERN_DIAGNOSTICS ? error_policy::log_error : error_policy::ignore;

		auto dggen = std::make_unique<distributed_graph_generator>(m_num_nodes, m_local_nid, *m_cdag, *m_task_mngr, m_command_recorder.get(), dggen_policy);

		m_schdlr = std::make_unique<scheduler>(is_dry_run(), std::move(dggen), *m_exec);
		m_task_mngr->register_task_callback([this](const task* tsk) { m_schdlr->notify_task_created(tsk); });

		CELERITY_INFO("Celerity runtime version {} running on {}. PID = {}, build type = {}, {}", get_version_string(), get_sycl_version(), get_pid(),
		    get_build_type(), get_mimalloc_string());
		m_d_queue->init(*m_cfg, user_device_or_selector);
	}

	runtime::~runtime() {
		m_schdlr.reset();
		m_cdag.reset();
		m_exec.reset();
		m_task_mngr.reset();
		m_reduction_mngr.reset();
		m_host_object_mngr.reset();
		// All buffers should have unregistered themselves by now.
		assert(!m_buffer_mngr->has_active_buffers());
		m_buffer_mngr.reset();
		m_d_queue.reset();
		m_h_queue.reset();
		m_command_recorder.reset();
		m_task_recorder.reset();

		cgf_diagnostics::teardown();

		m_user_bench.reset();

		if(!m_test_mode) { mpi_finalize_once(); }
	}

	void runtime::startup() {
		if(m_is_active) { throw runtime_already_started_error(); }
		m_is_active = true;
		m_schdlr->startup();
		m_exec->startup();
	}

	void runtime::shutdown() {
		assert(m_is_active);
		m_is_shutting_down = true;

		const auto shutdown_epoch = m_task_mngr->generate_epoch_task(epoch_action::shutdown);

		m_schdlr->shutdown();

		m_task_mngr->await_epoch(shutdown_epoch);

		m_exec->shutdown();
		m_d_queue->wait();
		m_h_queue->wait();

		if(spdlog::should_log(log_level::info) && m_cfg->should_print_graphs()) {
			if(m_local_nid == 0) { // It's the same across all nodes
				assert(m_task_recorder.get() != nullptr);
				const auto graph_str = detail::print_task_graph(*m_task_recorder);
				CELERITY_INFO("Task graph:\n\n{}\n", graph_str);
			}
			// must be called on all nodes
			auto cmd_graph = gather_command_graph();
			if(m_local_nid == 0) {
				std::this_thread::sleep_for(std::chrono::milliseconds(500)); // Avoid racing on stdout with other nodes (funneled through mpirun)
				CELERITY_INFO("Command graph:\n\n{}\n", cmd_graph);
			}
		}

		// Shutting down the task_manager will cause all buffers captured inside command group functions to unregister.
		// Since we check whether the runtime is still active upon unregistering, we have to set this to false first.
		m_is_active = false;
		m_task_mngr->shutdown();
		m_is_shutting_down = false;
		maybe_destroy_runtime();
	}

	void runtime::sync() {
		const auto epoch = m_task_mngr->generate_epoch_task(epoch_action::barrier);
		m_task_mngr->await_epoch(epoch);
	}

	task_manager& runtime::get_task_manager() const { return *m_task_mngr; }

	buffer_manager& runtime::get_buffer_manager() const { return *m_buffer_mngr; }

	reduction_manager& runtime::get_reduction_manager() const { return *m_reduction_mngr; }

	std::string runtime::gather_command_graph() const {
		assert(m_command_recorder.get() != nullptr);
		const auto graph_str = print_command_graph(m_local_nid, *m_command_recorder);

		// Send local graph to rank 0 on all other nodes
		if(m_local_nid != 0) {
			const uint64_t usize = graph_str.size();
			assert(usize < std::numeric_limits<int32_t>::max());
			const int32_t size = static_cast<int32_t>(usize);
			MPI_Send(&size, 1, MPI_INT32_T, 0, mpi_support::TAG_PRINT_GRAPH, MPI_COMM_WORLD);
			if(size > 0) MPI_Send(graph_str.data(), static_cast<int32_t>(size), MPI_BYTE, 0, mpi_support::TAG_PRINT_GRAPH, MPI_COMM_WORLD);
			return "";
		}
		// On node 0, receive and combine
		std::vector<std::string> graphs;
		graphs.push_back(graph_str);
		for(size_t i = 1; i < m_num_nodes; ++i) {
			int32_t size = 0;
			MPI_Recv(&size, 1, MPI_INT32_T, static_cast<int>(i), mpi_support::TAG_PRINT_GRAPH, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			if(size > 0) {
				std::string graph;
				graph.resize(size);
				MPI_Recv(graph.data(), size, MPI_BYTE, static_cast<int>(i), mpi_support::TAG_PRINT_GRAPH, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				graphs.push_back(std::move(graph));
			}
		}
		return combine_command_graphs(graphs);
	}

	void runtime::register_buffer(buffer_id bid, const range<3>& range, bool host_initialized) {
		m_task_mngr->notify_buffer_created(bid, range, host_initialized);
		m_schdlr->notify_buffer_created(bid, range, host_initialized);
	}

	void runtime::set_buffer_debug_name(const buffer_id bid, const std::string& debug_name) {
		m_buffer_mngr->set_debug_name(bid, debug_name);
		m_task_mngr->notify_buffer_debug_name_changed(bid, debug_name);
		m_schdlr->notify_buffer_debug_name_changed(bid, debug_name);
	}

	void runtime::destroy_buffer(const buffer_id bid) {
		m_schdlr->notify_buffer_destroyed(bid);
		m_task_mngr->notify_buffer_destroyed(bid);
		m_buffer_mngr->unregister_buffer(bid);
		maybe_destroy_runtime();
	}

	host_object_id runtime::create_host_object() {
		const auto hoid = m_host_object_mngr->create_host_object();
		m_task_mngr->notify_host_object_created(hoid);
		m_schdlr->notify_host_object_created(hoid);
		return hoid;
	}

	void runtime::destroy_host_object(const host_object_id hoid) {
		m_schdlr->notify_host_object_destroyed(hoid);
		m_task_mngr->notify_host_object_destroyed(hoid);
		m_host_object_mngr->destroy_host_object(hoid);
	}

	void runtime::maybe_destroy_runtime() const {
		if(m_test_active) return;
		if(m_is_active) return;
		if(m_is_shutting_down) return;
		if(m_buffer_mngr->has_active_buffers()) return;
		if(m_host_object_mngr->has_active_objects()) return;
		instance.reset();
	}

	void runtime::test_case_exit() {
		assert(m_test_mode && m_test_active);
		// We need to delete all tasks manually first, b/c objects that have their lifetime
		// extended by tasks (buffers, host objects) will attempt to shut down the runtime.
		if(instance != nullptr) { instance->m_task_mngr->shutdown(); }
		instance.reset();
		m_test_active = false;
	}

} // namespace detail
} // namespace celerity
