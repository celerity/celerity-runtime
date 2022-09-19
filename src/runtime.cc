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

#include "affinity.h"
#include "buffer.h"
#include "buffer_manager.h"
#include "command_graph.h"
#include "executor.h"
#include "graph_generator.h"
#include "graph_serializer.h"
#include "host_object.h"
#include "log.h"
#include "mpi_support.h"
#include "named_threads.h"
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

	static std::string get_sycl_version() {
#if defined(__COMPUTECPP__)
		return fmt::format("ComputeCpp {}.{}.{}", COMPUTECPP_VERSION_MAJOR, COMPUTECPP_VERSION_MINOR, COMPUTECPP_VERSION_PATCH);
#elif defined(__HIPSYCL__) || defined(__HIPSYCL_TRANSFORM__)
		return fmt::format("hipSYCL {}.{}.{}", HIPSYCL_VERSION_MAJOR, HIPSYCL_VERSION_MINOR, HIPSYCL_VERSION_PATCH);
#elif CELERITY_DPCPP
		return "DPC++ / Clang " __clang_version__;
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
		if(m_cfg->is_dry_run()) {
			if(m_num_nodes != 1) throw std::runtime_error("In order to run with CELERITY_DRY_RUN_NODES a single MPI process/rank must be used.\n");
			m_num_nodes = m_cfg->get_dry_run_nodes();
			CELERITY_WARN("Performing a dry run with {} simulated nodes", m_num_nodes);
		}

		int world_rank;
		MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
		m_local_nid = world_rank;

		spdlog::set_pattern(fmt::format("[%Y-%m-%d %H:%M:%S.%e] [{:0{}}] [%^%l%$] %v", world_rank, int(ceil(log10(world_size)))));

#ifndef __APPLE__
		if(const uint32_t cores = affinity_cores_available(); cores < min_cores_needed) {
			CELERITY_WARN("Celerity has detected that only {} logical cores are available to this process. It is recommended to assign at least {} "
			              "logical cores. Performance may be negatively impacted.",
			    cores, min_cores_needed);
		}
#endif
		m_user_bench = std::make_unique<experimental::bench::detail::user_benchmarker>(*m_cfg, static_cast<node_id>(world_rank));

		m_h_queue = std::make_unique<host_queue>();
		m_d_queue = std::make_unique<device_queue>();

		// Initialize worker classes (but don't start them up yet)
		m_buffer_mngr = std::make_unique<buffer_manager>(*m_d_queue, [this](buffer_manager::buffer_lifecycle_event event, buffer_id bid) {
			switch(event) {
			case buffer_manager::buffer_lifecycle_event::registered: handle_buffer_registered(bid); break;
			case buffer_manager::buffer_lifecycle_event::unregistered: handle_buffer_unregistered(bid); break;
			default: assert(false && "Unexpected buffer lifecycle event");
			}
		});

		m_reduction_mngr = std::make_unique<reduction_manager>();
		m_host_object_mngr = std::make_unique<host_object_manager>();
		m_task_mngr = std::make_unique<task_manager>(m_num_nodes, m_h_queue.get());
		m_exec = std::make_unique<executor>(m_local_nid, *m_h_queue, *m_d_queue, *m_task_mngr, *m_buffer_mngr, *m_reduction_mngr);
		if(is_master_node()) {
			m_cdag = std::make_unique<command_graph>();
			auto ggen = std::make_unique<graph_generator>(m_num_nodes, *m_cdag);
			auto gser = std::make_unique<graph_serializer>(
			    *m_cdag, [this](node_id target, unique_frame_ptr<command_frame> frame) { flush_command(target, std::move(frame)); });
			m_schdlr = std::make_unique<scheduler>(std::move(ggen), std::move(gser), m_num_nodes);
			m_task_mngr->register_task_callback([this](const task* tsk) { m_schdlr->notify_task_created(tsk); });
		}

		CELERITY_INFO(
		    "Celerity runtime version {} running on {}. PID = {}, build type = {}", get_version_string(), get_sycl_version(), get_pid(), get_build_type());
		m_d_queue->init(*m_cfg, user_device_or_selector);
	}

	runtime::~runtime() {
		if(is_master_node()) {
			m_schdlr.reset();
			m_cdag.reset();
		}

		m_exec.reset();
		m_task_mngr.reset();
		m_reduction_mngr.reset();
		m_host_object_mngr.reset();
		// All buffers should have unregistered themselves by now.
		assert(!m_buffer_mngr->has_active_buffers());
		m_buffer_mngr.reset();
		m_d_queue.reset();
		m_h_queue.reset();
		m_user_bench.reset();

		// Make sure we free all of our MPI transfers before we finalize
		while(!m_active_flushes.empty()) {
			int done;
			MPI_Test(&m_active_flushes.begin()->req, &done, MPI_STATUS_IGNORE);
			if(done) { m_active_flushes.pop_front(); }
		}

		if(!m_test_mode) { mpi_finalize_once(); }
	}

	void runtime::startup() {
		if(m_is_active) { throw runtime_already_started_error(); }
		m_is_active = true;
		if(is_master_node()) { m_schdlr->startup(); }
		m_exec->startup();
	}

	void runtime::shutdown() {
		assert(m_is_active);
		m_is_shutting_down = true;

		const auto shutdown_epoch = m_task_mngr->generate_epoch_task(epoch_action::shutdown);

		if(is_master_node()) { m_schdlr->shutdown(); }

		m_task_mngr->await_epoch(shutdown_epoch);

		m_exec->shutdown();
		m_d_queue->wait();
		m_h_queue->wait();

		if(is_master_node() && spdlog::should_log(log_level::trace)) {
			const auto print_max_nodes = m_cfg->get_graph_print_max_verts();
			{
				const auto graph_str = m_task_mngr->print_graph(print_max_nodes);
				if(graph_str.has_value()) {
					CELERITY_TRACE("Task graph:\n\n{}\n", *graph_str);
				} else {
					CELERITY_WARN("Task graph with {} vertices exceeds CELERITY_GRAPH_PRINT_MAX_VERTS={}. Skipping GraphViz output",
					    m_task_mngr->get_current_task_count(), print_max_nodes);
				}
			}
			{
				const auto graph_str = m_cdag->print_graph(print_max_nodes, *m_task_mngr, m_buffer_mngr.get());
				if(graph_str.has_value()) {
					CELERITY_TRACE("Command graph:\n\n{}\n", *graph_str);
				} else {
					CELERITY_WARN("Command graph with {} vertices exceeds CELERITY_GRAPH_PRINT_MAX_VERTS={}. Skipping GraphViz output", m_cdag->command_count(),
					    print_max_nodes);
				}
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

	host_object_manager& runtime::get_host_object_manager() const { return *m_host_object_mngr; }

	void runtime::handle_buffer_registered(buffer_id bid) {
		const auto& info = m_buffer_mngr->get_buffer_info(bid);
		m_task_mngr->add_buffer(bid, info.range, info.is_host_initialized);
		if(is_master_node()) m_schdlr->notify_buffer_registered(bid, info.range);
	}

	void runtime::handle_buffer_unregistered(buffer_id bid) { maybe_destroy_runtime(); }

	void runtime::maybe_destroy_runtime() const {
		if(m_test_active) return;
		if(m_is_active) return;
		if(m_is_shutting_down) return;
		if(m_buffer_mngr->has_active_buffers()) return;
		if(m_host_object_mngr->has_active_objects()) return;
		instance.reset();
	}

	void runtime::flush_command(node_id target, unique_frame_ptr<command_frame> frame) {
		if(is_dry_run()) {
			// We only want to send epochs to the master node for slow full sync and shutdown.
			if(target != 0 || frame->pkg.get_command_type() != command_type::epoch) return;
		}
		// Even though command packages are small enough to use a blocking send we want to be able to send to the master node as well,
		// which is why we have to use Isend after all. We also have to make sure that the buffer stays around until the send is complete.
		MPI_Request req;
		MPI_Isend(
		    frame.get_pointer(), static_cast<int>(frame.get_size_bytes()), MPI_BYTE, static_cast<int>(target), mpi_support::TAG_CMD, MPI_COMM_WORLD, &req);
		m_active_flushes.push_back(flush_handle{std::move(frame), req});

		// Cleanup finished transfers.
		// Just check the oldest flush. Since commands are small this will stay in equilibrium fairly quickly.
		int done;
		MPI_Test(&m_active_flushes.begin()->req, &done, MPI_STATUS_IGNORE);
		if(done) { m_active_flushes.pop_front(); }
	}

} // namespace detail
} // namespace celerity
