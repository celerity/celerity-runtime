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
#include "cgf_diagnostics.h"
#include "command_graph.h"
#include "distributed_graph_generator.h"
#include "executor.h"
#include "graph_serializer.h"
#include "host_object.h"
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

		cgf_diagnostics::make_available();

		m_local_devices = std::make_unique<local_devices>();
		m_local_devices->init(*m_cfg /*, user_device_or_selector*/);

		// Initialize worker classes (but don't start them up yet)
		m_buffer_mngr = std::make_unique<buffer_manager>(*m_local_devices, [this](buffer_manager::buffer_lifecycle_event event, buffer_id bid) {
			switch(event) {
			case buffer_manager::buffer_lifecycle_event::registered: handle_buffer_registered(bid); break;
			case buffer_manager::buffer_lifecycle_event::unregistered: handle_buffer_unregistered(bid); break;
			default: assert(false && "Unexpected buffer lifecycle event");
			}
		});

		m_reduction_mngr = std::make_unique<reduction_manager>();
		m_host_object_mngr = std::make_unique<host_object_manager>();
		m_task_mngr = std::make_unique<task_manager>(m_num_nodes, &m_local_devices->get_host_queue());
		m_exec = std::make_unique<executor>(m_local_nid, *m_local_devices, *m_task_mngr, *m_buffer_mngr, *m_reduction_mngr);
		m_cdag = std::make_unique<command_graph>();
		auto dggen = std::make_unique<distributed_graph_generator>(m_num_nodes, m_local_devices->num_compute_devices(), m_local_nid, *m_cdag, *m_task_mngr);
		m_schdlr = std::make_unique<scheduler>(is_dry_run(), std::move(dggen), *m_exec, m_num_nodes);
		m_task_mngr->register_task_callback([this](const task* tsk) { m_schdlr->notify_task_created(tsk); });

		CELERITY_INFO(
		    "Celerity runtime version {} running on {}. PID = {}, build type = {}", get_version_string(), get_sycl_version(), get_pid(), get_build_type());
#if TRACY_ENABLE
		CELERITY_WARN("Tracy integration is enabled (may incur overhead).");
#endif
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

		cgf_diagnostics::teardown();

		m_local_devices.reset();
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
		m_local_devices->wait_all();

		if(spdlog::should_log(log_level::trace)) {
			const auto print_max_nodes = m_cfg->get_graph_print_max_verts();
			if(m_local_nid == 0) { // It's the same across all nodes
				const auto graph_str = m_task_mngr->print_graph(print_max_nodes);
				if(graph_str.has_value()) {
					CELERITY_TRACE("Task graph:\n\n{}\n", *graph_str);
				} else {
					CELERITY_WARN("Task graph with {} vertices exceeds CELERITY_GRAPH_PRINT_MAX_VERTS={}. Skipping GraphViz output",
					    m_task_mngr->get_current_task_count(), print_max_nodes);
				}
			}
			{
				const auto graph_str = m_cdag->print_graph(m_local_nid, print_max_nodes, *m_task_mngr, m_buffer_mngr.get());
				if(!graph_str.has_value()) {
					CELERITY_WARN("Command graph with {} vertices exceeds CELERITY_GRAPH_PRINT_MAX_VERTS={}. Skipping GraphViz output", m_cdag->command_count(),
					    print_max_nodes);
				}

				// Send local graph to rank 0
				// FIXME: This actually deadlocks if CELERITY_LOG_LEVEL is not trace for every node!
				if(m_local_nid != 0) {
					const uint64_t size = graph_str.has_value() ? graph_str->size() : 0;
					MPI_Send(&size, 1, MPI_UINT64_T, 0, mpi_support::TAG_PRINT_GRAPH, MPI_COMM_WORLD);
					if(size > 0) MPI_Send(graph_str->data(), size, MPI_BYTE, 0, mpi_support::TAG_PRINT_GRAPH, MPI_COMM_WORLD);
				} else {
					std::vector<std::string> graphs;
					if(graph_str.has_value()) graphs.push_back(*graph_str);
					for(size_t i = 1; i < m_num_nodes; ++i) {
						uint64_t size = 0;
						MPI_Recv(&size, 1, MPI_UINT64_T, i, mpi_support::TAG_PRINT_GRAPH, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
						if(size > 0) {
							std::string graph;
							graph.resize(size);
							MPI_Recv(graph.data(), size, MPI_BYTE, i, mpi_support::TAG_PRINT_GRAPH, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
							graphs.push_back(std::move(graph));
						}
					}
					std::this_thread::sleep_for(std::chrono::milliseconds(500)); // Avoid racing stdout
					CELERITY_TRACE("Command graph:\n\n{}\n", combine_command_graphs(graphs));
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
		m_schdlr->notify_buffer_registered(bid, info.range, info.dims);
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
			// Only flush epochs (for slow_full_sync / shutdown) and horizons (for deleting tasks from the ring buffer).
			if(target != 0 || (frame->pkg.get_command_type() != command_type::epoch && frame->pkg.get_command_type() != command_type::horizon)) return;
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

	void runtime::test_case_exit() {
		assert(m_test_mode && m_test_active);
		// We need to delete all tasks manually first, b/c objects that have their lifetime
		// extended by tasks (buffers, host objects) will attempt to shut down the runtime.
		instance->m_task_mngr.reset();
		instance.reset();
		m_test_active = false;
	}

} // namespace detail
} // namespace celerity
