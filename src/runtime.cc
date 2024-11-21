#include "runtime.h"

#include "affinity.h"
#include "backend/sycl_backend.h"
#include "cgf_diagnostics.h"
#include "command_graph_generator.h"
#include "device_selection.h"
#include "dry_run_executor.h"
#include "host_object.h"
#include "instruction_graph_generator.h"
#include "live_executor.h"
#include "log.h"
#include "named_threads.h"
#include "print_graph.h"
#include "reduction.h"
#include "scheduler.h"
#include "system_info.h"
#include "task.h"
#include "task_manager.h"
#include "tracy.h"
#include "types.h"
#include "utils.h"
#include "version.h"

#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <future>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

#include <fmt/format.h>
#include <spdlog/spdlog.h>
#include <sycl/sycl.hpp>


#ifdef _MSC_VER
#include <process.h>
#else
#include <unistd.h>
#endif

#if CELERITY_USE_MIMALLOC
// override default new/delete operators to use the mimalloc memory allocator
#include <mimalloc-new-delete.h>
#endif

#if CELERITY_ENABLE_MPI
#include "mpi_communicator.h"
#include <mpi.h>
#else
#include "local_communicator.h"
#endif


namespace celerity {
namespace detail {

	class epoch_promise final : public task_promise {
	  public:
		std::future<void> get_future() { return m_promise.get_future(); }

		void fulfill() override { m_promise.set_value(); }

		allocation_id get_user_allocation_id() override { utils::panic("epoch_promise::get_user_allocation_id"); }

	  private:
		std::promise<void> m_promise;
	};

	std::unique_ptr<runtime> runtime::s_instance = nullptr;

	void runtime::mpi_initialize_once(int* argc, char*** argv) {
#if CELERITY_ENABLE_MPI
		CELERITY_DETAIL_TRACY_ZONE_SCOPED_V("mpi::init", LightSkyBlue, "MPI_Init");
		assert(!s_mpi_initialized);
		int provided;
		MPI_Init_thread(argc, argv, MPI_THREAD_MULTIPLE, &provided);
		assert(provided == MPI_THREAD_MULTIPLE);
#endif // CELERITY_ENABLE_MPI
		s_mpi_initialized = true;
	}

	void runtime::mpi_finalize_once() {
#if CELERITY_ENABLE_MPI
		CELERITY_DETAIL_TRACY_ZONE_SCOPED_V("mpi::finalize", LightSkyBlue, "MPI_Finalize");
		assert(s_mpi_initialized && !s_mpi_finalized && (!s_test_mode || !s_instance));
		MPI_Finalize();
#endif // CELERITY_ENABLE_MPI
		s_mpi_finalized = true;
	}

	void runtime::init(int* argc, char** argv[], const devices_or_selector& user_devices_or_selector) {
		assert(!s_instance);
		s_instance = std::unique_ptr<runtime>(new runtime(argc, argv, user_devices_or_selector));
		if(!s_test_mode) { atexit(shutdown); }
	}

	runtime& runtime::get_instance() {
		if(s_instance == nullptr) { throw std::runtime_error("Runtime has not been initialized"); }
		return *s_instance;
	}

	void runtime::shutdown() { s_instance.reset(); }

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
#if CELERITY_DETAIL_ENABLE_DEBUG
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
#if CELERITY_SYCL_IS_ACPP
		return fmt::format("AdaptiveCpp {}.{}.{}", HIPSYCL_VERSION_MAJOR, HIPSYCL_VERSION_MINOR, HIPSYCL_VERSION_PATCH);
#elif CELERITY_SYCL_IS_DPCPP
		return "DPC++ / Clang " __clang_version__;
#elif CELERITY_SYCL_IS_SIMSYCL
		return "SimSYCL " SIMSYCL_VERSION;
#else
#error "unknown SYCL implementation"
#endif
	}

	static std::string get_mpi_version() {
#if CELERITY_ENABLE_MPI
		char version[MPI_MAX_LIBRARY_VERSION_STRING];
		int len = -1;
		MPI_Get_library_version(version, &len);
		// try shortening the human-readable version string (so far tested on OpenMPI)
		if(const auto brk = /* find last of */ strpbrk(version, ",;")) { len = static_cast<int>(brk - version); }
		return std::string(version, static_cast<size_t>(len));
#else
		return "single node";
#endif
	}

	static host_config get_mpi_host_config() {
#if CELERITY_ENABLE_MPI
		// Determine the "host config", i.e., how many nodes are spawned on this host,
		// and what this node's local rank is. We do this by finding all world-ranks
		// that can use a shared-memory transport (if running on OpenMPI, use the
		// per-host split instead).
#ifdef OPEN_MPI
#define SPLIT_TYPE OMPI_COMM_TYPE_HOST
#else
		// TODO: Assert that shared memory is available (i.e. not explicitly disabled)
#define SPLIT_TYPE MPI_COMM_TYPE_SHARED
#endif
		MPI_Comm host_comm = nullptr;
		MPI_Comm_split_type(MPI_COMM_WORLD, SPLIT_TYPE, 0, MPI_INFO_NULL, &host_comm);

		int local_rank = 0;
		MPI_Comm_rank(host_comm, &local_rank);

		int node_count = 0;
		MPI_Comm_size(host_comm, &node_count);

		host_config host_cfg;
		host_cfg.local_rank = local_rank;
		host_cfg.node_count = node_count;

		MPI_Comm_free(&host_comm);

		return host_cfg;
#else  // CELERITY_ENABLE_MPI
		return host_config{1, 0};
#endif // CELERITY_ENABLE_MPI
	}

	runtime::runtime(int* argc, char** argv[], const devices_or_selector& user_devices_or_selector) {
		m_application_thread = std::this_thread::get_id();

		m_cfg = std::make_unique<config>(argc, argv);

		CELERITY_DETAIL_IF_TRACY_SUPPORTED(tracy_detail::g_tracy_mode = m_cfg->get_tracy_mode());
		CELERITY_DETAIL_TRACY_ZONE_SCOPED("runtime::startup", DarkGray);

		if(s_test_mode) {
			assert(s_test_active && "initializing the runtime from a test without a runtime_fixture");
			s_test_runtime_was_instantiated = true;
		} else {
			mpi_initialize_once(argc, argv);
		}

		int world_size = 1;
		int world_rank = 0;
#if CELERITY_ENABLE_MPI
		MPI_Comm_size(MPI_COMM_WORLD, &world_size);
		MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
#endif

		host_config host_cfg;
		if(m_cfg->is_dry_run()) {
			if(world_size != 1) throw std::runtime_error("In order to run with CELERITY_DRY_RUN_NODES a single MPI process/rank must be used.");
			m_num_nodes = static_cast<size_t>(m_cfg->get_dry_run_nodes());
			m_local_nid = 0;
			host_cfg.node_count = 1;
			host_cfg.local_rank = 0;
		} else {
			m_num_nodes = static_cast<size_t>(world_size);
			m_local_nid = static_cast<node_id>(world_rank);
			host_cfg = get_mpi_host_config();
		}

		// Do not touch logger settings in tests, where the full (trace) logs are captured
		if(!s_test_mode) {
			spdlog::set_level(m_cfg->get_log_level());
			spdlog::set_pattern(fmt::format("[%Y-%m-%d %H:%M:%S.%e] [{:0{}}] [%^%l%$] %v", m_local_nid, int(ceil(log10(double(m_num_nodes))))));
		}

		CELERITY_INFO("Celerity runtime version {} running on {} / {}. PID = {}, build type = {}, {}", get_version_string(), get_sycl_version(),
		    get_mpi_version(), get_pid(), get_build_type(), get_mimalloc_string());

		if(!s_test_mode && m_cfg->get_tracy_mode() != tracy_mode::off) {
			if constexpr(CELERITY_TRACY_SUPPORT) {
				CELERITY_WARN("Profiling with Tracy is enabled. Performance may be negatively impacted.");
			} else {
				CELERITY_WARN("CELERITY_TRACY is set, but Celerity was compiled without Tracy support. Ignoring.");
			}
		}

		cgf_diagnostics::make_available();

		std::vector<sycl::device> devices;
		{
			CELERITY_DETAIL_TRACY_ZONE_SCOPED("runtime::pick_devices", PaleVioletRed);
			devices = std::visit([&](const auto& value) { return pick_devices(host_cfg, value, sycl::platform::get_platforms()); }, user_devices_or_selector);
			assert(!devices.empty()); // postcondition of pick_devices
		}

		{
			const auto& pin_cfg = m_cfg->get_thread_pinning_config();
			const thread_pinning::runtime_configuration thread_pinning_cfg{
			    .enabled = pin_cfg.enabled,
			    .num_devices = static_cast<uint32_t>(devices.size()),
			    .use_backend_device_submission_threads = m_cfg->should_use_backend_device_submission_threads(),
			    .num_legacy_processes = static_cast<uint32_t>(host_cfg.node_count),
			    .legacy_process_index = static_cast<uint32_t>(host_cfg.local_rank),
			    .standard_core_start_id = pin_cfg.starting_from_core,
			    .hardcoded_core_ids = pin_cfg.hardcoded_core_ids,
			};
			m_thread_pinner = std::make_unique<thread_pinning::thread_pinner>(thread_pinning_cfg);
			name_and_pin_and_order_this_thread(named_threads::thread_type::application);
		}

		const sycl_backend::configuration backend_config = {
		    .per_device_submission_threads = m_cfg->should_use_backend_device_submission_threads(), .profiling = m_cfg->should_enable_device_profiling()};
		auto backend = make_sycl_backend(select_backend(sycl_backend_enumerator{}, devices), devices, backend_config);
		const auto system = backend->get_system_info(); // backend is about to be moved

		if(m_cfg->is_dry_run()) {
			m_exec = std::make_unique<dry_run_executor>(static_cast<executor::delegate*>(this));
		} else {
#if CELERITY_ENABLE_MPI
			auto comm = std::make_unique<mpi_communicator>(collective_clone_from, MPI_COMM_WORLD);
#else
			auto comm = std::make_unique<local_communicator>();
#endif
			m_exec = std::make_unique<live_executor>(std::move(backend), std::move(comm), static_cast<executor::delegate*>(this));
		}

		if(m_cfg->should_record()) {
			m_task_recorder = std::make_unique<task_recorder>();
			m_command_recorder = std::make_unique<command_recorder>();
			m_instruction_recorder = std::make_unique<instruction_recorder>();
		}

		task_manager::policy_set task_mngr_policy;
		// Merely _declaring_ an uninitialized read is legitimate as long as the kernel does not actually perform the read at runtime - this might happen in the
		// first iteration of a submit-loop. We could get rid of this case by making access-modes a runtime property of accessors (cf
		// https://github.com/celerity/meta/issues/74).
		task_mngr_policy.uninitialized_read_error = CELERITY_ACCESS_PATTERN_DIAGNOSTICS ? error_policy::log_warning : error_policy::ignore;

		m_task_mngr = std::make_unique<task_manager>(m_num_nodes, m_tdag, m_task_recorder.get(), static_cast<task_manager::delegate*>(this), task_mngr_policy);
		if(m_cfg->get_horizon_step()) m_task_mngr->set_horizon_step(m_cfg->get_horizon_step().value());
		if(m_cfg->get_horizon_max_parallelism()) m_task_mngr->set_horizon_max_parallelism(m_cfg->get_horizon_max_parallelism().value());

		scheduler::policy_set schdlr_policy;
		// Any uninitialized read that is observed on CDAG generation was already logged on task generation, unless we have a bug.
		schdlr_policy.command_graph_generator.uninitialized_read_error = error_policy::ignore;
		schdlr_policy.instruction_graph_generator.uninitialized_read_error = error_policy::ignore;
		schdlr_policy.command_graph_generator.overlapping_write_error = CELERITY_ACCESS_PATTERN_DIAGNOSTICS ? error_policy::log_error : error_policy::ignore;
		schdlr_policy.instruction_graph_generator.overlapping_write_error =
		    CELERITY_ACCESS_PATTERN_DIAGNOSTICS ? error_policy::log_error : error_policy::ignore;
		schdlr_policy.instruction_graph_generator.unsafe_oversubscription_error = error_policy::log_warning;

		// The scheduler references tasks by pointer, so we make sure its lifetime is shorter than the task_manager's.
		m_schdlr = std::make_unique<scheduler>(
		    m_num_nodes, m_local_nid, system, static_cast<scheduler::delegate*>(this), m_command_recorder.get(), m_instruction_recorder.get(), schdlr_policy);
		if(m_cfg->get_lookahead() != experimental::lookahead::automatic) { m_schdlr->set_lookahead(m_cfg->get_lookahead()); }

		// task_manager will pass generated tasks through its delegate, so generate the init epoch only after the scheduler has been initialized
		m_task_mngr->generate_epoch_task(epoch_action::init);

		m_num_local_devices = system.devices.size();
	}

	void runtime::require_call_from_application_thread() const {
		if(std::this_thread::get_id() != m_application_thread) {
			utils::panic("Celerity runtime, queue, handler, buffer and host_object types must only be constructed, used, and destroyed from the "
			             "application thread. Make sure that you did not accidentally capture one of these types in a host_task.");
		}
	}

	runtime::~runtime() {
		// LCOV_EXCL_START
		if(!is_unreferenced()) {
			// this call might originate from static destruction - we cannot assume spdlog to still be around
			utils::panic("Detected an attempt to destroy runtime while at least one queue, buffer or host_object was still alive. This likely means "
			             "that one of these objects was leaked, or at least its lifetime extended beyond the scope of main(). This is undefined.");
		}
		// LCOV_EXCL_STOP

		require_call_from_application_thread();

		CELERITY_DETAIL_TRACY_ZONE_SCOPED("runtime::shutdown", DimGray);

		// Create and await the shutdown epoch
		sync(epoch_action::shutdown);

		// The shutdown epoch is, by definition, the last task (and command / instruction) issued. Since it has now completed, no more scheduler -> executor
		// traffic will occur, and `runtime` can stop functioning as a scheduler_delegate (which would require m_exec to be live).
		m_exec.reset();

		// task_manager references the scheduler as its delegate, so we destroy it first.
		m_task_mngr.reset();

		// ~executor() joins its thread after notifying the scheduler that the shutdown epoch has been reached, which means that this notification is
		// sequenced-before the destructor return, and `runtime` can now stop functioning as an executor_delegate (which would require m_schdlr to be live).
		m_schdlr.reset();

		// With scheduler and executor threads gone, all recorders can be safely accessed from the runtime / application thread
		if(spdlog::should_log(log_level::info) && m_cfg->should_print_graphs()) {
			if(m_local_nid == 0) { // It's the same across all nodes
				assert(m_task_recorder.get() != nullptr);
				const auto tdag_str = detail::print_task_graph(*m_task_recorder);
				CELERITY_INFO("Task graph:\n\n{}\n", tdag_str);
			}

			assert(m_command_recorder.get() != nullptr);
			auto cdag_str = print_command_graph(m_local_nid, *m_command_recorder);
			if(!is_dry_run()) { cdag_str = gather_command_graph(cdag_str, m_num_nodes, m_local_nid); } // must be called on all nodes

			if(m_local_nid == 0) {
				// Avoid racing on stdout with other nodes (funneled through mpirun)
				if(!is_dry_run()) { std::this_thread::sleep_for(std::chrono::milliseconds(500)); }
				CELERITY_INFO("Command graph:\n\n{}\n", cdag_str);
			}

			// IDAGs become unreadable when all nodes print them at the same time - TODO attempt gathering them as well?
			if(m_local_nid == 0) {
				// we are allowed to deref m_instruction_recorder / m_command_recorder because the scheduler thread has exited at this point
				const auto idag_str = detail::print_instruction_graph(*m_instruction_recorder, *m_command_recorder, *m_task_recorder);
				CELERITY_INFO("Instruction graph on node 0:\n\n{}\n", idag_str);
			}
		}

		m_instruction_recorder.reset();
		m_command_recorder.reset();
		m_task_recorder.reset();

		cgf_diagnostics::teardown();

		if(!s_test_mode) { mpi_finalize_once(); }
	}

	task_id runtime::fence(buffer_access_map access_map, side_effect_map side_effects, std::unique_ptr<task_promise> fence_promise) {
		require_call_from_application_thread();
		maybe_prune_task_graph();
		return m_task_mngr->generate_fence_task(std::move(access_map), std::move(side_effects), std::move(fence_promise));
	}

	task_id runtime::sync(epoch_action action) {
		require_call_from_application_thread();

		maybe_prune_task_graph();
		auto promise = std::make_unique<epoch_promise>();
		const auto future = promise->get_future();
		const auto epoch = m_task_mngr->generate_epoch_task(action, std::move(promise));
		future.wait();
		return epoch;
	}

	void runtime::maybe_prune_task_graph() {
		require_call_from_application_thread();

		const auto current_epoch = m_latest_epoch_reached.load(std::memory_order_relaxed);
		if(current_epoch > m_last_epoch_pruned_before) {
			m_tdag.erase_before_epoch(current_epoch);
			m_last_epoch_pruned_before = current_epoch;
		}
	}

	std::string gather_command_graph(const std::string& graph_str, const size_t num_nodes, const node_id local_nid) {
#if CELERITY_ENABLE_MPI
		const auto comm = MPI_COMM_WORLD;
		const int tag = 0xCDA6; // aka 'CDAG' - Celerity does not perform any other peer-to-peer communication over MPI_COMM_WORLD

		// Send local graph to rank 0 on all other nodes
		if(local_nid != 0) {
			const uint64_t usize = graph_str.size();
			assert(usize < std::numeric_limits<int32_t>::max());
			const int32_t size = static_cast<int32_t>(usize);
			MPI_Send(&size, 1, MPI_INT32_T, 0, tag, comm);
			if(size > 0) MPI_Send(graph_str.data(), static_cast<int32_t>(size), MPI_BYTE, 0, tag, comm);
			return "";
		}
		// On node 0, receive and combine
		std::vector<std::string> graphs;
		graphs.push_back(graph_str);
		for(node_id peer = 1; peer < num_nodes; ++peer) {
			int32_t size = 0;
			MPI_Recv(&size, 1, MPI_INT32_T, static_cast<int>(peer), tag, comm, MPI_STATUS_IGNORE);
			if(size > 0) {
				std::string graph;
				graph.resize(size);
				MPI_Recv(graph.data(), size, MPI_BYTE, static_cast<int>(peer), tag, comm, MPI_STATUS_IGNORE);
				graphs.push_back(std::move(graph));
			}
		}
		return combine_command_graphs(graphs);
#else  // CELERITY_ENABLE_MPI
		assert(num_nodes == 1 && local_nid == 0);
		return graph_str;
#endif // CELERITY_ENABLE_MPI
	}

	// task_manager::delegate

	void runtime::task_created(const task* tsk) {
		assert(m_schdlr != nullptr);
		m_schdlr->notify_task_created(tsk);
	}

	// scheduler::delegate

	void runtime::flush(std::vector<const instruction*> instructions, std::vector<outbound_pilot> pilots) {
		// thread-safe
		assert(m_exec != nullptr);
		m_exec->submit(std::move(instructions), std::move(pilots));
	}

	// executor::delegate

	void runtime::horizon_reached(const task_id horizon_tid) {
		assert(!m_latest_horizon_reached || *m_latest_horizon_reached < horizon_tid);
		assert(m_latest_epoch_reached.load(std::memory_order::relaxed) < horizon_tid); // relaxed: written only by this thread

		if(m_latest_horizon_reached.has_value()) {
			m_latest_epoch_reached.store(*m_latest_horizon_reached, std::memory_order_relaxed);
			m_schdlr->notify_epoch_reached(*m_latest_horizon_reached);
		}
		m_latest_horizon_reached = horizon_tid;
	}

	void runtime::epoch_reached(const task_id epoch_tid) {
		// m_latest_horizon_reached does not need synchronization (see definition), all other accesses are implicitly synchronized.
		assert(!m_latest_horizon_reached || *m_latest_horizon_reached < epoch_tid);
		assert(epoch_tid == 0 || m_latest_epoch_reached.load(std::memory_order_relaxed) < epoch_tid);

		m_latest_epoch_reached.store(epoch_tid, std::memory_order_relaxed);
		m_schdlr->notify_epoch_reached(epoch_tid);
		m_latest_horizon_reached = std::nullopt; // Any non-applied horizon is now behind the epoch and will therefore never become an epoch itself
	}

	void runtime::create_queue() {
		require_call_from_application_thread();
		++m_num_live_queues;
	}

	void runtime::destroy_queue() {
		require_call_from_application_thread();

		assert(m_num_live_queues > 0);
		--m_num_live_queues;
	}

	allocation_id runtime::create_user_allocation(void* const ptr) {
		require_call_from_application_thread();
		const auto aid = allocation_id(user_memory_id, m_next_user_allocation_id++);
		m_exec->track_user_allocation(aid, ptr);
		return aid;
	}

	buffer_id runtime::create_buffer(const range<3>& range, const size_t elem_size, const size_t elem_align, const allocation_id user_aid) {
		require_call_from_application_thread();

		const auto bid = m_next_buffer_id++;
		m_live_buffers.emplace(bid);
		m_task_mngr->notify_buffer_created(bid, range, user_aid != null_allocation_id);
		m_schdlr->notify_buffer_created(bid, range, elem_size, elem_align, user_aid);
		return bid;
	}

	void runtime::set_buffer_debug_name(const buffer_id bid, const std::string& debug_name) {
		require_call_from_application_thread();

		assert(utils::contains(m_live_buffers, bid));
		m_task_mngr->notify_buffer_debug_name_changed(bid, debug_name);
		m_schdlr->notify_buffer_debug_name_changed(bid, debug_name);
	}

	void runtime::destroy_buffer(const buffer_id bid) {
		require_call_from_application_thread();

		assert(utils::contains(m_live_buffers, bid));
		m_schdlr->notify_buffer_destroyed(bid);
		m_task_mngr->notify_buffer_destroyed(bid);
		m_live_buffers.erase(bid);
	}

	host_object_id runtime::create_host_object(std::unique_ptr<host_object_instance> instance) {
		require_call_from_application_thread();

		const auto hoid = m_next_host_object_id++;
		m_live_host_objects.emplace(hoid);
		const bool owns_instance = instance != nullptr;
		if(owns_instance) { m_exec->track_host_object_instance(hoid, std::move(instance)); }
		m_task_mngr->notify_host_object_created(hoid);
		m_schdlr->notify_host_object_created(hoid, owns_instance);
		return hoid;
	}

	void runtime::destroy_host_object(const host_object_id hoid) {
		require_call_from_application_thread();

		assert(utils::contains(m_live_host_objects, hoid));
		m_schdlr->notify_host_object_destroyed(hoid);
		m_task_mngr->notify_host_object_destroyed(hoid);
		m_live_host_objects.erase(hoid);
	}

	reduction_id runtime::create_reduction(std::unique_ptr<reducer> reducer) {
		require_call_from_application_thread();

		const auto rid = m_next_reduction_id++;
		m_exec->track_reducer(rid, std::move(reducer));
		return rid;
	}

	void runtime::set_scheduler_lookahead(const experimental::lookahead lookahead) {
		require_call_from_application_thread();
		m_schdlr->set_lookahead(lookahead);
	}

	void runtime::flush_scheduler() {
		require_call_from_application_thread();
		m_schdlr->flush_commands();
	}

	bool runtime::is_unreferenced() const { return m_num_live_queues == 0 && m_live_buffers.empty() && m_live_host_objects.empty(); }

} // namespace detail
} // namespace celerity
