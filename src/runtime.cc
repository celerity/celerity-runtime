#include "runtime.h"

#include "affinity.h"
#include "backend/sycl_backend.h"
#include "cgf.h"
#include "cgf_diagnostics.h"
#include "command_graph_generator.h"
#include "dry_run_executor.h"
#include "host_object.h"
#include "instruction_graph_generator.h"
#include "live_executor.h"
#include "log.h"
#include "loop_template.h"
#include "named_threads.h"
#include "print_graph.h"
#include "print_utils.h"
#include "print_utils_internal.h"
#include "ranges.h"
#include "reduction.h"
#include "scheduler.h"
#include "select_devices.h"
#include "system_info.h"
#include "task.h"
#include "task_manager.h"
#include "testspy/runtime_testspy.h"
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

	class runtime::impl final : public runtime, private task_manager::delegate, private scheduler::delegate, private executor::delegate {
	  public:
		impl(int* argc, char** argv[], const devices_or_selector& user_devices_or_selector, const bool init_mpi);

		impl(const runtime::impl&) = delete;
		impl(runtime::impl&&) = delete;
		impl& operator=(const runtime::impl&) = delete;
		impl& operator=(runtime::impl&&) = delete;

		~impl();

		task_id submit(raw_command_group&& cg);

		task_id fence(buffer_access access, std::unique_ptr<task_promise> fence_promise);

		task_id fence(host_object_effect effect, std::unique_ptr<task_promise> fence_promise);

		task_id sync(detail::epoch_action action);

		void create_queue();

		void destroy_queue();

		allocation_id create_user_allocation(void* ptr);

		buffer_id create_buffer(const range<3>& range, size_t elem_size, size_t elem_align, allocation_id user_aid);

		void set_buffer_debug_name(buffer_id bid, const std::string& debug_name);

		void destroy_buffer(buffer_id bid);

		host_object_id create_host_object(std::unique_ptr<host_object_instance> instance /* optional */);

		void destroy_host_object(host_object_id hoid);

		reduction_id create_reduction(std::unique_ptr<reducer> reducer);

		bool is_dry_run() const;

		void set_scheduler_lookahead(experimental::lookahead lookahead);

		void flush_scheduler();

		void initialize_new_loop_template();

		void begin_loop_iteration();

		void complete_loop_iteration();

		void finalize_loop_template();

		backend* NOCOMMIT_backend_ptr;
		node_id NOCOMMIT_get_local_nid() const { return m_local_nid; }
		size_t NOCOMMIT_get_num_nodes() const { return m_num_nodes; }
		size_t NOCOMMIT_get_num_local_devices() const { return m_num_local_devices; }

	  private:
		friend struct runtime_testspy;

		bool m_external_mpi_init = false;

		// `runtime` is not thread safe except for its delegate implementations, so we store the id of the thread where it was instantiated (the application
		// thread) in order to throw if the user attempts to issue a runtime operation from any other thread. One case where this may happen unintentionally
		// is capturing a buffer into a host-task by value, where this capture is the last reference to the buffer: The runtime would attempt to destroy itself
		// from a thread that it also needs to await, which would at least cause a deadlock. This variable is immutable, so reading it from a different thread
		// for the purpose of the check is safe.
		std::thread::id m_application_thread;

		std::unique_ptr<config> m_cfg;
		size_t m_num_nodes = 0;
		node_id m_local_nid = 0;
		size_t m_num_local_devices = 0;

		// track all instances of celerity::queue, celerity::buffer and celerity::host_object to sanity-check runtime destruction
		size_t m_num_live_queues = 0;
		std::unordered_set<buffer_id> m_live_buffers;
		std::unordered_set<host_object_id> m_live_host_objects;

		buffer_id m_next_buffer_id = 0;
		raw_allocation_id m_next_user_allocation_id = 1;
		host_object_id m_next_host_object_id = 0;
		reduction_id m_next_reduction_id = no_reduction_id + 1;

		std::unique_ptr<loop_template> m_active_loop_template;

		task_graph m_tdag;
		std::unique_ptr<task_manager> m_task_mngr;
		std::unique_ptr<scheduler> m_schdlr;
		std::unique_ptr<executor> m_exec;

		std::optional<task_id> m_latest_horizon_reached; // only accessed by executor thread
		std::atomic<size_t> m_latest_epoch_reached;      // task_id, but cast to size_t to work with std::atomic
		task_id m_last_epoch_pruned_before = 0;

		std::unique_ptr<detail::task_recorder> m_task_recorder;                                       // accessed by task manager (application thread)
		std::unique_ptr<detail::command_recorder> m_command_recorder;                                 // accessed only by scheduler thread (until shutdown)
		std::unique_ptr<detail::instruction_recorder> m_instruction_recorder;                         // accessed only by scheduler thread (until shutdown)
		std::unique_ptr<detail::instruction_performance_recorder> m_instruction_performance_recorder; // accessed only by executor thread (until shutdown)

		std::unique_ptr<detail::thread_pinning::thread_pinner> m_thread_pinner; // thread safe, manages lifetime of thread pinning machinery

		/// Panic when not called from m_application_thread (see that variable for more info on the matter). Since there are thread-safe and non thread-safe
		/// member functions, we call this check at the beginning of all the non-safe ones.
		void require_call_from_application_thread() const;

		void maybe_prune_task_graph();

		// task_manager::delegate
		void task_created(const task* tsk) override;

		// scheduler::delegate
		void flush(std::vector<const instruction*> instructions, std::vector<outbound_pilot> pilot) override;
		void on_scheduler_idle() override;
		void on_scheduler_busy() override;

		// executor::delegate
		void horizon_reached(task_id horizon_tid) override;
		void epoch_reached(task_id epoch_tid) override;

		/// True when no buffers, host objects or queues are live that keep the runtime alive.
		bool is_unreferenced() const;
	};

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
		MPI_Comm host_comm = MPI_COMM_NULL;
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

	runtime::impl::impl(int* argc, char** argv[], const devices_or_selector& user_devices_or_selector, const bool init_mpi) {
		m_application_thread = std::this_thread::get_id();

		m_cfg = std::make_unique<config>(argc, argv);

		CELERITY_DETAIL_IF_TRACY_SUPPORTED(tracy_detail::g_tracy_mode = m_cfg->get_tracy_mode());
		CELERITY_DETAIL_TRACY_ZONE_SCOPED("runtime::startup", runtime_startup);

		if(s_test_mode) {
			assert(s_test_active && "initializing the runtime from a test without a runtime_fixture");
			s_test_runtime_was_instantiated = true;
		} else {
			if(init_mpi) {
				mpi_initialize_once(argc, argv);
			} else {
				m_external_mpi_init = true;
				int provided = 0;
				MPI_Query_thread(&provided);
				if(provided != MPI_THREAD_MULTIPLE) {
					throw std::runtime_error("MPI was not initialized with the required threading level MPI_THREAD_MULTIPLE");
				}
			}
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

#if !CELERITY_DETAIL_ENABLE_DEBUG && CELERITY_ACCESSOR_BOUNDARY_CHECK
		CELERITY_WARN("Celerity was configured with CELERITY_ACCESSOR_BOUNDARY_CHECK=ON. Kernel performance will be negatively impacted.");
#endif

		if(m_cfg->should_record()) {
			m_task_recorder = std::make_unique<task_recorder>();
			m_command_recorder = std::make_unique<command_recorder>();
			m_instruction_recorder = std::make_unique<instruction_recorder>();
			if(m_cfg->should_report_instruction_performance()) {
				m_instruction_performance_recorder = std::make_unique<instruction_performance_recorder>(m_num_nodes, m_local_nid);
			}
		}

		cgf_diagnostics::make_available();

		std::vector<sycl::device> devices;
		{
			CELERITY_DETAIL_TRACY_ZONE_SCOPED("runtime::select_devices", runtime_select_devices);
			devices = std::visit([&](const auto& value) { return select_devices(host_cfg, value, sycl::platform::get_platforms()); }, user_devices_or_selector);
			assert(!devices.empty()); // postcondition of select_devices
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
		NOCOMMIT_backend_ptr = backend.get();
		const auto system = backend->get_system_info(); // backend is about to be moved

		if(m_cfg->is_dry_run()) {
			m_exec = std::make_unique<dry_run_executor>(static_cast<executor::delegate*>(this));
		} else {
#if CELERITY_ENABLE_MPI
			auto comm = std::make_unique<mpi_communicator>(collective_clone_from, MPI_COMM_WORLD);
#else
			auto comm = std::make_unique<local_communicator>();
#endif
			m_exec = std::make_unique<live_executor>(
			    std::move(backend), std::move(comm), static_cast<executor::delegate*>(this), m_instruction_performance_recorder.get());
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

	void runtime::impl::require_call_from_application_thread() const {
		if(std::this_thread::get_id() != m_application_thread) {
			utils::panic("Celerity runtime, queue, handler, buffer and host_object types must only be constructed, used, and destroyed from the "
			             "application thread. Make sure that you did not accidentally capture one of these types in a host_task.");
		}
	}

	runtime::impl::~impl() {
		// LCOV_EXCL_START
		if(m_num_live_queues != 0 || !m_live_buffers.empty() || !m_live_host_objects.empty()) {
			// this call might originate from static destruction - we cannot assume spdlog to still be around
			utils::panic("Detected an attempt to destroy runtime while at least one queue, buffer or host_object was still alive. This likely means "
			             "that one of these objects was leaked, or at least its lifetime extended beyond the scope of main(). This is undefined.");
		}
		// LCOV_EXCL_STOP

		require_call_from_application_thread();

		CELERITY_DETAIL_TRACY_ZONE_SCOPED("runtime::shutdown", runtime_shutdown);

		// Create and await the shutdown epoch
		sync(epoch_action::shutdown);

		const auto starvation_time = m_exec->get_starvation_time();
		const auto active_time = m_exec->get_active_time();
		const auto ratio = static_cast<double>(starvation_time.count()) / static_cast<double>(active_time.count());
		CELERITY_DEBUG(
		    "Executor active time: {:.1f}. Starvation time: {:.1f} ({:.1f}%).", as_sub_second(active_time), as_sub_second(starvation_time), 100.0 * ratio);
		if(active_time > std::chrono::milliseconds(5) && ratio > 0.2) {
			CELERITY_WARN("The executor was starved for instructions for {:.1f}, or {:.1f}% of the total active time of {:.1f}. This may indicate that "
			              "your application is scheduler-bound. If you are interleaving Celerity tasks with other work, try flushing the queue.",
			    as_sub_second(starvation_time), 100.0 * ratio, as_sub_second(active_time));
		}

		// The shutdown epoch is, by definition, the last task (and command / instruction) issued. Since it has now completed, no more scheduler -> executor
		// traffic will occur, and `runtime` can stop functioning as a scheduler_delegate (which would require m_exec to be live).
		m_exec.reset();

		// task_manager references the scheduler as its delegate, so we destroy it first.
		m_task_mngr.reset();

		// ~executor() joins its thread after notifying the scheduler that the shutdown epoch has been reached, which means that this notification is
		// sequenced-before the destructor return, and `runtime` can now stop functioning as an executor_delegate (which would require m_schdlr to be live).
		m_schdlr.reset();

		// TODO: Add env var to control whether this is happening
		// TODO: Collect across all nodes?
		// Do this before filtering the records!
		if(m_cfg->should_report_instruction_performance()) {
			assert(m_instruction_recorder != nullptr);
			m_instruction_performance_recorder->print_summary(*m_instruction_recorder, *m_task_recorder);
		}

		// With scheduler and executor threads gone, all recorders can be safely accessed from the runtime / application thread
		if(spdlog::should_log(log_level::info) && m_cfg->should_print_graphs()) {
			const auto getoption = [](const std::string_view name) -> std::optional<size_t> {
				const auto cstr = getenv(name.data());
				if(cstr == nullptr) return std::nullopt;
				return atol(cstr);
			};

			// TODO: In a proper query language we would probably like to specify whether we want only true dependencies or all
			const auto filter_by_tid = getoption("GRAPH_QUERY_TID");
			const auto before = getoption("GRAPH_QUERY_BEFORE");
			const auto after = getoption("GRAPH_QUERY_AFTER");

			if(m_local_nid == 0) { // It's the same across all nodes
				assert(m_task_recorder.get() != nullptr);
				if(filter_by_tid.has_value()) { m_task_recorder->filter_by_task_id(*filter_by_tid, before.value_or(0), 0 /* unsupported */); }
				const auto tdag_str = detail::print_task_graph(*m_task_recorder);
				CELERITY_INFO("Task graph:\n\n{}\n", tdag_str);
			}

			assert(m_command_recorder.get() != nullptr);
			if(filter_by_tid.has_value()) { m_command_recorder->filter_by_task_id(*filter_by_tid, before.value_or(0), after.value_or(0)); }
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
				if(filter_by_tid.has_value()) { m_instruction_recorder->filter_by_task_id(*filter_by_tid, before.value_or(0), after.value_or(0)); }
				const auto idag_str =
				    detail::print_instruction_graph(*m_instruction_recorder, *m_command_recorder, *m_task_recorder, m_instruction_performance_recorder.get());
				CELERITY_INFO("Instruction graph on node 0:\n\n{}\n", idag_str);
			}
		}

		m_instruction_recorder.reset();
		m_command_recorder.reset();
		m_task_recorder.reset();

		cgf_diagnostics::teardown();

		if(!s_test_mode && !m_external_mpi_init) { mpi_finalize_once(); }
	}

	task_id runtime::impl::submit(raw_command_group&& cg) {
		require_call_from_application_thread();
		maybe_prune_task_graph();
		return m_task_mngr->generate_command_group_task(std::move(cg), m_active_loop_template.get());
	}

	task_id runtime::impl::fence(buffer_access access, std::unique_ptr<task_promise> fence_promise) {
		require_call_from_application_thread();
		maybe_prune_task_graph();
		return m_task_mngr->generate_fence_task(std::move(access), std::move(fence_promise));
	}

	task_id runtime::impl::fence(host_object_effect effect, std::unique_ptr<task_promise> fence_promise) {
		require_call_from_application_thread();
		maybe_prune_task_graph();
		return m_task_mngr->generate_fence_task(effect, std::move(fence_promise));
	}

	task_id runtime::impl::sync(epoch_action action) {
		require_call_from_application_thread();

		maybe_prune_task_graph();
		auto promise = std::make_unique<epoch_promise>();
		const auto future = promise->get_future();
		const auto epoch = m_task_mngr->generate_epoch_task(action, std::move(promise));
		future.wait();
		return epoch;
	}

	void runtime::impl::maybe_prune_task_graph() {
		require_call_from_application_thread();

		// Don't prune the task graph while there's an active loop template: task pointers must remain valid until the template is finalized.
		if(m_active_loop_template != nullptr) return;

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

	void runtime::impl::task_created(const task* tsk) {
		assert(m_schdlr != nullptr);
		m_schdlr->notify_task_created(tsk);
	}

	// scheduler::delegate

	void runtime::impl::flush(std::vector<const instruction*> instructions, std::vector<outbound_pilot> pilots) {
		// thread-safe
		assert(m_exec != nullptr);
		m_exec->submit(std::move(instructions), std::move(pilots));
	}

	void runtime::impl::on_scheduler_idle() {
		CELERITY_TRACE("Scheduler is idle");
		// The executor may have already been destroyed if we are currently shutting down
		if(m_exec != nullptr) { m_exec->notify_scheduler_idle(true); }
	}

	void runtime::impl::on_scheduler_busy() {
		CELERITY_TRACE("Scheduler is busy");
		// The executor may have already been destroyed if we are currently shutting down
		if(m_exec != nullptr) { m_exec->notify_scheduler_idle(false); }
	}

	// executor::delegate

	void runtime::impl::horizon_reached(const task_id horizon_tid) {
		assert(!m_latest_horizon_reached || *m_latest_horizon_reached < horizon_tid);
		assert(m_latest_epoch_reached.load(std::memory_order::relaxed) < horizon_tid); // relaxed: written only by this thread

		if(m_latest_horizon_reached.has_value()) {
			m_latest_epoch_reached.store(*m_latest_horizon_reached, std::memory_order_relaxed);
			m_schdlr->notify_epoch_reached(*m_latest_horizon_reached);
		}
		m_latest_horizon_reached = horizon_tid;
	}

	void runtime::impl::epoch_reached(const task_id epoch_tid) {
		// m_latest_horizon_reached does not need synchronization (see definition), all other accesses are implicitly synchronized.
		assert(!m_latest_horizon_reached || *m_latest_horizon_reached < epoch_tid);
		assert(epoch_tid == 0 || m_latest_epoch_reached.load(std::memory_order_relaxed) < epoch_tid);

		m_latest_epoch_reached.store(epoch_tid, std::memory_order_relaxed);
		m_schdlr->notify_epoch_reached(epoch_tid);
		m_latest_horizon_reached = std::nullopt; // Any non-applied horizon is now behind the epoch and will therefore never become an epoch itself
	}

	void runtime::impl::create_queue() {
		require_call_from_application_thread();
		++m_num_live_queues;
	}

	void runtime::impl::destroy_queue() {
		require_call_from_application_thread();

		assert(m_num_live_queues > 0);
		--m_num_live_queues;
	}

	bool runtime::impl::is_dry_run() const { return m_cfg->is_dry_run(); }

	allocation_id runtime::impl::create_user_allocation(void* const ptr) {
		require_call_from_application_thread();
		const auto aid = allocation_id(user_memory_id, m_next_user_allocation_id++);
		m_exec->track_user_allocation(aid, ptr);
		return aid;
	}

	buffer_id runtime::impl::create_buffer(const range<3>& range, const size_t elem_size, const size_t elem_align, const allocation_id user_aid) {
		require_call_from_application_thread();

		const auto bid = m_next_buffer_id++;
		m_live_buffers.emplace(bid);
		m_task_mngr->notify_buffer_created(bid, range, user_aid != null_allocation_id);
		m_schdlr->notify_buffer_created(bid, range, elem_size, elem_align, user_aid);
		return bid;
	}

	void runtime::impl::set_buffer_debug_name(const buffer_id bid, const std::string& debug_name) {
		require_call_from_application_thread();

		assert(utils::contains(m_live_buffers, bid));
		m_task_mngr->notify_buffer_debug_name_changed(bid, debug_name);
		m_schdlr->notify_buffer_debug_name_changed(bid, debug_name);
	}

	void runtime::impl::destroy_buffer(const buffer_id bid) {
		require_call_from_application_thread();

		assert(utils::contains(m_live_buffers, bid));
		m_schdlr->notify_buffer_destroyed(bid);
		m_task_mngr->notify_buffer_destroyed(bid);
		m_live_buffers.erase(bid);
	}

	host_object_id runtime::impl::create_host_object(std::unique_ptr<host_object_instance> instance) {
		require_call_from_application_thread();

		const auto hoid = m_next_host_object_id++;
		m_live_host_objects.emplace(hoid);
		const bool owns_instance = instance != nullptr;
		if(owns_instance) { m_exec->track_host_object_instance(hoid, std::move(instance)); }
		m_task_mngr->notify_host_object_created(hoid);
		m_schdlr->notify_host_object_created(hoid, owns_instance);
		return hoid;
	}

	void runtime::impl::destroy_host_object(const host_object_id hoid) {
		require_call_from_application_thread();

		assert(utils::contains(m_live_host_objects, hoid));
		m_schdlr->notify_host_object_destroyed(hoid);
		m_task_mngr->notify_host_object_destroyed(hoid);
		m_live_host_objects.erase(hoid);
	}

	reduction_id runtime::impl::create_reduction(std::unique_ptr<reducer> reducer) {
		require_call_from_application_thread();

		const auto rid = m_next_reduction_id++;
		m_exec->track_reducer(rid, std::move(reducer));
		return rid;
	}

	void runtime::impl::set_scheduler_lookahead(const experimental::lookahead lookahead) {
		require_call_from_application_thread();
		m_schdlr->set_lookahead(lookahead);
	}

	void runtime::impl::flush_scheduler() {
		require_call_from_application_thread();
		m_schdlr->flush_commands();
	}

	void runtime::impl::initialize_new_loop_template() {
		require_call_from_application_thread();
		assert(m_active_loop_template == nullptr);
		m_active_loop_template = std::make_unique<loop_template>();
		m_schdlr->enable_loop_template(*m_active_loop_template);
	}

	void runtime::impl::begin_loop_iteration() {
		require_call_from_application_thread();
		assert(m_active_loop_template != nullptr);
		m_task_mngr->begin_loop_template_iteration(*m_active_loop_template);
	}

	void runtime::impl::complete_loop_iteration() {
		require_call_from_application_thread();
		assert(m_active_loop_template != nullptr);
		m_active_loop_template->tdag.complete_iteration();
		m_schdlr->complete_loop_iteration();
	}

	void runtime::impl::finalize_loop_template() {
		require_call_from_application_thread();
		assert(m_active_loop_template != nullptr);
		m_task_mngr->finalize_loop_template(*m_active_loop_template);
		// NOCOMMIT TODO: Move unique_ptr into this so scheduler controls lifetime
		m_schdlr->finalize_loop_template(*m_active_loop_template);
		m_active_loop_template.release(); // NOCOMMIT LEAKY
		// We don't prune the task graph while a loop template is active - now is a good time.
		maybe_prune_task_graph();
	}

	bool runtime::s_mpi_initialized = false;
	bool runtime::s_mpi_finalized = false;

	runtime runtime::s_instance; // definition of static member

	void runtime::mpi_initialize_once(int* argc, char*** argv) {
#if CELERITY_ENABLE_MPI
		CELERITY_DETAIL_TRACY_ZONE_SCOPED_V("mpi::init", mpi_init, "MPI_Init");
		assert(!s_mpi_initialized);
		int provided = -1;
		MPI_Init_thread(argc, argv, MPI_THREAD_MULTIPLE, &provided);
		assert(provided == MPI_THREAD_MULTIPLE);
#endif // CELERITY_ENABLE_MPI
		s_mpi_initialized = true;
	}

	void runtime::mpi_finalize_once() {
#if CELERITY_ENABLE_MPI
		CELERITY_DETAIL_TRACY_ZONE_SCOPED_V("mpi::finalize", mpi_finalize, "MPI_Finalize");
		assert(s_mpi_initialized && !s_mpi_finalized && (!s_test_mode || !has_instance()));
		MPI_Finalize();
#endif // CELERITY_ENABLE_MPI
		s_mpi_finalized = true;
	}

	void runtime::init(int* argc, char** argv[], const devices_or_selector& user_devices_or_selector, const bool init_mpi) {
		assert(!has_instance());
		s_instance.m_impl = std::make_unique<runtime::impl>(argc, argv, user_devices_or_selector, init_mpi);
		if(!s_test_mode) { atexit(shutdown); }
	}

	runtime& runtime::get_instance() {
		if(!has_instance()) { throw std::runtime_error("Runtime has not been initialized"); }
		return s_instance;
	}

	void runtime::shutdown() { s_instance.m_impl.reset(); }

	task_id runtime::submit(raw_command_group&& cg) { return m_impl->submit(std::move(cg)); }

	task_id runtime::fence(buffer_access access, std::unique_ptr<task_promise> fence_promise) {
		return m_impl->fence(std::move(access), std::move(fence_promise));
	}

	task_id runtime::fence(host_object_effect effect, std::unique_ptr<task_promise> fence_promise) { return m_impl->fence(effect, std::move(fence_promise)); }

	task_id runtime::sync(detail::epoch_action action) { return m_impl->sync(action); }

	void runtime::create_queue() { m_impl->create_queue(); }

	void runtime::destroy_queue() { m_impl->destroy_queue(); }

	allocation_id runtime::create_user_allocation(void* const ptr) { return m_impl->create_user_allocation(ptr); }

	buffer_id runtime::create_buffer(const range<3>& range, const size_t elem_size, const size_t elem_align, const allocation_id user_aid) {
		return m_impl->create_buffer(range, elem_size, elem_align, user_aid);
	}

	void runtime::set_buffer_debug_name(const buffer_id bid, const std::string& debug_name) { m_impl->set_buffer_debug_name(bid, debug_name); }

	void runtime::destroy_buffer(const buffer_id bid) { m_impl->destroy_buffer(bid); }

	host_object_id runtime::create_host_object(std::unique_ptr<host_object_instance> instance) { return m_impl->create_host_object(std::move(instance)); }

	void runtime::destroy_host_object(const host_object_id hoid) { m_impl->destroy_host_object(hoid); }

	reduction_id runtime::create_reduction(std::unique_ptr<reducer> reducer) { return m_impl->create_reduction(std::move(reducer)); }

	bool runtime::is_dry_run() const { return m_impl->is_dry_run(); }

	void runtime::set_scheduler_lookahead(const experimental::lookahead lookahead) { m_impl->set_scheduler_lookahead(lookahead); }

	void runtime::flush_scheduler() { m_impl->flush_scheduler(); }

	void runtime::initialize_new_loop_template() { m_impl->initialize_new_loop_template(); }

	void runtime::begin_loop_iteration() { m_impl->begin_loop_iteration(); }

	void runtime::complete_loop_iteration() { m_impl->complete_loop_iteration(); }

	void runtime::finalize_loop_template() { m_impl->finalize_loop_template(); }

	backend* runtime::NOCOMMIT_get_backend_ptr() const { return m_impl->NOCOMMIT_backend_ptr; };
	node_id runtime::NOCOMMIT_get_local_nid() const { return m_impl->NOCOMMIT_get_local_nid(); }
	size_t runtime::NOCOMMIT_get_num_nodes() const { return m_impl->NOCOMMIT_get_num_nodes(); }
	size_t runtime::NOCOMMIT_get_num_local_devices() const { return m_impl->NOCOMMIT_get_num_local_devices(); }

	bool runtime::s_test_mode = false;
	bool runtime::s_test_active = false;
	bool runtime::s_test_runtime_was_instantiated = false;

} // namespace detail
} // namespace celerity


#define CELERITY_DETAIL_TAIL_INCLUDE
#include "testspy/runtime_testspy.inl"
