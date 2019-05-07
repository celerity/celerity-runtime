#include "runtime.h"

#include <queue>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>

#ifdef _MSC_VER
#include <process.h>
#else
#include <unistd.h>
#endif

#include <mpi.h>

#include "buffer.h"
#include "buffer_storage.h"
#include "command.h"
#include "executor.h"
#include "graph_generator.h"
#include "graph_utils.h"
#include "logger.h"
#include "mpi_support.h"
#include "scheduler.h"
#include "task_manager.h"
#include "user_bench.h"

namespace celerity {
namespace detail {

	std::unique_ptr<runtime> runtime::instance = nullptr;
	bool runtime::test_mode = false;

	void runtime::init(int* argc, char** argv[], cl::sycl::device* user_device) {
		if(test_mode) {
			instance.reset();
			instance = std::unique_ptr<runtime>(new runtime(argc, argv, user_device));
			return;
		}
		instance = std::unique_ptr<runtime>(new runtime(argc, argv, user_device));
	}

	runtime& runtime::get_instance() {
		if(instance == nullptr) { throw std::runtime_error("Runtime has not been initialized"); }
		return *instance;
	}

	auto get_pid() {
#ifdef _MSC_VER
		return _getpid();
#else
		return getpid();
#endif
	}

	const char* get_build_type() {
#ifdef NDEBUG
		return "release";
#else
		return "debug";
#endif
	}

	std::string get_sycl_version() {
		std::stringstream ss;
#if defined(__COMPUTECPP__)
		ss << "ComputeCpp " << COMPUTECPP_VERSION_MAJOR << "." << COMPUTECPP_VERSION_MINOR << "." << COMPUTECPP_VERSION_PATCH;
		return ss.str();
#elif defined(__HIPSYCL__) || defined(__HIPSYCL_TRANSFORM__)
		ss << "hipSYCL " << HIPSYCL_VERSION_MAJOR << "." << HIPSYCL_VERSION_MINOR << "." << HIPSYCL_VERSION_PATCH;
		return ss.str();
#else
		return "Unknown";
#endif
	}

	runtime::runtime(int* argc, char** argv[], cl::sycl::device* user_device) {
		if(!test_mode) {
			int provided;
			MPI_Init_thread(argc, argv, MPI_THREAD_MULTIPLE, &provided);
			assert(provided == MPI_THREAD_MULTIPLE);
		}

		int world_size;
		MPI_Comm_size(MPI_COMM_WORLD, &world_size);
		num_nodes = world_size;

		int world_rank;
		MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
		is_master = world_rank == 0;

		default_logger = logger("default").create_context({{"rank", std::to_string(world_rank)}});
		graph_logger = logger("graph").create_context({{"rank", std::to_string(world_rank)}});

		default_logger->info(
		    logger_map({{"event", "initialized"}, {"pid", std::to_string(get_pid())}, {"build", get_build_type()}, {"sycl", get_sycl_version()}}));

		experimental::bench::detail::user_benchmarker::initialize(static_cast<node_id>(world_rank));

		cfg = std::make_unique<config>(argc, argv, *default_logger);
		queue = std::make_unique<device_queue>(*default_logger);

		// Initialize worker classes (but don't start them up yet)
		task_mngr = std::make_shared<task_manager>(is_master);
		exec = std::make_unique<executor>(*queue, *task_mngr, default_logger);
		if(is_master) {
			ggen = std::make_shared<graph_generator>(num_nodes, *task_mngr,
			    [this](node_id target, const command_pkg& pkg, const std::vector<command_id>& dependencies) { flush_command(target, pkg, dependencies); });
			schdlr = std::make_unique<scheduler>(ggen, num_nodes);
			task_mngr->register_task_callback([this]() { schdlr->notify_task_created(); });
		}

		queue->init(*cfg, task_mngr.get(), user_device);
	}

	runtime::~runtime() {
		if(is_master) {
			schdlr.reset();
			ggen.reset();
		}

		exec.reset();
		task_mngr.reset();
		queue.reset();

		experimental::bench::detail::user_benchmarker::destroy();

		// All buffers should have unregistered themselves by now.
		assert(buffer_ptrs.empty());

		// Make sure we free all of our MPI custom types before we finalize
		active_flushes.clear();
		if(!test_mode) { MPI_Finalize(); }
	}

	void runtime::startup() {
		if(is_active) { throw runtime_already_started_error(); }
		is_active = true;
		if(is_master) { schdlr->startup(); }
		exec->startup();
	}

	void runtime::shutdown() noexcept {
		assert(is_active);
		is_shutting_down = true;
		if(is_master) {
			schdlr->shutdown();

			// Send shutdown commands to all worker nodes.
			// FIXME: This is a bit of hack, since we don't know the last command id issued. Maybe we should generate actual commands in the graph for this.
			command_id base_cmd_id = std::numeric_limits<command_id>::max() - num_nodes;
			for(auto n = 0u; n < num_nodes; ++n) {
				command_pkg pkg{0, base_cmd_id + n, command::SHUTDOWN, command_data{}};
				flush_command(n, pkg, {});
			}
		}

		exec->shutdown();
		queue->wait();

		if(is_master) {
			task_mngr->print_graph(*graph_logger);
			ggen->print_graph(*graph_logger);
		}

		// Shutting down the task_manager will cause all buffers captured inside command group functions to unregister.
		// Since we check whether the runtime is still active upon unregistering, we have to set this to false first.
		is_active = false;
		task_mngr->shutdown();
		is_shutting_down = false;
		maybe_destroy_runtime();
	}

	task_manager& runtime::get_task_manager() const { return *task_mngr; }

	buffer_id runtime::register_buffer(cl::sycl::range<3> range, std::shared_ptr<buffer_storage_base> buf_storage, bool host_initialized) {
		std::lock_guard<std::mutex> lock(buffer_mutex);
		const buffer_id bid = buffer_count++;
		buffer_ptrs[bid] = buf_storage;
		if(is_master) {
			task_mngr->add_buffer(bid, range, host_initialized);
			ggen->add_buffer(bid, range);
		}
		return bid;
	}

	void runtime::unregister_buffer(buffer_id bid) noexcept {
		assert(buffer_ptrs.find(bid) != buffer_ptrs.end());
		if(is_active) {
			// We cannot throw here, as this is being called from buffer destructors.
			default_logger->error(
			    "The Celerity runtime detected that a buffer is going out of scope before all tasks have been completed. This is not allowed.");
		}
		buffer_ptrs.erase(bid);
		maybe_destroy_runtime();
	}

	void runtime::maybe_destroy_runtime() const {
		if(is_active) return;
		if(is_shutting_down) return;
		if(!buffer_ptrs.empty()) return;
		instance.reset();
	}

	void runtime::flush_command(node_id target, const command_pkg& pkg, const std::vector<command_id>& dependencies) {
		// Even though command packages are small enough to use a blocking send we want to be able to send to the master node as well,
		// which is why we have to use Isend after all. We also have to make sure that the buffer stays around until the send is complete.
		active_flushes.push_back(flush_handle{pkg, dependencies, MPI_REQUEST_NULL, {}});
		auto it = active_flushes.rbegin();
		it->data_type = mpi_support::build_single_use_composite_type(
		    {{sizeof(command_pkg), &it->pkg}, {sizeof(command_id) * dependencies.size(), it->dependencies.data()}});
		MPI_Isend(MPI_BOTTOM, 1, *it->data_type, static_cast<int>(target), mpi_support::TAG_CMD, MPI_COMM_WORLD, &active_flushes.rbegin()->req);

		// Cleanup finished transfers.
		// Just check the oldest flush. Since commands are small this will stay in equilibrium fairly quickly.
		int done;
		MPI_Test(&active_flushes.begin()->req, &done, MPI_STATUS_IGNORE);
		if(done) { active_flushes.pop_front(); }
	}

} // namespace detail
} // namespace celerity
