#include "distr_queue.h"

#include <spdlog/fmt/fmt.h>

#include "graph_utils.h"
#include "logger.h"
#include "runtime.h"

// std::getenv is not thread-safe. MS provides an alternative.
std::string get_env(std::string key) {
	std::string result;
#ifdef _MSC_VER
	char* buf;
	_dupenv_s(&buf, nullptr, key.c_str());
	if(buf != nullptr) {
		result = buf;
		delete buf;
	}
#else
	auto value = std::getenv(key.c_str());
	if(value != nullptr) { result = value; }
#endif
	return result;
}

cl::sycl::device pick_device(int platform_id, int device_id, std::shared_ptr<celerity::logger> logger) {
	if(platform_id != -1 && device_id != -1) {
		cl_uint num_platforms;
		clGetPlatformIDs(0, nullptr, &num_platforms);
		std::vector<cl_platform_id> platforms(num_platforms);
		auto ret = clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
		assert(ret == CL_SUCCESS);
		logger->trace("Found {} platforms", num_platforms);
		for(auto i = 0u; i < num_platforms; ++i) {
			size_t name_length;
			ret = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 0, nullptr, &name_length);
			assert(ret == CL_SUCCESS);
			std::vector<char> platform_name(name_length);
			clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, name_length, platform_name.data(), nullptr);
			logger->trace("Platform {}: {}", i, std::string(platform_name.data()));
		}
		if(platform_id >= num_platforms) {
			throw std::runtime_error(fmt::format("Invalid platform id {}: Only {} platforms available", platform_id, num_platforms));
		}
		assert(platform_id < num_platforms);
		cl_uint num_devices;
		clGetDeviceIDs(platforms[platform_id], CL_DEVICE_TYPE_ALL, 0, nullptr, &num_devices);
		std::vector<cl_device_id> devices(num_devices);
		ret = clGetDeviceIDs(platforms[platform_id], CL_DEVICE_TYPE_ALL, num_devices, devices.data(), nullptr);
		assert(ret == CL_SUCCESS);
		if(device_id >= num_devices) {
			throw std::runtime_error(fmt::format("Invalid device id {}: Only {} devices available on platform {}", device_id, num_devices, platform_id));
		}
		logger->trace("Found {} devices on platform {}:", num_devices, platform_id);
		for(auto i = 0u; i < num_devices; ++i) {
			size_t name_length;
			ret = clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 0, nullptr, &name_length);
			assert(ret == CL_SUCCESS);
			std::vector<char> device_name(name_length);
			clGetDeviceInfo(devices[i], CL_DEVICE_NAME, name_length, device_name.data(), nullptr);
			logger->trace("Device {}: {}", i, std::string(device_name.data()));
		}
		return cl::sycl::device(devices[device_id]);
	}
	cl::sycl::gpu_selector selector;
	return selector.select_device();
}

/**
 * Attempts to retrieve the platform and device id from the CELERITY_DEVICES environment variable.
 * The variable has the form "P D0 [D1 ...]", where P is the platform index, followed by any number
 * of device indices. Each device is assigned to a different worker process on the same node, according
 * to their node-local rank.
 *
 * TODO: Should we support multiple platforms on the same node as well?
 */
bool try_get_platform_device_env(int& platform_id, int& device_id, std::shared_ptr<celerity::logger> logger) {
#ifdef OPEN_MPI
#define SPLIT_TYPE OMPI_COMM_TYPE_HOST
#else
#define SPLIT_TYPE MPI_COMM_TYPE_SHARED
#endif
	// Determine our per-node rank by finding all world-ranks that can use a shared-memory transport
	// (If running on OpenMPI, use the per-host split instead)
	// This is a collective call, so make sure we do this before checking the env var
	// TODO: Assert that shared memory is available (i.e. not explicitly disabled)
	MPI_Comm node_comm;
	MPI_Comm_split_type(MPI_COMM_WORLD, SPLIT_TYPE, 0, MPI_INFO_NULL, &node_comm);
	int node_rank = 0;
	MPI_Comm_rank(node_comm, &node_rank);

	const auto env_var = get_env("CELERITY_DEVICES");
	if(env_var.empty()) {
		logger->warn("CELERITY_DEVICES not set");
		return false;
	}

	std::vector<std::string> values;
	boost::split(values, env_var, boost::is_any_of(" "));

	if(node_rank > values.size() - 2) {
		throw std::runtime_error(fmt::format("Process has local rank {}, but CELERITY_DEVICES only includes {} device(s)", node_rank, values.size() - 1));
	}

	int node_size = 0;
	MPI_Comm_size(node_comm, &node_size);
	if(values.size() - 1 > node_size) {
		logger->warn("CELERITY_DEVICES contains {} device indices, but only {} worker processes were spawned on this node", values.size() - 1, node_size);
	}

	platform_id = atoi(values[0].c_str());
	assert(platform_id >= 0);
	device_id = atoi(values[node_rank + 1].c_str());
	assert(device_id >= 0);

	return true;
}

namespace celerity {

distr_queue::distr_queue() {
	init(nullptr);
}

distr_queue::distr_queue(cl::sycl::device& device) {
	init(&device);
}

void distr_queue::init(cl::sycl::device* device_ptr) {
	runtime::get_instance().register_queue(this);
	auto logger = runtime::get_instance().get_logger();

	cl::sycl::device device;
	std::string how_selected = "automatically selected";
	if(device_ptr != nullptr) {
		device = *device_ptr;
		how_selected = "specified by user";
	} else {
		auto platform_id = -1;
		auto device_id = -1;
		if(try_get_platform_device_env(platform_id, device_id, logger)) {
			how_selected = fmt::format("set by CELERITY_DEVICES: platform {}, device {}", platform_id, device_id);
		}
		device = pick_device(platform_id, device_id, logger);
	}

	const auto platform_name = device.get_platform().get_info<cl::sycl::info::platform::name>();
	const auto device_name = device.get_info<cl::sycl::info::device::name>();
	// The names returned by ComputeCpp seem to contain an additional null byte,
	// which causes problems (log files get interpreted as binary data etc), so we chop it off.
	logger->info("Using platform '{}', device '{}' ({})", platform_name.substr(0, platform_name.size() - 1), device_name.substr(0, device_name.size() - 1),
	    how_selected);

	ocl_profiling_enabled = get_env("CELERITY_PROFILE_OCL") == "1";
	if(ocl_profiling_enabled) { logger->info("OpenCL profiling enabled."); }

	// TODO: Do we need a queue on master nodes? (Only for single-node execution?)
	cl::sycl::property_list props;
	if(ocl_profiling_enabled) { props.push_back(std::make_shared<cl::sycl::property::queue::enable_profiling>()); }
	sycl_queue = std::make_unique<cl::sycl::queue>(device, handle_async_exceptions, props);
	task_graph[boost::graph_bundle].name = "TaskGraph";
}

distr_queue::~distr_queue() {
	sycl_queue->wait_and_throw();
}

void distr_queue::mark_task_as_processed(task_id tid) {
	graph_utils::mark_as_processed(tid, task_graph);
}

void distr_queue::debug_print_task_graph(std::shared_ptr<logger> graph_logger) const {
	graph_utils::print_graph(task_graph, graph_logger);
}

task_id distr_queue::add_task(std::shared_ptr<task> tsk) {
	const task_id tid = task_count++;
	task_map[tid] = tsk;
	boost::add_vertex(task_graph);
	task_graph[tid].label = fmt::format("Task {}", tid);
	return tid;
}

void distr_queue::add_requirement(task_id tid, buffer_id bid, cl::sycl::access::mode mode, std::unique_ptr<detail::range_mapper_base> rm) {
	assert(task_map.count(tid) != 0);
	assert(task_map[tid]->get_type() == task_type::COMPUTE);
	dynamic_cast<compute_task*>(task_map[tid].get())->add_range_mapper(bid, std::move(rm));
	update_dependencies(tid, bid, mode);
}

void distr_queue::add_requirement(task_id tid, buffer_id bid, cl::sycl::access::mode mode, cl::sycl::range<3> range, cl::sycl::id<3> offset) {
	assert(task_map.count(tid) != 0);
	assert(task_map[tid]->get_type() == task_type::MASTER_ACCESS);
	dynamic_cast<master_access_task*>(task_map[tid].get())->add_buffer_access(bid, mode, range, offset);
	update_dependencies(tid, bid, mode);
}

void distr_queue::set_task_data(task_id tid, any_range global_size, std::string debug_name) {
	assert(task_map.count(tid) != 0);
	assert(task_map[tid]->get_type() == task_type::COMPUTE);
	dynamic_cast<compute_task*>(task_map[tid].get())->set_global_size(global_size);
	task_graph[tid].label = fmt::format("{} ({})", task_graph[tid].label, debug_name);
}

void distr_queue::update_dependencies(task_id tid, buffer_id bid, cl::sycl::access::mode mode) {
	// TODO: Check if edge already exists (avoid double edges)
	// TODO: If we have dependencies "A -> B, B -> C, A -> C", we could get rid of
	// "A -> C", as it is transitively implicit in "B -> C".
	if(mode == cl::sycl::access::mode::read) {
		if(buffer_last_writer.find(bid) != buffer_last_writer.end()) {
			boost::add_edge(buffer_last_writer[bid], tid, task_graph);
			task_graph[tid].num_unsatisfied++;
		}
	}
	if(mode == cl::sycl::access::mode::write) { buffer_last_writer[bid] = tid; }
}

void distr_queue::handle_async_exceptions(cl::sycl::exception_list el) {
	for(auto& e : el) {
		try {
			std::rethrow_exception(e);
		} catch(cl::sycl::exception& e) {
			// TODO: We'd probably want to abort execution here
			runtime::get_instance().get_logger()->error("SYCL asynchronous exception: {}", e.what());
		}
	}
}

bool distr_queue::has_dependency(task_id task_a, task_id task_b) const {
	// TODO: Use DFS instead?
	bool found = false;
	graph_utils::search_vertex_bf(static_cast<vertex>(task_b), task_graph, [&found, task_a](vertex v, const task_dag&) {
		if(v == task_a) { found = true; }
		return found;
	});
	return found;
}

} // namespace celerity
