#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <SYCL/sycl.hpp>
#include <cstdlib>
#include <thread> // JUST FOR SLEEPING

#ifdef _MSC_VER
#include <process.h>
#else
#include <unistd.h>
#endif

#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>
#include <celerity.h>

// Use define instead of constexpr as MSVC seems to have some trouble getting it into nested closures
#define DEMO_DATA_SIZE (1024)

void print_pid() {
	std::cout << "PID: ";
#ifdef _MSC_VER
	std::cout << _getpid();
#else
	std::cout << getpid();
#endif
	std::cout << std::endl;
}

cl::sycl::device pick_device(int platform_id, int device_id) {
	if(platform_id != -1 && device_id != -1) {
		cl_platform_id platforms[10];
		cl::sycl::cl_uint num_platforms;
		assert(clGetPlatformIDs(10, platforms, &num_platforms) == CL_SUCCESS);
		std::cout << "Found " << num_platforms << " platforms:" << std::endl;

		for(auto i = 0u; i < num_platforms; ++i) {
			char platform_name[255];
			clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(platform_name), platform_name, nullptr);
			std::cout << "Platform " << i << ": " << platform_name << std::endl;
		}

		assert(platform_id < num_platforms);
		cl_device_id devices[10];
		cl::sycl::cl_uint num_devices;

		assert(clGetDeviceIDs(platforms[platform_id], CL_DEVICE_TYPE_ALL, 10, devices, &num_devices) == CL_SUCCESS);
		std::cout << "Found " << num_devices << " devices on platform #" << platform_id << ":" << std::endl;

		for(auto i = 0u; i < num_devices; ++i) {
			char device_name[255];
			clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(device_name), device_name, nullptr);
			std::cout << "Device " << i << ": " << device_name << std::endl;
		}

		std::cout << "Using device " << device_id << std::endl;
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
 * TODO: We should get something like this into the runtime itself
 * TODO: Should we support multiple platforms on the same node as well?
 */
void try_get_platform_device_env(int& platform_id, int& device_id) {
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

	const auto env_var = std::getenv("CELERITY_DEVICES");
	if(env_var == nullptr) {
		std::cout << "CELERITY_DEVICES not set" << std::endl;
		return;
	}

	std::vector<std::string> values;
	boost::split(values, env_var, boost::is_any_of(" "));

	if(node_rank > values.size() - 2) {
		throw std::runtime_error(
		    (boost::format("Process has local rank %d, but CELERITY_DEVICES only includes %d device(s)") % node_rank % (values.size() - 1)).str());
	}

	int node_size = 0;
	MPI_Comm_size(node_comm, &node_size);
	if(values.size() - 1 > node_size) {
		std::cout << boost::format("Warning: CELERITY_DEVICES contains %d device indices, but only %d worker processes were spawned on this node")
		                 % (values.size() - 1) % node_size
		          << std::endl;
	}

	platform_id = atoi(values[0].c_str());
	assert(platform_id >= 0);
	device_id = atoi(values[node_rank + 1].c_str());
	assert(device_id >= 0);

	std::cout << "Using platform / device set by CELERITY_DEVICES: " << platform_id << " " << device_id << std::endl;
}

// General notes:
// Spec version used: https://www.khronos.org/registry/SYCL/specs/sycl-1.2.1.pdf
//
// We will probably need some mechanism to detect whether all nodes are running
// the same version of the program (e.g. hash the source as a compile step).
//
// Device fission: The spec mentions (e.g. 3.4.1.2) that command groups without
// data overlap may be executed in parallel. Is this something we need to
// consider for our distributed scheduler?
int main(int argc, char* argv[]) {
	celerity::runtime::init(&argc, &argv);
	print_pid();
	// std::this_thread::sleep_for(std::chrono::seconds(5)); // Sleep so we have time to attach a debugger

	int platform_id = -1;
	int device_id = -1;
	// Always call this, even if we end up using command line args instead
	try_get_platform_device_env(platform_id, device_id);

	if(argc > 2) {
		platform_id = atoi(argv[1]);
		device_id = atoi(argv[2]);
	}

	std::vector<float> host_data_a(DEMO_DATA_SIZE);
	std::vector<float> host_data_b(DEMO_DATA_SIZE);
	std::vector<float> host_data_c(DEMO_DATA_SIZE);
	std::vector<float> host_data_d(DEMO_DATA_SIZE);

	try {
		const cl::sycl::device device = pick_device(platform_id, device_id);

		//// =========== DEVICE SELECTION END ===============

		// TODO: How is device selected for distributed queue?
		// The choice of device is obviously important to the master node, so we
		// have to inform it about our choice somehow. NOTE: We only support a
		// single queue per worker process! The whole idea of CELERITY is that we
		// don't have to address multiple devices (with multiple queues) manually.
		// If a node has multiple devices available, it can start multiple worker
		// processes to make use of them.
		celerity::distr_queue queue(device);

		// TODO: Do we support SYCL sub-buffers & images? Section 4.7.2
		celerity::buffer<float, 1> buf_a(host_data_a.data(), cl::sycl::range<1>(DEMO_DATA_SIZE));
		celerity::buffer<float, 1> buf_b(host_data_b.data(), cl::sycl::range<1>(DEMO_DATA_SIZE));
		celerity::buffer<float, 1> buf_c(host_data_c.data(), cl::sycl::range<1>(DEMO_DATA_SIZE));
		celerity::buffer<float, 1> buf_d(host_data_d.data(), cl::sycl::range<1>(DEMO_DATA_SIZE));

		// **** COMMAND GROUPS ****
		// The functor/lambda submitted to a SYCL queue is called a "command group".
		// Command groups specify a set of "requisites" (or sometimes called
		// "requirements") for a particular kernel, i.e. what buffers need to be
		// available for reading/writing etc. It's (likely) designed that way so
		// the kernel lambdas can capture the accessors from the command group
		// scope.
		//
		// See spec sections 3.4.1.2 and 4.8.2 for more info
		// TODO: This current approach requires C++14 (generic lambda)
		// TODO: We lose autocomplete from IDEs for cgh
		queue.submit([&](auto& cgh) {
			// Access-specifier scenario:
			// We have 2 worker nodes available, and 1000 work items
			// Then this functor is called twice:
			//    fn(nd_subrange(0, 500, 1000)), fn(nd_subrange(501, 999, 1000))
			// It returns the data range the kernel requires to compute it's result
			//
			// NOTE:
			// If returned subrange is out of buffer bounds (e.g. offset - 1 is
			// undefined at first item in block [0, n]), it simply gets clamped to
			// the valid range and it is assumed that the user handles edge cases
			// correctly within the kernel.
			auto a = buf_a.get_access<cl::sycl::access::mode::write>(cgh, [](celerity::subrange<1> range) -> celerity::subrange<1> {
				celerity::subrange<1> sr(range);
				// Write the opposite subrange
				// This is useful to demonstrate that the nodes are assigned to
				// chunks somewhat intelligently in order to minimize buffer
				// transfers. Remove this line and the node assignment in the
				// command graph should be flipped.
				sr.start = range.global_size - range.start - range.range;
				return sr;
			});

			// NOTES:
			// * We don't support explicit work group sizes etc., this should be
			//   handled by the runtime. We only specify the global size.
			// * We only support a single kernel call per command group (not sure if
			//   this is also the case in SYCL; spec doesn't mention it explicitly).
			cgh.template parallel_for<class produce_a>(cl::sycl::range<1>(DEMO_DATA_SIZE), [=](cl::sycl::item<1> item) {
				// TODO: Why doesn't this work? It appears like get_offset always returns 0? Bug? Investigate!
				// a[item.get_range() - item.get_offset() + item.get_id()] = 1.f;
				// a[1024 - item.get_range() - item.get_offset() + item.get_id()] = 1.f;

				auto id = item.get_id()[0];
				a[DEMO_DATA_SIZE - 1 - id] = 1.f;
			});
		});

		// TODO: How do we deal with branching queues, depending on buffer contents?
#define EPSILON 1e-10
		// ** OPTION A ***:
		// This would require blocking on subscript access to ensure that all nodes
		// have the same value here. (This is similar to SYCL "host accessors").
		if(buf_a[0] > EPSILON) {
			// queue.submit(...)
		}
		// *** OPTION B ***:
		// This is the more elaborate solution which would allow the task graph to
		// anticipate this read and preemptively broadcast the value to all nodes.
		// NOTE: In this example we still don't know the submitted command group(s)
		// until after the if().
		// Even more elaborate solution would make 2nd lambda a predicate, and
		// provide two additional lambdas containing the queue submit calls for both
		// branches.
		queue.branch([&](celerity::branch_handle& bh) { bh.get<float, 1>(buf_a, 256); },
		    [](float error) {
			    if(error > EPSILON) {
				    // queue.submit(...);
			    }
		    });
		// => Maybe provide both options? Users can go with A for simplicity, and
		// resort to B in case they identify a bottleneck.
		// ALSO: Do we need the same for loops? Can we unify branching & loops
		// somehow?
		// TODO: Create another example using these kinds of control structures
		// (E.g. some iterative algorithm minimizing an error)

		//// ==============================================

		queue.submit([&](auto& cgh) {
			auto a = buf_a.get_access<cl::sycl::access::mode::read>(cgh, [](celerity::subrange<1> range) -> celerity::subrange<1> {
				celerity::subrange<1> sr(range);
				// Add some overlap so we can generate pull commands
				// NOTE: JUST A DEMO. NOT HONORED IN KERNEL.

				// TODO: This overflows if sr.start == 0! Does user have to take
				// care or do we provide some safety mechanism?
				if(range.start[0] > 10) { sr.start -= 10; }
				sr.range += 20;
				return sr;
			});

			auto b = buf_b.get_access<cl::sycl::access::mode::write>(cgh, celerity::access::one_to_one<1>());
			cgh.template parallel_for<class compute_b>(cl::sycl::range<1>(DEMO_DATA_SIZE), [=](cl::sycl::item<1> item) {
				auto i = item.get_id();
				b[i] = a[i] * 2.f;
			});
		});

#define COMPUTE_C_ON_MASTER 1
#if COMPUTE_C_ON_MASTER
		celerity::with_master_access([&](auto& mah) {
			auto a = buf_a.get_access<cl::sycl::access::mode::read>(mah, cl::sycl::range<1>(DEMO_DATA_SIZE));
			auto c = buf_c.get_access<cl::sycl::access::mode::write>(mah, cl::sycl::range<1>(DEMO_DATA_SIZE));

			mah.run([=]() {
				for(int i = 0; i < DEMO_DATA_SIZE; ++i) {
					c[i] = 2.f - a[i];
				}
			});
		});
#else
		queue.submit([&](auto& cgh) {
			auto a = buf_a.get_access<cl::sycl::access::mode::read>(cgh, celerity::access::one_to_one<1>());
			auto c = buf_c.get_access<cl::sycl::access::mode::write>(cgh, celerity::access::one_to_one<1>());
			cgh.template parallel_for<class compute_c>(cl::sycl::range<1>(DEMO_DATA_SIZE), [=](cl::sycl::item<1> item) {
				auto i = item.get_id();
				c[i] = 2.f - a[i];
			});
		});

#endif

		queue.submit([&](auto& cgh) {
			auto b = buf_b.get_access<cl::sycl::access::mode::read>(cgh, celerity::access::one_to_one<1>());
			auto c = buf_c.get_access<cl::sycl::access::mode::read>(cgh, celerity::access::one_to_one<1>());
			auto d = buf_d.get_access<cl::sycl::access::mode::write>(cgh, celerity::access::one_to_one<1>());
			cgh.template parallel_for<class compute_d>(cl::sycl::range<1>(DEMO_DATA_SIZE), [=](cl::sycl::item<1> item) {
				auto i = item.get_id();
				d[i] = b[i] + c[i];
			});
		});

#if 0
		for(auto i = 0; i < 4; ++i) {
			queue.submit([&](auto& cgh) {
				auto c = buf_c.get_access<cl::sycl::access::mode::read>(cgh, celerity::access::one_to_one<1>());
				auto d = buf_d.get_access<cl::sycl::access::mode::write>(cgh, celerity::access::one_to_one<1>());
				cgh.template parallel_for<class compute_some_more>(cl::sycl::range<1>(DEMO_DATA_SIZE), [=](cl::sycl::item<1> item) {
					auto i = item.get_id();
					d[i] = 2.f - c[i];
				});
			});
		}
#endif

		celerity::with_master_access([&](auto& mah) {
			auto d = buf_d.get_access<cl::sycl::access::mode::read>(mah, cl::sycl::range<1>(DEMO_DATA_SIZE));

			mah.run([=]() {
				// Buffer contents can be accessed in here
				// This is indended for I/O and validation
				// No queue submissions are allowed (i.e. no branching)
				// If access mode is write, update valid buffer regions afterwards!
				size_t sum = 0;
				for(int i = 0; i < DEMO_DATA_SIZE; ++i) {
					sum += (size_t)d[i];
				}

				std::cout << "## RESULT: ";
				if(sum == 3 * DEMO_DATA_SIZE) {
					std::cout << "Success! Correct value was computed." << std::endl;
				} else {
					std::cout << "Fail! Value is " << sum << std::endl;
				}
			});
		});

		// Master: Compute task / command graph, distribute to workers
		// Workers: Wait for and execute commands
		// (In reality, this wouldn't be called explicitly)
		celerity::runtime::get_instance().TEST_do_work();

	} catch(std::exception e) {
		std::cerr << e.what();
		return EXIT_FAILURE;
	}
}
