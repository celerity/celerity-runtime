#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <SYCL/sycl.hpp>

#include "celerity_runtime.h"

// Prepend "//" to not break GraphViz format
// ...or output it to stderr... or both!
std::ostream& log() { return std::cerr << "// "; }

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
  //// ============= DEMO SETUP =================
  // Pause execution so we can attach debugger (for stepping w/ live plotting)
  if (argc > 1 && std::string("--pause") == argv[1]) {
    std::cout << "(Paused. Press return key to continue)" << std::endl;
    char a;
    std::cin >> std::noskipws >> a;
  }

  float host_data_a[1024];
  float host_data_b[1024];
  float host_data_c[1024];
  float host_data_d[1024];

  try {
    //// ============= DEVICE SELECTION =================

    cl_platform_id platforms[10];
    cl::sycl::cl_uint num_platforms;
    assert(clGetPlatformIDs(10, platforms, &num_platforms) == CL_SUCCESS);
    log() << "Found " << num_platforms << " platforms:" << std::endl;

    for (auto i = 0u; i < num_platforms; ++i) {
      char platform_name[255];
      clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(platform_name),
                        platform_name, nullptr);
      log() << "Platform " << i << ": " << platform_name << std::endl;
    }

    const size_t USE_PLATFORM = 2;
    assert(USE_PLATFORM < num_platforms);
    cl_device_id devices[10];
    cl::sycl::cl_uint num_devices;
    assert(clGetDeviceIDs(platforms[USE_PLATFORM], CL_DEVICE_TYPE_CPU, 10,
                          devices, &num_devices) == CL_SUCCESS);
    log() << "Found " << num_devices << " devices on platform #" << USE_PLATFORM
          << ":" << std::endl;

    for (auto i = 0u; i < num_devices; ++i) {
      char device_name[255];
      clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(device_name),
                      device_name, nullptr);
      log() << "Device " << i << ": " << device_name << std::endl;
    }

    // Create device explicitly as the default ComputeCpp device selector
    // selects a faulty device (on this machine - psalz 2018/02/06).
    cl::sycl::device myDevice(devices[0]);

    //// =========== DEVICE SELECTION END ===============

    // TODO: How is device selected for distributed queue?
    // The choice of device is obviously important to the master node, so we
    // have to inform it about our choice somehow. NOTE: We only support a
    // single queue per worker process! The whole idea of CELERITY is that we
    // don't have to address multiple devices (with multiple queues) manually.
    // If a node has multiple devices available, it can start multiple worker
    // processes to make use of them.
    celerity::distr_queue queue(myDevice);

    // TODO: Do we support SYCL sub-buffers & images? Section 4.7.2
    celerity::buffer<float, 1> buf_a =
        queue.create_buffer(host_data_a, cl::sycl::range<1>(1024));
    celerity::buffer<float, 1> buf_b =
        queue.create_buffer(host_data_b, cl::sycl::range<1>(1024));
    celerity::buffer<float, 1> buf_c =
        queue.create_buffer(host_data_c, cl::sycl::range<1>(1024));
    celerity::buffer<float, 1> buf_d =
        queue.create_buffer(host_data_d, cl::sycl::range<1>(1024));

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
      auto a = buf_a.get_access<cl::sycl::access::mode::write>(
          cgh, [](celerity::subrange<1> range) -> celerity::subrange<1> {
            celerity::subrange<1> sr(range);
            // 1-neighborhood
            sr.start -= 1;
            sr.range += 1;
            return sr;
          });

      // NOTES:
      // * We don't support explicit work group sizes etc., this should be
      //   handled by the runtime. We only specify the global size.
      // * We only support a single kernel call per command group (not sure if
      //   this is also the case in SYCL; spec doesn't mention it explicitly).
      // TODO: SYCL parallel_for allows specification of an offset - look into
      cgh.template parallel_for<class produce_a>(cl::sycl::range<1>(1024),
                                                 [=](cl::sycl::item<1> item) {
                                                   auto i = item.get_id();
                                                   a[i] = 1.f;
                                                 });
    });

    // TODO: How do we deal with branching queues, depending on buffer contents?
#define EPSILON 1e-10
    // ** OPTION A ***:
    // This would require blocking on subscript access to ensure that all nodes
    // have the same value here. (This is similar to SYCL "host accessors").
    if (buf_a[0] > EPSILON) {
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
    queue.branch(
        [&](celerity::branch_handle& bh) { bh.get<float, 1>(buf_a, 256); },
        [](float error) {
          if (error > EPSILON) {
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
      auto a = buf_a.get_access<cl::sycl::access::mode::read>(
          cgh, celerity::access::one_to_one<1>());
      auto b = buf_b.get_access<cl::sycl::access::mode::write>(
          cgh, celerity::access::one_to_one<1>());
      cgh.template parallel_for<class compute_b>(cl::sycl::range<1>(1024),
                                                 [=](cl::sycl::item<1> item) {
                                                   auto i = item.get_id();
                                                   b[i] = a[i] * 2.f;
                                                 });
    });

    queue.submit([&](auto& cgh) {
      auto a = buf_a.get_access<cl::sycl::access::mode::read>(
          cgh, celerity::access::one_to_one<1>());
      auto c = buf_c.get_access<cl::sycl::access::mode::write>(
          cgh, celerity::access::one_to_one<1>());
      cgh.template parallel_for<class compute_c>(cl::sycl::range<1>(1024),
                                                 [=](cl::sycl::item<1> item) {
                                                   auto i = item.get_id();
                                                   c[i] = 2.f - a[i];
                                                 });
    });

    queue.submit([&](auto& cgh) {
      auto b = buf_b.get_access<cl::sycl::access::mode::read>(
          cgh, celerity::access::one_to_one<1>());
      auto c = buf_c.get_access<cl::sycl::access::mode::read>(
          cgh, celerity::access::one_to_one<1>());
      auto d = buf_d.get_access<cl::sycl::access::mode::write>(
          cgh, celerity::access::one_to_one<1>());
      cgh.template parallel_for<class compute_d>(cl::sycl::range<1>(1024),
                                                 [=](cl::sycl::item<1> item) {
                                                   auto i = item.get_id();
                                                   d[i] = b[i] + c[i];
                                                 });
    });

#if 0
    for (auto i = 0; i < 4; ++i) {
      queue.submit([&](auto& cgh) {
        auto c = buf_c.get_access<cl::sycl::access::mode::read>(
            cgh, celerity::access::one_to_one<1>());
        auto d = buf_d.get_access<cl::sycl::access::mode::write>(
            cgh, celerity::access::one_to_one<1>());
        cgh.template parallel_for<class compute_some_more>(
            cl::sycl::range<1>(1024), [=](cl::sycl::item<1> item) {
              auto i = item.get_id();
              d[i] = 2.f - c[i];
            });
      });
    }
#endif

    queue.debug_print_task_graph();

    log() << "EXECUTE DEFERRED" << std::endl;
    queue.TEST_execute_deferred();

    // In reality, this would be called periodically by a worker thread
    queue.build_command_graph();

  } catch (std::exception e) {
    std::cerr << e.what();
    return EXIT_FAILURE;
  }

  // ================= VERIFY CORRECTNESS ==================

  float sum = 0.f;
  for (int i = 0; i < 1024; ++i) {
    sum += host_data_d[i];
  }

  if (sum == 3072.f) {
    log() << "Success!" << std::endl;
  } else {
    log() << "Fail! Value is " << sum << std::endl;
  }
}
