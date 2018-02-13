#include <iostream>

#include "celerity_runtime.h"

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
  // TODO: Do we support SYCL sub-buffers & images? Section 4.7.2
  celerity::buffer buf_a(1024);
  celerity::buffer buf_b(1024);
  celerity::buffer buf_c(1024);
  celerity::buffer buf_d(1024);

  // TODO: How is device selected for distributed queue?
  // The choice of device is obviously important to the master node, so we have
  // to inform it about our choice somehow.
  // NOTE: We only support a single queue per worker process!
  // The whole idea of CELERITY is that we don't have to address multiple
  // devices (with multiple queues) manually. If a node has multiple devices
  // available, it can start multiple worker processes to make use of them.
  celerity::distr_queue queue;

  // **** COMMAND GROUPS ****
  // The functor/lambda submitted to a SYCL queue is called a "command group".
  // Command groups specify a set of "requisites" (or sometimes called
  // "requirements") for a particular kernel, i.e. what buffers need to be
  // available for reading/writing etc. It's (likely) designed that way so
  // the kernel lambdas can capture the accessors from the command group scope.
  //
  // See spec sections 3.4.1.2 and 4.8.2 for more info
  queue.submit([&](celerity::handler& cgh) {
    // Access-specifier scenario:
    // We have 2 worker nodes available, and 1000 work items
    // Then this functor is called twice:
    //    fn(nd_subrange(0, 500, 1000)), fn(nd_subrange(501, 999, 1000))
    // It returns the data range the kernel requires to compute it's result
    //
    // NOTE:
    // If returned nd_subrange is out of buffer bounds (e.g. offset - 1 is
    // undefined at first item in block [0, n]), it simply gets clamped to
    // the valid range and it is assumed that the user handles edge cases
    // correctly within the kernel.
    auto a = buf_a.get_access<cl::sycl::access::mode::write>(
        cgh, [](celerity::nd_subrange range) -> celerity::nd_subrange {
          return range;
        });

    // NOTES:
    // * We don't support explicit work group sizes etc., this should be handled
    //   by the runtime. We only specify the global size.
    // * We only support a single kernel call per command group (not sure if
    //   this is also the case in SYCL; spec doesn't mention it explicitly).
    // TODO: SYCL parallel_for allows specification of an offset - look into
    // TODO: First parameter (count) should be nd_range in general case
    cgh.parallel_for<class produce_a>(1024, [=](cl::sycl::nd_item item) {
      auto i = item.get_global();
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
  // Even more elaborate solution would make 2nd lambda a predicate, and provide
  // two additional lambdas containing the queue submit calls for both branches.
  queue.branch([&](celerity::branch_handle& bh) { bh.get<0>(buf_a); },
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

  queue.submit([&](celerity::handler& cgh) {
    auto a = buf_a.get_access<cl::sycl::access::mode::read>(
        cgh, celerity::access::one_to_one());
    auto b = buf_b.get_access<cl::sycl::access::mode::write>(
        cgh, celerity::access::one_to_one());
    cgh.parallel_for<class compute_b>(1024, [=](cl::sycl::nd_item item) {
      auto i = item.get_global();
      b[i] = a[i] * 2.f;
    });
  });

  queue.submit([&](celerity::handler& cgh) {
    auto a = buf_a.get_access<cl::sycl::access::mode::read>(
        cgh, celerity::access::one_to_one());
    auto c = buf_c.get_access<cl::sycl::access::mode::write>(
        cgh, celerity::access::one_to_one());
    cgh.parallel_for<class compute_c>(1204, [=](cl::sycl::nd_item item) {
      auto i = item.get_global();
      c[i] = 2.f - a[i];
    });
  });

  queue.submit([&](celerity::handler& cgh) {
    auto b = buf_b.get_access<cl::sycl::access::mode::read>(
        cgh, celerity::access::one_to_one());
    auto c = buf_c.get_access<cl::sycl::access::mode::read>(
        cgh, celerity::access::one_to_one());
    auto d = buf_d.get_access<cl::sycl::access::mode::write>(
        cgh, celerity::access::one_to_one());
    cgh.parallel_for<class compute_d>(1024, [=](cl::sycl::nd_item item) {
      auto i = item.get_global();
      d[i] = b[i] + c[i];
    });
  });

#if 0
  for (auto i = 0; i < 4; ++i) {
    queue.submit([&](celerity::handler& cgh) {
      auto c = buf_c.get_access<cl::sycl::access::mode::read>(cgh, celerity::access::one_to_one());
      auto d = buf_d.get_access<cl::sycl::access::mode::write>(cgh, celerity::access::one_to_one());
      cgh.parallel_for<class compute_some_more>(1024,
                                                [=](cl::sycl::nd_item item) {
                                                  auto i = item.get_global();
                                                  c[i] = 2.f - d[i];
                                                });
    });
  }
#endif

  //// ==============================================

  queue.debug_print_task_graph();
}
