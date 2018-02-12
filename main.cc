#include <iostream>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graphviz.hpp>

#include "celerity_runtime.h"

namespace boost_graph {
using boost::adjacency_list;
using boost::bidirectionalS;
using boost::vecS;
}  // namespace boost_graph

const char* name[] = {"Produce Numbers", "Multiply by 3", "Add 5",
                      "Difference"};

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
  celerity::distr_queue queue;

  // TODO: Do we support multiple queues per node? SYCL certainly does allow it
  // => NO! The whole idea of CELERITY is that we don't have to address multiple
  // devices (with multiple queues) manually. If a node has multiple devices
  // available, it can start multiple worker processes to make use of them.

  // **** COMMAND GROUPS ****
  // The functor/lambda submitted to a SYCL queue is called a "command group".
  // Command groups specify a set of "requisites" (or sometimes called
  // "requirements") for a particular kernel, i.e. what buffers need to be
  // available for reading/writing etc.
  //
  // See spec sections 3.4.1.2 and 4.8.2 for more info
  //
  // TODO: Do we need the ability to reference command groups, or just kernels -
  // i.e. what gets the global ID?
  // If we don't re-run entire command groups on worker nodes, we have to
  // additionally encode buffer access modes within the command DAG / per-worker
  // command queues somehow - this might be unecessary overhead.
  queue.submit([&](celerity::handler& cgh) {
    // TODO: We may want to abstract command groups away altogether
    // According to spec, the command group is mainly intended as a
    // place to specify data requirements for the wrapped kernel.
    // 4.8.4 "SYCL functions for adding requirements" only lists one
    // function require(accessor).

    // Note that accessors can also be instantiated directly as
    // cl::sycl::access::accessor()
    auto a = buf_a.get_access<cl::sycl::access::mode::write>(cgh);

    // TODO: Figure out whether it is allowed to submit more than one kernel
    // per command group. ComputeCpp (v 0.5.1) doesn't allow it, TriSYCL does
    // (at least when running kernels on host). The spec doesn't seem to mention
    // it explicitly, but sec 3.2. mentions "all the requirements for *a* kernel
    // to execute are defined in this command group scope [...]".
    cgh.parallel_for<class produce_a>(
        1024, celerity::kernel_functor(
                  // General note on API: Why don't we specify access ranges
                  // beforehand (outside kernel call)?

                  // NOTE: "Range" is a bit of a misnomer here, as it actually
                  // specifies a point within our ND-buffers (which is why we
                  // need two different values, instead of a single "range").

                  // TODO: Do we also support explicit work group size etc.?
                  [&](celerity::range offset, celerity::range range) {
                    // a.access_range(offset, range);
                  },
                  [=](cl::sycl::nd_item item) {
                    auto i = item.get_global();
                    a[i] = 1.f;
                  }));
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
    auto a = buf_a.get_access<cl::sycl::access::mode::read>(cgh);
    auto b = buf_b.get_access<cl::sycl::access::mode::write>(cgh);
    cgh.parallel_for<class compute_b>(
        1024, celerity::kernel_functor(
                  [&](celerity::range offset, celerity::range range) {
                    // TODO
                  },
                  [=](cl::sycl::nd_item item) {
                    auto i = item.get_global();
                    b[i] = a[i] * 2.f;
                  }));
  });

  queue.submit([&](celerity::handler& cgh) {
    auto a = buf_a.get_access<cl::sycl::access::mode::read>(cgh);
    auto c = buf_b.get_access<cl::sycl::access::mode::write>(cgh);
    cgh.parallel_for<class compute_c>(
        1024, celerity::kernel_functor(
                  [&](celerity::range offset, celerity::range range) {
                    // TODO
                  },
                  [=](cl::sycl::nd_item item) {
                    auto i = item.get_global();
                    c[i] = 2.f - a[i];
                  }));
  });

  queue.submit([&](celerity::handler& cgh) {
    auto b = buf_b.get_access<cl::sycl::access::mode::read>(cgh);
    auto c = buf_c.get_access<cl::sycl::access::mode::read>(cgh);
    auto d = buf_d.get_access<cl::sycl::access::mode::write>(cgh);
    cgh.parallel_for<class compute_d>(
        1024, celerity::kernel_functor(
                  [&](celerity::range offset, celerity::range range) {
                    // TODO
                  },
                  [=](cl::sycl::nd_item item) {
                    auto i = item.get_global();
                    d[i] = b[i] + c[i];
                  }));
  });

  //// ==============================================

  using namespace boost_graph;

  typedef adjacency_list<vecS, vecS, bidirectionalS> Graph;
  typedef std::pair<int, int> Edge;

  Graph g(0);
  boost::add_edge(0, 1, g);
  boost::add_edge(0, 2, g);
  boost::add_edge(1, 3, g);
  boost::add_edge(2, 3, g);

  write_graphviz(std::cout, g, boost::make_label_writer(name));
}
