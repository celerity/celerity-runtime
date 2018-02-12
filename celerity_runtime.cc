#include "celerity_runtime.h"

#include <iostream>

namespace celerity {

// (Note: These are explicitly instantiated for now to fix the circular
// dependency between handler and distr_queue)
template <>
void handler::require(accessor<cl::sycl::access::mode::read> a) {
  queue.add_requirement(id, a.get_buffer().get_id(),
                        cl::sycl::access::mode::read);
}

template <>
void handler::require(accessor<cl::sycl::access::mode::write> a) {
  queue.add_requirement(id, a.get_buffer().get_id(),
                        cl::sycl::access::mode::write);
}

void distr_queue::submit(std::function<void(handler& cgh)> cgf) {
  handler h(*this);
  cgf(h);
}

void distr_queue::add_requirement(size_t task_id, size_t buffer_id,
                                  cl::sycl::access::mode mode) {
  auto mode_str = mode == cl::sycl::access::mode::write ? "WRITE" : "READ";
  std::cout << "Task " << task_id << " wants " << mode_str
            << " access on buffer " << buffer_id << std::endl;
};

}  // namespace celerity
