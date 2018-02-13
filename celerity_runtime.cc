#include "celerity_runtime.h"

#include <iostream>

#include <boost/format.hpp>

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

handler::handler(distr_queue& q) : id(instance_count++), queue(q) {
  debug_name = (boost::format("task%d") % id).str();
}

void distr_queue::submit(std::function<void(handler& cgh)> cgf) {
  auto h = std::unique_ptr<handler>(new handler(*this));
  cgf(*h);
  handlers[h->get_id()] = std::move(h);
}

void distr_queue::debug_print_task_graph() {
  std::vector<std::string> names;
  for (auto i = 0; i < task_graph.vertex_set().size(); ++i) {
    names.push_back(
        (boost::format("Task %d (%s)") % i % handlers[i]->get_debug_name())
            .str());
  }
  write_graphviz(std::cout, task_graph, boost::make_label_writer(&names[0]));
}

void distr_queue::add_requirement(size_t task_id, size_t buffer_id,
                                  cl::sycl::access::mode mode) {
  auto mode_str = mode == cl::sycl::access::mode::write ? "WRITE" : "READ";
  std::cout << "// Task " << task_id << " wants " << mode_str
            << " access on buffer " << buffer_id << std::endl;

  // TODO: Check if edge already exists (avoid double edges)
  // TODO: If we have dependencies "A -> B, B -> C, A -> C", we could get rid of
  // "A -> C", as it is transitively implicit in "B -> C".
  if (buffer_last_writer.find(buffer_id) != buffer_last_writer.end()) {
    boost::add_edge(buffer_last_writer[buffer_id], task_id, task_graph);
  }
  if (mode == cl::sycl::access::mode::write) {
    buffer_last_writer[buffer_id] = task_id;
  }
};

}  // namespace celerity
