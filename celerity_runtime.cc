#include "celerity_runtime.h"

#include <iostream>
#include <vector>

#include <boost/algorithm/string.hpp>

void print_graph(boost_graph::Graph& g) {
  using namespace boost_graph;
  std::stringstream ss;
  write_graphviz(
      ss, g,
      boost::make_label_writer(boost::get(&vertex_properties::label, g)));
  auto str = ss.str();
  std::vector<std::string> lines;
  boost::split(lines, str, boost::is_any_of("\n"));
  auto graph_name = g[boost::graph_bundle].name;
  for (auto l : lines) {
    std::cout << "#G:" << graph_name << "#" << l << std::endl;
  }
}

namespace celerity {

size_t buffer::instance_count = 0;

// (Note: These are explicitly instantiated for now to fix the circular
// dependency between handler and distr_queue)
template <>
void handler<is_prepass::true_t>::require(
    prepass_accessor<cl::sycl::access::mode::read> a, size_t buffer_id) {
  queue.add_requirement(task_id, buffer_id, cl::sycl::access::mode::read);
}

template <>
void handler<is_prepass::true_t>::require(
    prepass_accessor<cl::sycl::access::mode::write> a, size_t buffer_id) {
  queue.add_requirement(task_id, buffer_id, cl::sycl::access::mode::write);
}

template <>
void handler<is_prepass::false_t>::require(
    accessor<cl::sycl::access::mode::read> a, size_t buffer_id) {
  // TODO: This is where data dependencies would be resolved (PULL)
}

template <>
void handler<is_prepass::false_t>::require(
    accessor<cl::sycl::access::mode::write> a, size_t buffer_id) {
  // TODO: This is where data dependencies would be resolved (PULL)
}

distr_queue::distr_queue(cl::sycl::device device) : sycl_queue(device) {}

void distr_queue::debug_print_task_graph() {
  auto num_vertices = task_graph.vertex_set().size();
  if (num_vertices == 0) {
    // Write empty graph
    write_graphviz(std::cout, task_graph);
    return;
  }
  for (size_t i = 0; i < num_vertices; ++i) {
    task_graph[i].label =
        (boost::format("Task %d (%s)") % i % task_names[i]).str();
  }
  task_graph[boost::graph_bundle].name = "TaskGraph";
  print_graph(task_graph);
}

void distr_queue::add_requirement(task_id id, size_t buffer_id,
                                  cl::sycl::access::mode mode) {
  auto mode_str = mode == cl::sycl::access::mode::write ? "WRITE" : "READ";
  // TODO: Check if edge already exists (avoid double edges)
  // TODO: If we have dependencies "A -> B, B -> C, A -> C", we could get rid of
  // "A -> C", as it is transitively implicit in "B -> C".
  if (buffer_last_writer.find(buffer_id) != buffer_last_writer.end()) {
    boost::add_edge(buffer_last_writer[buffer_id], id, task_graph);
  }
  if (mode == cl::sycl::access::mode::write) {
    buffer_last_writer[buffer_id] = id;
  }
};

void distr_queue::TEST_execute_deferred() {
  for (size_t i = 0; i < task_command_groups.size(); ++i) {
    auto& cgf = task_command_groups[i];
    sycl_queue.submit([this, &cgf, i](cl::sycl::handler& sycl_handler) {
      handler<is_prepass::false_t> h(*this, i, &sycl_handler);
      (*cgf)(h);
    });
  }
}

void distr_queue::build_command_graph() {
  // NOTE: This must work only with the information contained within the task
  // graph!
}
}  // namespace celerity
