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

// (FIXME: These are explicitly instantiated for now to fix the circular
// dependency between handler and distr_queue)
template <>
void handler<is_prepass::true_t>::require(
    prepass_accessor<cl::sycl::access::mode::read> a, size_t buffer_id,
    range_mapper rm) {
  queue.add_requirement(task_id, buffer_id, cl::sycl::access::mode::read, rm);
}

template <>
void handler<is_prepass::true_t>::require(
    prepass_accessor<cl::sycl::access::mode::write> a, size_t buffer_id,
    range_mapper rm) {
  queue.add_requirement(task_id, buffer_id, cl::sycl::access::mode::write, rm);
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

// TODO: Initialize SYCL queue lazily
distr_queue::distr_queue(cl::sycl::device device) : sycl_queue(device) {
  // Include an additional node 0 (= master)
  for (auto i = 0; i < CELERITY_NUM_WORKER_NODES + 1; ++i) {
    nodes[i] = detail::node();
  }
}

void distr_queue::debug_print_task_graph() {
  auto num_vertices = task_graph.vertex_set().size();
  if (num_vertices == 0) {
    return;
  }
  for (size_t i = 0; i < num_vertices; ++i) {
    task_graph[i].label =
        (boost::format("Task %d (%s)") % (i + 1) % task_names[i + 1]).str();
  }
  task_graph[boost::graph_bundle].name = "TaskGraph";
  print_graph(task_graph);
}

void distr_queue::add_requirement(task_id tid, buffer_id bid,
                                  cl::sycl::access::mode mode,
                                  range_mapper rm) {
  auto mode_str = mode == cl::sycl::access::mode::write ? "WRITE" : "READ";
  // TODO: Check if edge already exists (avoid double edges)
  // TODO: If we have dependencies "A -> B, B -> C, A -> C", we could get rid of
  // "A -> C", as it is transitively implicit in "B -> C".
  if (buffer_last_writer.find(bid) != buffer_last_writer.end()) {
    boost::add_edge(buffer_last_writer[bid] - 1, tid - 1, task_graph);
  }
  if (mode == cl::sycl::access::mode::write) {
    buffer_last_writer[bid] = tid;
  }
  task_range_mappers[tid].insert(std::make_pair(bid, rm));
};

void distr_queue::TEST_execute_deferred() {
  for (auto& it : task_command_groups) {
    task_id tid = it.first;
    auto& cgf = it.second;
    sycl_queue.submit([this, &cgf, tid](cl::sycl::handler& sycl_handler) {
      handler<is_prepass::false_t> h(*this, tid, &sycl_handler);
      (*cgf)(h);
    });
  }
}

void distr_queue::build_command_graph() {
  // NOTE: This must work only with the information contained within the task
  // graph!

  // Potential commands:
  // - Move region (buffer_id, region, from_node_id, to_node_id)
  // - Compute subtask (task_id, work items (offset + size?), node_id)
  // - Complete task (task_id)
  //      This will cause the buffer version bump of all affected buffers

  // => Is it a reasonable requirement to have only "Complete task" leaf-nodes
  // in command graph? These could then act as synchronization points with the
  // task graph.

  if (active_tasks.size() == 0) {
    // TODO: Find all "root" tasks in task DAG

    std::cout << "Constructing CMD-DAG for task 1" << std::endl;
    auto& rms = task_range_mappers[1];
    std::cout << "Found " << rms.size() << " range mappers" << std::endl;
  }
}
}  // namespace celerity
