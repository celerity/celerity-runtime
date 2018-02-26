#include "celerity_runtime.h"

#include <iostream>
#include <queue>
#include <vector>

#include <allscale/utils/string_utils.h>
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
    std::unique_ptr<detail::range_mapper_base> rm) {
  queue.add_requirement(task_id, buffer_id, cl::sycl::access::mode::read,
                        std::move(rm));
}

template <>
void handler<is_prepass::true_t>::require(
    prepass_accessor<cl::sycl::access::mode::write> a, size_t buffer_id,
    std::unique_ptr<detail::range_mapper_base> rm) {
  queue.add_requirement(task_id, buffer_id, cl::sycl::access::mode::write,
                        std::move(rm));
}

handler<is_prepass::true_t>::~handler() {
  int dimensions = global_size.which() + 1;
  switch (dimensions) {
    case 1:
      queue.set_task_data(task_id, boost::get<cl::sycl::range<1>>(global_size),
                          debug_name);
      break;
    case 2:
      queue.set_task_data(task_id, boost::get<cl::sycl::range<2>>(global_size),
                          debug_name);
      break;
    case 3:
      queue.set_task_data(task_id, boost::get<cl::sycl::range<3>>(global_size),
                          debug_name);
      break;
    default:
      // Can't happen
      assert(false);
      break;
  }
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
distr_queue::distr_queue(cl::sycl::device device)
    : sycl_queue(device),
      // Include an additional node 0 (= master)
      num_nodes(CELERITY_NUM_WORKER_NODES + 1) {
  task_graph[boost::graph_bundle].name = "TaskGraph";
  command_graph[boost::graph_bundle].name = "CommandGraph";
}

void distr_queue::debug_print_task_graph() { print_graph(task_graph); }

void distr_queue::add_requirement(
    task_id tid, buffer_id bid, cl::sycl::access::mode mode,
    std::unique_ptr<detail::range_mapper_base> rm) {
  // TODO: Check if edge already exists (avoid double edges)
  // TODO: If we have dependencies "A -> B, B -> C, A -> C", we could get rid of
  // "A -> C", as it is transitively implicit in "B -> C".
  if (buffer_last_writer.find(bid) != buffer_last_writer.end()) {
    boost::add_edge(buffer_last_writer[bid], tid, task_graph);
  }
  if (mode == cl::sycl::access::mode::write) {
    buffer_last_writer[bid] = tid;
  }
  task_range_mappers[tid][bid].push_back(std::move(rm));
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

std::vector<subrange<1>> split_equal(const subrange<1>& sr, size_t num_splits) {
  subrange<1> split;
  split.global_size = sr.global_size;
  split.start = cl::sycl::range<1>(0);
  split.range = cl::sycl::range<1>(sr.range.size() / num_splits);

  std::vector<subrange<1>> result;
  for (auto i = 0u; i < num_splits; ++i) {
    result.push_back(split);
    split.start = split.start + split.range;
    if (i == num_splits - 1) {
      result[i].range += sr.range.size() % num_splits;
    }
  }
  return result;
}

void distr_queue::build_command_graph() {
  // Potential commands:
  // - Move region (buffer_id, region, from_node_id, to_node_id)
  // - Wait for pull (replace with PUSH later on?)
  // - Compute subtask (task_id, work items (offset + size?), node_id)
  // - Complete task (task_id)

  // => Is it a reasonable requirement to have only "Complete task" leaf-nodes
  // in command graph? These could then act as synchronization points with the
  // task graph.

  auto num_tasks = task_graph.vertex_set().size();
  std::set<task_id> queued_tasks;
  std::queue<task_id> task_queue;
  // FIXME: Assuming single root task 0
  task_queue.push(0);
  queued_tasks.insert(0);

  std::map<task_id, size_t> task_complete_vertices;

  while (!task_queue.empty()) {
    const task_id tid = task_queue.front();
    task_queue.pop();
    queued_tasks.erase(tid);
    auto& rms = task_range_mappers[tid];
    std::cout << "Task " << tid << " has range mappers for " << rms.size()
              << " buffers" << std::endl;

    auto begin_task_v = boost::add_vertex(command_graph);
    command_graph[begin_task_v].label =
        (boost::format("Begin Task %d") % tid).str();

    {
      // Find all tasks this task is dependending on
      // TODO: Move into separate function
      boost::graph_traits<Graph>::in_edge_iterator eit, eit_end;
      for (std::tie(eit, eit_end) = boost::in_edges(tid, task_graph);
           eit != eit_end; ++eit) {
        auto dep = boost::source(*eit, task_graph);
        boost::add_edge(task_complete_vertices[dep], begin_task_v,
                        command_graph);
      }
    }

    std::vector<size_t> split_compute_vertices;
    for (auto i = 0u; i < num_nodes - 1; ++i) {
      auto v = boost::add_vertex(command_graph);
      split_compute_vertices.push_back(v);
    }

    for (auto& it : rms) {
      buffer_id bid = it.first;
      // TODO: We have to distinguish between read and write range mappers!
      for (auto& rm : it.second) {
        // Outline (TODO)
        // For every requirement, determine the most suitable node
        //    => Add data pulls for all missing regions
        //    => Add wait-for-pulls on nodes that contain that data (but
        //    where do we add those in the DAG??)
        // For every split, add a execution command

        auto dims = rm->get_dimensions();
        if (dims != 1) {
          throw new std::runtime_error("2D/3D splits NYI");
        }

        auto sr = subrange<1>();
        // FIXME: Assuming task has same dimensionality
        sr.global_size = boost::get<cl::sycl::range<1>>(task_global_sizes[tid]);
        sr.range = sr.global_size;
        auto splits = split_equal(sr, num_nodes - 1);
        std::vector<subrange<1>> reqs(splits.size());
        std::transform(splits.cbegin(), splits.cend(), reqs.begin(),
                       [&rm](auto& split) { return (*rm)(split); });

        for (auto i = 0u; i < splits.size(); ++i) {
          auto v = boost::add_vertex(command_graph);
          boost::add_edge(begin_task_v, v, command_graph);
          boost::add_edge(v, split_compute_vertices[i], command_graph);
          // TODO: Find suitable source(s) node for pull
          //   => This may mean we'll have multiple pulls or non at all
          // TODO: Add wait-for-pull command in source node
          command_graph[v].label =
              (boost::format("Node %d:\\nPULL %d\\n %s") % (i + 1) % bid %
               toString(detail::subrange_to_grid_region(reqs[i])))
                  .str();

          // FIXME: No need to set this on every range-mapper iteration
          // (We just don't have the split available above)
          command_graph[split_compute_vertices[i]].label =
              (boost::format("Node %d:\\nCompute\\n%s") % (i + 1) %
               toString(detail::subrange_to_grid_region(splits[i])))
                  .str();
        }
      }
    }

    auto complete_task_v = boost::add_vertex(command_graph);
    task_complete_vertices[tid] = complete_task_v;
    command_graph[complete_task_v].label =
        (boost::format("Complete Task %d") % tid).str();
    for (auto& v : split_compute_vertices) {
      boost::add_edge(v, complete_task_v, command_graph);
    }

    {
      // Find all tasks depending on this task
      // TODO: Move into separate function
      boost::graph_traits<Graph>::out_edge_iterator eit, eit_end;
      // FIXME: This doesn't check if the task has other unresolved
      // dependencies!!
      for (std::tie(eit, eit_end) = boost::out_edges(tid, task_graph);
           eit != eit_end; ++eit) {
        auto tid = boost::target(*eit, task_graph);
        if (queued_tasks.find(tid) == queued_tasks.end()) {
          task_queue.push(tid);
          queued_tasks.insert(tid);
        }
      }
    }
  }

  print_graph(command_graph);
}  // namespace celerity
}  // namespace celerity
