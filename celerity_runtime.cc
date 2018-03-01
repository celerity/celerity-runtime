#include "celerity_runtime.h"

#include <iostream>
#include <queue>
#include <vector>

#include <allscale/utils/string_utils.h>
#include <boost/algorithm/string.hpp>

template <typename Graph>
void print_graph(const Graph& g) {
  std::stringstream ss;
  write_graphviz(ss, g,
                 boost::make_label_writer(
                     boost::get(&Graph::vertex_property_type::label, g)));
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

// TODO: Move elsewhere
namespace graph_utils {
using task_vertices = std::pair<vertex, vertex>;

template <typename Graph, typename Functor>
void for_predecessors(const Graph& graph, vertex v, Functor f) {
  boost::graph_traits<Graph>::in_edge_iterator eit, eit_end;
  for (std::tie(eit, eit_end) = boost::in_edges(v, graph); eit != eit_end;
       ++eit) {
    vertex pre = boost::source(*eit, graph);
    f(pre);
  }
}

template <typename Graph, typename Functor>
void for_successors(const Graph& graph, vertex v, Functor f) {
  boost::graph_traits<Graph>::out_edge_iterator eit, eit_end;
  for (std::tie(eit, eit_end) = boost::out_edges(v, graph); eit != eit_end;
       ++eit) {
    vertex suc = boost::target(*eit, graph);
    f(suc);
  }
}

// Note that we don't check whether the edge u->v actually existed
template <typename Graph>
vertex insert_vertex_on_edge(vertex u, vertex v, Graph& graph) {
  auto e = boost::edge(u, v, graph);
  auto w = boost::add_vertex(graph);
  boost::remove_edge(u, v, graph);
  boost::add_edge(u, w, graph);
  boost::add_edge(w, v, graph);
  return w;
}

task_vertices add_task(task_id tid, const task_dag& tdag, command_dag& cdag) {
  vertex begin_task_v = boost::add_vertex(cdag);
  cdag[begin_task_v].label = (boost::format("Begin Task %d") % tid).str();

  // Add all task requirements
  for_predecessors(tdag, tid, [&cdag, begin_task_v](vertex requirement) {
    boost::add_edge(
        cdag[boost::graph_bundle].task_complete_vertices[requirement],
        begin_task_v, cdag);
  });

  vertex complete_task_v = boost::add_vertex(cdag);
  cdag[boost::graph_bundle].task_complete_vertices[tid] = complete_task_v;
  cdag[complete_task_v].label = (boost::format("Complete Task %d") % tid).str();

  return task_vertices(begin_task_v, complete_task_v);
}

template <int Dims>
vertex add_compute_cmd(node_id nid, const task_vertices& tv,
                       const subrange<Dims>& sr, command_dag& cdag) {
  auto v = boost::add_vertex(cdag);
  boost::add_edge(tv.first, v, cdag);
  boost::add_edge(v, tv.second, cdag);
  cdag[v].label = (boost::format("Node %d:\\nCompute\\n%s") % nid %
                   toString(detail::subrange_to_grid_region(sr)))
                      .str();
  return v;
}

template <int Dims>
vertex add_pull_cmd(node_id nid, buffer_id bid, const task_vertices& tv,
                    vertex compute_cmd, const subrange<Dims>& req,
                    command_dag& cdag) {
  // TODO: Once we have actual commands attached to vertices, assert compute_cmd
  auto v = graph_utils::insert_vertex_on_edge(tv.first, compute_cmd, cdag);
  cdag[v].label = (boost::format("Node %d:\\nPULL %d\\n %s") % nid % bid %
                   toString(detail::subrange_to_grid_region(req)))
                      .str();
  return v;
}

}  // namespace graph_utils

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

  while (!task_queue.empty()) {
    const task_id tid = task_queue.front();
    task_queue.pop();
    queued_tasks.erase(tid);
    auto& rms = task_range_mappers[tid];

    graph_utils::task_vertices taskv =
        graph_utils::add_task(tid, task_graph, command_graph);

    // Split task into equal chunks for every compute node
    // TODO: In the future, we may want to adjust our split based on the range
    // mapper results and data location!
    auto sr = subrange<1>();
    // FIXME: We assume task dimensionality 1 here
    sr.global_size = boost::get<cl::sycl::range<1>>(task_global_sizes[tid]);
    sr.range = sr.global_size;
    auto splits = split_equal(sr, num_nodes - 1);

    std::vector<vertex> split_compute_vertices;
    for (auto i = 0u; i < splits.size(); ++i) {
      auto cv =
          graph_utils::add_compute_cmd(i + 1, taskv, splits[i], command_graph);
      split_compute_vertices.push_back(cv);
    }

    for (auto& it : rms) {
      buffer_id bid = it.first;

      for (auto& rm : it.second) {
        auto dims = rm->get_dimensions();
        if (dims != 1) {
          throw new std::runtime_error("2D/3D splits NYI");
        }

        std::vector<subrange<1>> reqs(splits.size());
        std::transform(splits.cbegin(), splits.cend(), reqs.begin(),
                       [&rm](auto& split) { return (*rm)(split); });

        // **************** NEXT STEPS **************
        // [x] Distinguish read and write access
        //     Write access is only relevant for "await pull" commands (these
        //     should come before the write)
        // [ ] Figure out from which node to pull
        // [ ] Insert "await pull" commands
        // [ ] Try to execute the sub-ranges only with data from the CMD-DA
        //     (If it's not much work, consider using sub-buffers for the
        //     per-node buffer ranges!) (pulls are no-ops obviously, but we need
        //     for example the exact execution ranges within the DAG)

        if (rm->get_access_mode() == cl::sycl::access::mode::write) continue;

        for (auto i = 0u; i < splits.size(); ++i) {
          // TODO: Find suitable source(s) node for pull
          //   => This may mean we'll have multiple pulls or none at all
          // TODO: Add wait-for-pull command in source node
          //   => How do we determine WHERE to add that command?
          //      Source node might not (yet) have a command subgraph at this
          //      point
          graph_utils::add_pull_cmd(i + 1, bid, taskv,
                                    split_compute_vertices[i], reqs[i],
                                    command_graph);
        }
      }
    }

    // Find all tasks depending on this task
    graph_utils::for_successors(
        task_graph, tid, [&task_queue, &queued_tasks](vertex successor) {
          // FIXME: This doesn't check if the task has other unresolved
          // dependencies!!
          if (queued_tasks.find(successor) == queued_tasks.end()) {
            task_queue.push(successor);
            queued_tasks.insert(successor);
          }
        });
  }

  print_graph(command_graph);
}  // namespace celerity
}  // namespace celerity
