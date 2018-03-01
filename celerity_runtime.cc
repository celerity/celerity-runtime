#include "celerity_runtime.h"

#include <iostream>
#include <queue>
#include <type_traits>
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
  if (mode == cl::sycl::access::mode::read) {
    if (buffer_last_writer.find(bid) != buffer_last_writer.end()) {
      boost::add_edge(buffer_last_writer[bid], tid, task_graph);
      task_graph[tid].num_unsatisfied++;
    }
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

std::vector<subrange<1>> split_equal(const subrange<1>& sr, size_t num_chunks) {
  subrange<1> chunk;
  chunk.global_size = sr.global_size;
  chunk.start = cl::sycl::range<1>(0);
  chunk.range = cl::sycl::range<1>(sr.range.size() / num_chunks);

  std::vector<subrange<1>> result;
  for (auto i = 0u; i < num_chunks; ++i) {
    result.push_back(chunk);
    chunk.start = chunk.start + chunk.range;
    if (i == num_chunks - 1) {
      result[i].range += sr.range.size() % num_chunks;
    }
  }
  return result;
}

// TODO: Move elsewhere
namespace graph_utils {
using task_vertices = std::pair<vertex, vertex>;

template <typename Functor>
bool call_for_vertex_fn(const Functor& fn, vertex v, std::true_type) {
  return fn(v);
}

template <typename Functor>
bool call_for_vertex_fn(const Functor& fn, vertex v, std::false_type) {
  fn(v);
  return true;
}

/*
 * Calls a functor on every predecessor of vertex v within the graph.
 * The functor can optionally return a boolean indicating whether the
 * loop should abort.
 *
 * Returns false if the loop was aborted.
 */
template <typename Graph, typename Functor>
bool for_predecessors(const Graph& graph, vertex v, const Functor& f) {
  typename boost::graph_traits<Graph>::in_edge_iterator eit, eit_end;
  for (std::tie(eit, eit_end) = boost::in_edges(v, graph); eit != eit_end;
       ++eit) {
    vertex pre = boost::source(*eit, graph);
    if (call_for_vertex_fn(f, pre, std::is_same<bool, decltype(f(pre))>()) ==
        false) {
      return false;
    }
  }
  return true;
}

/*
 * Calls a functor on every successor of vertex v within the graph.
 * The functor can optionally return a boolean indicating whether the
 * loop should abort.
 *
 * Returns false if the loop was aborted.
 */
template <typename Graph, typename Functor>
bool for_successors(const Graph& graph, vertex v, const Functor& f) {
  typename boost::graph_traits<Graph>::out_edge_iterator eit, eit_end;
  for (std::tie(eit, eit_end) = boost::out_edges(v, graph); eit != eit_end;
       ++eit) {
    vertex suc = boost::target(*eit, graph);
    if (call_for_vertex_fn(f, suc, std::is_same<bool, decltype(f(suc))>()) ==
        false) {
      return false;
    }
  }
  return true;
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
                       const subrange<Dims>& chunk, command_dag& cdag) {
  auto v = boost::add_vertex(cdag);
  boost::add_edge(tv.first, v, cdag);
  boost::add_edge(v, tv.second, cdag);
  cdag[v].label = (boost::format("Node %d:\\nCompute\\n%s") % nid %
                   toString(detail::subrange_to_grid_region(chunk)))
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

/**
 * Returns a set of tasks that
 *  (1) have all their requirements satisfied (i.e., all predecessors are
 *      marked as processed)
 *  (2) don't have any unsatisfied siblings.
 *
 *  Note that "siblingness" can be transitive, meaning that not every pair
 *  of returned tasks necessarily has common parents. All siblings are
 *  however connected through some child->parent->child->[...] chain.
 */
std::vector<task_id> get_satisfied_sibling_set(const task_dag& tdag) {
  for (auto v : tdag.vertex_set()) {
    if (tdag[v].processed || tdag[v].num_unsatisfied > 0) continue;

    std::unordered_set<task_id> checked_predecessors;
    std::unordered_set<task_id> candidates;
    std::queue<task_id> unchecked_siblings;
    candidates.insert(v);
    unchecked_siblings.push(v);

    bool abort = false;
    while (!abort && !unchecked_siblings.empty()) {
      task_id sib = unchecked_siblings.front();
      unchecked_siblings.pop();

      abort = !for_predecessors(tdag, sib, [&](vertex pre) {
        if (!tdag[pre].processed) return false;
        if (checked_predecessors.find(pre) != checked_predecessors.end()) {
          return true;
        }
        checked_predecessors.insert(pre);

        abort = !for_successors(tdag, pre, [&](vertex suc) {
          if (candidates.find(suc) == candidates.end()) {
            if (tdag[suc].processed || tdag[suc].num_unsatisfied > 0) {
              return false;
            }
            candidates.insert(suc);
            unchecked_siblings.push(suc);
          }
          return true;
        });

        // abort if v has unsatisfied sibling
        return !abort;
      });
    }

    if (!abort) {
      std::vector<task_id> result;
      result.insert(result.end(), candidates.begin(), candidates.end());
      return result;
    }
  }

  return std::vector<task_id>();
}

void mark_as_processed(task_id tid, task_dag& tdag) {
  tdag[tid].processed = true;
  for_successors(tdag, tid, [&tdag](vertex suc) {
    assert(tdag[suc].num_unsatisfied >= 1);
    tdag[suc].num_unsatisfied--;
  });
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
  auto sibling_set = graph_utils::get_satisfied_sibling_set(task_graph);
  std::cout << "Found sibling set of size " << sibling_set.size() << ": [ ";
  for (auto s : sibling_set) {
    std::cout << s << " ";
  }
  std::cout << " ]" << std::endl;

  for (task_id tid : sibling_set) {
    auto& rms = task_range_mappers[tid];

    const graph_utils::task_vertices taskv =
        graph_utils::add_task(tid, task_graph, command_graph);

    // Split task into equal chunks for every compute node
    // TODO: In the future, we may want to adjust our split based on the range
    // mapper results and data location!
    auto sr = subrange<1>();
    // FIXME: We assume task dimensionality 1 here
    sr.global_size = boost::get<cl::sycl::range<1>>(task_global_sizes[tid]);
    sr.range = sr.global_size;
    auto chunks = split_equal(sr, num_nodes - 1);

    std::vector<vertex> chunk_compute_vertices;
    for (auto i = 0u; i < chunks.size(); ++i) {
      auto cv =
          graph_utils::add_compute_cmd(i + 1, taskv, chunks[i], command_graph);
      chunk_compute_vertices.push_back(cv);
    }

    // Things to do / consider for buffer region handling:
    // * For every buffer and chunk, make union region of all reads and writes
    // * Use per-chunk read-region to determine most suitable node to compute on
    //   (i.e., the node with the least amount of data transfers required)
    //   Possible initial simplifications:
    //    - First-come-first serve, i.e. assign a chunk to the most suitable
    //      node even though a later chunk might be even better on that
    //      particular node
    //    - Distribute every task over all nodes (which isn't necessarily the
    //      best strategy, e.g. it might be better to assign two entire tasks to
    //      two nodes)
    // * Update buffer regions based on per-chunk write-region
    //
    //   ISSUE: We cannot update the regions after processing a single task,
    //   since sibling tasks might depend on previous data locations (unless
    //   tasks are processed in the same order they are submitted to the
    //   distributed queue, which is a restriction we may not want to have ->
    //   investigate). This means we can only update after a COMPLETE set of
    //   sibling tasks has been processed. This in turn means that we can only
    //   start processing a set of siblings, if we can ensure that all
    //   requirement tasks for those siblings have already been fulfilled! So
    //   the choice of where to continue processing the task graph can certainly
    //   be important!

    for (auto& it : rms) {
      buffer_id bid = it.first;

      for (auto& rm : it.second) {
        auto dims = rm->get_dimensions();
        if (dims != 1) {
          throw std::runtime_error("2D/3D splits NYI");
        }

        std::vector<subrange<1>> reqs(chunks.size());
        std::transform(chunks.cbegin(), chunks.cend(), reqs.begin(),
                       [&rm](auto& chunk) { return (*rm)(chunk); });

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

        for (auto i = 0u; i < chunks.size(); ++i) {
          // TODO: Find suitable source(s) node for pull
          //   => This may mean we'll have multiple pulls or none at all
          // TODO: Add wait-for-pull command in source node
          //   => How do we determine WHERE to add that command?
          //      Source node might not (yet) have a command subgraph at this
          //      point
          graph_utils::add_pull_cmd(i + 1, bid, taskv,
                                    chunk_compute_vertices[i], reqs[i],
                                    command_graph);
        }
      }
    }

    graph_utils::mark_as_processed(tid, task_graph);
  }

  // HACK: We recursively call this until all tasks have been processed
  // In the future, we may want to do this periodically in a worker thread
  if (graph_utils::get_satisfied_sibling_set(task_graph).size() > 0) {
    build_command_graph();
  } else {
    print_graph(command_graph);
  }
}  // namespace celerity
}  // namespace celerity
