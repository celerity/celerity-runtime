#include "celerity_runtime.h"

#include <iostream>
#include <queue>
#include <type_traits>
#include <vector>

#include <allscale/utils/string_utils.h>
#include <boost/algorithm/string.hpp>
#include <boost/graph/breadth_first_search.hpp>

template <typename Graph, typename VertexPropertiesWriter,
          typename EdgePropertiesWriter>
void write_graph_mux(const Graph& g, VertexPropertiesWriter vpw,
                     EdgePropertiesWriter epw) {
  std::stringstream ss;
  write_graphviz(ss, g, vpw, epw);
  auto str = ss.str();
  std::vector<std::string> lines;
  boost::split(lines, str, boost::is_any_of("\n"));
  auto graph_name = g[boost::graph_bundle].name;
  for (auto l : lines) {
    std::cout << "#G:" << graph_name << "#" << l << std::endl;
  }
}

void print_graph(const celerity::task_dag& tdag) {
  write_graph_mux(tdag,
                  boost::make_label_writer(boost::get(
                      &celerity::tdag_vertex_properties::label, tdag)),
                  boost::default_writer());
}

void print_graph(const celerity::command_dag& cdag) {
  using namespace celerity;
  write_graph_mux(cdag,
                  [&](std::ostream& out, vertex v) {
                    const char* colors[] = {"black",       "crimson",
                                            "dodgerblue4", "goldenrod",
                                            "maroon4",     "springgreen2",
                                            "tan1",        "chartreuse2"};

                    std::unordered_map<std::string, std::string> props;
                    props["label"] = boost::escape_dot_string(cdag[v].label);

                    props["fontcolor"] = colors[cdag[v].nid % sizeof(colors)];

                    switch (cdag[v].cmd) {
                      case cdag_command::NOP:
                        props["color"] = "gray50";
                        props["fontcolor"] = "gray50";
                        break;
                      case cdag_command::COMPUTE:
                        props["shape"] = "box";
                        break;
                      default:
                        break;
                    }

                    out << "[";
                    for (auto it : props) {
                      out << " " << it.first << "=" << it.second;
                    }
                    out << "]";
                  },
                  [&](std::ostream& out, auto e) {
                    vertex v0 = boost::source(e, cdag);
                    vertex v1 = boost::target(e, cdag);
                    if ((cdag[v0].cmd == cdag_command::PULL ||
                         cdag[v0].cmd == cdag_command::WAIT_FOR_PULL) &&
                        (cdag[v1].cmd == cdag_command::PULL ||
                         cdag[v1].cmd == cdag_command::WAIT_FOR_PULL)) {
                      out << "[color=gray50]";
                    }
                  });
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
  const int dimensions = global_size.which() + 1;
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
    const task_id tid = it.first;
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
  const auto e = boost::edge(u, v, graph);
  const auto w = boost::add_vertex(graph);
  boost::remove_edge(u, v, graph);
  boost::add_edge(u, w, graph);
  boost::add_edge(w, v, graph);
  return w;
}

class abort_search_exception : public std::runtime_error {
 public:
  abort_search_exception()
      : std::runtime_error("Abort search (not an error)") {}
};

template <typename Functor>
class bfs_visitor : public boost::default_bfs_visitor {
 public:
  bfs_visitor(Functor f) : f(f) {}

  template <typename Graph>
  void discover_vertex(vertex v, const Graph& graph) const {
    if (f(v, graph) == true) {
      throw abort_search_exception();
    }
  }

 private:
  Functor f;
};

/*
 * Search vertices using a breadth-first-search.
 * The functor receives the current vertex as well as the graph by reference.
 * The search is aborted if the functor returns true.
 */
template <typename Graph, typename Functor>
void search_vertex_bf(vertex start, const Graph& graph, Functor f) {
  try {
    bfs_visitor<Functor> vis(f);
    boost::breadth_first_search(graph, start, boost::visitor(vis));
  } catch (abort_search_exception&) {
    // Nop
  }
}

task_vertices add_task(task_id tid, const task_dag& tdag, command_dag& cdag) {
  const vertex begin_task_v = boost::add_vertex(cdag);
  cdag[begin_task_v].label =
      (boost::format("Begin %s") % tdag[tid].label).str();

  // Add all task requirements
  for_predecessors(tdag, tid, [&cdag, begin_task_v](vertex requirement) {
    boost::add_edge(
        cdag[boost::graph_bundle].task_complete_vertices[requirement],
        begin_task_v, cdag);
  });

  const vertex complete_task_v = boost::add_vertex(cdag);
  cdag[boost::graph_bundle].task_complete_vertices[tid] = complete_task_v;
  cdag[complete_task_v].label =
      (boost::format("Complete %s") % tdag[tid].label).str();

  return task_vertices(begin_task_v, complete_task_v);
}

template <int Dims>
vertex add_compute_cmd(node_id nid, const task_vertices& tv,
                       const subrange<Dims>& chunk, command_dag& cdag) {
  const auto v = boost::add_vertex(cdag);
  boost::add_edge(tv.first, v, cdag);
  boost::add_edge(v, tv.second, cdag);
  cdag[v].cmd = cdag_command::COMPUTE;
  cdag[v].nid = nid;
  cdag[v].label = (boost::format("Node %d:\\COMPUTE %s") % nid %
                   toString(detail::subrange_to_grid_region(chunk)))
                      .str();
  return v;
}

template <int Dims>
vertex add_pull_cmd(node_id nid, node_id source_nid, buffer_id bid,
                    const task_vertices& tv, const task_vertices& source_tv,
                    vertex compute_cmd, const GridBox<Dims>& req,
                    command_dag& cdag) {
  assert(cdag[compute_cmd].cmd == cdag_command::COMPUTE);
  const auto v =
      graph_utils::insert_vertex_on_edge(tv.first, compute_cmd, cdag);
  cdag[v].cmd = cdag_command::PULL;
  cdag[v].nid = nid;
  cdag[v].label = (boost::format("Node %d:\\nPULL %d from %d\\n %s") % nid %
                   bid % source_nid % toString(req))
                      .str();

  // Find the compute command for the source node in the writing task (or this
  // task, if no writing task has been found)
  vertex source_compute_v = 0;
  search_vertex_bf(
      source_tv.first, cdag,
      [source_nid, source_tv, &source_compute_v](vertex v,
                                                 const command_dag& cdag) {
        if (cdag[v].cmd == cdag_command::COMPUTE && cdag[v].nid == source_nid) {
          source_compute_v = v;
          return true;
        }
        return false;
      });
  assert(source_compute_v != 0);

  const auto w = graph_utils::insert_vertex_on_edge(source_tv.first,
                                                    source_compute_v, cdag);
  cdag[w].cmd = cdag_command::WAIT_FOR_PULL;
  cdag[w].nid = source_nid;
  cdag[w].label = (boost::format("Node %d:\\nWAIT FOR PULL %d by %d\\n %s") %
                   source_nid % bid % nid % toString(req))
                      .str();

  // Add edges in both directions
  boost::add_edge(w, v, cdag);
  boost::add_edge(v, w, cdag);

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
      const task_id sib = unchecked_siblings.front();
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

/**
 * Computes a command graph from the task graph, in batches of sibling sets.
 *
 * This currently (= likely to change in the future!) works as follows:
 *
 * 1) Obtain a suitable satisfied sibling set from the task graph.
 * 2) For every task within that sibling set:
 *    a) Split the task into equally sized chunks.
 *    b) Obtain all range mappers for that task and iterate over them,
 *       determining the read and write regions for every chunk. Note that a
 *       task may contain several read/write accessors for the same buffer,
 *       which is why we first have to compute their union regions.
 *    c) Iterate over all per-chunk read regions and try to find the most
 *       suitable node to execute that chunk on, i.e. the node that requires
 *       the least amount of data-transfer in order to execute that chunk.
 *       Note that currently only the first read buffer is considered, and
 *       nodes are assigned greedily.
 *    d) Insert compute commands for every node into the command graph.
 *       It is important to create these before pull-commands are inserted
 *       (see below).
 *    e) Iterate over per-chunk reads & writes to (i) store per-buffer per-node
 *       written regions and (ii) create pull / wait-for-pull commands for
 *       all nodes, inserting them as requirements for their respective
 *       compute commands. If no task in the sibling set writes to a specific
 *       buffer, the wait-for-pull command for that buffer will be inserted in
 *       the command subgraph for the current task (which is why it's important
 *       that all compute commands already exist).
 * 3) Finally, all per-buffer per-node written regions are used to update the
 *    data structure that keeps track of valid buffer regions.
 */
void distr_queue::build_command_graph() {
  using chunk_id = size_t;

  auto sibling_set = graph_utils::get_satisfied_sibling_set(task_graph);
  std::sort(sibling_set.begin(), sibling_set.end());

  std::unordered_map<task_id, graph_utils::task_vertices> taskvs;

  // FIXME: Dimensions. Also, containers much??
  std::unordered_map<
      buffer_id, std::unordered_map<
                     node_id, std::vector<std::pair<task_id, GridRegion<1>>>>>
      buffer_writers;

  // Iterate over tasks in reverse order so we can determine kernels which
  // write to certain buffer ranges before generating the pull commands for
  // those ranges, which allows us to insert "wait-for-pull"s before writes.
  for (auto it = sibling_set.crbegin(); it != sibling_set.crend(); ++it) {
    const task_id tid = *it;
    const auto& rms = task_range_mappers[tid];

    taskvs[tid] = graph_utils::add_task(tid, task_graph, command_graph);

    // Split task into equal chunks for every worker node
    // TODO: In the future, we may want to adjust our split based on the range
    // mapper results and data location!
    auto sr = subrange<1>();
    // FIXME: We assume task dimensionality 1 here
    sr.global_size = boost::get<cl::sycl::range<1>>(task_global_sizes[tid]);
    sr.range = sr.global_size;
    auto chunks = split_equal(sr, num_nodes - 1);

    // FIXME: Dimensions
    std::unordered_map<
        chunk_id,
        std::unordered_map<buffer_id, std::unordered_map<cl::sycl::access::mode,
                                                         GridRegion<1>>>>
        chunk_reqs;

    for (auto& it : rms) {
      const buffer_id bid = it.first;

      for (auto& rm : it.second) {
        auto mode = rm->get_access_mode();
        assert(mode == cl::sycl::access::mode::read ||
               mode == cl::sycl::access::mode::write);

        for (auto i = 0u; i < chunks.size(); ++i) {
          // FIXME: Dimensions
          subrange<1> req = (*rm)(chunks[i]);
          chunk_reqs[i][bid][mode] = GridRegion<1>::merge(
              chunk_reqs[i][bid][mode], detail::subrange_to_grid_region(req));
        }
      }
    }

    std::unordered_map<chunk_id, node_id> chunk_nodes;
    std::unordered_set<node_id> free_nodes;
    for (auto i = 1u; i < num_nodes; ++i) {
      free_nodes.insert(i);
    }

    // FIXME: Dimensions
    std::unordered_map<
        chunk_id,
        std::unordered_map<
            buffer_id,
            std::vector<std::pair<GridBox<1>, std::unordered_set<node_id>>>>>
        chunk_buffer_sources;

    // Find per-chunk per-buffer sources and assign nodes to chunks
    for (auto i = 0u; i < chunks.size(); ++i) {
      bool node_assigned = false;
      node_id nid = 0;

      for (auto& it : chunk_reqs[i]) {
        const buffer_id bid = it.first;
        const auto& read_req = it.second[cl::sycl::access::mode::read];

        // FIXME Dimensions
        auto bs = dynamic_cast<detail::buffer_state<1>*>(
            valid_buffer_regions[bid].get());

        const auto sn = bs->get_source_nodes(read_req);
        chunk_buffer_sources[i][bid] = sn;

        if (!node_assigned) {
          assert(free_nodes.size() > 0);

          // If the chunk doesn't have any read requirements (for this buffer!),
          // we also won't get any source nodes
          if (sn.size() > 0) {
            const auto& source_nodes = sn[0].second;

            // We simply pick the first node that contains the largest chunk of
            // the first requested buffer, given it is still available.
            // Otherwise we simply pick the first available node.
            // TODO: We should probably consider all buffers, not just the first
            std::vector<node_id> intersection;
            std::set_intersection(free_nodes.cbegin(), free_nodes.cend(),
                                  source_nodes.cbegin(), source_nodes.cend(),
                                  std::back_inserter(intersection));
            if (!intersection.empty()) {
              nid = intersection[0];
            } else {
              nid = *free_nodes.cbegin();
            }
          } else {
            nid = *free_nodes.cbegin();
          }

          assert(nid != 0);
          node_assigned = true;
          free_nodes.erase(nid);
          chunk_nodes[i] = nid;
        }
      }
    }

    // Create a compute command for every chunk
    std::vector<vertex> chunk_compute_vertices;
    for (chunk_id i = 0u; i < chunks.size(); ++i) {
      const node_id nid = chunk_nodes[i];
      const auto cv = graph_utils::add_compute_cmd(nid, taskvs[tid], chunks[i],
                                                   command_graph);
      chunk_compute_vertices.push_back(cv);
    }

    // Process writes and create pull / wait-for-pull commands
    for (auto i = 0u; i < chunks.size(); ++i) {
      const node_id nid = chunk_nodes[i];

      for (auto& it : chunk_reqs[i]) {
        const buffer_id bid = it.first;

        // Add read to compute node label for debugging
        const auto& read_req = it.second[cl::sycl::access::mode::read];
        if (read_req.area() > 0) {
          command_graph[chunk_compute_vertices[i]].label =
              (boost::format("%s\\nRead %d %s") %
               command_graph[chunk_compute_vertices[i]].label % bid %
               toString(read_req))
                  .str();
        }

        // ==== Writes ====
        const auto& write_req = it.second[cl::sycl::access::mode::write];
        if (write_req.area() > 0) {
          buffer_writers[bid][nid].push_back(std::make_pair(tid, write_req));

          // Add to compute node label for debugging
          command_graph[chunk_compute_vertices[i]].label =
              (boost::format("%s\\nWrite %d %s") %
               command_graph[chunk_compute_vertices[i]].label % bid %
               toString(write_req))
                  .str();
        }

        // ==== Reads ====
        const auto buffer_sources = chunk_buffer_sources[i][bid];

        for (auto& box_sources : buffer_sources) {
          const auto& box = box_sources.first;
          const auto& box_src_nodes = box_sources.second;

          if (box_src_nodes.count(nid) == 1) {
            // No need to pull
            continue;
          }

          // We just pick the first source node for now
          const node_id source_nid = *box_src_nodes.cbegin();

          // Figure out where/when (and if) source node writes to that buffer
          // TODO: For now we just store the writer's task id since we assume
          // that every node has exactly one compute command per task. In the
          // future this may not be true.
          bool has_writer = false;
          task_id writer_tid = 0;
          for (const auto& bw : buffer_writers[bid][source_nid]) {
            if (GridRegion<1>::intersect(bw.second, GridRegion<1>(box)).area() >
                0) {
#ifdef _DEBUG
              // We assume at most one sibling writes to that exact region
              // TODO: Is there any (useful) scenario where this isn't true?
              assert(!has_writer);
#endif
              has_writer = true;
              writer_tid = bw.first;
#ifndef _DEBUG
              break;
#endif
            }
          }

          // If we haven't found a writer, simply add the "wait-for-pull" in
          // the current task
          const auto source_tv = has_writer ? taskvs[writer_tid] : taskvs[tid];

          // TODO: Update buffer regions since we copied some stuff!!
          graph_utils::add_pull_cmd(nid, source_nid, bid, taskvs[tid],
                                    source_tv, chunk_compute_vertices[i], box,
                                    command_graph);
        }
      }
    }

    graph_utils::mark_as_processed(tid, task_graph);
  }

  // Update buffer regions
  // FIXME Dimensions
  for (auto it : buffer_writers) {
    const buffer_id bid = it.first;
    auto bs =
        static_cast<detail::buffer_state<1>*>(valid_buffer_regions[bid].get());

    for (auto jt : it.second) {
      const node_id nid = jt.first;
      GridRegion<1> region;

      for (const auto& kt : jt.second) {
        region = GridRegion<1>::merge(region, kt.second);
      }

      bs->update_region(region, {nid});
    }
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
