#pragma once

#include <stdexcept>
#include <utility>

#include <boost/algorithm/string.hpp>
#include <boost/graph/breadth_first_search.hpp>

#include "graph.h"
#include "grid.h"
#include "subrange.h"
#include "types.h"

namespace celerity {

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

	/**
	 * Calls a functor on every predecessor of vertex v within the graph.
	 * The functor can optionally return a boolean indicating whether the
	 * loop should abort.
	 *
	 * Returns false if the loop was aborted.
	 */
	template <typename Graph, typename Functor>
	bool for_predecessors(const Graph& graph, vertex v, const Functor& f) {
		typename boost::graph_traits<Graph>::in_edge_iterator eit, eit_end;
		for(std::tie(eit, eit_end) = boost::in_edges(v, graph); eit != eit_end; ++eit) {
			vertex pre = boost::source(*eit, graph);
			if(call_for_vertex_fn(f, pre, std::is_same<bool, decltype(f(pre))>()) == false) { return false; }
		}
		return true;
	}

	/**
	 * Calls a functor on every successor of vertex v within the graph.
	 * The functor can optionally return a boolean indicating whether the
	 * loop should abort.
	 *
	 * Returns false if the loop was aborted.
	 */
	template <typename Graph, typename Functor>
	bool for_successors(const Graph& graph, vertex v, const Functor& f) {
		typename boost::graph_traits<Graph>::out_edge_iterator eit, eit_end;
		for(std::tie(eit, eit_end) = boost::out_edges(v, graph); eit != eit_end; ++eit) {
			vertex suc = boost::target(*eit, graph);
			if(call_for_vertex_fn(f, suc, std::is_same<bool, decltype(f(suc))>()) == false) { return false; }
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
		abort_search_exception() : std::runtime_error("Abort search (not an error)") {}
	};

	template <typename Functor>
	class bfs_visitor : public boost::default_bfs_visitor {
	  public:
		bfs_visitor(Functor f) : f(f) {}

		template <typename Graph>
		void discover_vertex(vertex v, const Graph& graph) const {
			if(f(v, graph) == true) { throw abort_search_exception(); }
		}

	  private:
		Functor f;
	};

	/**
	 * Search vertices using a breadth-first-search.
	 * The functor receives the current vertex as well as the graph by reference.
	 * The search is aborted if the functor returns true.
	 */
	template <typename Graph, typename Functor>
	void search_vertex_bf(vertex start, const Graph& graph, Functor f) {
		try {
			bfs_visitor<Functor> vis(f);
			boost::breadth_first_search(graph, start, boost::visitor(vis));
		} catch(abort_search_exception&) {
			// Nop
		}
	}

	task_vertices add_task(task_id tid, const task_dag& tdag, command_dag& cdag);

	template <int Dims>
	vertex add_compute_cmd(node_id nid, const task_vertices& tv, const subrange<Dims>& chunk, command_dag& cdag) {
		const auto v = boost::add_vertex(cdag);
		boost::add_edge(tv.first, v, cdag);
		boost::add_edge(v, tv.second, cdag);
		cdag[v].cmd = cdag_command::COMPUTE;
		cdag[v].nid = nid;
		cdag[v].label = (boost::format("Node %d:\\COMPUTE %s") % nid % toString(detail::subrange_to_grid_region(chunk))).str();
		return v;
	}

	template <std::size_t Dims>
	vertex add_pull_cmd(node_id nid, node_id source_nid, buffer_id bid, const task_vertices& tv, const task_vertices& source_tv, vertex compute_cmd,
	    const GridBox<Dims>& req, command_dag& cdag) {
		assert(cdag[compute_cmd].cmd == cdag_command::COMPUTE);
		const auto v = graph_utils::insert_vertex_on_edge(tv.first, compute_cmd, cdag);
		cdag[v].cmd = cdag_command::PULL;
		cdag[v].nid = nid;
		cdag[v].label = (boost::format("Node %d:\\nPULL %d from %d\\n %s") % nid % bid % source_nid % toString(req)).str();

		// Find the compute command for the source node in the writing task (or this
		// task, if no writing task has been found)
		vertex source_compute_v = 0;
		search_vertex_bf(source_tv.first, cdag, [source_nid, source_tv, &source_compute_v](vertex v, const command_dag& cdag) {
			if(cdag[v].cmd == cdag_command::COMPUTE && cdag[v].nid == source_nid) {
				source_compute_v = v;
				return true;
			}
			return false;
		});
		assert(source_compute_v != 0);

		const auto w = graph_utils::insert_vertex_on_edge(source_tv.first, source_compute_v, cdag);
		cdag[w].cmd = cdag_command::AWAIT_PULL;
		cdag[w].nid = source_nid;
		cdag[w].label = (boost::format("Node %d:\\nAWAIT PULL %d by %d\\n %s") % source_nid % bid % nid % toString(req)).str();

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
	std::vector<task_id> get_satisfied_sibling_set(const task_dag& tdag);

	void mark_as_processed(task_id tid, task_dag& tdag);


	// --------------------------- Graph printing ---------------------------


	template <typename Graph, typename VertexPropertiesWriter, typename EdgePropertiesWriter>
	void write_graph_mux(const Graph& g, VertexPropertiesWriter vpw, EdgePropertiesWriter epw) {
		std::stringstream ss;
		write_graphviz(ss, g, vpw, epw);
		auto str = ss.str();
		std::vector<std::string> lines;
		boost::split(lines, str, boost::is_any_of("\n"));
		auto graph_name = g[boost::graph_bundle].name;
		for(auto l : lines) {
			std::cout << "#G:" << graph_name << "#" << l << std::endl;
		}
	}

	void print_graph(const celerity::task_dag& tdag);
	void print_graph(const celerity::command_dag& cdag);

} // namespace graph_utils

} // namespace celerity
