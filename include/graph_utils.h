#pragma once

#include <memory>
#include <stdexcept>
#include <utility>

#include <boost/algorithm/string.hpp>
#include <boost/graph/breadth_first_search.hpp>

#include "graph.h"
#include "grid.h"
#include "logger.h"
#include "ranges.h"
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

	vertex add_compute_cmd(command_id& next_cmd_id, node_id nid, const task_vertices& tv, const chunk<3>& chnk, command_dag& cdag);
	vertex add_master_access_cmd(command_id& next_cmd_id, const task_vertices& tv, command_dag& cdag);
	vertex add_push_cmd(command_id& next_cmd_id, node_id to_nid, node_id from_nid, buffer_id bid, const task_vertices& tv, vertex req_cmd,
	    const GridBox<3>& req, command_dag& cdag);

	/**
	 * Finds the next (= in the global list of task vertices) task with no unsatisfied dependencies.
	 * Returns false if no satisfied task was found.
	 */
	bool get_satisfied_task(const task_dag& tdag, task_id& tid);

	void mark_as_processed(task_id tid, task_dag& tdag);


	// --------------------------- Graph printing ---------------------------


	template <typename Graph, typename VertexPropertiesWriter, typename EdgePropertiesWriter>
	void write_graph_mux(const Graph& g, VertexPropertiesWriter vpw, EdgePropertiesWriter epw, std::shared_ptr<logger> graph_logger) {
		std::stringstream ss;
		write_graphviz(ss, g, vpw, epw);
		auto str = ss.str();
		boost::replace_all(str, "\n", "\\n");
		boost::replace_all(str, "\"", "\\\"");
		graph_logger->info(logger_map({{"name", g[boost::graph_bundle].name}, {"data", str}}));
	}

	void print_graph(const celerity::task_dag& tdag, std::shared_ptr<logger> graph_logger);
	void print_graph(const celerity::command_dag& cdag, std::shared_ptr<logger> graph_logger);

} // namespace graph_utils

} // namespace celerity
