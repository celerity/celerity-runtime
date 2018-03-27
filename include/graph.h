#pragma once

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/type_index.hpp>

#include "command.h"
#include "types.h"

namespace celerity {

struct tdag_vertex_properties {
	std::string label;

	// Whether this task has been processed into the command dag
	bool processed = false;

	// The number of unsatisfied (= unprocessed) dependencies this task has
	size_t num_unsatisfied = 0;
};

struct tdag_graph_properties {
	std::string name;
};

using task_dag = boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS, tdag_vertex_properties, boost::no_property, tdag_graph_properties>;

struct cdag_vertex_properties {
	std::string label;
	command cmd = command::NOP;
	node_id nid = 0;
	task_id tid;
	command_data data = {};
};

struct cdag_graph_properties {
	std::string name;
	std::unordered_map<task_id, vertex> task_complete_vertices;
};

using command_dag = boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS, cdag_vertex_properties, boost::no_property, cdag_graph_properties>;

} // namespace celerity
