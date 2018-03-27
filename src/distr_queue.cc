#include "distr_queue.h"

#include "graph_utils.h"
#include "runtime.h"

namespace celerity {

// TODO: Initialize SYCL queue lazily
distr_queue::distr_queue(cl::sycl::device device) : sycl_queue(device) {
	runtime::get_instance().register_queue(this);
	task_graph[boost::graph_bundle].name = "TaskGraph";
}

void distr_queue::mark_task_as_processed(task_id tid) {
	graph_utils::mark_as_processed(tid, task_graph);
}

void distr_queue::debug_print_task_graph() {
	graph_utils::print_graph(task_graph);
}

void distr_queue::add_requirement(task_id tid, buffer_id bid, cl::sycl::access::mode mode, std::unique_ptr<detail::range_mapper_base> rm) {
	// TODO: Check if edge already exists (avoid double edges)
	// TODO: If we have dependencies "A -> B, B -> C, A -> C", we could get rid of
	// "A -> C", as it is transitively implicit in "B -> C".
	if(mode == cl::sycl::access::mode::read) {
		if(buffer_last_writer.find(bid) != buffer_last_writer.end()) {
			boost::add_edge(buffer_last_writer[bid], tid, task_graph);
			task_graph[tid].num_unsatisfied++;
		}
	}
	if(mode == cl::sycl::access::mode::write) { buffer_last_writer[bid] = tid; }
	task_range_mappers[tid][bid].push_back(std::move(rm));
};

bool distr_queue::has_dependency(task_id task_a, task_id task_b) const {
	// TODO: Use DFS instead?
	bool found = false;
	graph_utils::search_vertex_bf(task_b, task_graph, [&found, task_a](vertex v, const task_dag&) {
		if(v == task_a) { found = true; }
		return found;
	});
	return found;
}

} // namespace celerity
