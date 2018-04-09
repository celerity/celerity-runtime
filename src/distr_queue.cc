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

void distr_queue::debug_print_task_graph() const {
	graph_utils::print_graph(task_graph);
}

task_id distr_queue::add_task(std::shared_ptr<task> tsk) {
	const task_id tid = task_count++;
	task_map[tid] = tsk;
	boost::add_vertex(task_graph);
	task_graph[tid].label = (boost::format("Task %d") % tid).str();
	return tid;
}

void distr_queue::add_requirement(task_id tid, buffer_id bid, cl::sycl::access::mode mode, std::unique_ptr<detail::range_mapper_base> rm) {
	assert(task_map.count(tid) != 0);
	assert(task_map[tid]->get_type() == task_type::COMPUTE);
	dynamic_cast<compute_task*>(task_map[tid].get())->add_range_mapper(bid, std::move(rm));
	update_dependencies(tid, bid, mode);
}

void distr_queue::add_requirement(task_id tid, buffer_id bid, cl::sycl::access::mode mode, any_range range, any_range offset) {
	assert(task_map.count(tid) != 0);
	assert(task_map[tid]->get_type() == task_type::MASTER_ACCESS);
	dynamic_cast<master_access_task*>(task_map[tid].get())->add_buffer_access(bid, mode, range, offset);
	update_dependencies(tid, bid, mode);
}

void distr_queue::set_task_data(task_id tid, any_range global_size, std::string debug_name) {
	assert(task_map.count(tid) != 0);
	assert(task_map[tid]->get_type() == task_type::COMPUTE);
	dynamic_cast<compute_task*>(task_map[tid].get())->set_global_size(global_size);
	task_graph[tid].label = (boost::format("%s (%s)") % task_graph[tid].label % debug_name).str();
}

void distr_queue::update_dependencies(task_id tid, buffer_id bid, cl::sycl::access::mode mode) {
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
}

bool distr_queue::has_dependency(task_id task_a, task_id task_b) const {
	// TODO: Use DFS instead?
	bool found = false;
	graph_utils::search_vertex_bf(static_cast<vertex>(task_b), task_graph, [&found, task_a](vertex v, const task_dag&) {
		if(v == task_a) { found = true; }
		return found;
	});
	return found;
}

} // namespace celerity
