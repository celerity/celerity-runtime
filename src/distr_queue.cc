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

void distr_queue::TEST_execute_deferred() {
	for(auto& it : task_command_groups) {
		const task_id tid = it.first;
		auto& cgf = it.second;
		sycl_queue.submit([this, &cgf, tid](cl::sycl::handler& sycl_handler) {
			handler<is_prepass::false_t> h(*this, tid, &sycl_handler);
			(*cgf)(h);
		});
	}
}

} // namespace celerity
