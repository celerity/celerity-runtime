#include "task_manager.h"

#include "graph_utils.h"

namespace celerity {
namespace detail {
	locked_graph<const task_dag> task_manager::get_task_graph() const { return locked_graph<const task_dag>{task_graph, task_mutex}; }

	bool task_manager::has_task(task_id tid) const {
		std::lock_guard<std::mutex> lock(task_mutex);
		return task_map.count(tid) != 0;
	}

	// Note that we assume tasks are not modified after their initial creation, which is why
	// we don't need to worry about thread-safety after returning the task pointer.
	std::shared_ptr<const task> task_manager::get_task(task_id tid) const {
		std::lock_guard<std::mutex> lock(task_mutex);
		assert(task_map.count(tid) != 0);
		return task_map.at(tid);
	}

	void task_manager::mark_task_as_processed(task_id tid) {
		std::lock_guard<std::mutex> lock(task_mutex);
		graph_utils::mark_as_processed(tid, task_graph);
	}

	bool task_manager::has_dependency(task_id task_a, task_id task_b) const {
		auto tdag = get_task_graph();
		// TODO: Use DFS instead?
		bool found = false;
		graph_utils::search_vertex_bf(static_cast<tdag_vertex>(task_b), *tdag, [&found, task_a](tdag_vertex v, const task_dag&) {
			if(v == task_a) { found = true; }
			return found;
		});
		return found;
	}

	void task_manager::print_graph(std::shared_ptr<logger>& graph_logger) const {
		const auto locked_tdag = get_task_graph();
		if((*locked_tdag).m_vertices.size() < 200) {
			graph_utils::print_graph(*locked_tdag, graph_logger);
		} else {
			graph_logger->warn("Task graph is very large ({} vertices). Skipping GraphViz output", (*locked_tdag).m_vertices.size());
		}
	}

	void task_manager::add_requirement(task_id tid, buffer_id bid, cl::sycl::access::mode mode, std::unique_ptr<detail::range_mapper_base> rm) {
		assert(task_map.count(tid) != 0);
		assert(task_map[tid]->get_type() == task_type::COMPUTE);
		dynamic_cast<compute_task*>(task_map[tid].get())->add_range_mapper(bid, std::move(rm));
		update_dependencies(tid, bid, mode);
	}

	void task_manager::add_requirement(task_id tid, buffer_id bid, cl::sycl::access::mode mode, cl::sycl::range<3> range, cl::sycl::id<3> offset) {
		assert(task_map.count(tid) != 0);
		assert(task_map[tid]->get_type() == task_type::MASTER_ACCESS);
		dynamic_cast<master_access_task*>(task_map[tid].get())->add_buffer_access(bid, mode, range, offset);
		update_dependencies(tid, bid, mode);
	}

	void task_manager::set_compute_task_data(task_id tid, int dimensions, cl::sycl::range<3> global_size, std::string debug_name) {
		assert(task_map.count(tid) != 0);
		assert(task_map[tid]->get_type() == task_type::COMPUTE);
		auto ctsk = dynamic_cast<compute_task*>(task_map[tid].get());
		ctsk->set_dimensions(dimensions);
		ctsk->set_global_size(global_size);
		task_graph[tid].label = fmt::format("{} ({})", task_graph[tid].label, debug_name);
	}

	task_id task_manager::add_task(std::shared_ptr<task> tsk) {
		const task_id tid = task_count++;
		task_map[tid] = tsk;
		boost::add_vertex(task_graph);
		task_graph[tid].label = fmt::format("Task {}", static_cast<size_t>(tid));
		return tid;
	}

	void task_manager::update_dependencies(task_id tid, buffer_id bid, cl::sycl::access::mode mode) {
		// TODO: If we have dependencies "A -> B, B -> C, A -> C", we could get rid of
		// "A -> C", as it is transitively implicit in "B -> C".
		if(mode == cl::sycl::access::mode::read) {
			if(buffer_last_writer.find(bid) != buffer_last_writer.end()) {
				boost::add_edge(buffer_last_writer[bid], tid, task_graph);
				if(!task_graph[buffer_last_writer[bid]].processed) { task_graph[tid].num_unsatisfied++; }
			}
		}
		if(mode == cl::sycl::access::mode::write) { buffer_last_writer[bid] = tid; }
	}

	void task_manager::invoke_callbacks() {
		for(auto& cb : task_callbacks) {
			cb();
		}
	}

} // namespace detail
} // namespace celerity
