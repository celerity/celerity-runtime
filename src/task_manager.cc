#include "task_manager.h"

#include "access_modes.h"
#include "graph_utils.h"

namespace celerity {
namespace detail {
	task_manager::task_manager(bool is_master_node) : is_master_node(is_master_node), init_task_id(next_task_id++) {
		// We add a special init task for initializing buffers. This task is marked as processed right away.
		// This is useful so we can correctly generate anti-dependencies for tasks that read host initialized buffers.
		// TODO: Not the cleanest solution, especially since it doesn't have an associated task object.
		task_map[init_task_id] = nullptr;
		boost::add_vertex(task_graph);
		task_graph[init_task_id].label = fmt::format("Task {} <INIT>", static_cast<size_t>(init_task_id));
		task_graph[init_task_id].processed = true;
	}

	void task_manager::add_buffer(buffer_id bid, const cl::sycl::range<3>& range, bool host_initialized) {
		std::lock_guard<std::mutex> lock(task_mutex);
		buffers_last_writers.emplace(bid, range);
		if(host_initialized) { buffers_last_writers.at(bid).update_region(subrange_to_grid_region(subrange<3>({}, range)), init_task_id); }
	}

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

	void task_manager::print_graph(logger& graph_logger) const {
		const auto locked_tdag = get_task_graph();
		if((*locked_tdag).m_vertices.size() < 200) {
			graph_utils::print_graph(*locked_tdag, graph_logger);
		} else {
			graph_logger.warn("Task graph is very large ({} vertices). Skipping GraphViz output", (*locked_tdag).m_vertices.size());
		}
	}

	GridRegion<3> get_requirements(task const* tsk, buffer_id bid, const std::vector<cl::sycl::access::mode> modes) {
		GridRegion<3> result;
		switch(tsk->get_type()) {
		case task_type::COMPUTE: {
			const auto ctsk = dynamic_cast<compute_task const*>(tsk);
			// Determine the requirements for the full kernel global size
			const subrange<3> full_range{ctsk->get_global_offset(), ctsk->get_global_size()};
			for(auto m : modes) {
				result = GridRegion<3>::merge(result, ctsk->get_requirements(bid, m, full_range));
			}
		} break;
		case task_type::MASTER_ACCESS: {
			const auto mtsk = dynamic_cast<master_access_task const*>(tsk);
			for(auto m : modes) {
				result = GridRegion<3>::merge(result, mtsk->get_requirements(bid, m));
			}
		} break;
		default: assert(false);
		}
		return result;
	}

	void task_manager::compute_dependencies(task_id tid) {
		using namespace cl::sycl::access;

		const auto add_dependency = [this](task_id dependant, task_id dependency, bool anti) {
			// Check if edge already exists
			const auto ed = boost::edge(dependency, dependant, task_graph);
			const bool exists = ed.second;

			if(exists) {
				// If it already exists, make sure true dependencies take precedence
				if(!anti) { task_graph[ed.first].anti_dependency = false; }
			} else {
				const auto result = boost::add_edge(dependency, dependant, task_graph);
				task_graph[result.first].anti_dependency = anti;
				if(!task_graph[dependency].processed) { task_graph[dependant].num_unsatisfied++; }
			}
		};

		const auto& tsk = task_map[tid];
		const auto buffers = tsk->get_accessed_buffers();

		for(const auto bid : buffers) {
			const auto modes = tsk->get_access_modes(bid);

			// Determine reader dependencies
			if(std::any_of(modes.cbegin(), modes.cend(), access::detail::mode_traits::is_consumer)) {
				const auto read_requirements =
				    get_requirements(tsk.get(), bid, {access::detail::consumer_modes.cbegin(), access::detail::consumer_modes.cend()});
				const auto last_writers = buffers_last_writers.at(bid).get_region_values(read_requirements);

				for(auto& p : last_writers) {
					// This indicates that the buffer is being used for the first time by this task, or all previous tasks also only read from it.
					// A valid use case (i.e., not reading garbage) for this is when the buffer has been initialized using a host pointer.
					if(p.second == boost::none) continue;
					const task_id last_writer = *p.second;
					add_dependency(tid, last_writer, false);
				}
			}

			// Update last writers and determine anti-dependencies
			if(std::any_of(modes.cbegin(), modes.cend(), access::detail::mode_traits::is_producer)) {
				const auto write_requirements =
				    get_requirements(tsk.get(), bid, {access::detail::producer_modes.cbegin(), access::detail::producer_modes.cend()});
				assert(!write_requirements.empty() && "Task specified empty buffer range requirement. This indicates potential anti-pattern.");
				const auto last_writers = buffers_last_writers.at(bid).get_region_values(write_requirements);

				for(auto& p : last_writers) {
					if(p.second == boost::none) continue;
					task_id last_writer = *p.second;

					// Determine anti-dependencies by looking at all the dependants of the last writing task
					bool has_anti_dependants = false;
					graph_utils::for_successors(
					    task_graph, last_writer, [tid, bid, add_dependency, &write_requirements, &has_anti_dependants, this](tdag_vertex v, tdag_edge) {
						    const auto dependant_tid = static_cast<task_id>(v);
						    if(dependant_tid == tid) {
							    // This can happen
							    // - if a task writes to two or more buffers with the same last writer
							    // - if the task itself also needs read access to that buffer (R/W access)
							    return;
						    }
						    const auto dependant_read_requirements = get_requirements(
						        task_map[dependant_tid].get(), bid, {access::detail::consumer_modes.cbegin(), access::detail::consumer_modes.cend()});
						    // Only add an anti-dependency if we are really writing over the region read by this task
						    if(!GridRegion<3>::intersect(write_requirements, dependant_read_requirements).empty()) {
							    add_dependency(tid, dependant_tid, true);
							    has_anti_dependants = true;
						    }
					    });

					if(!has_anti_dependants) {
						// If no intermediate consumers exist, add an anti-dependency on the last writer directly.
						// Note that unless this task is a pure producer, a true dependency will be created and this is a no-op.
						// While it might not always make total sense to have anti-dependencies between (pure) producers without an
						// intermediate consumer, we at least have a defined behavior, and the thus enforced ordering of tasks
						// likely reflects what the user expects.
						add_dependency(tid, last_writer, true);
					}
				}

				buffers_last_writers.at(bid).update_region(write_requirements, tid);
			}
		}
	}

	void task_manager::invoke_callbacks() {
		for(auto& cb : task_callbacks) {
			cb();
		}
	}

} // namespace detail
} // namespace celerity
