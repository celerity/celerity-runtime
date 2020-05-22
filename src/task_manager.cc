#include "task_manager.h"

#include "access_modes.h"
#include "logger.h"
#include "print_graph.h"

namespace celerity {
namespace detail {
	task_manager::task_manager(bool is_master_node) : is_master_node(is_master_node), init_task_id(next_task_id++) {
		// We add a special init task for initializing buffers.
		// This is useful so we can correctly generate anti-dependencies for tasks that read host initialized buffers.
		task_map[init_task_id] = std::make_shared<nop_task>(init_task_id);
	}

	void task_manager::add_buffer(buffer_id bid, const cl::sycl::range<3>& range, bool host_initialized) {
		std::lock_guard<std::mutex> lock(task_mutex);
		buffers_last_writers.emplace(bid, range);
		if(host_initialized) { buffers_last_writers.at(bid).update_region(subrange_to_grid_region(subrange<3>({}, range)), init_task_id); }
	}

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

	void task_manager::print_graph(logger& graph_logger) const {
		std::lock_guard<std::mutex> lock(task_mutex);
		if(task_map.size() < 200) {
			detail::print_graph(task_map, graph_logger);
		} else {
			graph_logger.warn("Task graph is very large ({} vertices). Skipping GraphViz output", task_map.size());
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
					if(p.second == std::nullopt) continue;
					const task_id last_writer = *p.second;
					tsk->add_dependency({task_map[last_writer].get(), false});
				}
			}

			// Update last writers and determine anti-dependencies
			if(std::any_of(modes.cbegin(), modes.cend(), access::detail::mode_traits::is_producer)) {
				const auto write_requirements =
				    get_requirements(tsk.get(), bid, {access::detail::producer_modes.cbegin(), access::detail::producer_modes.cend()});
				assert(!write_requirements.empty() && "Task specified empty buffer range requirement. This indicates potential anti-pattern.");
				const auto last_writers = buffers_last_writers.at(bid).get_region_values(write_requirements);

				for(auto& p : last_writers) {
					if(p.second == std::nullopt) continue;
					auto last_writer = task_map[*p.second];

					// Determine anti-dependencies by looking at all the dependents of the last writing task
					bool has_anti_dependents = false;

					for(auto dependent : last_writer->get_dependents()) {
						if(dependent.node->get_id() == tid) {
							// This can happen
							// - if a task writes to two or more buffers with the same last writer
							// - if the task itself also needs read access to that buffer (R/W access)
							continue;
						}
						const auto dependent_read_requirements =
						    get_requirements(dependent.node, bid, {access::detail::consumer_modes.cbegin(), access::detail::consumer_modes.cend()});
						// Only add an anti-dependency if we are really writing over the region read by this task
						if(!GridRegion<3>::intersect(write_requirements, dependent_read_requirements).empty()) {
							tsk->add_dependency({dependent.node, true});
							has_anti_dependents = true;
						}
					}

					if(!has_anti_dependents) {
						// If no intermediate consumers exist, add an anti-dependency on the last writer directly.
						// Note that unless this task is a pure producer, a true dependency will be created and this is a no-op.
						// While it might not always make total sense to have anti-dependencies between (pure) producers without an
						// intermediate consumer, we at least have a defined behavior, and the thus enforced ordering of tasks
						// likely reflects what the user expects.
						tsk->add_dependency({last_writer.get(), true});
					}
				}

				buffers_last_writers.at(bid).update_region(write_requirements, tid);
			}
		}
	}

	void task_manager::invoke_callbacks(task_id tid) {
		for(auto& cb : task_callbacks) {
			cb(tid);
		}
	}

} // namespace detail
} // namespace celerity
