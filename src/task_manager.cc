#include "task_manager.h"

#include "access_modes.h"
#include "logger.h"
#include "print_graph.h"

namespace celerity {
namespace detail {

	task_manager::task_manager(size_t num_collective_nodes, host_queue* queue, reduction_manager* reduction_mgr)
	    : num_collective_nodes(num_collective_nodes), queue(queue), reduction_mngr(reduction_mgr), init_task_id(next_task_id++) {
		// We add a special init task for initializing buffers.
		// This is useful so we can correctly generate anti-dependencies for tasks that read host initialized buffers.
		task_map[init_task_id] = task::make_nop(init_task_id);
	}

	void task_manager::add_buffer(buffer_id bid, const cl::sycl::range<3>& range, bool host_initialized) {
		std::lock_guard<std::mutex> lock(task_mutex);
		buffers_last_writers.emplace(bid, range);
		if(host_initialized) { buffers_last_writers.at(bid).update_region(subrange_to_grid_box(subrange<3>({}, range)), init_task_id); }
	}

	bool task_manager::has_task(task_id tid) const {
		std::lock_guard<std::mutex> lock(task_mutex);
		return task_map.count(tid) != 0;
	}

	// Note that we assume tasks are not modified after their initial creation, which is why
	// we don't need to worry about thread-safety after returning the task pointer.
	const task* task_manager::get_task(task_id tid) const {
		std::lock_guard<std::mutex> lock(task_mutex);
		assert(task_map.count(tid) != 0);
		return task_map.at(tid).get();
	}

	void task_manager::print_graph(logger& graph_logger) const {
		std::lock_guard<std::mutex> lock(task_mutex);
		if(task_map.size() < 200) {
			detail::print_graph(task_map, graph_logger);
		} else {
			graph_logger.warn("Task graph is very large ({} vertices). Skipping GraphViz output", task_map.size());
		}
	}

	void task_manager::notify_horizon_executed(task_id tid) {
		executed_horizons.push(tid);

		if(executed_horizons.size() >= horizon_deletion_lag) {
			// actual cleanup happens on new task creation
			horizon_task_id_for_deletion = executed_horizons.front();
			executed_horizons.pop();
		}
	}

	GridRegion<3> get_requirements(task const* tsk, buffer_id bid, const std::vector<cl::sycl::access::mode> modes) {
		const auto& access_map = tsk->get_buffer_access_map();
		const subrange<3> full_range{tsk->get_global_offset(), tsk->get_global_size()};
		GridRegion<3> result;
		for(auto m : modes) {
			result = GridRegion<3>::merge(result, access_map.get_requirements_for_access(bid, m, tsk->get_dimensions(), full_range, tsk->get_global_size()));
		}
		return result;
	}

	void task_manager::compute_dependencies(task_id tid) {
		using namespace cl::sycl::access;

		const auto& tsk = task_map[tid];
		const auto& access_map = tsk->get_buffer_access_map();

		auto buffers = access_map.get_accessed_buffers();
		for(auto rid : tsk->get_reductions()) {
			assert(reduction_mngr != nullptr);
			buffers.emplace(reduction_mngr->get_reduction(rid).output_buffer_id);
		}

		for(const auto bid : buffers) {
			const auto modes = access_map.get_access_modes(bid);

			std::optional<reduction_info> reduction;
			for(auto maybe_rid : tsk->get_reductions()) {
				auto maybe_reduction = reduction_mngr->get_reduction(maybe_rid);
				if(maybe_reduction.output_buffer_id == bid) {
					if(reduction) { throw std::runtime_error(fmt::format("Multiple reductions attempt to write buffer {} in task {}", bid, tid)); }
					reduction = maybe_reduction;
				}
			}

			if(reduction && !modes.empty()) {
				throw std::runtime_error(fmt::format("Buffer {} is both required through an accessor and used as a reduction output in task {}", bid, tid));
			}

			// Determine reader dependencies
			if(std::any_of(modes.cbegin(), modes.cend(), detail::access::mode_traits::is_consumer)
			    || (reduction.has_value() && reduction->initialize_from_buffer)) {
				auto read_requirements = get_requirements(tsk.get(), bid, {detail::access::consumer_modes.cbegin(), detail::access::consumer_modes.cend()});
				if(reduction.has_value()) { read_requirements = GridRegion<3>::merge(read_requirements, GridRegion<3>{{1, 1, 1}}); }
				const auto last_writers = buffers_last_writers.at(bid).get_region_values(read_requirements);

				for(auto& p : last_writers) {
					// This indicates that the buffer is being used for the first time by this task, or all previous tasks also only read from it.
					// A valid use case (i.e., not reading garbage) for this is when the buffer has been initialized using a host pointer.
					if(p.second == std::nullopt) continue;
					const task_id last_writer = *p.second;
					add_dependency(tsk.get(), task_map[last_writer].get(), dependency_kind::TRUE_DEP);
				}
			}

			// Update last writers and determine anti-dependencies
			if(std::any_of(modes.cbegin(), modes.cend(), detail::access::mode_traits::is_producer) || reduction.has_value()) {
				auto write_requirements = get_requirements(tsk.get(), bid, {detail::access::producer_modes.cbegin(), detail::access::producer_modes.cend()});
				if(reduction.has_value()) { write_requirements = GridRegion<3>::merge(write_requirements, GridRegion<3>{{1, 1, 1}}); }
				assert(!write_requirements.empty() && "Task specified empty buffer range requirement. This indicates potential anti-pattern.");
				const auto last_writers = buffers_last_writers.at(bid).get_region_values(write_requirements);

				for(auto& p : last_writers) {
					if(p.second == std::nullopt) continue;
					auto& last_writer = *task_map[*p.second];

					// Determine anti-dependencies by looking at all the dependents of the last writing task
					bool has_anti_dependents = false;

					for(auto dependent : last_writer.get_dependents()) {
						if(dependent.node->get_id() == tid) {
							// This can happen
							// - if a task writes to two or more buffers with the same last writer
							// - if the task itself also needs read access to that buffer (R/W access)
							continue;
						}
						const auto dependent_read_requirements =
						    get_requirements(dependent.node, bid, {detail::access::consumer_modes.cbegin(), detail::access::consumer_modes.cend()});
						// Only add an anti-dependency if we are really writing over the region read by this task
						if(!GridRegion<3>::intersect(write_requirements, dependent_read_requirements).empty()) {
							add_dependency(tsk.get(), dependent.node, dependency_kind::ANTI_DEP);
							has_anti_dependents = true;
						}
					}

					if(!has_anti_dependents) {
						// If no intermediate consumers exist, add an anti-dependency on the last writer directly.
						// Note that unless this task is a pure producer, a true dependency will be created and this is a no-op.
						// While it might not always make total sense to have anti-dependencies between (pure) producers without an
						// intermediate consumer, we at least have a defined behavior, and the thus enforced ordering of tasks
						// likely reflects what the user expects.
						add_dependency(tsk.get(), &last_writer, dependency_kind::ANTI_DEP);
					}
				}

				buffers_last_writers.at(bid).update_region(write_requirements, tid);
			}
		}

		if(auto cgid = tsk->get_collective_group_id(); cgid != 0) {
			if(auto prev = last_collective_tasks.find(cgid); prev != last_collective_tasks.end()) {
				add_dependency(tsk.get(), task_map.at(prev->second).get(), dependency_kind::ORDER_DEP);
				last_collective_tasks.erase(prev);
			}
			last_collective_tasks.emplace(cgid, tid);
		}
	}

	task& task_manager::register_task_internal(std::unique_ptr<task> task) {
		auto& task_ref = *task;
		assert(task != nullptr);
		task_map.emplace(task->get_id(), std::move(task));
		execution_front.insert(&task_ref);
		return task_ref;
	}

	void task_manager::invoke_callbacks(task_id tid, task_type type) {
		for(auto& cb : task_callbacks) {
			cb(tid, type);
		}
	}

	void task_manager::add_dependency(task* depender, task* dependee, dependency_kind kind) {
		assert(depender != dependee);
		assert(depender != nullptr && dependee != nullptr);
		depender->add_dependency({dependee, kind});
		execution_front.erase(dependee);
		max_pseudo_critical_path_length = std::max(max_pseudo_critical_path_length, depender->get_pseudo_critical_path_length());
	}

	void task_manager::generate_task_horizon() {
		// we are probably overzealous in locking here
		task_id tid;
		{
			std::lock_guard lock(task_mutex);
			previous_horizon_critical_path_length = max_pseudo_critical_path_length;

			// create horizon task
			tid = get_new_tid();
			task* horizon_task_ptr = nullptr;
			{
				auto horizon_task = task::make_horizon_task(tid);
				horizon_task_ptr = &register_task_internal(std::move(horizon_task));
			}

			// add dependencies from a copy of the front to this task
			auto current_front = get_execution_front();
			for(task* front_task : current_front) {
				if(front_task != horizon_task_ptr) { add_dependency(horizon_task_ptr, front_task); }
			}

			// apply the previous horizon to buffers_last_writers data struct
			if(previous_horizon_task != nullptr) {
				for(auto& [_, buffer_region_map] : buffers_last_writers) {
					task_id prev_hid = previous_horizon_task->get_id();
					buffer_region_map.apply_to_values([prev_hid](std::optional<task_id> tid) -> std::optional<task_id> {
						if(!tid) return tid;
						return {std::max(prev_hid, *tid)};
					});
				}
			}
			previous_horizon_task = horizon_task_ptr;
		}

		// it's important that we don't hold the lock while doing this
		invoke_callbacks(tid, task_type::HORIZON);
	}

	void task_manager::clean_up_pre_horizon_tasks() {
		task_id deletion_task_id = horizon_task_id_for_deletion.exchange(nothing_to_delete);
		if(deletion_task_id != nothing_to_delete) {
			for(auto iter = task_map.begin(); iter != task_map.end();) {
				if(iter->first < deletion_task_id) {
					iter = task_map.erase(iter);
				} else {
					++iter;
				}
			}
		}
	}

} // namespace detail
} // namespace celerity
