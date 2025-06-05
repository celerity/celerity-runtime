#include "task_manager.h"

#include "cgf.h"
#include "grid.h"
#include "intrusive_graph.h"
#include "log.h"
#include "ranges.h"
#include "recorders.h"
#include "task.h"
#include "types.h"
#include "utils.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include <fmt/format.h>


namespace celerity {
namespace detail {

	task_manager::task_manager(
	    size_t num_collective_nodes, task_graph& tdag, detail::task_recorder* recorder, task_manager::delegate* const dlg, const policy_set& error_policy)
	    : m_delegate(dlg), m_num_collective_nodes(num_collective_nodes), m_policy(error_policy), m_tdag(tdag), m_task_recorder(recorder) {}

	void task_manager::notify_buffer_created(const buffer_id bid, const range<3>& range, const bool host_initialized) {
		const auto [iter, inserted] = m_buffers.emplace(bid, range);
		assert(inserted);
		auto& buffer = iter->second;
		if(host_initialized) { buffer.last_writers.update_box(box<3>::full_range(range), m_epoch_for_new_tasks); }
	}

	void task_manager::notify_buffer_debug_name_changed(const buffer_id bid, const std::string& debug_name) { m_buffers.at(bid).debug_name = debug_name; }

	void task_manager::notify_buffer_destroyed(const buffer_id bid) {
		assert(m_buffers.count(bid) != 0);
		m_buffers.erase(bid);
	}
	void task_manager::notify_host_object_created(const host_object_id hoid) { m_host_objects.emplace(hoid, host_object_state()); }

	void task_manager::notify_host_object_destroyed(const host_object_id hoid) {
		assert(m_host_objects.count(hoid) != 0);
		m_host_objects.erase(hoid);
	}

	void task_manager::compute_dependencies(task& tsk) {
		const auto& access_map = tsk.get_buffer_access_map();

		auto buffers = access_map.get_accessed_buffers();
		for(const auto& reduction : tsk.get_reductions()) {
			if(buffers.contains(reduction.bid)) {
				throw std::runtime_error(
				    fmt::format("Buffer {} is both required through an accessor and used as a reduction output in task {}", reduction.bid, tsk.get_id()));
			}
			buffers.emplace(reduction.bid);
		}

		const box<3> scalar_box({0, 0, 0}, {1, 1, 1});

		for(const auto bid : buffers) {
			auto& buffer = m_buffers.at(bid);
			std::optional<reduction_info> reduction;
			for(const auto& maybe_reduction : tsk.get_reductions()) {
				if(maybe_reduction.bid == bid) {
					if(reduction) { throw std::runtime_error(fmt::format("Multiple reductions attempt to write buffer {} in task {}", bid, tsk.get_id())); }
					reduction = maybe_reduction;
				}
			}

			// Determine reader dependencies
			auto read_requirements = access_map.get_task_consumed_region(bid);
			if(!read_requirements.empty() || (reduction.has_value() && reduction->init_from_buffer)) {
				if(reduction.has_value()) { read_requirements = region_union(read_requirements, scalar_box); }
				const auto last_writers = buffer.last_writers.get_region_values(read_requirements);

				region_builder<3> uninitialized_reads;
				for(const auto& [box, writer] : last_writers) {
					// host-initialized buffers are last-written by the current epoch
					if(writer != nullptr) {
						add_dependency(tsk, *writer, dependency_kind::true_dep, dependency_origin::dataflow);
					} else if(m_policy.uninitialized_read_error != error_policy::ignore) {
						uninitialized_reads.add(box);
					}
				}
				if(!uninitialized_reads.empty()) {
					bool is_pure_consumer_access = true;
					for(size_t i = 0; i < access_map.get_num_accesses(); ++i) {
						const auto [b, mode] = access_map.get_nth_access(i);
						if(b == bid && is_producer_mode(mode)) {
							is_pure_consumer_access = false;
							break;
						}
					}
					const auto verb = is_pure_consumer_access ? "reading" : "consuming";
					const auto advice = is_pure_consumer_access ? "" : " Make sure to construct the accessor with no_init if this was unintentional.";

					utils::report_error(m_policy.uninitialized_read_error, "{} declares a {} access on uninitialized {} {}.{}",
					    print_task_debug_label(tsk, true /* title case */), verb, print_buffer_debug_label(bid), std::move(uninitialized_reads).into_region(),
					    advice);
				}
			}

			// Update last writers and determine anti-dependencies
			auto write_requirements = tsk.get_buffer_access_map().get_task_produced_region(bid);
			if(!write_requirements.empty() || reduction.has_value()) {
				if(reduction.has_value()) { write_requirements = region_union(write_requirements, scalar_box); }
				if(write_requirements.empty()) continue;

				const auto last_writers = buffer.last_writers.get_region_values(write_requirements);
				for(const auto& [box, writer] : last_writers) {
					if(writer == nullptr) continue;

					// Determine anti-dependencies by looking at all the dependents of the last writing task
					bool has_anti_dependents = false;

					for(const auto dependent : writer->get_dependents()) {
						if(dependent.node->get_id() == tsk.get_id()) {
							// This can happen
							// - if a task writes to two or more buffers with the same last writer
							// - if the task itself also needs read access to that buffer (R/W access)
							continue;
						}
						const auto dependent_read_requirements = dependent.node->get_buffer_access_map().get_task_consumed_region(bid);
						// Only add an anti-dependency if we are really writing over the region read by this task
						if(!region_intersection(write_requirements, dependent_read_requirements).empty()) {
							add_dependency(tsk, *dependent.node, dependency_kind::anti_dep, dependency_origin::dataflow);
							has_anti_dependents = true;
						}
					}

					if(!has_anti_dependents) {
						// If no intermediate consumers exist, add an anti-dependency on the last writer directly.
						// Note that unless this task is a pure producer, a true dependency will be created and this is a no-op.
						// While it might not always make total sense to have anti-dependencies between (pure) producers without an
						// intermediate consumer, we at least have a defined behavior, and the thus enforced ordering of tasks
						// likely reflects what the user expects.
						add_dependency(tsk, *writer, dependency_kind::anti_dep, dependency_origin::dataflow);
					}
				}

				buffer.last_writers.update_region(write_requirements, &tsk);
			}
		}

		for(const auto& side_effect : tsk.get_side_effect_map()) {
			const auto [hoid, order] = side_effect;
			auto& host_object = m_host_objects.at(hoid);
			if(host_object.last_side_effect != nullptr) {
				add_dependency(tsk, *host_object.last_side_effect, dependency_kind::true_dep, dependency_origin::dataflow);
			}
			host_object.last_side_effect = &tsk;
		}

		if(auto cgid = tsk.get_collective_group_id(); cgid != 0) {
			if(const auto prev = m_last_collective_tasks.find(cgid); prev != m_last_collective_tasks.end()) {
				add_dependency(tsk, *prev->second, dependency_kind::true_dep, dependency_origin::collective_group_serialization);
				m_last_collective_tasks.erase(prev);
			}
			m_last_collective_tasks.emplace(cgid, &tsk);
		}

		// Tasks without any other true-dependency must depend on the last epoch to ensure they cannot be re-ordered before the epoch
		// Exception is the initial epoch, which is the only TDAG node without a predecessor.
		assert(m_epoch_for_new_tasks != nullptr || (tsk.get_type() == task_type::epoch && tsk.get_epoch_action() == epoch_action::init));
		if(m_epoch_for_new_tasks != nullptr) {
			if(const auto deps = tsk.get_dependencies();
			    std::none_of(deps.begin(), deps.end(), [](const task::dependency d) { return d.kind == dependency_kind::true_dep; })) {
				add_dependency(tsk, *m_epoch_for_new_tasks, dependency_kind::true_dep, dependency_origin::last_epoch);
			}
		}
	}

	task& task_manager::register_task_internal(std::unique_ptr<task> task) {
		// register_task_internal() is called for all task types, so we use this location to assert that the init epoch is submitted first and exactly once
		assert((task->get_id() == initial_epoch_task) == (task->get_type() == task_type::epoch && task->get_epoch_action() == epoch_action::init)
		       && "first task submitted is not an init epoch, or init epoch is not the first task submitted");

		const auto tsk = m_tdag.retain_in_current_epoch(std::move(task));
		m_execution_front.insert(tsk);
		return *tsk;
	}

	void task_manager::invoke_callbacks(const task* tsk) const {
		if(m_delegate != nullptr) { m_delegate->task_created(tsk); }
		if(m_task_recorder != nullptr) {
			m_task_recorder->record(std::make_unique<task_record>(*tsk, [this](const buffer_id bid) { return m_buffers.at(bid).debug_name; }));
		}
	}

	void task_manager::add_dependency(task& depender, task& dependee, dependency_kind kind, dependency_origin origin) {
		assert(&depender != &dependee);
		depender.add_dependency({&dependee, kind, origin});
		m_execution_front.erase(&dependee);
		m_max_pseudo_critical_path_length = std::max(m_max_pseudo_critical_path_length, depender.get_pseudo_critical_path_length());
		if(m_task_recorder != nullptr) { m_task_recorder->record_dependency({dependee.get_id(), depender.get_id(), kind, origin}); }
	}

	bool task_manager::need_new_horizon() const {
		const bool need_seq_horizon = m_max_pseudo_critical_path_length - m_current_horizon_critical_path_length >= m_task_horizon_step_size;
		const bool need_para_horizon = static_cast<int>(m_execution_front.size()) >= m_task_horizon_max_parallelism;
		const bool need_horizon = need_seq_horizon || need_para_horizon;
		CELERITY_TRACE("Horizon decision: {} - seq: {} para: {} - crit_p: {} exec_f: {}", need_horizon, need_seq_horizon, need_para_horizon,
		    m_current_horizon_critical_path_length, m_execution_front.size());
		return need_horizon;
	}

	task& task_manager::reduce_execution_front(std::unique_ptr<task> new_front) {
		// add dependencies from a copy of the front to this task
		const auto current_front = m_execution_front;
		for(task* front_task : current_front) {
			add_dependency(*new_front, *front_task, dependency_kind::true_dep, dependency_origin::execution_front);
		}
		assert(m_execution_front.empty());
		return register_task_internal(std::move(new_front));
	}

	void task_manager::set_epoch_for_new_tasks(task* const epoch) {
		// apply the new epoch to buffers_last_writers and last_collective_tasks data structs
		for(auto& [_, buffer] : m_buffers) {
			buffer.last_writers.apply_to_values([epoch](task* const tsk) -> task* {
				if(tsk == nullptr) return nullptr;
				return tsk->get_id() < epoch->get_id() ? epoch : tsk;
			});
		}
		for(auto& [_, tsk] : m_last_collective_tasks) {
			if(tsk->get_id() < epoch->get_id()) { tsk = epoch; }
		}
		for(auto& [_, host_object] : m_host_objects) {
			if(host_object.last_side_effect != nullptr && host_object.last_side_effect->get_id() < epoch->get_id()) { host_object.last_side_effect = epoch; }
		}

		m_epoch_for_new_tasks = epoch;
	}

	task_id task_manager::generate_horizon_task() {
		const auto tid = m_next_tid++;
		m_tdag.begin_epoch(tid);
		auto unique_horizon = task::make_horizon(tid);

		m_current_horizon_critical_path_length = m_max_pseudo_critical_path_length;
		const auto previous_horizon = m_current_horizon;
		m_current_horizon = unique_horizon.get();

		const task& new_horizon = reduce_execution_front(std::move(unique_horizon));
		if(previous_horizon != nullptr) { set_epoch_for_new_tasks(previous_horizon); }

		invoke_callbacks(&new_horizon);
		return tid;
	}

	task_id task_manager::generate_epoch_task(epoch_action action, std::unique_ptr<task_promise> promise) {
		const auto tid = m_next_tid++;
		m_tdag.begin_epoch(tid);

		task& new_epoch = reduce_execution_front(task::make_epoch(tid, action, std::move(promise)));
		compute_dependencies(new_epoch);
		set_epoch_for_new_tasks(&new_epoch);

		m_current_horizon = nullptr; // this horizon is now behind the epoch_for_new_tasks, so it will never become an epoch itself
		m_current_horizon_critical_path_length = m_max_pseudo_critical_path_length; // the explicit epoch resets the need to create horizons

		invoke_callbacks(&new_epoch);

		// On shutdown, attempt to detect suspiciously high numbers of previous user-generated epochs
		if(action != epoch_action::shutdown) {
			m_num_user_epochs_generated++;
		} else if(m_num_user_command_groups_submitted > 100 && m_num_user_epochs_generated * 10 >= m_num_user_command_groups_submitted) {
			CELERITY_WARN("Your program appears to call queue::wait() excessively, which may lead to performance degradation. Consider using queue::fence() "
			              "for data-dependent branching and employ queue::wait() for timing only on a very coarse granularity.");
		}

		return tid;
	}

	task_id task_manager::generate_fence_task(buffer_access access, std::unique_ptr<task_promise> fence_promise) {
		const auto tid = m_next_tid++;
		std::vector<buffer_access> buffer_accesses;
		buffer_accesses.push_back(std::move(access));
		buffer_access_map bam({std::move(buffer_accesses)}, task_geometry{});
		task& tsk = register_task_internal(task::make_fence(tid, std::move(bam), {}, std::move(fence_promise)));
		compute_dependencies(tsk);
		invoke_callbacks(&tsk);
		return tid;
	}

	task_id task_manager::generate_fence_task(host_object_effect effect, std::unique_ptr<task_promise> fence_promise) {
		const auto tid = m_next_tid++;
		task& tsk = register_task_internal(task::make_fence(tid, {}, {{effect}}, std::move(fence_promise)));
		compute_dependencies(tsk);
		invoke_callbacks(&tsk);
		return tid;
	}

	std::string task_manager::print_buffer_debug_label(const buffer_id bid) const { return utils::make_buffer_debug_label(bid, m_buffers.at(bid).debug_name); }

} // namespace detail
} // namespace celerity
