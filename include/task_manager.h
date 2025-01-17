#pragma once

#include "cgf.h"
#include "intrusive_graph.h"
#include "loop_template.h" // NOCOMMIT Forward declare and move to impl
#include "ranges.h"
#include "region_map.h"
#include "task.h"
#include "types.h"

#include <cassert>
#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>


namespace celerity {
namespace detail {

	class task_recorder;

	// TODO rename to task_graph_generator eventually
	class task_manager {
		friend struct task_manager_testspy;

	  public:
		/// Implement this as the owner of task_manager to receive callbacks on generated tasks.
		class delegate {
		  protected:
			delegate() = default;
			delegate(const delegate&) = default;
			delegate(delegate&&) = default;
			delegate& operator=(const delegate&) = default;
			delegate& operator=(delegate&&) = default;
			~delegate() = default;

		  public:
			/// Called whenever new tasks have been generated and inserted into the tasks graph.
			virtual void task_created(const task* tsk) = 0;
		};

		struct policy_set {
			error_policy uninitialized_read_error = error_policy::panic;
		};

		task_manager(size_t num_collective_nodes, task_graph& tdag, detail::task_recorder* recorder, task_manager::delegate* dlg,
		    const policy_set& policy = default_policy_set());

		// TODO pimpl this - ctors are explicitly deleted / defaulted to avoid lint on `m_tdag` reference member
		task_manager(const task_manager&) = delete;
		task_manager(task_manager&&) = delete;
		task_manager& operator=(const task_manager) = delete;
		task_manager& operator=(task_manager&&) = delete;
		~task_manager() = default;

		void finalize_loop_template(loop_template& loop_templ) {
			// CELERITY_CRITICAL("TDAG: Finalizing template!"); //

			const auto repl_map = loop_templ.tdag.get_replacement_map();

			// TODO: Could optimize this to only update touched buffers
			for(auto& [bid, buffer] : m_buffers) {
				buffer.last_writers.apply_to_values([&repl_map](task* const tsk) -> task* {
					if(tsk == nullptr) return nullptr;
					const auto it = repl_map.find(tsk);
					if(it == repl_map.end()) return tsk;
					return it->second;
				});
			}

			for(const auto& [from, to] : repl_map) {
				if(m_execution_front.contains(from)) {
					m_execution_front.erase(from);
					m_execution_front.insert(to);
				}
			}
		}

		void begin_loop_template_iteration(loop_template& templ) {
			// NOCOMMIT Do we just ignore horizon step setting while loop template is active? Always generate one horizon at end of loop?
			// => Makes things more difficult to test though (need to adjust horizon step for ground truth)
			generate_horizon_task(&templ);
		}

		// NOCOMMIT TODO: Other task types as well
		task_id generate_command_group_task(raw_command_group&& cg, loop_template* loop_templ = nullptr) {
			const auto tid = m_next_tid++;
			auto unique_tsk = detail::make_command_group_task(tid, m_num_collective_nodes, std::move(cg));
			auto& tsk = register_task_internal(std::move(unique_tsk));

			if(loop_templ != nullptr) {
				if(!loop_templ->tdag.is_primed) {
					compute_dependencies(tsk);
					loop_templ->tdag.prime(tsk);
				} else if(!loop_templ->tdag.is_verified) {
					compute_dependencies(tsk);
					loop_templ->tdag.verify(tsk);
				} else {
					loop_templ->tdag.apply(
					    tsk, [this](task* from, task* to, dependency_kind kind, dependency_origin origin) { add_dependency(*from, *to, kind, origin); });
				}
			} else {
				compute_dependencies(tsk);
			}

			invoke_callbacks(&tsk);
			// Only generate a horizon if we are not in a loop template. Otherwise we generate a horizon at the end of the loop.
			if(need_new_horizon() && loop_templ == nullptr) { generate_horizon_task(); }
			++m_num_user_command_groups_submitted;
			return tid;
		}

		/**
		 * Inserts an epoch task that depends on the entire execution front and that immediately becomes the current epoch_for_new_tasks and the last writer
		 * for all buffers.
		 */
		task_id generate_epoch_task(epoch_action action, std::unique_ptr<task_promise> promise = nullptr);

		task_id generate_fence_task(buffer_access access, std::unique_ptr<task_promise> fence_promise);

		task_id generate_fence_task(host_object_effect effect, std::unique_ptr<task_promise> fence_promise);

		/**
		 * @brief Adds a new buffer for dependency tracking
		 * @param host_initialized Whether this buffer has been initialized using a host pointer (i.e., it contains useful data before any write-task)
		 */
		void notify_buffer_created(buffer_id bid, const range<3>& range, bool host_initialized);

		void notify_buffer_debug_name_changed(buffer_id bid, const std::string& name);

		void notify_buffer_destroyed(buffer_id bid);

		void notify_host_object_created(host_object_id hoid);

		void notify_host_object_destroyed(host_object_id hoid);

		void set_horizon_step(const int step) {
			assert(step >= 0);
			m_task_horizon_step_size = step;
		}

		void set_horizon_max_parallelism(const int para) {
			assert(para >= 1);
			m_task_horizon_max_parallelism = para;
		}

	  private:
		// default-constructs a policy_set - this must be a function because we can't use the implicit default constructor of policy_set, which has member
		// initializers, within its surrounding class (Clang)
		constexpr static policy_set default_policy_set() { return {}; }

		struct buffer_state {
			std::string debug_name;
			region_map<task*> last_writers; ///< nullopt for uninitialized regions

			explicit buffer_state(const range<3>& range) : last_writers(range) {}
		};

		struct host_object_state {
			task* last_side_effect = nullptr;
		};

		static constexpr task_id initial_epoch_task = 0;

		task_manager::delegate* m_delegate;

		size_t m_num_collective_nodes;
		policy_set m_policy;

		task_graph& m_tdag;
		task_id m_next_tid = initial_epoch_task;

		// The active epoch is used as the last writer for host-initialized buffers.
		// This is useful so we can correctly generate anti-dependencies onto tasks that read host-initialized buffers.
		// To ensure correct ordering, all tasks that have no other true-dependencies depend on this task.
		task* m_epoch_for_new_tasks = nullptr;

		std::unordered_map<buffer_id, buffer_state> m_buffers;

		std::unordered_map<collective_group_id, task*> m_last_collective_tasks;

		std::unordered_map<host_object_id, host_object_state> m_host_objects;

		// Maximum critical path length in the task graph before inserting a horizon
		// While it seems like this value could have a significant performance impact and might need to be tweaked per-platform,
		// benchmarking results so far indicate that as long as some sufficiently small value is chosen, there is a broad range
		// of values which all provide good performance results across a variety of workloads.
		// More information can be found in this paper: https://link.springer.com/chapter/10.1007/978-3-031-32316-4_2
		int m_task_horizon_step_size = 4;

		// Maximum number of independent tasks (task graph breadth) allowed in a single horizon step
		int m_task_horizon_max_parallelism = 64;

		// This only (potentially) grows when adding dependencies,
		// it never shrinks and does not take into account later changes further up in the dependency chain
		int m_max_pseudo_critical_path_length = 0;

		// The maximum critical path length of tasks within the current horizon frame
		int m_current_horizon_critical_path_length = 0;

		// The latest horizon task created. Will be applied as the epoch for new tasks once the next horizon is created.
		task* m_current_horizon = nullptr;

		// Track the number of user-generated task and epochs to heuristically detect programs that lose performance by frequently calling `queue::wait()`.
		size_t m_num_user_command_groups_submitted = 0;
		size_t m_num_user_epochs_generated = 0;

		// Set of tasks with no dependents
		std::unordered_set<task*> m_execution_front;

		// An optional task_recorder which records information about tasks for e.g. printing graphs.
		detail::task_recorder* m_task_recorder;

		task& register_task_internal(std::unique_ptr<task> task);

		void invoke_callbacks(const task* tsk) const;

		void add_dependency(task& depender, task& dependee, dependency_kind kind, dependency_origin origin);

		bool need_new_horizon() const;

		task& reduce_execution_front(std::unique_ptr<task> new_front);

		void set_epoch_for_new_tasks(task* epoch);

		// NOCOMMIT Figure out a better way than passing the loop template around
		task_id generate_horizon_task(loop_template* const templ = nullptr);

		void compute_dependencies(task& tsk);

		std::string print_buffer_debug_label(buffer_id bid) const;
	};

} // namespace detail
} // namespace celerity
