#pragma once

#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

#include "region_map.h"
#include "task.h"
#include "types.h"

namespace celerity {
namespace detail {

	class task_recorder;

	// Allows other threads to await an epoch change in the task manager.
	// This is worth a separate class to encapsulate the synchronization behavior.
	class epoch_monitor {
	  public:
		explicit epoch_monitor(const task_id epoch) : m_this_epoch(epoch) {}

		task_id get() const {
			std::lock_guard lock{m_mutex};
			return m_this_epoch;
		}

		task_id await(const task_id min_tid_reached) const {
			std::unique_lock lock{m_mutex};
			m_epoch_changed.wait(lock, [&] { return m_this_epoch >= min_tid_reached; });
			return m_this_epoch;
		}

		void set(const task_id epoch) {
			{
				std::lock_guard lock{m_mutex};
				assert(epoch >= m_this_epoch);
				m_this_epoch = epoch;
			}
			m_epoch_changed.notify_all();
		}

	  private:
		task_id m_this_epoch = 0;
		mutable std::mutex m_mutex;
		mutable std::condition_variable m_epoch_changed;
	};

	// definition is in handler.h to avoid circular dependency
	template <typename CGF>
	std::unique_ptr<task> invoke_command_group_function(const task_id tid, size_t num_collective_nodes, CGF&& cgf);

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

		task_manager(
		    size_t num_collective_nodes, detail::task_recorder* recorder, task_manager::delegate* dlg, const policy_set& policy = default_policy_set());

		template <typename CGF>
		task_id submit_command_group(CGF&& cgf) {
			const auto tid = m_next_tid++;
			auto unique_tsk = invoke_command_group_function(tid, m_num_collective_nodes, std::forward<CGF>(cgf));
			auto& tsk = register_task_internal(std::move(unique_tsk));
			compute_dependencies(tsk);
			invoke_callbacks(&tsk);
			if(need_new_horizon()) { generate_horizon_task(); }
			++m_num_user_command_groups_submitted;
			return tid;
		}

		/**
		 * Inserts an epoch task that depends on the entire execution front and that immediately becomes the current epoch_for_new_tasks and the last writer
		 * for all buffers.
		 */
		task_id generate_epoch_task(epoch_action action);

		task_id generate_fence_task(buffer_access_map access_map, side_effect_map side_effects, std::unique_ptr<fence_promise> fence_promise);

		/**
		 * @brief Adds a new buffer for dependency tracking
		 * @param host_initialized Whether this buffer has been initialized using a host pointer (i.e., it contains useful data before any write-task)
		 */
		void notify_buffer_created(buffer_id bid, const range<3>& range, bool host_initialized);

		void notify_buffer_debug_name_changed(buffer_id bid, const std::string& name);

		void notify_buffer_destroyed(buffer_id bid);

		void notify_host_object_created(host_object_id hoid);

		void notify_host_object_destroyed(host_object_id hoid);

		/**
		 * Blocks until an epoch task has executed on this node (or all nodes, if the epoch_for_new_tasks was created with `epoch_action::barrier`).
		 */
		void await_epoch(task_id epoch);

		void set_horizon_step(const int step) {
			assert(step >= 0);
			m_task_horizon_step_size = step;
		}

		void set_horizon_max_parallelism(const int para) {
			assert(para >= 1);
			m_task_horizon_max_parallelism = para;
		}

		/**
		 * @brief Notifies the task manager that the given horizon has been executed (used for task deletion).
		 *
		 * notify_horizon_reached and notify_epoch_reached must only ever be called from a single thread, but that thread does not have to be the main
		 * thread.
		 */
		void notify_horizon_reached(task_id horizon_tid);

		/**
		 * @brief Notifies the task manager that the given epoch has been executed on this node.
		 *
		 * notify_horizon_reached and notify_epoch_reached must only ever be called from a single thread, but that thread does not have to be the main
		 * thread.
		 */
		void notify_epoch_reached(task_id epoch_tid);

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

		const size_t m_num_collective_nodes;
		policy_set m_policy;

		task_graph m_tdag;
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

		// The last horizon processed by the executor will become the latest_epoch_reached once the next horizon is completed as well.
		// Only accessed in task_manager::notify_*, which are always called from the executor thread - no locking needed.
		std::optional<task_id> m_latest_horizon_reached;

		// The last epoch task that has been processed by the executor. Behind a monitor to allow awaiting this change from the main thread.
		epoch_monitor m_latest_epoch_reached{initial_epoch_task};

		// Track the number of user-generated task and epochs to heuristically detect programs that lose performance by frequently calling `queue::wait()`.
		size_t m_num_user_command_groups_submitted = 0;
		size_t m_num_user_epochs_generated = 0;

		// Set of tasks with no dependents
		std::unordered_set<task*> m_execution_front;

		// An optional task_recorder which records information about tasks for e.g. printing graphs.
		mutable detail::task_recorder* m_task_recorder;

		task& register_task_internal(std::unique_ptr<task> task);

		void invoke_callbacks(const task* tsk) const;

		void add_dependency(task& depender, task& dependee, dependency_kind kind, dependency_origin origin);

		bool need_new_horizon() const;

		task& reduce_execution_front(std::unique_ptr<task> new_front);

		void set_epoch_for_new_tasks(task* epoch);

		task_id generate_horizon_task();

		void compute_dependencies(task& tsk);

		std::string print_buffer_debug_label(buffer_id bid) const;
	};

} // namespace detail
} // namespace celerity
