#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>

#include "handler.h"
#include "host_queue.h"
#include "region_map.h"
#include "task.h"
#include "types.h"

namespace celerity {
namespace detail {

	class logger;
	using task_callback = std::function<void(task_id)>;

	class task_manager {
		using buffer_writers_map = std::unordered_map<buffer_id, region_map<std::optional<task_id>>>;

	  public:
		/**
		 * The task_manager operates differently based on whether this is the master node or not.
		 * In the first case, in addition to storing command groups within task objects, the task graph is additionally computed.
		 *
		 * TODO: This is a bit of a code smell. Maybe we should split simple task management and task graph generation into separate classes?
		 */
		task_manager(size_t num_collective_nodes, host_queue* queue, bool is_master_node);
		virtual ~task_manager() = default;

		template <typename CGF, typename... Hints>
		task_id create_task(CGF cgf, Hints... hints) {
			task_id tid;
			{
				std::lock_guard lock(task_mutex);
				tid = next_task_id++;
				prepass_handler cgh(tid, std::make_unique<command_group_storage<CGF>>(cgf), num_collective_nodes);
				cgf(cgh);
				auto task = std::move(cgh).into_task();
				assert(task != nullptr);
				task_map.emplace(tid, task);
				if(is_master_node) { compute_dependencies(tid); }
				if(queue) queue->require_collective_group(task->get_collective_group_id());
			}
			invoke_callbacks(tid);
			return tid;
		}

		/**
		 * @brief Registers a new callback that will be called whenever a new task is created.
		 */
		void register_task_callback(task_callback cb) { task_callbacks.push_back(cb); }

		/**
		 * @brief Adds a new buffer for dependency tracking
		 * @arg host_initialized Whether this buffer has been initialized using a host pointer (i.e., it contains useful data before any write-task)
		 */
		void add_buffer(buffer_id bid, const cl::sycl::range<3>& range, bool host_initialized);

		/**
		 * @brief Checks whether a task has already been registered with the queue.
		 *
		 * This is useful for scenarios where the master node sends out commands concerning tasks
		 * that have not yet been registered through the local execution of the user program.
		 */
		bool has_task(task_id tid) const;

		std::shared_ptr<const task> get_task(task_id tid) const;

		/**
		 * @brief Returns the id of the INIT task which acts as a surrogate for the host-initialization of buffers.
		 *
		 * While this is always 0, having this method makes code dealing with the INIT task more explicit.
		 */
		task_id get_init_task_id() const {
			assert(init_task_id == 0);
			return init_task_id;
		}

		void print_graph(logger& graph_logger) const;

		/**
		 * @brief Shuts down the task_manager, freeing all stored tasks.
		 */
		void shutdown() { task_map.clear(); }

	  private:
		const size_t num_collective_nodes;
		host_queue* queue;
		const bool is_master_node;
		task_id next_task_id = 0;
		const task_id init_task_id;
		std::unordered_map<task_id, std::shared_ptr<task>> task_map;

		// We store a map of which task last wrote to a certain region of a buffer.
		// NOTE: This represents the state after the latest performed pre-pass.
		buffer_writers_map buffers_last_writers;

		std::unordered_map<collective_group_id, task_id> last_collective_tasks;

		// For simplicity we use a single mutex to control access to all task-related (i.e. the task graph, task_map, ...) data structures.
		mutable std::mutex task_mutex;

		std::vector<task_callback> task_callbacks;

		void invoke_callbacks(task_id tid);

	  protected:
		virtual void compute_dependencies(task_id tid);
	};

} // namespace detail
} // namespace celerity
