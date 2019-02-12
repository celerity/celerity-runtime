#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>

#include <boost/optional.hpp>

#include "graph.h"
#include "handler.h"
#include "region_map.h"
#include "task.h"
#include "types.h"

namespace celerity {
namespace detail {

	class logger;
	using task_callback = std::function<void(void)>;

	class task_manager {
		using buffer_writers_map = std::unordered_map<buffer_id, region_map<boost::optional<task_id>>>;

	  public:
		task_manager();
		virtual ~task_manager() = default;

		template <typename CGF>
		void create_compute_task(CGF cgf) {
			{
				std::lock_guard<std::mutex> lock(task_mutex);
				const auto task = create_task<compute_task>(std::make_unique<cgf_storage<CGF>>(cgf));
				compute_prepass_handler h(*task);
				cgf(h);
				task_graph[task->get_id()].label = fmt::format("{} ({})", task_graph[task->get_id()].label, task->get_debug_name());
				compute_dependencies(task->get_id());
			}
			invoke_callbacks();
		}

		template <typename MAF>
		void create_master_access_task(MAF maf) {
			{
				std::lock_guard<std::mutex> lock(task_mutex);
				const auto task = create_task<master_access_task>(std::make_unique<maf_storage<MAF>>(maf));
				master_access_prepass_handler h(*task);
				maf(h);
				task_graph[task->get_id()].label = fmt::format("{} ({})", task_graph[task->get_id()].label, "master-access");
				compute_dependencies(task->get_id());
			}
			invoke_callbacks();
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

		// TODO: See if we can get rid of this entirely, effectively making the task graph an implementation detail.
		locked_graph<const task_dag> get_task_graph() const;

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

		void mark_task_as_processed(task_id tid);

		void print_graph(logger& graph_logger) const;

	  private:
		task_id next_task_id = 0;
		const task_id init_task_id;
		std::unordered_map<task_id, std::shared_ptr<task>> task_map;

		// We store a map of which task last wrote to a certain region of a buffer.
		// NOTE: This represents the state after the latest performed pre-pass.
		buffer_writers_map buffers_last_writers;

		task_dag task_graph;

		// For simplicity we use a single mutex to control access to all task-related (i.e. the task graph, task_map, ...) data structures.
		mutable std::mutex task_mutex;

		std::vector<task_callback> task_callbacks;

		template <typename Task, typename... Args>
		std::shared_ptr<Task> create_task(Args... args) {
			const task_id tid = next_task_id++;
			const auto task = std::make_shared<Task>(tid, std::forward<Args...>(args...));
			task_map[tid] = task;
			boost::add_vertex(task_graph);
			task_graph[tid].label = fmt::format("Task {}", static_cast<size_t>(tid));
			return task;
		}

		void invoke_callbacks();

	  protected:
		virtual void compute_dependencies(task_id tid);
	};

	// The simple_task_manager is intended for use on worker_nodes, omitting task graph generation.
	// TODO: This is a bit of a code smell. Maybe we should split the simple task management and task graph generation into separate classes
	class simple_task_manager : public task_manager {
	  protected:
		void compute_dependencies(task_id) override{};
	};

} // namespace detail
} // namespace celerity
