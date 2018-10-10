#pragma once

#include <functional>
#include <unordered_map>

#include "graph.h"
#include "handler.h"
#include "task.h"
#include "types.h"

namespace celerity {

class logger;

namespace detail {

	using task_callback = std::function<void(void)>;

	class task_manager {
	  public:
		template <typename CGF>
		void create_compute_task(CGF cgf) {
			{
				std::lock_guard<std::mutex> lock(task_mutex);
				const auto tid = add_task(std::make_shared<compute_task>(std::make_unique<cgf_storage<CGF>>(cgf)));
				compute_prepass_handler h(*this, tid);
				cgf(h);
			}
			invoke_callbacks();
		}

		template <typename MAF>
		void create_master_access_task(MAF maf) {
			{
				std::lock_guard<std::mutex> lock(task_mutex);
				const auto tid = add_task(std::make_shared<master_access_task>(std::make_unique<maf_storage<MAF>>(maf)));
				task_graph[tid].label = fmt::format("{} ({})", task_graph[tid].label, "master-access");
				master_access_prepass_handler h(*this, tid);
				maf(h);
			}
			invoke_callbacks();
		}

		/**
		 * @brief Registers a new callback that will be called whenever a new task is created.
		 */
		void register_task_callback(task_callback cb) { task_callbacks.push_back(cb); }

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

		void mark_task_as_processed(task_id tid);

		/**
		 * Returns true iff task_a has a dependency on task_b within the task graph.
		 */
		bool has_dependency(task_id task_a, task_id task_b) const;

		void print_graph(std::shared_ptr<logger>& graph_logger) const;

		/**
		 * Adds requirement for a compute task, including a range mapper.
		 */
		void add_requirement(task_id tid, buffer_id bid, cl::sycl::access::mode mode, std::unique_ptr<detail::range_mapper_base> rm);

		/**
		 * Adds requirement for a master-access task, with plain ranges.
		 */
		void add_requirement(task_id tid, buffer_id bid, cl::sycl::access::mode mode, cl::sycl::range<3> range, cl::sycl::id<3> offset);

		void set_compute_task_data(task_id tid, int dimensions, cl::sycl::range<3> global_size, std::string debug_name);

	  private:
		std::unordered_map<task_id, std::shared_ptr<task>> task_map;

		// This is a high-level view on buffer writers, for creating the task graph
		// NOTE: This represents the state after the latest performed pre-pass, i.e.
		// it corresponds to the leaf nodes of the current task graph.
		std::unordered_map<buffer_id, task_id> buffer_last_writer;

		size_t task_count = 0;
		task_dag task_graph;

		// For simplicity we use a single mutex to control access to all task-related (i.e. the task graph, task_map, ...) data structures.
		mutable std::mutex task_mutex;

		std::vector<task_callback> task_callbacks;

		task_id add_task(std::shared_ptr<task> tsk);

		void update_dependencies(task_id tid, buffer_id bid, cl::sycl::access::mode mode);

		void invoke_callbacks();
	};

} // namespace detail
} // namespace celerity
