#pragma once

#include <functional>
#include <memory>
#include <unordered_map>

#include <SYCL/sycl.hpp>
#include <boost/variant.hpp>

#include "graph.h"
#include "handler.h"
#include "logger.h"
#include "task.h"
#include "types.h"

namespace celerity {

template <typename DataT, int Dims>
class buffer;

/*
 * TODO: distr_queue has a bit of an identity crisis
 * On the one hand, it encapsulates the SYCL queue and is thus responsible for compute tasks (kernels), which makes
 * it a reasonable location for storing tasks as well as the task graph. Things are complicated by the introduction of
 * "single-tasks" (i.e., celerity::with_master_access, or branching later on): distr_queue needs to know about these tasks,
 * as they also affect the task graph. However, the execution of single-tasks is entirely out of SYCL scope and much better
 * suited to be handled by the runtime class, which has access to all buffers. So distr_queue now stores all tasks, but
 * execution is handled by distr_queue OR the runtime, depending on the task type.
 *
 * ==> Maybe distr_queue should really only be a thin wrapper around the SYCL queue, and task management should be delegated
 * to a dedicated class.
 */
class distr_queue {
  public:
	distr_queue();
	distr_queue(cl::sycl::device& device);

	~distr_queue();

	distr_queue(const distr_queue& other) = delete;
	distr_queue(distr_queue&& other) = delete;

	template <typename CGF>
	void submit(CGF cgf) {
		const auto tid = add_task(std::make_shared<compute_task>(std::make_unique<detail::cgf_storage<CGF>>(cgf)));
		compute_prepass_handler h(*this, tid);
		cgf(h);
	}

	// ----------- CELERITY INTERNAL -----------
	// TODO: Consider employing the pimpl-pattern to hide all these internal member functions

	template <typename MAF>
	task_id create_master_access_task(MAF maf) {
		auto tid = add_task(std::make_shared<master_access_task>(std::make_unique<detail::maf_storage<MAF>>(maf)));
		task_graph[tid].label = fmt::format("{} ({})", task_graph[tid].label, "master-access");
		return tid;
	}

	const task_dag& get_task_graph() const { return task_graph; }

	std::shared_ptr<const task> get_task(task_id tid) const {
		assert(task_map.count(tid) != 0);
		return task_map.at(tid);
	}

	void mark_task_as_processed(task_id tid);

	/**
	 * Returns true iff task_a has a dependency on task_b within the task graph.
	 */
	bool has_dependency(task_id task_a, task_id task_b) const;

	/**
	 * @brief Executes the kernel associated with task @p tid over the chunk @p chnk.
	 */
	template <int Dims>
	cl::sycl::event execute(task_id tid, chunk<Dims> chnk) {
		assert(task_map.count(tid) != 0);
		assert(task_map[tid]->get_type() == task_type::COMPUTE);
		auto task = std::static_pointer_cast<compute_task>(task_map[tid]);
		auto& cgf = task->get_command_group();
		return sycl_queue->submit([this, &cgf, tid, task, chnk](cl::sycl::handler& sycl_handler) {
			compute_livepass_handler h(*this, tid, task, chnk, &sycl_handler);
			cgf(h);
		});
	}

	bool is_ocl_profiling_enabled() const { return ocl_profiling_enabled; }

  private:
	friend compute_prepass_handler;
	friend master_access_prepass_handler;

	bool ocl_profiling_enabled = false;

	std::unordered_map<task_id, std::shared_ptr<task>> task_map;

	// This is a high-level view on buffer writers, for creating the task graph
	// NOTE: This represents the state after the latest performed pre-pass, i.e.
	// it corresponds to the leaf nodes of the current task graph.
	std::unordered_map<buffer_id, task_id> buffer_last_writer;

	size_t task_count = 0;
	task_dag task_graph;

	std::unique_ptr<cl::sycl::queue> sycl_queue;

	void init(cl::sycl::device* device_ptr);

	task_id add_task(std::shared_ptr<task> tsk);

	/**
	 * Adds requirement for a compute task, including a range mapper.
	 */
	void add_requirement(task_id tid, buffer_id bid, cl::sycl::access::mode mode, std::unique_ptr<detail::range_mapper_base> rm);

	/**
	 * Adds requirement for a master-access task, with plain ranges.
	 */
	void add_requirement(task_id tid, buffer_id bid, cl::sycl::access::mode mode, cl::sycl::range<3> range, cl::sycl::id<3> offset);

	void set_task_data(task_id tid, any_range global_size, std::string debug_name);

	void update_dependencies(task_id tid, buffer_id bid, cl::sycl::access::mode mode);

	static void handle_async_exceptions(cl::sycl::exception_list el);
};

} // namespace celerity
