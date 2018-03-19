#pragma once

#include <functional>
#include <unordered_map>

#include <SYCL/sycl.hpp>
#include <boost/variant.hpp>

#include "buffer.h"
#include "graph.h"
#include "handler.h"
#include "range_mapper.h"
#include "types.h"

namespace celerity {

// experimental / NYI
class branch_handle {
  public:
	template <typename DataT, int Dims>
	void get(buffer<DataT, Dims>, cl::sycl::range<Dims>){};
};

namespace detail {

	// This is a workaround that let's us store a command group functor with auto&
	// parameter, which we require in order to be able to pass different
	// celerity::handlers (celerity::is_prepass::true_t/false_t) for prepass and
	// live invocations.
	struct cgf_storage_base {
		virtual void operator()(handler<is_prepass::true_t>) = 0;
		virtual void operator()(handler<is_prepass::false_t>) = 0;
		virtual ~cgf_storage_base(){};
	};

	template <typename CGF>
	struct cgf_storage : cgf_storage_base {
		CGF cgf;

		cgf_storage(CGF cgf) : cgf(cgf) {}

		void operator()(handler<is_prepass::true_t> cgh) override { cgf(cgh); }
		void operator()(handler<is_prepass::false_t> cgh) override { cgf(cgh); }
	};

} // namespace detail

class distr_queue {
  public:
	// TODO: Device should be selected transparently
	distr_queue(cl::sycl::device device);

	template <typename CGF>
	void distr_queue::submit(CGF cgf) {
		const task_id tid = task_count++;
		boost::add_vertex(task_graph);
		handler<is_prepass::true_t> h(*this, tid);
		cgf(h);
		task_command_groups[tid] = std::make_unique<detail::cgf_storage<CGF>>(cgf);
	}

	const task_dag& get_task_graph() const { return task_graph; }
	const auto& get_task_range_mappers() const { return task_range_mappers; }
	const auto& get_task_global_sizes() const { return task_global_sizes; }

	void mark_task_as_processed(task_id tid);

	// experimental
	// TODO: Can we derive 2nd lambdas args from requested values in 1st?
	void branch(std::function<void(branch_handle& bh)>, std::function<void(float)>) {}

	void debug_print_task_graph();
	void TEST_execute_deferred();

  private:
	friend handler<is_prepass::true_t>;
	// TODO: We may want to move all these task maps into a dedicated struct
	std::unordered_map<task_id, std::unique_ptr<detail::cgf_storage_base>> task_command_groups;
	std::unordered_map<task_id, boost::variant<cl::sycl::range<1>, cl::sycl::range<2>, cl::sycl::range<3>>> task_global_sizes;
	std::unordered_map<task_id, std::unordered_map<buffer_id, std::vector<std::unique_ptr<detail::range_mapper_base>>>> task_range_mappers;

	// This is a high-level view on buffer writers, for creating the task graph
	// NOTE: This represents the state after the latest performed pre-pass, i.e.
	// it corresponds to the leaf nodes of the current task graph.
	std::unordered_map<buffer_id, task_id> buffer_last_writer;

	size_t task_count = 0;
	task_dag task_graph;

	cl::sycl::queue sycl_queue;

	void add_requirement(task_id tid, buffer_id bid, cl::sycl::access::mode mode, std::unique_ptr<detail::range_mapper_base> rm);

	template <int Dims>
	void set_task_data(task_id tid, cl::sycl::range<Dims> global_size, std::string debug_name) {
		task_global_sizes[tid] = global_size;
		task_graph[tid].label = (boost::format("Task %d (%s)") % tid % debug_name).str();
	}
};

} // namespace celerity
