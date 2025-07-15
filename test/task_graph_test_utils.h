#pragma once

#include "graph_test_utils.h"


namespace celerity::test_utils {

template <typename Record>
struct task_matcher {
	static bool matches(const Record& tsk, const task_id tid) { return tsk.id == tid; }

	static std::string print_filter(const task_id tid) { return fmt::format("\"T{}\"", tid); }
};

template <typename R>
using task_query = graph_query<R, task_record, task_recorder, task_matcher>;

// TODO: Can we make this the base class of cdag / idag test contexts?
class tdag_test_context final : private task_manager::delegate {
	friend class task_builder<tdag_test_context>;

  public:
	struct policy_set {
		task_manager::policy_set tm;
	};

	tdag_test_context(const size_t num_collective_nodes, const policy_set& policy = {})
	    : m_tm(num_collective_nodes, m_tdag, &m_task_recorder, static_cast<task_manager::delegate*>(this), policy.tm) {
		m_initial_epoch_tid = m_tm.generate_epoch_task(epoch_action::init);
	}

	~tdag_test_context() { maybe_print_graphs(); }

	tdag_test_context(const tdag_test_context&) = delete;
	tdag_test_context(tdag_test_context&&) = delete;
	tdag_test_context& operator=(const tdag_test_context&) = delete;
	tdag_test_context& operator=(tdag_test_context&&) = delete;

	void task_created(const task* tsk) override {}

	template <int Dims>
	test_utils::mock_buffer<Dims> create_buffer(range<Dims> size, bool mark_as_host_initialized = false) {
		const buffer_id bid = m_next_buffer_id++;
		const auto buf = test_utils::mock_buffer<Dims>(bid, size);
		m_tm.notify_buffer_created(bid, range_cast<3>(size), mark_as_host_initialized);
		return buf;
	}

	test_utils::mock_host_object create_host_object(const bool owns_instance = true) {
		const host_object_id hoid = m_next_host_object_id++;
		m_tm.notify_host_object_created(hoid);
		return test_utils::mock_host_object(hoid);
	}

	template <typename Name = unnamed_kernel, int Dims>
	auto device_compute(const range<Dims>& global_size, const id<Dims>& global_offset = {}) {
		return task_builder(*this).template device_compute<Name>(global_size, global_offset);
	}

	template <typename Name = unnamed_kernel, int Dims>
	auto device_compute(const nd_range<Dims>& execution_range) {
		return task_builder(*this).template device_compute<Name>(execution_range);
	}

	template <int Dims>
	auto host_task(const range<Dims>& global_size) {
		return task_builder(*this).host_task(global_size);
	}

	auto master_node_host_task() { return task_builder(*this).master_node_host_task(); }

	auto collective_host_task(experimental::collective_group group = detail::default_collective_group) {
		return task_builder(*this).collective_host_task(group);
	}

	task_id fence(test_utils::mock_host_object ho) {
		host_object_effect effect{ho.get_id(), experimental::side_effect_order::sequential};
		return m_tm.generate_fence_task(effect, nullptr);
	}

	template <int Dims>
	task_id fence(test_utils::mock_buffer<Dims> buf, subrange<Dims> sr) {
		buffer_access access{buf.get_id(), access_mode::read,
		    std::make_unique<range_mapper<Dims, celerity::access::fixed<Dims>>>(celerity::access::fixed<Dims>(sr), buf.get_range())};
		return m_tm.generate_fence_task(std::move(access), nullptr);
	}

	template <int Dims>
	task_id fence(test_utils::mock_buffer<Dims> buf) {
		return fence(buf, {{}, buf.get_range()});
	}

	task_id epoch(epoch_action action) { return m_tm.generate_epoch_task(action); }

	// TODO: This is of limited usefulness until we convert task records into a hierarchy
	template <typename SpecificRecord = task_record, typename... Filters>
	task_query<SpecificRecord> query_tasks(Filters... filters) {
		return task_query<task_record>(m_task_recorder).template select_all<SpecificRecord>(std::forward<Filters>(filters)...);
	}

	void set_horizon_step(const int step) { m_tm.set_horizon_step(step); }

	task_graph& get_task_graph() { return m_tdag; }

	task_manager& get_task_manager() { return m_tm; }

	const task_recorder& get_task_recorder() const { return m_task_recorder; }

	task_id get_initial_epoch_task() const { return m_initial_epoch_tid; }

	[[nodiscard]] std::string print_task_graph() { return detail::print_task_graph(m_task_recorder, make_test_graph_title("Task Graph")); }

  private:
	buffer_id m_next_buffer_id = 0;
	host_object_id m_next_host_object_id = 0;
	reduction_id m_next_reduction_id = 1; // Start from 1 as rid 0 designates "no reduction" in push commands
	task_graph m_tdag;
	task_manager m_tm;
	task_recorder m_task_recorder;
	task_id m_initial_epoch_tid = 0;

	reduction_info create_reduction(const buffer_id bid, const bool include_current_buffer_value) {
		return reduction_info{m_next_reduction_id++, bid, include_current_buffer_value};
	}

	template <typename CGF>
	task_id submit_command_group(CGF cgf) {
		return m_tm.generate_command_group_task(invoke_command_group_function(cgf));
	}

	void maybe_print_graphs() {
		if(test_utils::g_print_graphs) { fmt::print("\n{}\n", print_task_graph()); }
	}
};

} // namespace celerity::test_utils
