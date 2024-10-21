#pragma once

#include "command_graph_generator_test_utils.h"
#include "instruction_graph_test_utils.h"

namespace celerity::test_utils {

class scheduler_test_context final : private task_manager::delegate {
	friend class task_builder<scheduler_test_context>;

  public:
	scheduler_test_context(const size_t num_nodes, const node_id local_nid, const size_t num_devices_per_node)
	    : m_num_nodes(num_nodes), m_local_nid(local_nid), m_num_devices_per_node(num_devices_per_node), m_tm(num_nodes, m_tdag, &m_task_recorder, this),
	      m_cmd_recorder(), m_instr_recorder(),
	      m_scheduler(std::make_unique<scheduler>(num_nodes, local_nid, make_system_info(num_devices_per_node, true /* supports d2d copies */),
	          nullptr /* delegate */, &m_cmd_recorder, &m_instr_recorder)) //
	{
		REQUIRE(local_nid < num_nodes);
		REQUIRE(num_devices_per_node > 0);
		m_initial_epoch_tid = m_tm.generate_epoch_task(epoch_action::init);
	}

	~scheduler_test_context() {
		// instruction-graph-generator has no exception guarantees, so we must not call further member functions if one of them threw an exception
		finish();
		maybe_print_graphs();
	}

	scheduler_test_context(const scheduler_test_context&) = delete;
	scheduler_test_context(scheduler_test_context&&) = delete;
	scheduler_test_context& operator=(const scheduler_test_context&) = delete;
	scheduler_test_context& operator=(scheduler_test_context&&) = delete;

	/// Call this after issuing all submissions in order to trigger the shutdown epoch together with all cleanup instructions.
	void finish() {
		if(m_scheduler == nullptr) return;
		for(auto iter = m_managed_objects.rbegin(); iter != m_managed_objects.rend(); ++iter) {
			matchbox::match(
			    *iter,
			    [&](const buffer_id bid) {
				    m_tm.notify_buffer_destroyed(bid);
				    m_scheduler->notify_buffer_destroyed(bid);
			    },
			    [&](const host_object_id hoid) {
				    m_tm.notify_host_object_destroyed(hoid);
				    m_scheduler->notify_host_object_destroyed(hoid);
			    });
		}
		epoch(epoch_action::shutdown);
		m_scheduler.reset();
	}

	template <typename DataT, int Dims>
	test_utils::mock_buffer<Dims> create_buffer(range<Dims> size, bool mark_as_host_initialized = false) {
		if(m_scheduler == nullptr) { FAIL("scheduler_test_context already finish()ed"); }
		const buffer_id bid = m_next_buffer_id++;
		const auto buf = test_utils::mock_buffer<Dims>(bid, size);
		m_tm.notify_buffer_created(bid, range_cast<3>(size), mark_as_host_initialized);
		m_scheduler->notify_buffer_created(bid, range_cast<3>(size), sizeof(DataT), alignof(DataT),
		    mark_as_host_initialized ? detail::allocation_id(detail::user_memory_id, m_next_user_allocation_id++) : detail::null_allocation_id);
		m_managed_objects.emplace_back(bid);
		return buf;
	}

	template <int Dims>
	test_utils::mock_buffer<Dims> create_buffer(range<Dims> size, bool mark_as_host_initialized = false) {
		return create_buffer<float, Dims>(size, mark_as_host_initialized);
	}

	test_utils::mock_host_object create_host_object(const bool owns_instance = true) {
		if(m_scheduler == nullptr) { FAIL("scheduler_test_context already finish()ed"); }
		const host_object_id hoid = m_next_host_object_id++;
		m_tm.notify_host_object_created(hoid);
		m_scheduler->notify_host_object_created(hoid, owns_instance);
		m_managed_objects.emplace_back(hoid);
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
		side_effect_map side_effects;
		side_effects.add_side_effect(ho.get_id(), experimental::side_effect_order::sequential);
		return fence({}, std::move(side_effects), std::make_unique<mock_host_object_fence_promise>());
	}

	template <int Dims>
	task_id fence(test_utils::mock_buffer<Dims> buf, subrange<Dims> sr) {
		std::vector<buffer_access> accesses;
		accesses.push_back(buffer_access{buf.get_id(), access_mode::read,
		    std::make_unique<range_mapper<Dims, celerity::access::fixed<Dims>>>(celerity::access::fixed<Dims>(sr), buf.get_range())});
		return fence(buffer_access_map(std::move(accesses), task_geometry{}), {}, std::make_unique<mock_buffer_fence_promise>(create_user_allocation()));
	}

	template <int Dims>
	task_id fence(test_utils::mock_buffer<Dims> buf) {
		return fence(buf, {{}, buf.get_range()});
	}

	task_id epoch(epoch_action action) {
		if(m_scheduler == nullptr) { FAIL("scheduler_test_context already finish()ed"); }
		const auto tid = m_tm.generate_epoch_task(action);
		m_scheduler->notify_epoch_reached(tid);
		return tid;
	}

	void set_horizon_step(const int step) { m_tm.set_horizon_step(step); }

	task_id get_initial_epoch_task() const { return m_initial_epoch_tid; }

	void flush() {
		if(m_scheduler == nullptr) { FAIL("scheduler_test_context already finish()ed"); }
		m_scheduler->flush_commands();
	}

	void set_lookahead(const experimental::lookahead lookahead) {
		if(m_scheduler == nullptr) { FAIL("scheduler_test_context already finish()ed"); }
		m_scheduler->set_lookahead(lookahead);
	}

	template <typename Inspector>
	void inspect_commands(Inspector&& inspect) const {
		const auto do_inspect = [&](const auto&...) { inspect(command_query(m_cmd_recorder)); };
		m_scheduler != nullptr ? scheduler_testspy::inspect_thread(*m_scheduler, do_inspect) : do_inspect();
	}

	template <typename Inspector>
	void inspect_instructions(Inspector&& inspect) const {
		const auto do_inspect = [&](const auto&...) { inspect(instruction_query(m_instr_recorder)); };
		m_scheduler != nullptr ? scheduler_testspy::inspect_thread(*m_scheduler, do_inspect) : do_inspect();
	}

  private:
	size_t m_num_nodes;
	node_id m_local_nid;
	size_t m_num_devices_per_node;
	buffer_id m_next_buffer_id = 0;
	host_object_id m_next_host_object_id = 0;
	reduction_id m_next_reduction_id = no_reduction_id + 1;
	detail::raw_allocation_id m_next_user_allocation_id = 1;
	std::vector<std::variant<buffer_id, host_object_id>> m_managed_objects;
	task_graph m_tdag;
	task_recorder m_task_recorder;
	task_manager m_tm;
	command_recorder m_cmd_recorder;
	instruction_recorder m_instr_recorder;
	std::unique_ptr<scheduler> m_scheduler;
	task_id m_initial_epoch_tid = 0;

	allocation_id create_user_allocation() { return detail::allocation_id(detail::user_memory_id, m_next_user_allocation_id++); }

	reduction_info create_reduction(const buffer_id bid, const bool include_current_buffer_value) {
		return reduction_info{m_next_reduction_id++, bid, include_current_buffer_value};
	}

	template <typename CGF, typename... Hints>
	task_id submit_command_group(CGF cgf, Hints... hints) {
		if(m_scheduler == nullptr) { FAIL("scheduler_test_context already finish()ed"); }
		return m_tm.submit_command_group(cgf, hints...);
	}

	void task_created(const task* tsk) override {
		if(m_scheduler == nullptr) { FAIL("scheduler_test_context already finish()ed"); }
		m_scheduler->notify_task_created(tsk);
	}

	[[nodiscard]] std::string print_task_graph() const {
		assert(m_scheduler == nullptr);
		return detail::print_task_graph(m_task_recorder, make_test_graph_title("Task Graph"));
	}
	[[nodiscard]] std::string print_command_graph() const {
		assert(m_scheduler == nullptr);
		return detail::print_command_graph(m_local_nid, m_cmd_recorder, make_test_graph_title("Command Graph", m_num_nodes, m_local_nid));
	}
	[[nodiscard]] std::string print_instruction_graph() const {
		assert(m_scheduler == nullptr);
		return detail::print_instruction_graph(
		    m_instr_recorder, m_cmd_recorder, m_task_recorder, make_test_graph_title("Instruction Graph", m_num_nodes, m_local_nid, m_num_devices_per_node));
	}

	void maybe_print_graphs() const {
		assert(m_scheduler == nullptr);
		if(test_utils::g_print_graphs) {
			fmt::print("\n{}\n", print_task_graph());
			fmt::print("\n{}\n", print_command_graph());
			fmt::print("\n{}\n", print_instruction_graph());
		}
	}

	task_id fence(buffer_access_map access_map, side_effect_map side_effects, std::unique_ptr<task_promise> promise) {
		if(m_scheduler == nullptr) { FAIL("scheduler_test_context already finish()ed"); }
		return m_tm.generate_fence_task(std::move(access_map), std::move(side_effects), std::move(promise));
	}
};

} // namespace celerity::test_utils
