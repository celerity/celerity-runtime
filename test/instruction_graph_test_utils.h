#pragma once

#include "graph_test_utils.h"
#include "instruction_graph_generator.h"

namespace celerity::test_utils {

template <typename Record>
struct instruction_matcher {
	static bool matches(const Record& instr, const instruction_id iid) { return instr.id == iid; }

	static bool matches(const Record& instr, const task_id tid) {
		return matchbox::match(
		    instr,                                                                                                 //
		    [=](const device_kernel_instruction_record& dkinstr) { return dkinstr.command_group_task_id == tid; }, //
		    [=](const host_task_instruction_record& htinstr) { return htinstr.command_group_task_id == tid; },     //
		    [=](const fence_instruction_record& finstr) { return finstr.tid == tid; },                             //
		    [=](const horizon_instruction_record& hinstr) { return hinstr.horizon_task_id == tid; },               //
		    [=](const epoch_instruction_record& einstr) { return einstr.epoch_task_id == tid; },                   //
		    [](const auto& /* other */) { return false; });
	}

	static bool matches(const Record& instr, const device_id did) {
		return utils::isa<device_kernel_instruction_record>(&instr) && utils::as<device_kernel_instruction_record>(&instr)->device_id == did;
	}

	static bool matches(const Record& instr, const std::string& debug_name) {
		return matchbox::match(
		    instr,                                                                                             //
		    [&](const device_kernel_instruction_record& dkinstr) { return dkinstr.debug_name == debug_name; }, //
		    [&](const host_task_instruction_record& htinstr) { return htinstr.debug_name == debug_name; },     //
		    [](const auto& /* other */) { return false; });
	}

	template <typename Predicate, std::enable_if_t<std::is_invocable_r_v<bool, const Predicate&, const Record&>, int> = 0>
	static bool matches(const Record& instr, const Predicate& predicate) {
		return predicate(instr);
	}

	static std::string print_filter(const instruction_id iid) { return fmt::format("I{}", iid); }
	static std::string print_filter(const task_id iid) { return fmt::format("T{}", iid); }
	static std::string print_filter(const device_id iid) { return fmt::format("D{}", iid); }
	static std::string print_filter(const std::string& debug_name) { return fmt::format("\"{}\"", debug_name); }

	template <typename Predicate, std::enable_if_t<std::is_invocable_r_v<bool, const Predicate&, const Record&>, int> = 0>
	static std::string print_filter(const Predicate& /* predicate */) {
		return "<lambda>";
	}
};

using instruction_query = graph_query<instruction_record, instruction_record, instruction_recorder, instruction_matcher>;

/// A view on an `instruction_recorder` like `instruction_query`, but for `outbound_pilot`s.
class pilot_query {
  public:
	explicit pilot_query(const instruction_recorder& recorder) : pilot_query(&recorder, recorder.get_outbound_pilots()) {}

	/// Returns the subset-query of all pilots that match the provided filters.
	template <typename... Filters>
	pilot_query select_all(const Filters&... filters) const {
		std::vector<outbound_pilot> filtered;
		std::copy_if(
		    m_result.begin(), m_result.end(), std::back_inserter(filtered), [&](const outbound_pilot& pilot) { return matches_all(pilot, filters...); });
		return pilot_query(m_recorder, std::move(filtered));
	}

	/// Like `select_all`, but asserts that the resulting query contains exactly one pilot.
	template <typename... Filters>
	pilot_query select_unique(const Filters&... filters) const {
		return select_all(filters...).assert_unique();
	}

	/// Returns the number of pilots in this query.
	size_t count() const { return m_result.size(); }

	/// Returns the number of pilots in this query that satisfy all provided filters.
	template <typename... Filters>
	size_t count(const Filters&... filters) const {
		return static_cast<size_t>(std::count_if(m_result.begin(), m_result.end(), [&](const outbound_pilot& p) { return matches_all(p, filters...); }));
	}

	/// Returns whether all pilots in this query match all provided filters.
	template <typename... Filters>
	bool all_match(const Filters&... filters) const {
		const auto num_non_matching = std::count_if(m_result.begin(), m_result.end(), [&](const outbound_pilot& p) { return !matches_all(p, filters...); });
		if(num_non_matching > 0) { UNSCOPED_INFO(fmt::format("non-matching: {} pilots", num_non_matching)); }
		return num_non_matching == 0;
	}

	/// Returns whether this query contains exactly one pilot, and whether that pilot matches all provided filters.
	template <typename... Filters>
	bool is_unique(const Filters&... filters) const {
		if(m_result.size() != 1) { UNSCOPED_INFO(fmt::format("{} pilots in query result", m_result.size())); }
		return m_result.size() == 1 && all_match(filters...);
	}

	/// Asserts that `is_unique` holds and returns the query unchanged.
	template <typename... Filters>
	pilot_query assert_unique(const Filters&... filters) const {
		REQUIRE(is_unique(filters...));
		return *this;
	}

	/// Asserts that `is_unique` holds and provides direct access to the single pilot.
	const outbound_pilot* operator->() const {
		REQUIRE(is_unique());
		return &m_result.front();
	}

  private:
	pilot_query(const instruction_recorder* recorder, std::vector<outbound_pilot> result) : m_recorder(recorder), m_result(std::move(result)) {}

	static bool matches(const outbound_pilot& pilot, const node_id nid) { return pilot.to == nid; }
	static bool matches(const outbound_pilot& pilot, const transfer_id& trid) { return pilot.message.transfer_id == trid; }
	static bool matches(const outbound_pilot& pilot, const box<3>& box) { return pilot.message.box == box; }

	template <typename Predicate, std::enable_if_t<std::is_invocable_r_v<bool, const Predicate&, const outbound_pilot&>, int> = 0>
	static bool matches(const outbound_pilot& pilot, const Predicate& predicate) {
		return predicate(pilot);
	}

	template <typename... Filters>
	static bool matches_all(const outbound_pilot& pilot, const Filters&... filters) {
		return (matches(pilot, filters) && ...);
	}

	const instruction_recorder* m_recorder;
	std::vector<outbound_pilot> m_result;
};

class mock_host_object_fence_promise : public fence_promise {
  public:
	void fulfill() override { FAIL("unimplemented"); }
	allocation_id get_user_allocation_id() override {
		FAIL("unimplemented");
		return null_allocation_id;
	}
};

class mock_buffer_fence_promise : public fence_promise {
  public:
	mock_buffer_fence_promise() = default;
	explicit mock_buffer_fence_promise(allocation_id user_allocation_id) : m_user_aid(user_allocation_id) {}

	void fulfill() override { FAIL("unimplemented"); }

	allocation_id get_user_allocation_id() override { return m_user_aid; }

  private:
	allocation_id m_user_aid;
};

class idag_test_context final : private task_manager::delegate {
	friend class task_builder<idag_test_context>;

  private:
	class uncaught_exception_guard {
	  public:
		explicit uncaught_exception_guard(idag_test_context* ictx) : m_ictx(ictx), m_uncaught_exceptions_before(std::uncaught_exceptions()) {}

		~uncaught_exception_guard() {
			if(std::uncaught_exceptions() > m_uncaught_exceptions_before) {
				m_ictx->m_uncaught_exceptions_before = -1; // always destroy idag_test_context under the uncaught-exception condition
			}
		}

		uncaught_exception_guard(const uncaught_exception_guard&) = delete;
		uncaught_exception_guard(uncaught_exception_guard&&) = delete;
		uncaught_exception_guard& operator=(const uncaught_exception_guard&) = delete;
		uncaught_exception_guard& operator=(uncaught_exception_guard&&) = delete;

	  private:
		idag_test_context* m_ictx;
		int m_uncaught_exceptions_before;
	};

  public:
	struct policy_set {
		task_manager::policy_set tm;
		command_graph_generator::policy_set cggen;
		instruction_graph_generator::policy_set iggen;
	};

	idag_test_context(
	    const size_t num_nodes, const node_id local_nid, const size_t num_devices_per_node, bool supports_d2d_copies = true, const policy_set& policy = {})
	    : m_num_nodes(num_nodes), m_local_nid(local_nid), m_num_devices_per_node(num_devices_per_node),
	      m_uncaught_exceptions_before(std::uncaught_exceptions()), m_tm(num_nodes, &m_task_recorder, this, policy.tm), m_cmd_recorder(), m_cdag(),
	      m_cggen(num_nodes, local_nid, m_cdag, &m_cmd_recorder, policy.cggen), m_instr_recorder(),
	      m_iggen(num_nodes, local_nid, make_system_info(num_devices_per_node, supports_d2d_copies), m_idag, nullptr /* delegate */, &m_instr_recorder,
	          policy.iggen) {
		REQUIRE(local_nid < num_nodes);
		REQUIRE(num_devices_per_node > 0);
		m_initial_epoch_tid = m_tm.generate_epoch_task(epoch_action::init);
	}

	~idag_test_context() {
		// instruction-graph-generator has no exception guarantees, so we must not call further member functions if one of them threw an exception
		if(m_uncaught_exceptions_before == std::uncaught_exceptions()) { finish(); }
		maybe_print_graphs();
	}

	idag_test_context(const idag_test_context&) = delete;
	idag_test_context(idag_test_context&&) = delete;
	idag_test_context& operator=(const idag_test_context&) = delete;
	idag_test_context& operator=(idag_test_context&&) = delete;

	static memory_id get_native_memory(const device_id did) { return first_device_memory_id + did; }

	void task_created(const task* tsk) override {
		if(m_finished) { FAIL("idag_test_context already finish()ed"); }
		const uncaught_exception_guard guard(this);
		for(const auto cmd : m_cggen.build_task(*tsk)) {
			m_iggen.compile(*cmd);
		}
	}

	/// Call this after issuing all submissions in order to trigger the shutdown epoch together with all cleanup instructions.
	void finish() {
		if(m_finished) return;

		for(auto iter = m_managed_objects.rbegin(); iter != m_managed_objects.rend(); ++iter) {
			matchbox::match(
			    *iter,
			    [&](const buffer_id bid) {
				    m_iggen.notify_buffer_destroyed(bid);
				    m_cggen.notify_buffer_destroyed(bid);
				    m_tm.notify_buffer_destroyed(bid);
			    },
			    [&](const host_object_id hoid) {
				    m_iggen.notify_host_object_destroyed(hoid);
				    m_cggen.notify_host_object_destroyed(hoid);
				    m_tm.notify_host_object_destroyed(hoid);
			    });
		}
		m_tm.generate_epoch_task(epoch_action::shutdown);

		m_finished = true;
	}

	template <typename DataT, int Dims>
	test_utils::mock_buffer<Dims> create_buffer(range<Dims> size, bool mark_as_host_initialized = false) {
		if(m_finished) { FAIL("idag_test_context already finish()ed"); } // if{FAIL} instead of REQUIRE so we don't count these as passed assertions
		const uncaught_exception_guard guard(this);
		const buffer_id bid = m_next_buffer_id++;
		const auto buf = test_utils::mock_buffer<Dims>(bid, size);
		m_tm.notify_buffer_created(bid, range_cast<3>(size), mark_as_host_initialized);
		m_cggen.notify_buffer_created(bid, range_cast<3>(size), mark_as_host_initialized);
		m_iggen.notify_buffer_created(bid, range_cast<3>(size), sizeof(DataT), alignof(DataT),
		    mark_as_host_initialized ? detail::allocation_id(detail::user_memory_id, m_next_user_allocation_id++) : detail::null_allocation_id);
		m_managed_objects.emplace_back(bid);
		return buf;
	}

	template <int Dims>
	test_utils::mock_buffer<Dims> create_buffer(range<Dims> size, bool mark_as_host_initialized = false) {
		return create_buffer<float, Dims>(size, mark_as_host_initialized);
	}

	test_utils::mock_host_object create_host_object(const bool owns_instance = true) {
		if(m_finished) { FAIL("idag_test_context already finish()ed"); }
		const uncaught_exception_guard guard(this);
		const host_object_id hoid = m_next_host_object_id++;
		m_tm.notify_host_object_created(hoid);
		m_cggen.notify_host_object_created(hoid);
		m_iggen.notify_host_object_created(hoid, owns_instance);
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
		buffer_access_map access_map;
		access_map.add_access(buf.get_id(),
		    std::make_unique<range_mapper<Dims, celerity::access::fixed<Dims>>>(celerity::access::fixed<Dims>(sr), access_mode::read, buf.get_range()));
		return fence(std::move(access_map), {}, std::make_unique<mock_buffer_fence_promise>(create_user_allocation()));
	}

	template <int Dims>
	task_id fence(test_utils::mock_buffer<Dims> buf) {
		return fence(buf, {{}, buf.get_range()});
	}

	task_id epoch(epoch_action action) {
		if(m_finished) { FAIL("idag_test_context already finish()ed"); }
		const uncaught_exception_guard guard(this);
		return m_tm.generate_epoch_task(action);
	}

	void set_horizon_step(const int step) { m_tm.set_horizon_step(step); }

	task_manager& get_task_manager() { return m_tm; }

	command_graph_generator& get_graph_generator() { return m_cggen; }

	task_id get_initial_epoch_task() const { return m_initial_epoch_tid; }

	instruction_query query_instructions() const { return instruction_query(m_instr_recorder); }

	pilot_query query_outbound_pilots() const { return pilot_query(m_instr_recorder); }

	[[nodiscard]] std::string print_task_graph() { //
		return detail::print_task_graph(m_task_recorder, make_test_graph_title("Task Graph"));
	}
	[[nodiscard]] std::string print_command_graph() {
		return detail::print_command_graph(m_local_nid, m_cmd_recorder, make_test_graph_title("Command Graph", m_num_nodes, m_local_nid));
	}
	[[nodiscard]] std::string print_instruction_graph() {
		return detail::print_instruction_graph(
		    m_instr_recorder, m_cmd_recorder, m_task_recorder, make_test_graph_title("Instruction Graph", m_num_nodes, m_local_nid, m_num_devices_per_node));
	}

  private:
	size_t m_num_nodes;
	node_id m_local_nid;
	size_t m_num_devices_per_node;
	bool m_finished = false;
	int m_uncaught_exceptions_before;
	buffer_id m_next_buffer_id = 0;
	host_object_id m_next_host_object_id = 0;
	reduction_id m_next_reduction_id = no_reduction_id + 1;
	detail::raw_allocation_id m_next_user_allocation_id = 1;
	std::vector<std::variant<buffer_id, host_object_id>> m_managed_objects;
	task_recorder m_task_recorder;
	task_manager m_tm;
	command_recorder m_cmd_recorder;
	command_graph m_cdag;
	command_graph_generator m_cggen;
	instruction_graph m_idag;
	instruction_recorder m_instr_recorder;
	instruction_graph_generator m_iggen;
	task_id m_initial_epoch_tid = 0;

	allocation_id create_user_allocation() { return detail::allocation_id(detail::user_memory_id, m_next_user_allocation_id++); }

	reduction_info create_reduction(const buffer_id bid, const bool include_current_buffer_value) {
		return reduction_info{m_next_reduction_id++, bid, include_current_buffer_value};
	}

	template <typename CGF, typename... Hints>
	task_id submit_command_group(CGF cgf, Hints... hints) {
		if(m_finished) { FAIL("idag_test_context already finish()ed"); }
		const uncaught_exception_guard guard(this);
		return m_tm.submit_command_group(cgf, hints...);
	}

	void maybe_print_graphs() {
		if(test_utils::g_print_graphs) {
			fmt::print("\n{}\n", print_task_graph());
			fmt::print("\n{}\n", print_command_graph());
			fmt::print("\n{}\n", print_instruction_graph());
		}
	}

	task_id fence(buffer_access_map access_map, side_effect_map side_effects, std::unique_ptr<fence_promise> promise) {
		if(m_finished) { FAIL("idag_test_context already finish()ed"); }
		const uncaught_exception_guard guard(this);
		return m_tm.generate_fence_task(std::move(access_map), std::move(side_effects), std::move(promise));
	}
};

} // namespace celerity::test_utils
