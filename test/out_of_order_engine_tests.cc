#include "instruction_graph.h"
#include "launcher.h"
#include "out_of_order_engine.h"
#include "test_utils.h"
#include "types.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_range.hpp>

using namespace celerity;
using namespace celerity::detail;


class assigned_set_query {
  public:
	explicit assigned_set_query(std::vector<out_of_order_engine::assignment> isq_vec) : m_assignments(std::move(isq_vec)) {}

	/// Has `instr` been assigned with `target::immediate`?
	bool executed_immediately(const instruction* instr) const {
		const auto it = std::find_if(m_assignments.begin(), m_assignments.end(),
		    [&](const out_of_order_engine::assignment& isq) { return isq.instruction == instr && isq.target == out_of_order_engine::target::immediate; });
		return it != m_assignments.end();
	}

	/// Has `instr` been assigned with `target::alloc_queue`?
	bool queued_for_alloc(const instruction* instr) const {
		const auto it = std::find_if(m_assignments.begin(), m_assignments.end(),
		    [&](const out_of_order_engine::assignment& isq) { return isq.instruction == instr && isq.target == out_of_order_engine::target::alloc_queue; });
		return it != m_assignments.end();
	}

	/// Has `instr` been assigned with `target::host_queue`?
	bool queued_on_host(const instruction* instr) const {
		const auto it = std::find_if(m_assignments.begin(), m_assignments.end(),
		    [&](const out_of_order_engine::assignment& isq) { return isq.instruction == instr && isq.target == out_of_order_engine::target::host_queue; });
		return it != m_assignments.end();
	}

	/// Has `instr` been assigned with `target::device_queue` for `device`?
	bool queued_on_device(const instruction* instr, const device_id device) const {
		const auto it = std::find_if(m_assignments.begin(), m_assignments.end(), [&](const out_of_order_engine::assignment& isq) {
			return isq.instruction == instr && isq.target == out_of_order_engine::target::device_queue && isq.device == device;
		});
		return it != m_assignments.end();
	}

	/// True if exactly `expected` instructions were assigned and no more.
	bool assigned_instructions_are(std::vector<const instruction*> expected) const {
		std::vector<const instruction*> actual(m_assignments.size());
		std::transform(m_assignments.begin(), m_assignments.end(), actual.begin(), [](const out_of_order_engine::assignment& isq) { return isq.instruction; });
		std::sort(actual.begin(), actual.end());
		std::sort(expected.begin(), expected.end());
		return actual == expected;
	}

	/// True iff all instructions in `sequence` are assigned to the same lane in the given order.
	bool assigned_in_order(std::vector<const instruction*> sequence) const {
		auto isq_it = m_assignments.begin();
		std::optional<out_of_order_engine::target> prev_target;
		std::optional<device_id> prev_device;
		std::optional<size_t> prev_lane;
		for(auto seq_it = sequence.begin(); seq_it != sequence.end(); ++seq_it) {
			while(isq_it != m_assignments.end() && isq_it->instruction != *seq_it) {
				++isq_it;
			}
			if(isq_it == m_assignments.end()) return false;

			if(prev_target.has_value() && prev_target != isq_it->target) return false;
			if(prev_device.has_value() && prev_device != isq_it->device) return false;
			if(prev_lane.has_value() && prev_lane != isq_it->lane) return false;
			prev_target = isq_it->target;
			prev_device = isq_it->device;
			prev_lane = isq_it->lane;
		}
		return true;
	}

	/// True iff all instructions in `left` and `right` are assigned, but never any from `left` and `right` both on the same lane.
	bool assigned_concurrently(const std::vector<const instruction*>& left, const std::vector<const instruction*>& right) const {
		for(const auto l : left) {
			for(const auto r : right) {
				const auto al = std::find_if(m_assignments.begin(), m_assignments.end(), //
				    [&](const out_of_order_engine::assignment& isq) { return isq.instruction == l; });
				const auto ar = std::find_if(m_assignments.begin(), m_assignments.end(), //
				    [&](const out_of_order_engine::assignment& isq) { return isq.instruction == r; });
				if(al == m_assignments.end() || ar == m_assignments.end()) return false;
				return al->target != ar->target || al->device != ar->device || al->lane != ar->lane;
			}
		}
		return true;
	}

	/// Requires that `instr` is assigned and returns the original `assignment` for that instruction.
	const out_of_order_engine::assignment& assignment_for(const instruction* instr) const {
		const auto it =
		    std::find_if(m_assignments.begin(), m_assignments.end(), [=](const out_of_order_engine::assignment& a) { return a.instruction == instr; });
		REQUIRE(it != m_assignments.end());
		return *it;
	}

  private:
	std::vector<out_of_order_engine::assignment> m_assignments;
};

/// Creates instructions directly (without going through graph generation) and submits them to an out_of_order_engine.
class out_of_order_test_context {
  public:
	explicit out_of_order_test_context(const size_t num_devices) : m_engine(test_utils::make_system_info(num_devices, true /* supports_d2d_copies */)) {}

	const instruction* alloc(const std::vector<const instruction*>& dependencies, const memory_id mid, const int priority = 0) {
		return create<alloc_instruction>(dependencies, priority, allocation_id(mid, 1), 1024, 1);
	}

	const instruction* free(const std::vector<const instruction*>& dependencies, const memory_id mid, const int priority = 0) {
		return create<free_instruction>(dependencies, priority, allocation_id(mid, 1));
	}

	const instruction* device_kernel(const std::vector<const instruction*>& dependencies, const device_id did, const int priority = 0) {
		return create<device_kernel_instruction>(
		    dependencies, priority, did, device_kernel_launcher{}, box<3>(), buffer_access_allocation_map{},
		    buffer_access_allocation_map {} //
		    CELERITY_DETAIL_IF_ACCESSOR_BOUNDARY_CHECK(, task_type::device_compute, task_id(0), "task"));
	}

	const instruction* copy(const std::vector<const instruction*>& dependencies, const memory_id source, const memory_id dest, const int priority = 0) {
		const box<3> box(id(0, 0, 0), id(1, 1, 1));
		return create<copy_instruction>(dependencies, priority, allocation_id(source, 1), allocation_id(dest, 1), box, box, box, sizeof(int));
	}

	const instruction* host_task(const std::vector<const instruction*>& dependencies, const int priority = 0) {
		return create<host_task_instruction>(
		    dependencies, priority, host_task_launcher{}, box<3>(), range<3>(), buffer_access_allocation_map{},
		    collective_group_id {} CELERITY_DETAIL_IF_ACCESSOR_BOUNDARY_CHECK(, task_type::device_compute, task_id(0), "task"));
	}

	const instruction* epoch(const std::vector<const instruction*>& dependencies, const int priority = 0) {
		return create<epoch_instruction>(dependencies, priority, task_id(0), epoch_action::none, instruction_garbage{});
	}

	void complete(const instruction* instr) { m_engine.complete_assigned(instr); }

	std::optional<out_of_order_engine::assignment> assign_one() {
		const auto assignment = m_engine.assign_one();
		if(assignment.has_value()) {
			CHECK(assignment->instruction != nullptr);
			switch(assignment->target) {
			case out_of_order_engine::target::immediate: //
				CHECK(assignment->device == std::nullopt);
				CHECK(assignment->lane == std::nullopt);
				break;
			case out_of_order_engine::target::alloc_queue: //
				CHECK(assignment->lane == std::nullopt);
				break;
			case out_of_order_engine::target::host_queue: //
				CHECK(assignment->device == std::nullopt);
				CHECK(assignment->lane.has_value());
				break;
			case out_of_order_engine::target::device_queue: //
				CHECK(assignment->device.has_value());
				CHECK(assignment->lane.has_value());
				break;
			default: FAIL();
			}
		}
		return assignment;
	}

	assigned_set_query assign_all() {
		std::vector<out_of_order_engine::assignment> isq_vec;
		while(const auto assignment = assign_one()) {
			isq_vec.push_back(*assignment);
		}
		return assigned_set_query(std::move(isq_vec));
	}

	bool is_idle() const { return m_engine.is_idle(); }

  private:
	instruction_id m_next_iid = 0;
	std::vector<std::unique_ptr<instruction>> m_instrs;
	out_of_order_engine m_engine;

	template <typename Instruction, typename... CtorParams>
	const instruction* create(const std::vector<const instruction*>& dependencies, const int priority, const CtorParams&... ctor_args) {
		const auto iid = m_next_iid++;
		const auto instr = m_instrs.emplace_back(std::make_unique<Instruction>(iid, priority, ctor_args...)).get();
		for(const auto dep : dependencies) {
			instr->add_dependency(dep->get_id());
		}
		m_engine.submit(instr);
		return instr;
	}
};

TEST_CASE("out_of_order_engine schedules independent chains concurrently", "[out_of_order_engine]") {
	out_of_order_test_context octx(4);
	const auto d0_k0 = octx.device_kernel({}, device_id(0));
	const auto d1_k0 = octx.device_kernel({}, device_id(1));
	const auto d0_k1 = octx.device_kernel({d0_k0}, device_id(0));
	const auto d1_k1 = octx.device_kernel({d1_k0}, device_id(1));
	const auto h0 = octx.host_task({d0_k1, d1_k1});
	const auto h1 = octx.host_task({h0});
	const auto h2 = octx.host_task({h0, h1});

	{
		const auto iq = octx.assign_all();
		CHECK(iq.assigned_instructions_are({d0_k0, d0_k1, d1_k0, d1_k1}));
		CHECK(iq.queued_on_device(d0_k0, device_id(0)));
		CHECK(iq.queued_on_device(d0_k1, device_id(0)));
		CHECK(iq.queued_on_device(d1_k0, device_id(1)));
		CHECK(iq.queued_on_device(d1_k1, device_id(1)));
		CHECK(iq.assigned_in_order({d0_k0, d0_k1}));
		CHECK(iq.assigned_in_order({d1_k0, d1_k1}));
		CHECK(iq.assigned_concurrently({d0_k0, d0_k1}, {d1_k0, d1_k1}));
	}

	octx.complete(d0_k0);
	octx.complete(d0_k1);
	octx.complete(d1_k0);

	{
		const auto iq = octx.assign_all();
		CHECK(iq.assigned_instructions_are({}));
	}

	octx.complete(d1_k1);

	{
		const auto iq = octx.assign_all();
		CHECK(iq.assigned_instructions_are({h0, h1, h2}));
		CHECK(iq.assigned_in_order({h0, h1, h2}));
	}
}

TEST_CASE("out_of_order_engine eagerly assigns copy-instructions to the lanes of their dependencies", "[out_of_order_engine]") {
	out_of_order_test_context octx(4);
	const auto d0_k0 = octx.device_kernel({}, device_id(0));
	const auto d1_k0 = octx.device_kernel({}, device_id(1));
	const auto d2_k0 = octx.device_kernel({}, device_id(2));
	const auto d3_k0 = octx.device_kernel({}, device_id(3));
	const auto copy_dep0 = octx.copy({d0_k0}, first_device_memory_id, first_device_memory_id + 1);
	const auto copy_dep1 = octx.copy({d1_k0}, first_device_memory_id, first_device_memory_id + 1);
	const auto copy_dep2 = octx.copy({d2_k0}, host_memory_id, first_device_memory_id + 2);
	const auto copy_dep3 = octx.copy({d3_k0}, first_device_memory_id + 3, host_memory_id);
	const auto copy_host = octx.copy({copy_dep3}, host_memory_id, host_memory_id); // can't be enqueued on a device in-order queue

	CHECK_FALSE(octx.is_idle());

	{
		const auto iq = octx.assign_all();
		CHECK(iq.assigned_instructions_are({d0_k0, d1_k0, d2_k0, d3_k0, copy_dep0, copy_dep1, copy_dep2, copy_dep3}));
		CHECK(iq.queued_on_device(d0_k0, device_id(0)));
		CHECK(iq.queued_on_device(d1_k0, device_id(1)));
		CHECK(iq.queued_on_device(d2_k0, device_id(2)));
		CHECK(iq.queued_on_device(d3_k0, device_id(3)));
		CHECK(iq.assigned_in_order({d0_k0, copy_dep0}));
		CHECK(iq.assigned_in_order({d1_k0, copy_dep1}));
		CHECK(iq.assigned_in_order({d2_k0, copy_dep2}));
		CHECK(iq.assigned_in_order({d3_k0, copy_dep3}));
	}

	octx.complete(d0_k0);
	octx.complete(d1_k0);
	octx.complete(d2_k0);
	octx.complete(d3_k0);
	CHECK_FALSE(octx.is_idle());

	const auto copy_indep0 = octx.copy({d0_k0 /* already complete */}, first_device_memory_id + 1, first_device_memory_id);
	const auto copy_indep1 = octx.copy({d1_k0 /* already complete */}, first_device_memory_id, first_device_memory_id + 1);
	const auto copy_indep2 = octx.copy({d2_k0 /* already complete */}, first_device_memory_id + 1, first_device_memory_id + 2);
	const auto copy_indep3 = octx.copy({d3_k0 /* already complete */}, first_device_memory_id, first_device_memory_id + 3);

	{
		const auto iq = octx.assign_all();
		CHECK(iq.assigned_instructions_are({copy_indep0, copy_indep1, copy_indep2, copy_indep3}));
		CHECK((iq.queued_on_device(copy_indep0, device_id(1)) || iq.queued_on_device(copy_indep0, device_id(0))));
		CHECK((iq.queued_on_device(copy_indep1, device_id(0)) || iq.queued_on_device(copy_indep1, device_id(1))));
		CHECK((iq.queued_on_device(copy_indep2, device_id(1)) || iq.queued_on_device(copy_indep2, device_id(2))));
		CHECK((iq.queued_on_device(copy_indep3, device_id(0)) || iq.queued_on_device(copy_indep3, device_id(3))));
	}

	octx.complete(copy_dep2);
	octx.complete(copy_dep1);
	octx.complete(copy_dep0);
	octx.complete(copy_dep3);
	CHECK_FALSE(octx.is_idle());

	{
		const auto iq = octx.assign_all();
		CHECK(iq.assigned_instructions_are({copy_host}));
		CHECK(iq.queued_on_host(copy_host));
	}

	octx.complete(copy_indep2);
	octx.complete(copy_indep1);
	octx.complete(copy_indep0);
	octx.complete(copy_indep3);
	CHECK_FALSE(octx.is_idle());

	octx.complete(copy_host);
	CHECK(octx.is_idle());
}

TEST_CASE("alloc / free instructions are scheduled on alloc_queue instead of host_queue", "[out_of_order_engine]") {
	out_of_order_test_context octx(1);
	const auto init_epoch = octx.epoch({});
	const auto alloc_host = octx.alloc({init_epoch}, host_memory_id);
	const auto alloc_device = octx.alloc({init_epoch}, first_device_memory_id);
	const auto host_task = octx.host_task({alloc_host});
	const auto device_kernel = octx.device_kernel({alloc_device}, device_id(0));
	const auto free_host = octx.free({host_task}, host_memory_id);
	const auto free_device = octx.free({device_kernel}, first_device_memory_id);
	CHECK_FALSE(octx.is_idle());

	{
		const auto iq = octx.assign_all();
		CHECK(iq.assigned_instructions_are({init_epoch}));
		CHECK(iq.executed_immediately(init_epoch));
	}

	octx.complete(init_epoch);
	CHECK_FALSE(octx.is_idle());

	{
		const auto iq = octx.assign_all();
		CHECK(iq.assigned_instructions_are({alloc_host, alloc_device}));
		CHECK(iq.queued_for_alloc(alloc_host));
		CHECK(iq.queued_for_alloc(alloc_device));
		CHECK(iq.assignment_for(alloc_host).device == std::nullopt);
		CHECK(iq.assignment_for(alloc_device).device == device_id(0));
	}

	octx.complete(alloc_host);
	octx.complete(alloc_device);
	CHECK_FALSE(octx.is_idle());

	{
		const auto iq = octx.assign_all();
		CHECK(iq.assigned_instructions_are({host_task, device_kernel}));
		CHECK(iq.queued_on_host(host_task));
		CHECK(iq.queued_on_device(device_kernel, device_id(0)));
	}

	octx.complete(host_task);
	octx.complete(device_kernel);

	{
		const auto iq = octx.assign_all();
		CHECK(iq.assigned_instructions_are({free_host, free_device}));
		CHECK(iq.queued_for_alloc(free_host));
		CHECK(iq.queued_for_alloc(free_device));
	}

	octx.complete(free_host);
	octx.complete(free_device);
	CHECK(octx.is_idle());
}

TEST_CASE("out_of_order_engine does not attempt to assign instructions more than once in the presence of eager assignment", "[out_of_order_engine]") {
	out_of_order_test_context octx(1);
	auto k1 = octx.device_kernel({}, device_id(0));
	auto k2 = octx.device_kernel({k1}, device_id(0));

	const auto assigned = octx.assign_one();
	REQUIRE(assigned.has_value());
	CHECK(assigned->instruction == k1);

	octx.complete(k1);

	const auto other = octx.assign_all();
	CHECK(other.assigned_instructions_are({k2}));

	octx.complete(k2);
	CHECK(octx.is_idle());
}

// Even though instructions on an in-order queue / thread queue logically always finish in the order they were submitted, we might not poll events in the same
// order within the executor (without additional effort), so out_of_order_engine instead supports completing instructions out-of-order (at little extra cost).
TEST_CASE("assigned sets of instructions with internal dependencies can be completed out-of-order", "[out_of_order_engine]") {
	out_of_order_test_context octx(1);
	auto k1 = octx.device_kernel({}, device_id(0));
	auto k2 = octx.device_kernel({k1}, device_id(0));

	octx.assign_all();
	CHECK_FALSE(octx.is_idle());

	octx.complete(k2);
	CHECK_FALSE(octx.is_idle());

	octx.complete(k1);
	CHECK(octx.is_idle());
}

TEST_CASE("concurrent instructions are assigned in decreasing priority", "[out_of_order_engine]") {
	out_of_order_test_context octx(1);
	auto k1 = octx.device_kernel({}, device_id(0), /* priority */ 1);
	auto k3 = octx.device_kernel({}, device_id(0), /* priority */ 3);
	auto k2 = octx.device_kernel({}, device_id(0), /* priority */ 2);

	const auto first = octx.assign_one();
	REQUIRE(first.has_value());
	CHECK(first->instruction == k3);

	const auto second = octx.assign_one();
	REQUIRE(second.has_value());
	CHECK(second->instruction == k2);

	const auto third = octx.assign_one();
	REQUIRE(third.has_value());
	CHECK(third->instruction == k1);
}

TEST_CASE("eagerly-assignable instructions become immediately assignable once their predecessors complete", "[out_of_order_engine]") {
	out_of_order_test_context octx(1);
	auto k1 = octx.device_kernel({}, device_id(0), /* priority */ 0);
	auto k2 = octx.device_kernel({k1}, device_id(0), /* priority */ 1);
	auto k3 = octx.device_kernel({k1}, device_id(0), /* priority */ 0);
	auto k4 = octx.device_kernel({k2}, device_id(0), /* priority */ 2);

	const auto first = octx.assign_one();
	REQUIRE(first.has_value());
	CHECK(first->instruction == k1);

	const auto second = octx.assign_one();
	REQUIRE(second.has_value());
	CHECK(second->instruction == k2);

	// complete out-of-order so we are able to distinguish between eager and immediate assignment of k4
	octx.complete(k2);

	const auto third = octx.assign_one();
	REQUIRE(third.has_value());
	CHECK(third->instruction == k4); // immediately assignable now; highest priority
}

TEST_CASE("eagerly-assignable instructions remain so as long as their predecessors are partially complete", "[out_of_order_engine]") {
	out_of_order_test_context octx(1);
	const auto k1 = octx.device_kernel({}, device_id(0));
	const auto k2 = octx.device_kernel({k1}, device_id(0));
	const auto k3 = octx.device_kernel({k2}, device_id(0));
	const auto k4 = octx.device_kernel({k3}, device_id(0));

	const auto first = octx.assign_one();
	REQUIRE(first.has_value());
	CHECK(first->instruction == k1);

	const auto second = octx.assign_one();
	REQUIRE(second.has_value());
	CHECK(second->instruction == k2);

	const auto complete_first = GENERATE(values({1 /* in-order completion */, 2 /* out-of-order completion */}));
	CAPTURE(complete_first);

	octx.complete(complete_first == 1 ? k1 : k2);

	const auto rest = octx.assign_all();
	CHECK(rest.assigned_instructions_are({k3, k4}));
	CHECK(rest.assigned_in_order({k3, k4}));
}

TEST_CASE("instructions are not eagerly assigned if their predecessors are assigned to a different device", "[out_of_order_engine]") {
	out_of_order_test_context octx(2);
	const auto k1 = octx.device_kernel({}, device_id(0));
	const auto k2 = octx.device_kernel({k1}, device_id(1));

	{
		const auto iq = octx.assign_all();
		CHECK(iq.assigned_instructions_are({k1}));
		CHECK(iq.queued_on_device(k1, device_id(0)));
	}

	octx.complete(k1);

	{
		const auto iq = octx.assign_all();
		CHECK(iq.assigned_instructions_are({k2}));
		CHECK(iq.queued_on_device(k2, device_id(1)));
	}

	octx.complete(k2);
	CHECK(octx.is_idle());
}

TEST_CASE("instructions are not eagerly assigned if their predecessors are assigned to a different lanes", "[out_of_order_engine]") {
	out_of_order_test_context octx(2);
	const auto k1 = octx.host_task({});
	const auto k2 = octx.host_task({});
	const auto k3 = octx.host_task({k1, k2});

	{
		const auto iq = octx.assign_all();
		CHECK(iq.assigned_instructions_are({k1, k2}));
		CHECK(iq.queued_on_host(k1));
		CHECK(iq.queued_on_host(k2));
	}

	const auto k_complete_first = GENERATE(values({1, 2}));
	octx.complete(k_complete_first == 1 ? k1 : k2);

	{
		const auto iq = octx.assign_all();
		CHECK(iq.assigned_instructions_are({k3}));
	}

	octx.complete(k_complete_first == 1 ? k2 : k1);
	octx.complete(k3);
	CHECK(octx.is_idle());
}

TEST_CASE("instructions with predecessors on multiple devices are eagerly assigned only if the last remaining dependency is on the same device",
    "[out_of_order_engine]") //
{
	out_of_order_test_context octx(2);
	const auto k1_d0 = octx.device_kernel({}, device_id(0));
	const auto k2_d1 = octx.device_kernel({}, device_id(1));
	const auto k3_d0 = octx.device_kernel({k1_d0, k2_d1}, device_id(0));

	const auto iq_first = octx.assign_all();
	CHECK(iq_first.assigned_instructions_are({k1_d0, k2_d1}));

	const auto k_complete_first = GENERATE(values({1, 2}));
	octx.complete(k_complete_first == 1 ? k1_d0 : k2_d1);

	const auto iq_last = octx.assign_all();
	if(k_complete_first == 2) {
		// k2_d1 was completed first, so k3_d0 should be assigned to the same lane as k1_d0
		CHECK(iq_last.assigned_instructions_are({k3_d0}));
		CHECK(iq_last.queued_on_device(k3_d0, device_id(0)));
		CHECK(iq_last.assignment_for(k3_d0).lane == iq_first.assignment_for(k1_d0).lane);
	} else {
		iq_last.assigned_instructions_are({/* none */});
	}

	octx.complete(k_complete_first == 1 ? k2_d1 : k1_d0);
	octx.assign_all();
	octx.complete(k3_d0);
	CHECK(octx.is_idle());
}

TEST_CASE("eligibility for eager assignment is retracted when an instruction is overtaken by a higher-priority sibling", "[out_of_order_engine]") {
	const auto higher_prio = GENERATE(values({2, 3}));

	out_of_order_test_context octx(1);
	const auto k1 = octx.host_task({});
	const auto k2 = octx.host_task({k1}, higher_prio == 2 ? 1 : 0);
	const auto k3 = octx.host_task({k1}, higher_prio == 3 ? 1 : 0);

	const auto iq_first = octx.assign_all();
	CHECK(iq_first.assigned_instructions_are({k1, higher_prio == 2 ? k2 : k3}));
	CHECK(iq_first.assigned_in_order({k1, higher_prio == 2 ? k2 : k3}));

	octx.complete(k1);

	const auto iq_second = octx.assign_all();
	CHECK(iq_second.assigned_instructions_are({higher_prio == 2 ? k3 : k2}));
	// k2 and k3 are concurrent, check that they are assigned to different lanes now (eager assignment would have placed both on k1's lane)
	CHECK(iq_second.assignment_for(higher_prio == 2 ? k3 : k2).lane != iq_first.assignment_for(k1).lane);

	octx.complete(k2);
	octx.complete(k3);
	CHECK(octx.is_idle());
}

TEST_CASE("eager assignment is only attempted when the one incomplete dependency is last in the target lane", "[out_of_order_engine]") {
	out_of_order_test_context octx(1);
	const auto k1 = octx.device_kernel({}, device_id(0));
	const auto k2 = octx.device_kernel({k1}, device_id(0));

	const auto iq_12 = octx.assign_all();
	CHECK(iq_12.queued_on_device(k1, device_id(0)));
	CHECK(iq_12.queued_on_device(k2, device_id(0)));
	CHECK(iq_12.assigned_in_order({k1, k2}));

	const auto k3 = octx.device_kernel({k1}, device_id(0));

	const auto iq_3_before = octx.assign_all();
	CHECK(iq_3_before.assigned_instructions_are({})); // can't be queued eagerly

	octx.complete(k1);

	const auto iq_3 = octx.assign_all();
	CHECK(iq_3.assigned_instructions_are({k3}));
	CHECK(iq_3.queued_on_device(k3, device_id(0)));
	CHECK(iq_12.assignment_for(k1).lane != iq_3.assignment_for(k3).lane);
	CHECK(iq_12.assignment_for(k2).lane != iq_3.assignment_for(k3).lane);
}
