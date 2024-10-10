#include "out_of_order_engine.h"
#include "dense_map.h"
#include "instruction_graph.h"
#include "system_info.h"
#include "tracy.h"
#include "utils.h"

#include <queue>
#include <unordered_map>
#include <variant>
#include <vector>

#include <matchbox.hh>

namespace celerity::detail::out_of_order_engine_detail {

using target = out_of_order_engine::target;
using lane_id = out_of_order_engine::lane_id;
using assignment = out_of_order_engine::assignment;

/// Comparison operator to make priority_queue<instruction*> return concurrent instructions in decreasing priority.
struct instruction_priority_less {
	bool operator()(const instruction* lhs, const instruction* rhs) const { return lhs->get_priority() < rhs->get_priority(); }
};

/// Instruction is not eligible for assignment yet. Implies `num_incomplete_predecessors > 0`.
struct unassigned_state {
	/// Flag allowing us to shortcut the eager-assignment check if we know that it cannot succeed (anymore). Set to false statically for the immediate and
	/// alloc_queue target, and also set to false if the instruction was in `conditional_eagerly_assignable_state` but eager assignment had to be aborted
	/// because the lane was not in the expected state anymore.
	bool probe_for_eager_assignment = true;
};

/// Instruction has been inserted into `assignment_queue`, but has not yet been assigned. After popping it from the queue, it can potentially be assigned by
/// eagerly enqueueing it to the lane where all its remaining predecessors are already assigned to. If a third instruction is submitted to the target lane in
/// the meantime however, assignment will fail, and we revert back to `unassigned_state`. Implies `num_unassigned_predecessors == 0`.
struct conditional_eagerly_assignable_state {
	std::optional<device_id> device;
	lane_id lane = -1;
	std::optional<instruction_id> expected_last_submission_on_lane; // instruction_id, because past instruction pointers may dangle
};

/// Instruction is inserted into `assignment_queue` and will be unconditionally assigned after being popped.
/// Implies `num_incomplete_predecessors == 0 && num_unassigned_predecessors == 0`.
struct unconditional_assignable_state {};

/// Instruction is assigned and waiting for a `complete_assigned()` call. Implies `num_incomplete_predecessors == 0 && num_unassigned_predecessors == 0`.
struct assigned_state {
	std::optional<device_id> device;
	std::optional<lane_id> lane;
};

/// State maintained for every instruction between its submission and completion.
struct incomplete_instruction_state {
	const instruction* instr = nullptr;
	out_of_order_engine::target target = out_of_order_engine::target::immediate;
	gch::small_vector<device_id, 2> eligible_devices; ///< device-to-device copies can be submitted to host or destination device.
	gch::small_vector<instruction_id> successors;     ///< we collect successors as they are submitted

	/// An instruction with no incomplete predecessors is ready for immediate assignment.
	size_t num_incomplete_predecessors = 0;

	/// An instruction with no unassigned (but some incomplete) predecessors may be eligible for eager assignment (see try_mark_for_assignment).
	size_t num_unassigned_predecessors = 0;

	/// Data that only exist in specific assignment states.
	std::variant<unassigned_state, conditional_eagerly_assignable_state, unconditional_assignable_state, assigned_state> assignment;

	explicit incomplete_instruction_state(const instruction* instr) : instr(instr), assignment(unassigned_state()) {}
};

/// State maintained per host thread queue or device in-order queue.
struct lane_state {
	size_t num_in_flight_assigned_instructions = 0;
	std::optional<instruction_id> last_incomplete_submission; // instruction_id, because past instruction pointers may dangle
};

/// State maintained for every "multi-lane" target, i.e., `target::host_queue` and `target::device_queue[num_devices]`.
struct target_state {
	std::vector<lane_state> lanes;
};

// Implementation is behind a pimpl because we expect it to grow in complexity when adding support for waiting on inter-queue events to increase scheduling
// "eagerness" when an instruction needs to wait on multiple incomplete predecessors. This can already be implemented with AdaptiveCpp using the
// enqueue_custom_operation extension, but requires a similar interop story in DPC++, see https://github.com/intel/llvm/issues/13706 .
struct engine_impl {
	const system_info system;

	target_state host_queue_target_state;
	dense_map<device_id, target_state> device_queue_target_states;

	/// The set of all instructions between submit() and complete_assigned(). Keyed by `instruction_id` to allow collecting successors through iterating over a
	/// newly submitted instruction's dependencies, which are given in terms of instruction ids. Any dependency that is not found in `incomplete_instructions`
	/// is assumed to have completed earlier (triggering its removal from the map).
	std::unordered_map<instruction_id, incomplete_instruction_state> incomplete_instructions;

	/// Queue of all instructions in `conditional_eagerly_assignable_state` and `unconditional_assignable_state`, in decreasing order of instruction priority.
	std::priority_queue<const instruction*, std::vector<const instruction*>, instruction_priority_less> assignment_queue;

	explicit engine_impl(const system_info& system);

	// Non-copyable: Only used through unique_ptr.
	~engine_impl() = default;
	engine_impl(const engine_impl&) = delete;
	engine_impl(engine_impl&&) = delete;
	engine_impl& operator=(const engine_impl&) = delete;
	engine_impl& operator=(engine_impl&&) = delete;

	target_state& get_target_state(const target tgt, const std::optional<device_id>& device);

	/// Retrieve state for an existing lane.
	lane_state& get_lane_state(const target tgt, const std::optional<device_id>& device, const lane_id lane);

	/// Linearly search for a lane that has no in-flight instructions within a target_state. If none exists, add an additional lane.
	lane_id get_free_lane_id(const target tgt, const std::optional<device_id>& device);

	/// Attempt to replace assignment state for an incomplete instruction with `conditional_eagerly_assignable_state` or `unconditional_assignable_state` in
	/// response to either the initial submission or to completion of one or more predecessors. If successful, the instruction ends up in `assignment_queue`.
	void try_mark_for_assignment(incomplete_instruction_state& node);

	void submit(const instruction* const instr);

	void complete(const instruction_id iid);

	bool is_idle() const;

	/// Return the highest-priority instruction from `assignment_queue` that is unconditionally assignable (helper).
	incomplete_instruction_state* pop_assignable();

	std::optional<assignment> assign_one();
};

engine_impl::engine_impl(const system_info& system) : system(system), device_queue_target_states(system.devices.size()) {}

target_state& engine_impl::get_target_state(const target tgt, const std::optional<device_id>& device) {
	switch(tgt) {
	case target::host_queue: assert(!device.has_value()); return host_queue_target_state;
	case target::device_queue: assert(device.has_value()); return device_queue_target_states[*device];
	default: utils::unreachable();
	}
}

lane_state& engine_impl::get_lane_state(const target tgt, const std::optional<device_id>& device, const lane_id lane) {
	return get_target_state(tgt, device).lanes.at(lane);
}

lane_id engine_impl::get_free_lane_id(const target tgt, const std::optional<device_id>& device) {
	auto& target_state = get_target_state(tgt, device);
	for(lane_id lid = 0; lid < target_state.lanes.size(); ++lid) {
		if(target_state.lanes[lid].num_in_flight_assigned_instructions == 0) return lid;
	}
	target_state.lanes.emplace_back();
	return target_state.lanes.size() - 1;
}

void engine_impl::try_mark_for_assignment(incomplete_instruction_state& node) {
	if(std::holds_alternative<assigned_state>(node.assignment)                     // already assigned
	    || std::holds_alternative<unconditional_assignable_state>(node.assignment) // no upgrade path from here
	    || node.num_unassigned_predecessors > 0) {                                 // an instruction cannot be assigned before its predecessors
		return;
	}

	if(std::holds_alternative<conditional_eagerly_assignable_state>(node.assignment)) {
		// already in assignment_queue - try upgrading to `unconditional_assignable_state`
		if(node.num_incomplete_predecessors == 0) { node.assignment = unconditional_assignable_state{}; }
		return;
	}

	assert(std::holds_alternative<unassigned_state>(node.assignment));
	if(node.num_incomplete_predecessors == 0) {
		node.assignment = unconditional_assignable_state{};
		assignment_queue.push(node.instr);
		return;
	}

	auto& unassigned = std::get<unassigned_state>(node.assignment);
	if(!unassigned.probe_for_eager_assignment) return; // shortcut: we know this to be false ahead-of-time in many cases

	// We don't do eager assignment for the immediate target (because there is no in-order queueing behavior) nor for the `alloc_queue` target (because
	// allocations currently never form dependency chains where this would be beneficial - this might change though when we implement IDAG allocation pooling).
	assert(node.target == target::device_queue || node.target == target::host_queue);

	// The instruction still has pending dependencies, so it can't be assigned immediately, but might be assigned eagerly to a thread queue or in-order queue if
	// (1) all its dependencies are on the same device and lane and (2) one of the dependencies is the last submitted instruction within that lane. We still
	// require it to go through `assignment_queue` first for priority ordering, so condition (2) can change if another, higher-priority instruction is
	// popped and assigned before this instruction - in that case, we revert back to `unassigned_state` (see pop_assignable).
	std::optional<conditional_eagerly_assignable_state>
	    eagerly_assignable_now; // accumulator for verifying that _all_ incomplete dependencies are on the same lane
	for(auto dep_iid : node.instr->get_dependencies()) {
		const auto dep_it = incomplete_instructions.find(dep_iid);
		if(dep_it == incomplete_instructions.end()) continue; // dependency has completed before

		auto& [_, dep] = *dep_it;
		auto& dep_assigned = std::get<assigned_state>(dep.assignment); // otherwise num_unassigned_predecessors would have been > 0 above

		if(dep.target != node.target) return; // incompatible targets

		assert(dep_assigned.device.has_value() != dep.eligible_devices.empty());
		if(dep_assigned.device.has_value()
		    && std::find(node.eligible_devices.begin(), node.eligible_devices.end(), *dep_assigned.device) == node.eligible_devices.end()) {
			return; // dependency's device is not eligible for this instruction
		}

		if(eagerly_assignable_now.has_value()) {
			if(eagerly_assignable_now->device != dep_assigned.device) return; // there are dependencies on more than one device
			if(eagerly_assignable_now->lane != dep_assigned.lane) return;     // there are dependencies on multiple lanes
		} else {
			assert(dep_assigned.lane.has_value());
			eagerly_assignable_now = conditional_eagerly_assignable_state{dep_assigned.device, *dep_assigned.lane, std::nullopt};
		}

		auto& lane = get_lane_state(node.target, eagerly_assignable_now->device, eagerly_assignable_now->lane);
		if(lane.last_incomplete_submission == dep_iid) { eagerly_assignable_now->expected_last_submission_on_lane = dep_iid; }
	}

	// If we didn't return early so far (1) all incomplete dependencies are on the same lane
	assert(eagerly_assignable_now.has_value()); // otherwise num_incomplete_predecessors would have been == 0 above

	// Only if (2) one of the incomplete dependencies was last in the target lane will this be non-null
	if(!eagerly_assignable_now->expected_last_submission_on_lane.has_value()) return;

	node.assignment = *eagerly_assignable_now;
	assignment_queue.push(node.instr);
}

void engine_impl::submit(const instruction* const instr) {
	CELERITY_DETAIL_TRACY_ZONE_SCOPED("out_of_order_engine::submit", Blue3);

	const auto iid = instr->get_id();
	auto [node_it, inserted] = incomplete_instructions.emplace(iid, incomplete_instruction_state(instr));
	assert(inserted);
	auto& node = node_it->second;

	const auto add_eligible_devices_by_memory_id = [&](const memory_id mid) {
		/// We assume that there's either a 1:1 device <-> memory mapping for device-accessible memory, or when it's not, it's irrelevant which device we
		/// dispatch alloc / free instructions to.
		for(device_id did = 0; did < system.devices.size(); ++did) {
			if(system.devices[did].native_memory == mid) { node.eligible_devices.push_back(did); }
		}
	};

	// Perform target / queue assignment here instead of instruction graph generation time because the IDAG is too abstract of a description to know
	// about in-order queues, and because at least for copy-instruction we want to retain dynamic queue assignment to submit to either the source or
	// destination device and lengthen the path that can be scheduled onto a single in-order queue.
	matchbox::match(
	    *instr,                                //
	    [&](const alloc_instruction& ainstr) { //
		    node.target = target::alloc_queue;
		    add_eligible_devices_by_memory_id(ainstr.get_allocation_id().get_memory_id());
	    },
	    [&](const free_instruction& finstr) { //
		    node.target = target::alloc_queue;
		    add_eligible_devices_by_memory_id(finstr.get_allocation_id().get_memory_id());
	    },
	    [&](const copy_instruction& cinstr) {
		    const auto source_mid = cinstr.get_source_allocation_id().get_memory_id();
		    const auto dest_mid = cinstr.get_dest_allocation_id().get_memory_id();

		    // Eligible devices are tried in array-order within `assign_one`. If no other constraints exist, prefer assigning this instruction to the
		    // dest-memory device to allow eager assignment of a subsequent kernel launch to the same in-order queue.
		    add_eligible_devices_by_memory_id(dest_mid);
		    add_eligible_devices_by_memory_id(source_mid);

		    if(!node.eligible_devices.empty()) {
			    node.target = target::device_queue;
		    } else {
			    assert(source_mid <= host_memory_id && dest_mid <= host_memory_id);
			    node.target = target::host_queue;
		    }
	    },
	    [&](const device_kernel_instruction& dkinstr) {
		    node.target = target::device_queue;
		    node.eligible_devices.push_back(dkinstr.get_device_id());
	    },
	    [&](const host_task_instruction& htinstr) { //
		    node.target = target::host_queue;
	    },
	    [&](const clone_collective_group_instruction& /* other */) { node.target = target::immediate; },
	    [&](const send_instruction& /* other */) { node.target = target::immediate; },
	    [&](const receive_instruction& /* other */) { node.target = target::immediate; },
	    [&](const split_receive_instruction& /* other */) { node.target = target::immediate; },
	    [&](const await_receive_instruction& /* other */) { node.target = target::immediate; },
	    [&](const gather_receive_instruction& /* other */) { node.target = target::immediate; },
	    [&](const fill_identity_instruction& /* other */) { node.target = target::immediate; },
	    [&](const reduce_instruction& /* other */) { node.target = target::immediate; },
	    [&](const fence_instruction& /* other */) { node.target = target::immediate; },
	    [&](const destroy_host_object_instruction& /* other */) { node.target = target::immediate; },
	    [&](const horizon_instruction& /* other */) { node.target = target::immediate; },
	    [&](const epoch_instruction& /* other */) { node.target = target::immediate; });

	auto& unassigned = node.assignment.emplace<unassigned_state>();
	// target::immediate is not backed by an in-order queue, and alloc/free_instructions do not generate dependency chains frequently enough to justify
	// maintaining an in_order_queue_state
	unassigned.probe_for_eager_assignment = node.target == target::device_queue || node.target == target::host_queue;

	for(const auto pred_iid : instr->get_dependencies()) {
		// If predecessor is not found in `incomplete_instructions`, it has completed previously
		if(const auto pred_it = incomplete_instructions.find(pred_iid); pred_it != incomplete_instructions.end()) {
			auto& predecessor = pred_it->second;
			predecessor.successors.push_back(iid);
			++node.num_incomplete_predecessors;
			if(!std::holds_alternative<assigned_state>(predecessor.assignment)) { ++node.num_unassigned_predecessors; }
		}
	}

	// It might be possible to immediately assign the new instruction
	try_mark_for_assignment(node);
}

void engine_impl::complete(const instruction_id iid) {
	CELERITY_DETAIL_TRACY_ZONE_SCOPED("out_of_order_engine::complete", Blue3);

	const auto node_it = incomplete_instructions.find(iid);
	assert(node_it != incomplete_instructions.end());
	auto deleted_node = std::move(node_it->second); // move so we can access members / iterate successors after erasure
	incomplete_instructions.erase(node_it);

	auto& was_assigned = std::get<assigned_state>(deleted_node.assignment);
	if(deleted_node.target == target::host_queue || deleted_node.target == target::device_queue) {
		// "remove" instruction from assigned lane
		assert(was_assigned.lane.has_value());
		auto& lane = get_lane_state(deleted_node.target, was_assigned.device, *was_assigned.lane);
		assert(lane.num_in_flight_assigned_instructions > 0);
		lane.num_in_flight_assigned_instructions -= 1;
		if(lane.last_incomplete_submission == iid) { lane.last_incomplete_submission = std::nullopt; }
	}

	for(const auto succ_iid : deleted_node.successors) {
		// When multiple instructions complete in the same time step, out_of_order_engine explicitly allows them to be marked as completed in arbitrary
		// order even if that would violate the internal dependency relationship. This avoids the need for duplicating dependency tracking elsewhere.
		// As a consequence, some successors might have already been removed from the set and we need to call find() instead of at() here.
		if(const auto succ_it = incomplete_instructions.find(succ_iid); succ_it != incomplete_instructions.end()) {
			auto& successor = succ_it->second;
			assert(successor.num_incomplete_predecessors > 0);
			--successor.num_incomplete_predecessors;
			try_mark_for_assignment(successor);
		}
	}
}

bool engine_impl::is_idle() const {
#ifndef NDEBUG
	if(incomplete_instructions.empty()) {
		assert(assignment_queue.empty());
		for(const auto& device_state : device_queue_target_states) {
			for(const auto& lane : device_state.lanes) {
				assert(lane.num_in_flight_assigned_instructions == 0);
			}
		}
		for(const auto& lane : host_queue_target_state.lanes) {
			assert(lane.num_in_flight_assigned_instructions == 0);
		}
	}
#endif
	return incomplete_instructions.empty();
}

incomplete_instruction_state* engine_impl::pop_assignable() {
	while(!assignment_queue.empty()) {
		const auto instr = assignment_queue.top();
		assignment_queue.pop();

		auto& node = incomplete_instructions.at(instr->get_id());
		assert(node.num_unassigned_predecessors == 0);

		if(const auto eagerly_assignable_when_pushed = std::get_if<conditional_eagerly_assignable_state>(&node.assignment)) {
			assert(node.num_incomplete_predecessors > 0); // otherwise this would be an immediately_assignable_state
			assert(eagerly_assignable_when_pushed->expected_last_submission_on_lane.has_value());
			const auto& lane = get_lane_state(node.target, eagerly_assignable_when_pushed->device, eagerly_assignable_when_pushed->lane);
			if(lane.last_incomplete_submission == eagerly_assignable_when_pushed->expected_last_submission_on_lane) {
				// Our preferred lane is still in the required state to go through with eager assignment
				return &node;
			} else {
				// One of two conditions is met:
				// a) lane.last_incomplete_submission != nullptr: A third instruction has been submitted to our preferred lane since, so we drop eager
				//    assignment and don't need to attempt it again, since none of our dependencies can now end up last in the queue anymore.
				// b) lane.last_incomplete_submission == nullptr: We have been overtaken by a higher-priority instruction which has since completed. This rare
				//    case transitively means that all our predecessors have completed as well, but since we still await a call to complete(), we abort eager
				//    assignment to avoid dealing with the special case of an unconditional_assignable_state with num_incomplete_predecessors > 0.
				node.assignment.emplace<unassigned_state>().probe_for_eager_assignment = false;
				continue;
			}
		} else {
			assert(std::holds_alternative<unconditional_assignable_state>(node.assignment));
			assert(node.num_incomplete_predecessors == 0);
			return &node;
		}
	}

	return nullptr;
}

std::optional<assignment> engine_impl::assign_one() {
	if(assignment_queue.empty()) return std::nullopt; // Don't begin a Tracy zone if there is nothing to assign
	CELERITY_DETAIL_TRACY_ZONE_SCOPED("out_of_order_engine::assign", Blue3);

	const auto node_ptr = pop_assignable();
	if(node_ptr == nullptr) return std::nullopt;

	auto& node = *node_ptr;

	assigned_state assigned;
	if(const auto& eagerly_assignable = std::get_if<conditional_eagerly_assignable_state>(&node.assignment)) {
		// After an instruction is popped from the assignment queue, "eager-assignability" is unconditional. Force-assign to the same lane to implicitly fulfill
		// all remaining dependencies through queue ordering.
		assigned.device = eagerly_assignable->device;
		assigned.lane = eagerly_assignable->lane;
		assert(
		    assigned.lane.has_value() && eagerly_assignable->expected_last_submission_on_lane.has_value()
		    && get_lane_state(node.target, assigned.device, *assigned.lane).last_incomplete_submission == eagerly_assignable->expected_last_submission_on_lane);
	} else {
		if(!node.eligible_devices.empty()) {
			// "Heuristically" pick a device
			assert(node.target == target::alloc_queue || node.target == target::device_queue);
			assigned.device = node.eligible_devices.front();
		}
		if(node.target == target::host_queue || node.target == target::device_queue) { //
			// Select a free existing lane or create a new one. This might cause excessive numbers of threads or in-order queues to be constructed in the
			// backend (even though this number is indirectly bounded by breadth horizons). TODO in the future consider limiting the number of lanes while
			// avoiding potential deadlocks that can then arise from stalling / temporally re-ordering collective host tasks between nodes.
			assigned.lane = get_free_lane_id(node.target, assigned.device);
		}
	}
	node.assignment = assigned;

	if(assigned.lane.has_value()) {
		auto& lane = get_lane_state(node.target, assigned.device, *assigned.lane);
		lane.num_in_flight_assigned_instructions += 1;
		lane.last_incomplete_submission = node.instr->get_id();
	}

	for(const auto successor : node.successors) {
		auto& successor_node = incomplete_instructions.at(successor);
		assert(successor_node.num_unassigned_predecessors > 0);
		if(--successor_node.num_unassigned_predecessors == 0) {
			// might now be eligible for eager assignment
			try_mark_for_assignment(successor_node);
		}
	}

	return assignment{node.instr, node.target, assigned.device, assigned.lane};
}

} // namespace celerity::detail::out_of_order_engine_detail

namespace celerity::detail {

out_of_order_engine::out_of_order_engine(const system_info& system) : m_impl(new out_of_order_engine_detail::engine_impl(system)) {}
out_of_order_engine::~out_of_order_engine() = default;
bool out_of_order_engine::is_idle() const { return m_impl->is_idle(); }
size_t out_of_order_engine::get_assignment_queue_length() const { return m_impl->assignment_queue.size(); }
void out_of_order_engine::submit(const instruction* const instr) { m_impl->submit(instr); }
void out_of_order_engine::complete_assigned(const instruction_id iid) { m_impl->complete(iid); }
std::optional<out_of_order_engine::assignment> out_of_order_engine::assign_one() { return m_impl->assign_one(); }

} // namespace celerity::detail
