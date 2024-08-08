#include "live_executor.h"
#include "backend/backend.h"
#include "closure_hydrator.h"
#include "communicator.h"
#include "host_object.h"
#include "instruction_graph.h"
#include "named_threads.h"
#include "out_of_order_engine.h"
#include "receive_arbiter.h"
#include "system_info.h"
#include "types.h"
#include "utils.h"

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <matchbox.hh>

namespace celerity::detail::live_executor_detail {

#if CELERITY_ACCESSOR_BOUNDARY_CHECK
struct boundary_check_info {
	struct accessor_info {
		detail::buffer_id buffer_id = 0;
		std::string buffer_name;
		box<3> accessible_box;
	};

	detail::task_type task_type;
	detail::task_id task_id;
	std::string task_name;

	oob_bounding_box* illegal_access_bounding_boxes = nullptr;
	std::vector<accessor_info> accessors;

	boundary_check_info(detail::task_type tt, detail::task_id tid, const std::string& task_name) : task_type(tt), task_id(tid), task_name(task_name) {}
};
#endif

struct async_instruction_state {
	instruction_id iid = -1;
	allocation_id alloc_aid = null_allocation_id; ///< non-null iff instruction is an alloc_instruction
	async_event event;
	CELERITY_DETAIL_IF_ACCESSOR_BOUNDARY_CHECK(std::unique_ptr<boundary_check_info> oob_info;) // unique_ptr: oob_info is optional and rather large
};

struct executor_impl {
	const std::unique_ptr<detail::backend> backend;
	communicator* const root_communicator;
	double_buffered_queue<submission>* const submission_queue;
	live_executor::delegate* const delegate;
	const live_executor::policy_set policy;

	receive_arbiter recv_arbiter{*root_communicator};
	out_of_order_engine engine{backend->get_system_info()};

	bool expecting_more_submissions = true; ///< shutdown epoch has not been executed yet
	std::vector<async_instruction_state> in_flight_async_instructions;
	std::unordered_map<allocation_id, void*> allocations{{null_allocation_id, nullptr}}; ///< obtained from alloc_instruction or track_user_allocation
	std::unordered_map<host_object_id, std::unique_ptr<host_object_instance>> host_object_instances; ///< passed in through track_host_object_instance
	std::unordered_map<collective_group_id, std::unique_ptr<communicator>> cloned_communicators;     ///< transitive clones of root_communicator
	std::unordered_map<reduction_id, std::unique_ptr<reducer>> reducers; ///< passed in through track_reducer, erased on epochs / horizons

	std::optional<std::chrono::steady_clock::time_point> last_progress_timestamp; ///< last successful call to check_progress
	bool made_progress = false;                                                   ///< progress was made since `last_progress_timestamp`
	bool progress_warning_emitted = false;                                        ///< no progress was made since warning was emitted

	executor_impl(std::unique_ptr<detail::backend> backend, communicator* root_comm, double_buffered_queue<submission>& submission_queue,
	    live_executor::delegate* dlg, const live_executor::policy_set& policy);

	void run();
	void poll_in_flight_async_instructions();
	void poll_submission_queue();
	void try_issue_one_instruction();
	void retire_async_instruction(async_instruction_state& async);
	void check_progress();

	// Instruction types that complete synchronously within the executor.
	void issue(const clone_collective_group_instruction& ccginstr);
	void issue(const split_receive_instruction& srinstr);
	void issue(const fill_identity_instruction& fiinstr);
	void issue(const reduce_instruction& rinstr);
	void issue(const fence_instruction& finstr);
	void issue(const destroy_host_object_instruction& dhoinstr);
	void issue(const horizon_instruction& hinstr);
	void issue(const epoch_instruction& einstr);

	template <typename Instr>
	auto dispatch(const Instr& instr, const out_of_order_engine::assignment& assignment)
	    // SFINAE: there is a (synchronous) `issue` overload above for the concrete Instr type
	    -> decltype(issue(instr));

	// Instruction types that complete asynchronously via async_event, outside the executor.
	void issue_async(const alloc_instruction& ainstr, const out_of_order_engine::assignment& assignment, async_instruction_state& async);
	void issue_async(const free_instruction& finstr, const out_of_order_engine::assignment& assignment, async_instruction_state& async);
	void issue_async(const copy_instruction& cinstr, const out_of_order_engine::assignment& assignment, async_instruction_state& async);
	void issue_async(const device_kernel_instruction& dkinstr, const out_of_order_engine::assignment& assignment, async_instruction_state& async);
	void issue_async(const host_task_instruction& htinstr, const out_of_order_engine::assignment& assignment, async_instruction_state& async);
	void issue_async(const send_instruction& sinstr, const out_of_order_engine::assignment& assignment, async_instruction_state& async);
	void issue_async(const receive_instruction& rinstr, const out_of_order_engine::assignment& assignment, async_instruction_state& async);
	void issue_async(const await_receive_instruction& arinstr, const out_of_order_engine::assignment& assignment, async_instruction_state& async);
	void issue_async(const gather_receive_instruction& grinstr, const out_of_order_engine::assignment& assignment, async_instruction_state& async);

	template <typename Instr>
	auto dispatch(const Instr& instr, const out_of_order_engine::assignment& assignment)
	    // SFINAE: there is an `issue_async` overload above for the concrete Instr type
	    -> decltype(issue_async(instr, assignment, std::declval<async_instruction_state&>()));

	std::vector<closure_hydrator::accessor_info> make_accessor_infos(const buffer_access_allocation_map& amap) const;

#if CELERITY_ACCESSOR_BOUNDARY_CHECK
	std::unique_ptr<boundary_check_info> attach_boundary_check_info(std::vector<closure_hydrator::accessor_info>& accessor_infos,
	    const buffer_access_allocation_map& amap, task_type tt, task_id tid, const std::string& task_name) const;
#endif

	void collect(const instruction_garbage& garbage);
};

executor_impl::executor_impl(std::unique_ptr<detail::backend> backend, communicator* const root_comm, double_buffered_queue<submission>& submission_queue,
    live_executor::delegate* const dlg, const live_executor::policy_set& policy)
    : backend(std::move(backend)), root_communicator(root_comm), submission_queue(&submission_queue), delegate(dlg), policy(policy) {}

void executor_impl::run() {
	closure_hydrator::make_available();

	uint8_t check_overflow_counter = 0;
	for(;;) {
		if(engine.is_idle()) {
			if(!expecting_more_submissions) break; // shutdown complete
			submission_queue->wait_while_empty();  // we are stalled on the scheduler, suspend thread
			last_progress_timestamp.reset();       // do not treat suspension as being stuck
		}

		recv_arbiter.poll_communicator();
		poll_in_flight_async_instructions();
		poll_submission_queue();
		try_issue_one_instruction(); // potentially expensive, so only issue one per loop to continue checking for async completion in between

		if(++check_overflow_counter == 0) { // once every 256 iterations
			backend->check_async_errors();
			check_progress();
		}
	}

	assert(in_flight_async_instructions.empty());
	// check that for each alloc_instruction, we executed a corresponding free_instruction
	assert(std::all_of(allocations.begin(), allocations.end(),
	    [](const std::pair<allocation_id, void*>& p) { return p.first == null_allocation_id || p.first.get_memory_id() == user_memory_id; }));
	// check that for each track_host_object_instance, we executed a destroy_host_object_instruction
	assert(host_object_instances.empty());

	closure_hydrator::teardown();
}

void executor_impl::poll_in_flight_async_instructions() {
	utils::erase_if(in_flight_async_instructions, [&](async_instruction_state& async) {
		if(!async.event.is_complete()) return false;
		retire_async_instruction(async);
		made_progress = true;
		return true;
	});
}

void executor_impl::poll_submission_queue() {
	for(auto& submission : submission_queue->pop_all()) {
		matchbox::match(
		    submission,
		    [&](const instruction_pilot_batch& batch) {
			    for(const auto incoming_instr : batch.instructions) {
				    engine.submit(incoming_instr);
			    }
			    for(const auto& pilot : batch.pilots) {
				    root_communicator->send_outbound_pilot(pilot);
			    }
		    },
		    [&](const user_allocation_transfer& uat) {
			    assert(uat.aid != null_allocation_id);
			    assert(uat.aid.get_memory_id() == user_memory_id);
			    assert(allocations.count(uat.aid) == 0);
			    allocations.emplace(uat.aid, uat.ptr);
		    },
		    [&](host_object_transfer& hot) {
			    assert(host_object_instances.count(hot.hoid) == 0);
			    host_object_instances.emplace(hot.hoid, std::move(hot.instance));
		    },
		    [&](reducer_transfer& rt) {
			    assert(reducers.count(rt.rid) == 0);
			    reducers.emplace(rt.rid, std::move(rt.reduction));
		    });
	}
}

void executor_impl::retire_async_instruction(async_instruction_state& async) {
#if CELERITY_ACCESSOR_BOUNDARY_CHECK
	if(async.oob_info != nullptr) {
		const auto& oob_info = *async.oob_info;
		for(size_t i = 0; i < oob_info.accessors.size(); ++i) {
			if(const auto oob_box = oob_info.illegal_access_bounding_boxes[i].into_box(); !oob_box.empty()) {
				const auto& accessor_info = oob_info.accessors[i];
				CELERITY_ERROR("Out-of-bounds access detected in {}: accessor {} attempted to access buffer {} indicies between {} and outside the "
				               "declared range {}.",
				    utils::make_task_debug_label(oob_info.task_type, oob_info.task_id, oob_info.task_name), i,
				    utils::make_buffer_debug_label(accessor_info.buffer_id, accessor_info.buffer_name), oob_box, accessor_info.accessible_box);
			}
		}
		if(oob_info.illegal_access_bounding_boxes != nullptr /* i.e. there was at least one accessor */) {
			backend->debug_free(oob_info.illegal_access_bounding_boxes);
		}
	}
#endif

	if(spdlog::should_log(spdlog::level::trace)) {
		if(const auto native_time = async.event.get_native_execution_time(); native_time.has_value()) {
			CELERITY_TRACE("[executor] retired I{} after {:.2f}", async.iid, as_sub_second(*native_time));
		} else {
			CELERITY_TRACE("[executor] retired I{}", async.iid);
		}
	}

	if(async.alloc_aid != null_allocation_id) {
		const auto ptr = async.event.get_result();
		assert(ptr != nullptr && "backend allocation returned nullptr");
		CELERITY_TRACE("[executor] {} allocated as {}", async.alloc_aid, ptr);
		assert(allocations.count(async.alloc_aid) == 0);
		allocations.emplace(async.alloc_aid, ptr);
	}

	engine.complete_assigned(async.iid);
}

template <typename Instr>
auto executor_impl::dispatch(const Instr& instr, const out_of_order_engine::assignment& assignment)
    // SFINAE: there is a (synchronous) `issue` overload above for the concrete Instr type
    -> decltype(issue(instr)) //
{
	assert(assignment.target == out_of_order_engine::target::immediate);
	assert(!assignment.lane.has_value());

	const auto iid = instr.get_id(); // instr may dangle after issue()
	issue(instr);                    // completes immediately
	engine.complete_assigned(iid);
}

template <typename Instr>
auto executor_impl::dispatch(const Instr& instr, const out_of_order_engine::assignment& assignment)
    // SFINAE: there is an `issue_async` overload above for the concrete Instr type
    -> decltype(issue_async(instr, assignment, std::declval<async_instruction_state&>())) //
{
	auto& async = in_flight_async_instructions.emplace_back();
	async.iid = assignment.instruction->get_id();
	issue_async(instr, assignment, async); // stores event in `async` and completes asynchronously
}

void executor_impl::try_issue_one_instruction() {
	auto assignment = engine.assign_one();
	if(!assignment.has_value()) return;

	matchbox::match(*assignment->instruction, [&](const auto& instr) { dispatch(instr, *assignment); });
	made_progress = true;
}

void executor_impl::check_progress() {
	if(!policy.progress_warning_timeout.has_value()) return;

	if(made_progress) {
		last_progress_timestamp = std::chrono::steady_clock::now();
		progress_warning_emitted = false;
		made_progress = false;
	} else if(last_progress_timestamp.has_value()) {
		// being stuck either means a deadlock in the user application, or a bug in Celerity.
		const auto elapsed_since_last_progress = std::chrono::steady_clock::now() - *last_progress_timestamp;
		if(elapsed_since_last_progress > *policy.progress_warning_timeout && !progress_warning_emitted) {
			std::string instr_list;
			for(auto& in_flight : in_flight_async_instructions) {
				if(!instr_list.empty()) instr_list += ", ";
				fmt::format_to(std::back_inserter(instr_list), "I{}", in_flight.iid);
			}
			CELERITY_WARN("[executor] no progress for {:.0f}, might be stuck. Active instructions: {}", as_sub_second(elapsed_since_last_progress),
			    in_flight_async_instructions.empty() ? "none" : instr_list);
			progress_warning_emitted = true;
		}
	}
}

void executor_impl::issue(const clone_collective_group_instruction& ccginstr) {
	const auto original_cgid = ccginstr.get_original_collective_group_id();
	assert(original_cgid != non_collective_group_id);
	assert(original_cgid == root_collective_group_id || cloned_communicators.count(original_cgid) != 0);

	const auto new_cgid = ccginstr.get_new_collective_group_id();
	assert(new_cgid != non_collective_group_id && new_cgid != root_collective_group_id);
	assert(cloned_communicators.count(new_cgid) == 0);

	CELERITY_TRACE("[executor] I{}: clone collective group CG{} -> CG{}", ccginstr.get_id(), original_cgid, new_cgid);

	const auto original_communicator = original_cgid == root_collective_group_id ? root_communicator : cloned_communicators.at(original_cgid).get();
	cloned_communicators.emplace(new_cgid, original_communicator->collective_clone());
}


void executor_impl::issue(const split_receive_instruction& srinstr) {
	CELERITY_TRACE("[executor] I{}: split receive {} {}x{} bytes into {} ({}),", srinstr.get_id(), srinstr.get_transfer_id(), srinstr.get_requested_region(),
	    srinstr.get_element_size(), srinstr.get_dest_allocation_id(), srinstr.get_allocated_box());

	const auto allocation = allocations.at(srinstr.get_dest_allocation_id());
	recv_arbiter.begin_split_receive(
	    srinstr.get_transfer_id(), srinstr.get_requested_region(), allocation, srinstr.get_allocated_box(), srinstr.get_element_size());
}

void executor_impl::issue(const fill_identity_instruction& fiinstr) {
	CELERITY_TRACE("[executor] I{}: fill identity {} x{} values for R{}", fiinstr.get_id(), fiinstr.get_allocation_id(), fiinstr.get_num_values(),
	    fiinstr.get_reduction_id());

	const auto allocation = allocations.at(fiinstr.get_allocation_id());
	const auto& reduction = *reducers.at(fiinstr.get_reduction_id());
	reduction.fill_identity(allocation, fiinstr.get_num_values());
}

void executor_impl::issue(const reduce_instruction& rinstr) {
	CELERITY_TRACE("[executor] I{}: reduce {} x{} values into {} for R{}", rinstr.get_id(), rinstr.get_source_allocation_id(), rinstr.get_num_source_values(),
	    rinstr.get_dest_allocation_id(), rinstr.get_reduction_id());

	const auto gather_allocation = allocations.at(rinstr.get_source_allocation_id());
	const auto dest_allocation = allocations.at(rinstr.get_dest_allocation_id());
	const auto& reduction = *reducers.at(rinstr.get_reduction_id());
	reduction.reduce(dest_allocation, gather_allocation, rinstr.get_num_source_values());
}

void executor_impl::issue(const fence_instruction& finstr) { // NOLINT(readability-make-member-function-const, readability-convert-member-functions-to-static)
	CELERITY_TRACE("[executor] I{}: fence", finstr.get_id());

	finstr.get_promise()->fulfill();
}

void executor_impl::issue(const destroy_host_object_instruction& dhoinstr) {
	assert(host_object_instances.count(dhoinstr.get_host_object_id()) != 0);
	CELERITY_TRACE("[executor] I{}: destroy H{}", dhoinstr.get_id(), dhoinstr.get_host_object_id());

	host_object_instances.erase(dhoinstr.get_host_object_id());
}

void executor_impl::issue(const horizon_instruction& hinstr) {
	CELERITY_TRACE("[executor] I{}: horizon", hinstr.get_id());

	if(delegate != nullptr) { delegate->horizon_reached(hinstr.get_horizon_task_id()); }
	collect(hinstr.get_garbage());
}

void executor_impl::issue(const epoch_instruction& einstr) {
	switch(einstr.get_epoch_action()) {
	case epoch_action::none: //
		CELERITY_TRACE("[executor] I{}: epoch", einstr.get_id());
		break;
	case epoch_action::barrier: //
		CELERITY_TRACE("[executor] I{}: epoch (barrier)", einstr.get_id());
		root_communicator->collective_barrier();
		break;
	case epoch_action::shutdown: //
		CELERITY_TRACE("[executor] I{}: epoch (shutdown)", einstr.get_id());
		expecting_more_submissions = false;
		break;
	}
	if(delegate != nullptr && einstr.get_epoch_task_id() != 0 /* TODO task_manager doesn't expect us to actually execute the init epoch */) {
		delegate->epoch_reached(einstr.get_epoch_task_id());
	}
	collect(einstr.get_garbage());
}

void executor_impl::issue_async(const alloc_instruction& ainstr, const out_of_order_engine::assignment& assignment, async_instruction_state& async) {
	assert(ainstr.get_allocation_id().get_memory_id() != user_memory_id);
	assert(assignment.target == out_of_order_engine::target::alloc_queue);
	assert(!assignment.lane.has_value());
	assert(assignment.device.has_value() == (ainstr.get_allocation_id().get_memory_id() > host_memory_id));

	CELERITY_TRACE(
	    "[executor] I{}: alloc {}, {} % {} bytes", ainstr.get_id(), ainstr.get_allocation_id(), ainstr.get_size_bytes(), ainstr.get_alignment_bytes());

	if(assignment.device.has_value()) {
		async.event = backend->enqueue_device_alloc(*assignment.device, ainstr.get_size_bytes(), ainstr.get_alignment_bytes());
	} else {
		async.event = backend->enqueue_host_alloc(ainstr.get_size_bytes(), ainstr.get_alignment_bytes());
	}
	async.alloc_aid = ainstr.get_allocation_id(); // setting alloc_aid != null will make `retire_async_instruction` insert the result into `allocations`
}

void executor_impl::issue_async(const free_instruction& finstr, const out_of_order_engine::assignment& assignment, async_instruction_state& async) {
	const auto it = allocations.find(finstr.get_allocation_id());
	assert(it != allocations.end());
	const auto ptr = it->second;
	allocations.erase(it);

	CELERITY_TRACE("[executor] I{}: free {}", finstr.get_id(), finstr.get_allocation_id());

	if(assignment.device.has_value()) {
		async.event = backend->enqueue_device_free(*assignment.device, ptr);
	} else {
		async.event = backend->enqueue_host_free(ptr);
	}
}

void executor_impl::issue_async(const copy_instruction& cinstr, const out_of_order_engine::assignment& assignment, async_instruction_state& async) {
	assert(assignment.target == out_of_order_engine::target::host_queue || assignment.target == out_of_order_engine::target::device_queue);
	assert((assignment.target == out_of_order_engine::target::device_queue) == assignment.device.has_value());
	assert(assignment.lane.has_value());

	CELERITY_TRACE("[executor] I{}: copy {} ({}) -> {} ({}), {}x{} bytes", cinstr.get_id(), cinstr.get_source_allocation(), cinstr.get_source_box(),
	    cinstr.get_dest_allocation(), cinstr.get_dest_box(), cinstr.get_copy_region(), cinstr.get_element_size());

	const auto source_base = static_cast<const std::byte*>(allocations.at(cinstr.get_source_allocation().id)) + cinstr.get_source_allocation().offset_bytes;
	const auto dest_base = static_cast<std::byte*>(allocations.at(cinstr.get_dest_allocation().id)) + cinstr.get_dest_allocation().offset_bytes;

	if(assignment.device.has_value()) {
		async.event = backend->enqueue_device_copy(*assignment.device, *assignment.lane, source_base, dest_base, cinstr.get_source_box(), cinstr.get_dest_box(),
		    cinstr.get_copy_region(), cinstr.get_element_size());
	} else {
		async.event = backend->enqueue_host_copy(
		    *assignment.lane, source_base, dest_base, cinstr.get_source_box(), cinstr.get_dest_box(), cinstr.get_copy_region(), cinstr.get_element_size());
	}
}

std::string format_access_log(const buffer_access_allocation_map& map) {
	std::string acc_log;
	for(size_t i = 0; i < map.size(); ++i) {
		auto& aa = map[i];
		const auto accessed_box_in_allocation = box(aa.accessed_box_in_buffer.get_min() - aa.allocated_box_in_buffer.get_offset(),
		    aa.accessed_box_in_buffer.get_max() - aa.allocated_box_in_buffer.get_offset());
		fmt::format_to(std::back_inserter(acc_log), "{} {} {}", i == 0 ? ", accessing" : ",", aa.allocation_id, accessed_box_in_allocation);
	}
	return acc_log;
}

void executor_impl::issue_async(const device_kernel_instruction& dkinstr, const out_of_order_engine::assignment& assignment, async_instruction_state& async) {
	assert(assignment.target == out_of_order_engine::target::device_queue);
	assert(assignment.device == dkinstr.get_device_id());
	assert(assignment.lane.has_value());

	CELERITY_TRACE("[executor] I{}: launch device kernel on D{}, {}{}", dkinstr.get_id(), dkinstr.get_device_id(), dkinstr.get_execution_range(),
	    format_access_log(dkinstr.get_access_allocations()));

	auto accessor_infos = make_accessor_infos(dkinstr.get_access_allocations());
#if CELERITY_ACCESSOR_BOUNDARY_CHECK
	async.oob_info = attach_boundary_check_info(
	    accessor_infos, dkinstr.get_access_allocations(), dkinstr.get_oob_task_type(), dkinstr.get_oob_task_id(), dkinstr.get_oob_task_name());
#endif

	const auto& reduction_allocs = dkinstr.get_reduction_allocations();
	std::vector<void*> reduction_ptrs(reduction_allocs.size());
	for(size_t i = 0; i < reduction_allocs.size(); ++i) {
		reduction_ptrs[i] = allocations.at(reduction_allocs[i].allocation_id);
	}

	async.event = backend->enqueue_device_kernel(
	    dkinstr.get_device_id(), *assignment.lane, dkinstr.get_launcher(), std::move(accessor_infos), dkinstr.get_execution_range(), reduction_ptrs);
}

void executor_impl::issue_async(const host_task_instruction& htinstr, const out_of_order_engine::assignment& assignment, async_instruction_state& async) {
	assert(assignment.target == out_of_order_engine::target::host_queue);
	assert(!assignment.device.has_value());
	assert(assignment.lane.has_value());

	CELERITY_TRACE(
	    "[executor] I{}: launch host task, {}{}", htinstr.get_id(), htinstr.get_execution_range(), format_access_log(htinstr.get_access_allocations()));

	auto accessor_infos = make_accessor_infos(htinstr.get_access_allocations());
#if CELERITY_ACCESSOR_BOUNDARY_CHECK
	async.oob_info = attach_boundary_check_info(
	    accessor_infos, htinstr.get_access_allocations(), htinstr.get_oob_task_type(), htinstr.get_oob_task_id(), htinstr.get_oob_task_name());
#endif

	const auto& execution_range = htinstr.get_execution_range();
	const auto collective_comm =
	    htinstr.get_collective_group_id() != non_collective_group_id ? cloned_communicators.at(htinstr.get_collective_group_id()).get() : nullptr;

	async.event = backend->enqueue_host_task(*assignment.lane, htinstr.get_launcher(), std::move(accessor_infos), execution_range, collective_comm);
}

void executor_impl::issue_async(
    const send_instruction& sinstr, [[maybe_unused]] const out_of_order_engine::assignment& assignment, async_instruction_state& async) //
{
	assert(assignment.target == out_of_order_engine::target::immediate);

	CELERITY_TRACE("[executor] I{}: send {}+{}, {}x{} bytes to N{} (MSG{})", sinstr.get_id(), sinstr.get_source_allocation_id(),
	    sinstr.get_offset_in_source_allocation(), sinstr.get_send_range(), sinstr.get_element_size(), sinstr.get_dest_node_id(), sinstr.get_message_id());

	const auto allocation_base = allocations.at(sinstr.get_source_allocation_id());
	const communicator::stride stride{
	    sinstr.get_source_allocation_range(),
	    subrange<3>{sinstr.get_offset_in_source_allocation(), sinstr.get_send_range()},
	    sinstr.get_element_size(),
	};
	async.event = root_communicator->send_payload(sinstr.get_dest_node_id(), sinstr.get_message_id(), allocation_base, stride);
}

void executor_impl::issue_async(
    const receive_instruction& rinstr, [[maybe_unused]] const out_of_order_engine::assignment& assignment, async_instruction_state& async) //
{
	assert(assignment.target == out_of_order_engine::target::immediate);

	CELERITY_TRACE("[executor] I{}: receive {} {}x{} bytes into {} ({})", rinstr.get_id(), rinstr.get_transfer_id(), rinstr.get_requested_region(),
	    rinstr.get_element_size(), rinstr.get_dest_allocation_id(), rinstr.get_allocated_box());

	const auto allocation = allocations.at(rinstr.get_dest_allocation_id());
	async.event =
	    recv_arbiter.receive(rinstr.get_transfer_id(), rinstr.get_requested_region(), allocation, rinstr.get_allocated_box(), rinstr.get_element_size());
}

void executor_impl::issue_async(
    const await_receive_instruction& arinstr, [[maybe_unused]] const out_of_order_engine::assignment& assignment, async_instruction_state& async) //
{
	assert(assignment.target == out_of_order_engine::target::immediate);

	CELERITY_TRACE("[executor] I{}: await receive {} {}", arinstr.get_id(), arinstr.get_transfer_id(), arinstr.get_received_region());

	async.event = recv_arbiter.await_split_receive_subregion(arinstr.get_transfer_id(), arinstr.get_received_region());
}

void executor_impl::issue_async(
    const gather_receive_instruction& grinstr, [[maybe_unused]] const out_of_order_engine::assignment& assignment, async_instruction_state& async) //
{
	assert(assignment.target == out_of_order_engine::target::immediate);

	CELERITY_TRACE("[executor] I{}: gather receive {} into {}, {} bytes / node", grinstr.get_id(), grinstr.get_transfer_id(), grinstr.get_dest_allocation_id(),
	    grinstr.get_node_chunk_size());

	const auto allocation = allocations.at(grinstr.get_dest_allocation_id());
	async.event = recv_arbiter.gather_receive(grinstr.get_transfer_id(), allocation, grinstr.get_node_chunk_size());
}

void executor_impl::collect(const instruction_garbage& garbage) {
	for(const auto rid : garbage.reductions) {
		assert(reducers.count(rid) != 0);
		reducers.erase(rid);
	}
	for(const auto aid : garbage.user_allocations) {
		assert(aid.get_memory_id() == user_memory_id);
		assert(allocations.count(aid) != 0);
		allocations.erase(aid);
	}
}

std::vector<closure_hydrator::accessor_info> executor_impl::make_accessor_infos(const buffer_access_allocation_map& amap) const {
	std::vector<closure_hydrator::accessor_info> accessor_infos(amap.size());
	for(size_t i = 0; i < amap.size(); ++i) {
		const auto ptr = allocations.at(amap[i].allocation_id);
		accessor_infos[i] = closure_hydrator::accessor_info{ptr, amap[i].allocated_box_in_buffer, amap[i].accessed_box_in_buffer};
	}
	return accessor_infos;
}

#if CELERITY_ACCESSOR_BOUNDARY_CHECK
std::unique_ptr<boundary_check_info> executor_impl::attach_boundary_check_info(std::vector<closure_hydrator::accessor_info>& accessor_infos,
    const buffer_access_allocation_map& amap, task_type tt, task_id tid, const std::string& task_name) const //
{
	if(amap.empty()) return nullptr;

	auto oob_info = std::make_unique<boundary_check_info>(tt, tid, task_name);

	oob_info->illegal_access_bounding_boxes = static_cast<oob_bounding_box*>(backend->debug_alloc(amap.size() * sizeof(oob_bounding_box)));
	std::uninitialized_default_construct_n(oob_info->illegal_access_bounding_boxes, amap.size());

	oob_info->accessors.resize(amap.size());
	for(size_t i = 0; i < amap.size(); ++i) {
		oob_info->accessors[i] = boundary_check_info::accessor_info{amap[i].oob_buffer_id, amap[i].oob_buffer_name, amap[i].accessed_box_in_buffer};
		accessor_infos[i].out_of_bounds_indices = oob_info->illegal_access_bounding_boxes + i;
	}
	return oob_info;
}
#endif // CELERITY_ACCESSOR_BOUNDARY_CHECK

} // namespace celerity::detail::live_executor_detail

namespace celerity::detail {

live_executor::live_executor(std::unique_ptr<backend> backend, std::unique_ptr<communicator> root_comm, delegate* const dlg, const policy_set& policy)
    : m_root_comm(std::move(root_comm)), m_thread(&live_executor::thread_main, this, std::move(backend), dlg, policy) //
{
	set_thread_name(m_thread.native_handle(), "cy-executor");
}

live_executor::~live_executor() {
	m_thread.join(); // thread_main will exit only after executing shutdown epoch
}

void live_executor::track_user_allocation(const allocation_id aid, void* const ptr) {
	m_submission_queue.push(live_executor_detail::user_allocation_transfer{aid, ptr});
}

void live_executor::track_host_object_instance(const host_object_id hoid, std::unique_ptr<host_object_instance> instance) {
	assert(instance != nullptr);
	m_submission_queue.push(live_executor_detail::host_object_transfer{hoid, std::move(instance)});
}

void live_executor::track_reducer(const reduction_id rid, std::unique_ptr<reducer> reducer) {
	assert(reducer != nullptr);
	m_submission_queue.push(live_executor_detail::reducer_transfer{rid, std::move(reducer)});
}

void live_executor::submit(std::vector<const instruction*> instructions, std::vector<outbound_pilot> pilots) {
	m_submission_queue.push(live_executor_detail::instruction_pilot_batch{std::move(instructions), std::move(pilots)});
}

void live_executor::thread_main(std::unique_ptr<backend> backend, delegate* const dlg, const policy_set& policy) {
	try {
		live_executor_detail::executor_impl(std::move(backend), m_root_comm.get(), m_submission_queue, dlg, policy).run();
	}
	// LCOV_EXCL_START
	catch(const std::exception& e) {
		CELERITY_CRITICAL("[executor] {}", e.what());
		std::abort();
	}
	// LCOV_EXCL_STOP
}

} // namespace celerity::detail
