#include "instruction_executor.h"
#include "closure_hydrator.h"
#include "communicator.h"
#include "host_object.h"
#include "instruction_graph.h"
#include "mpi_communicator.h" // TODO
#include "nd_memory.h"
#include "receive_arbiter.h"
#include "tracy.h"
#include "types.h"

#include <queue>

#include <matchbox.hh>

namespace celerity::detail {

// TODO --v move into PIMPL?

struct instruction_executor::pending_instruction_info {
	size_t n_unmet_dependencies;
};

struct instruction_executor::active_instruction_info {
	async_event operation;
	CELERITY_DETAIL_TRACY_DECLARE_ASYNC_LANE(tracy_lane)
};

struct instruction_executor::incomplete_instruction_info {
	// we need to track successors ourselves, because the intrusive_graph dependents are still being updated in the scheduler thread after we start processing
	// the instruction, so calling get_dependents() would cause a data race
	std::vector<const instruction*> dependents;
};

instruction_executor::instruction_executor(std::unique_ptr<backend::queue> backend_queue, std::unique_ptr<communicator> comm, delegate* dlg)
    : m_delegate(dlg), m_communicator(std::move(comm)), m_backend_queue(std::move(backend_queue)), m_recv_arbiter(*m_communicator),
      m_thread(&instruction_executor::thread_main, this), m_alloc_pool(4) {
	set_thread_name(m_thread.native_handle(), "cy-executor");
}

instruction_executor::~instruction_executor() { wait(); }

void instruction_executor::wait() {
	if(m_thread.joinable()) { m_thread.join(); }
}

void instruction_executor::submit_instruction(const instruction* instr) { m_submission_queue.push_back(instr); }

void instruction_executor::submit_pilot(const outbound_pilot& pilot) { m_submission_queue.push_back(pilot); }

void instruction_executor::announce_user_allocation(const allocation_id aid, void* const ptr) {
	m_submission_queue.push_back(user_allocation_announcement{aid, ptr});
}

void instruction_executor::announce_host_object_instance(const host_object_id hoid, std::unique_ptr<host_object_instance> instance) {
	assert(instance != nullptr);
	m_submission_queue.push_back(host_object_instance_announcement{hoid, std::move(instance)});
}

void instruction_executor::announce_reduction(const reduction_id rid, std::unique_ptr<runtime_reduction> reduction) {
	assert(reduction != nullptr);
	m_submission_queue.push_back(reduction_announcement{rid, std::move(reduction)});
}

void instruction_executor::thread_main() {
	CELERITY_DETAIL_TRACY_SET_CURRENT_THREAD_NAME("cy-executor");
	try {
		loop();
	} catch(const std::exception& e) {
		CELERITY_CRITICAL("[executor] {}", e.what());
		std::abort();
	}
}

struct instruction_priority_less {
	bool operator()(const instruction* lhs, const instruction* rhs) const { return lhs->get_priority() < rhs->get_priority(); }
};

// TODO dupe of host_queue::future_event
template <typename Result = void>
class future_event final : public async_event_base {
  public:
	future_event(std::future<Result> future) : m_future(std::move(future)) {}

	bool is_complete() const override { return m_future.wait_for(std::chrono::seconds(0)) == std::future_status::ready; }

	std::any take_result() override {
		if constexpr(!std::is_void_v<Result>) { return m_future.get(); }
		return std::any();
	}

  private:
	std::future<Result> m_future;
};


void instruction_executor::loop() {
	m_backend_queue->init();

	closure_hydrator::make_available();

	m_allocations.emplace(null_allocation_id, nullptr);
	m_collective_groups.emplace(root_collective_group_id, m_communicator->get_collective_root());
	m_host_queue.require_collective_group(root_collective_group_id);

	std::vector<submission> loop_submission_queue;
	std::unordered_map<const instruction*, pending_instruction_info> pending_instructions; // TODO chould be a vector?
	std::priority_queue<const instruction*, std::vector<const instruction*>, instruction_priority_less> ready_instructions;
	std::unordered_map<const instruction*, active_instruction_info> active_instructions; // TODO why is this a map and not a vector?
	std::unordered_map<instruction_id, incomplete_instruction_info> incomplete_instructions;
	std::optional<std::chrono::steady_clock::time_point> last_progress_timestamp;
	bool progress_warning_emitted = false;
	while(m_expecting_more_submissions || !incomplete_instructions.empty()) {
		m_recv_arbiter.poll_communicator();

		bool made_progress = false;

		for(auto active_it = active_instructions.begin(); active_it != active_instructions.end();) {
			auto& [active_instr, active_info] = *active_it;
			if(active_info.operation.is_complete()) {
#if CELERITY_ENABLE_TRACY
				if(active_info.tracy_lane) {
					CELERITY_DETAIL_TRACY_ASYNC_ZONE_RESUME(active_info.tracy_lane);
					const auto bytes_processed = matchbox::match(
					    *active_instr, //
					    [](const alloc_instruction& ainstr) { return ainstr.get_size_bytes(); },
					    [](const copy_instruction& cinstr) { return cinstr.get_copy_region().get_area() * cinstr.get_element_size(); },
					    [](const send_instruction& sinstr) { return sinstr.get_send_range().size() * sinstr.get_element_size(); },
					    [](const auto& /* other */) { return 0; });
					if(bytes_processed > 0) {
						const auto seconds = CELERITY_DETAIL_TRACY_ASYNC_ELAPSED_TIME_SECONDS(active_info.tracy_lane);
						CELERITY_DETAIL_TRACY_ASYNC_ZONE_TEXT(active_info.tracy_lane, "throughput: {:.2f} MB/s", bytes_processed / (1024.0 * 1024.0 * seconds));
					}
					CELERITY_DETAIL_TRACY_ASYNC_ZONE_END(active_info.tracy_lane);
				}
#endif

				CELERITY_DEBUG("[executor] completed I{}", active_instr->get_id());

				const auto incomplete_it = incomplete_instructions.find(active_instr->get_id());
				assert(incomplete_it != incomplete_instructions.end());
				for(const auto successor : incomplete_it->second.dependents) {
					if(const auto pending_it = pending_instructions.find(successor); pending_it != pending_instructions.end()) {
						auto& [pending_instr, pending_info] = *pending_it;
						assert(pending_info.n_unmet_dependencies > 0);
						pending_info.n_unmet_dependencies -= 1;
						if(pending_info.n_unmet_dependencies == 0) {
							ready_instructions.push(pending_instr);
							pending_instructions.erase(pending_it);
						}
					}
				}
				made_progress = true;
				incomplete_instructions.erase(active_instr->get_id());
				active_it = active_instructions.erase(active_it);
			} else {
				++active_it;
			}
		}

		if(m_submission_queue.swap_if_nonempty(loop_submission_queue)) {
			CELERITY_DETAIL_TRACY_SCOPED_ZONE("executor_dequeue", Gray, "process submissions");
			for(auto& submission : loop_submission_queue) {
				matchbox::match(
				    submission,
				    [&](const instruction* incoming_instr) {
					    incomplete_instruction_info incomplete_info;
					    size_t n_unmet_dependencies = 0;
					    for(const auto dep : incoming_instr->get_dependencies()) {
						    const auto predecessor = incomplete_instructions.find(dep);
						    if(predecessor != incomplete_instructions.end()) {
							    predecessor->second.dependents.push_back(incoming_instr);
							    ++n_unmet_dependencies;
						    }
					    }
					    if(n_unmet_dependencies > 0) {
						    pending_instructions.emplace(incoming_instr, pending_instruction_info{n_unmet_dependencies});
					    } else {
						    ready_instructions.push(incoming_instr);
					    }
					    incomplete_instructions.emplace(incoming_instr->get_id(), incomplete_instruction_info{});
				    },
				    [&](const outbound_pilot& pilot) { //
					    m_communicator->send_outbound_pilot(pilot);
				    },
				    [&](const user_allocation_announcement& ann) {
					    assert(ann.aid != null_allocation_id);
					    assert(ann.aid.get_memory_id() == user_memory_id);
					    assert(m_allocations.count(ann.aid) == 0);
					    m_allocations.emplace(ann.aid, ann.ptr);
				    },
				    [&](host_object_instance_announcement& ann) {
					    assert(m_host_object_instances.count(ann.hoid) == 0);
					    m_host_object_instances.emplace(ann.hoid, std::move(ann.instance));
				    },
				    [&](reduction_announcement& ann) {
					    assert(m_reductions.count(ann.rid) == 0);
					    m_reductions.emplace(ann.rid, std::move(ann.reduction));
				    });
			}
			loop_submission_queue.clear();
		}

		if(!ready_instructions.empty()) {
			const auto ready_instr = ready_instructions.top();
			ready_instructions.pop();
			active_instructions.emplace(ready_instr, begin_executing(*ready_instr));
			made_progress = true;
		}

		// TODO consider rate-limiting this (e.g. with an overflow counter) if steady_clock::now() turns out to have measurable latency
		if(made_progress) {
			last_progress_timestamp = std::chrono::steady_clock::now();
			progress_warning_emitted = false;
		} else if(last_progress_timestamp.has_value()) {
			const auto assume_stuck_after = std::chrono::seconds(3);
			const auto elapsed_since_last_progress = std::chrono::steady_clock::now() - *last_progress_timestamp;
			if(elapsed_since_last_progress > assume_stuck_after && !progress_warning_emitted) {
				std::string instr_list;
				for(auto& [instr, _] : active_instructions) {
					if(!instr_list.empty()) instr_list += ", ";
					fmt::format_to(std::back_inserter(instr_list), "I{}", instr->get_id());
				}
				CELERITY_WARN("[executor] no progress for {:.3f} seconds, potentially stuck. Active instructions: {}",
				    std::chrono::duration_cast<std::chrono::duration<double>>(elapsed_since_last_progress).count(),
				    active_instructions.empty() ? "none" : instr_list);
				progress_warning_emitted = true;
			}
		}
	}

	assert(std::all_of(m_allocations.begin(), m_allocations.end(),
	    [](const std::pair<allocation_id, void*>& p) { return p.first == null_allocation_id || p.first.get_memory_id() == user_memory_id; }));
	assert(m_host_object_instances.empty());
}

void instruction_executor::collect(const instruction_garbage& garbage) {
	for(const auto rid : garbage.reductions) {
		assert(m_reductions.count(rid) != 0);
		m_reductions.erase(rid);
	}
	for(const auto aid : garbage.user_allocations) {
		assert(aid.get_memory_id() == user_memory_id);
		assert(m_allocations.count(aid) != 0);
		m_allocations.erase(aid);
	}
}

void instruction_executor::prepare_accessor_hydration(
    target target, const buffer_access_allocation_map& amap CELERITY_DETAIL_IF_ACCESSOR_BOUNDARY_CHECK(, const boundary_check_info& oob_info)) {
	std::vector<closure_hydrator::accessor_info> accessor_infos;
	accessor_infos.reserve(amap.size());
	for(size_t i = 0; i < amap.size(); ++i) {
		const auto ptr = m_allocations.at(amap[i].allocation_id);
		accessor_infos.push_back(closure_hydrator::accessor_info{ptr, amap[i].allocated_box_in_buffer,
		    amap[i].accessed_box_in_buffer CELERITY_DETAIL_IF_ACCESSOR_BOUNDARY_CHECK(, &oob_info.illegal_access_bounding_boxes[i])});
	}

	closure_hydrator::get_instance().arm(target, std::move(accessor_infos));
}

#if CELERITY_ACCESSOR_BOUNDARY_CHECK

instruction_executor::boundary_check_info instruction_executor::prepare_accessor_boundary_check(
    const buffer_access_allocation_map& amap, const task_id tid, const std::string& task_name, const target target) {
	boundary_check_info info;
	if(!amap.empty()) {
		info.illegal_access_bounding_boxes =
		    static_cast<oob_bounding_box*>(m_backend_queue->alloc(host_memory_id, amap.size() * sizeof(oob_bounding_box), alignof(oob_bounding_box)));
		std::uninitialized_default_construct_n(info.illegal_access_bounding_boxes, amap.size());
	}
	for(size_t i = 0; i < amap.size(); ++i) {
		info.accessors.push_back({amap[i].oob_buffer_id, amap[i].oob_buffer_name, amap[i].accessed_box_in_buffer});
	}
	info.task_id = tid;
	info.task_name = task_name;
	info.target = target;
	return info;
}

bool instruction_executor::boundary_checked_event::is_complete() const {
	if(!m_state.has_value()) return true; // we clear `state` completion to make is_complete() idempotent
	if(!m_state->launch_event.is_complete()) return false;

	const auto& info = m_state->oob_info;
	for(size_t i = 0; i < info.accessors.size(); ++i) {
		if(const auto oob_box = info.illegal_access_bounding_boxes[i].into_box(); !oob_box.empty()) {
			const auto& accessor_info = info.accessors[i];
			CELERITY_ERROR(
			    "Out-of-bounds access detected in {} T{}{}: accessor {} attempted to access buffer {} indicies between {} and outside the declared range {}.",
			    info.target == target::device ? "device kernel" : "host task", info.task_id,
			    (!info.task_name.empty() ? fmt::format(" \"{}\"", info.task_name) : ""), i,
			    utils::make_buffer_debug_label(accessor_info.buffer_id, accessor_info.buffer_name), oob_box, accessor_info.accessible_box);
		}
	}

	if(info.illegal_access_bounding_boxes != nullptr /* i.e. there is at least one accessor */) {
		m_state->executor->m_backend_queue->free(host_memory_id, info.illegal_access_bounding_boxes);
	}

	m_state = {}; // make is_complete() idempotent
	return true;
}

#endif // CELERITY_ACCESSOR_BOUNDARY_CHECK

instruction_executor::active_instruction_info instruction_executor::begin_executing(const instruction& instr) {
	static constexpr auto log_accesses = [](const buffer_access_allocation_map& map) {
		std::string acc_log;
		for(size_t i = 0; i < map.size(); ++i) {
			auto& aa = map[i];
			const auto accessed_box_in_allocation = box(aa.accessed_box_in_buffer.get_min() - aa.allocated_box_in_buffer.get_offset(),
			    aa.accessed_box_in_buffer.get_max() - aa.allocated_box_in_buffer.get_offset());
			fmt::format_to(std::back_inserter(acc_log), "{} {} {}", i == 0 ? ", accessing" : ",", aa.allocation_id, accessed_box_in_allocation);
		}
		return acc_log;
	};

#if CELERITY_ENABLE_TRACY
	std::string tracy_dependency_list;
	for(const auto dep : instr.get_dependencies()) {
		if(!tracy_dependency_list.empty()) tracy_dependency_list += ", ";
		fmt::format_to(std::back_inserter(tracy_dependency_list), "I{}", dep);
	}
#endif

#define CELERITY_DETAIL_TRACY_ZONE_METADATA() CELERITY_DETAIL_TRACY_ZONE_TEXT("depends: {}\npriority: {}", tracy_dependency_list, instr.get_priority())
#define CELERITY_DETAIL_TRACY_ASYNC_ZONE_METADATA()                                                                                                            \
	CELERITY_DETAIL_TRACY_ASYNC_ZONE_TEXT(active_instruction.tracy_lane, "depends: {}\npriority: {}", tracy_dependency_list, instr.get_priority())

	active_instruction_info active_instruction;
	active_instruction.operation = matchbox::match<async_event>(
	    instr,
	    [&](const clone_collective_group_instruction& ccginstr) {
		    const auto new_cgid = ccginstr.get_new_collective_group_id();
		    const auto origin_cgid = ccginstr.get_original_collective_group_id();

		    CELERITY_DEBUG("[executor] I{}: clone collective group CG{} -> CG{}", ccginstr.get_id(), origin_cgid, new_cgid);
		    CELERITY_DETAIL_TRACY_SCOPED_ZONE("executor::clone_collective_group", Brown, "I{} clone collective", ccginstr.get_id());

		    assert(m_collective_groups.count(new_cgid) == 0);
		    const auto new_group = m_collective_groups.at(origin_cgid)->clone();
		    m_collective_groups.emplace(new_cgid, new_group);
		    m_host_queue.require_collective_group(new_cgid);
		    return make_complete_event();
	    },
	    [&](const alloc_instruction& ainstr) {
		    CELERITY_DEBUG(
		        "[executor] I{}: alloc {}, {}%{} bytes", ainstr.get_id(), ainstr.get_allocation_id(), ainstr.get_size_bytes(), ainstr.get_alignment_bytes());

		    void* ptr;
		    {
			    CELERITY_DETAIL_TRACY_SCOPED_ZONE("executor::alloc", Turquoise, "I{} alloc", ainstr.get_id());
			    CELERITY_DETAIL_TRACY_ZONE_METADATA()
			    CELERITY_DETAIL_TRACY_ZONE_TEXT("alloc {}, {}%{} bytes", ainstr.get_allocation_id(), ainstr.get_size_bytes(), ainstr.get_alignment_bytes());
			    ptr = m_backend_queue->alloc(ainstr.get_allocation_id().get_memory_id(), ainstr.get_size_bytes(), ainstr.get_alignment_bytes());
		    }

		    CELERITY_DEBUG("[executor] {} allocated as {}", ainstr.get_allocation_id(), ptr);
		    m_allocations.emplace(ainstr.get_allocation_id(), ptr);
		    return make_complete_event();
	    },
	    [&](const free_instruction& finstr) {
		    const auto it = m_allocations.find(finstr.get_allocation_id());
		    assert(it != m_allocations.end());
		    const auto ptr = it->second;
		    m_allocations.erase(it);

		    CELERITY_DEBUG("[executor] I{}: free {}", finstr.get_id(), finstr.get_allocation_id());
		    CELERITY_DETAIL_TRACY_SCOPED_ZONE("executor::free", Turquoise, "I{} free", finstr.get_id());
		    CELERITY_DETAIL_TRACY_ZONE_METADATA()
		    CELERITY_DETAIL_TRACY_ZONE_TEXT("free {}", finstr.get_allocation_id());

		    m_backend_queue->free(finstr.get_allocation_id().get_memory_id(), ptr);
		    return make_complete_event();
	    },
	    [&](const copy_instruction& cinstr) {
		    CELERITY_DEBUG("[executor] I{}: copy {} ({}) -> {} ({}), {} x{} bytes", cinstr.get_id(), cinstr.get_source_allocation(), cinstr.get_source_box(),
		        cinstr.get_dest_allocation(), cinstr.get_dest_box(), cinstr.get_copy_region(), cinstr.get_element_size());

		    const auto source_mid = cinstr.get_source_allocation().id.get_memory_id();
		    const auto dest_mid = cinstr.get_dest_allocation().id.get_memory_id();
		    const auto source_base =
		        static_cast<const std::byte*>(m_allocations.at(cinstr.get_source_allocation().id)) + cinstr.get_source_allocation().offset_bytes;
		    const auto dest_base = static_cast<std::byte*>(m_allocations.at(cinstr.get_dest_allocation().id)) + cinstr.get_dest_allocation().offset_bytes;
		    if((source_mid == user_memory_id || source_mid == host_memory_id) && (dest_mid == user_memory_id || dest_mid == host_memory_id)) {
			    // TODO into thread pool
			    CELERITY_DETAIL_TRACY_SCOPED_ZONE("executor::copy", Lime, "I{} copy", cinstr.get_id());
			    CELERITY_DETAIL_TRACY_ZONE_METADATA()
			    CELERITY_DETAIL_TRACY_ZONE_TEXT("copy {} -> {}, {} x{} bytes\n{} bytes total", cinstr.get_source_allocation(), cinstr.get_dest_allocation(),
			        cinstr.get_copy_region(), cinstr.get_element_size(), cinstr.get_copy_region().get_area() * cinstr.get_element_size());
			    copy_region_host(source_base, dest_base, cinstr.get_source_box(), cinstr.get_dest_box(), cinstr.get_copy_region(), cinstr.get_element_size());
			    return make_complete_event();
		    } else {
			    assert(source_mid != user_memory_id && dest_mid != user_memory_id);
			    CELERITY_DETAIL_TRACY_ASYNC_ZONE_BEGIN_SCOPED(
			        active_instruction.tracy_lane, "cy-executor", "executor::copy", Lime, "I{} copy", cinstr.get_id());
			    CELERITY_DETAIL_TRACY_ASYNC_ZONE_METADATA()
			    CELERITY_DETAIL_TRACY_ASYNC_ZONE_TEXT(active_instruction.tracy_lane, "copy {} -> {}, {} x{} bytes\n{} bytes total",
			        cinstr.get_source_allocation(), cinstr.get_dest_allocation(), cinstr.get_copy_region(), cinstr.get_element_size(),
			        cinstr.get_copy_region().get_area() * cinstr.get_element_size());
			    return m_backend_queue->copy_region(source_mid, dest_mid, source_base, dest_base, cinstr.get_source_box(), cinstr.get_dest_box(),
			        cinstr.get_copy_region(), cinstr.get_element_size());
		    }
	    },
	    [&](const device_kernel_instruction& dkinstr) {
		    CELERITY_DEBUG("[executor] I{}: launch device kernel on D{}, {}{}", dkinstr.get_id(), dkinstr.get_device_id(), dkinstr.get_execution_range(),
		        log_accesses(dkinstr.get_access_allocations()));
		    CELERITY_DETAIL_TRACY_ASYNC_ZONE_BEGIN_SCOPED(
		        active_instruction.tracy_lane, "cy-executor", "executor::device_kernel", Orange, "I{} device kernel", dkinstr.get_id());
		    CELERITY_DETAIL_TRACY_ASYNC_ZONE_METADATA()
		    CELERITY_DETAIL_TRACY_ASYNC_ZONE_TEXT(active_instruction.tracy_lane, "kernel on D{}", dkinstr.get_device_id());

#if CELERITY_ACCESSOR_BOUNDARY_CHECK
		    auto oob_info =
		        prepare_accessor_boundary_check(dkinstr.get_access_allocations(), dkinstr.get_oob_task_id(), dkinstr.get_oob_task_name(), target::device);
#endif
		    prepare_accessor_hydration(target::device, dkinstr.get_access_allocations() CELERITY_DETAIL_IF_ACCESSOR_BOUNDARY_CHECK(, oob_info));

		    std::vector<void*> reduction_ptrs;
		    reduction_ptrs.reserve(dkinstr.get_reduction_allocations().size());
		    for(const auto& ra : dkinstr.get_reduction_allocations()) {
			    reduction_ptrs.push_back(m_allocations.at(ra.allocation_id));
		    }

		    auto evt = m_backend_queue->launch_kernel(dkinstr.get_device_id(), dkinstr.get_launcher(), dkinstr.get_execution_range(), reduction_ptrs);

#if CELERITY_ACCESSOR_BOUNDARY_CHECK
		    return make_async_event<boundary_checked_event>(this, std::move(evt), std::move(oob_info));
#else
		    return evt;
#endif
	    },
	    [&](const host_task_instruction& htinstr) {
		    CELERITY_DEBUG(
		        "[executor] I{}: launch host task, {}{}", htinstr.get_id(), htinstr.get_execution_range(), log_accesses(htinstr.get_access_allocations()));
		    CELERITY_DETAIL_TRACY_ASYNC_ZONE_BEGIN_SCOPED(
		        active_instruction.tracy_lane, "cy-executor", "executor::host_task", Orange, "I{} host task", htinstr.get_id());
		    CELERITY_DETAIL_TRACY_ASYNC_ZONE_METADATA()
		    CELERITY_DETAIL_TRACY_ASYNC_ZONE_TEXT(active_instruction.tracy_lane, "host task");

#if CELERITY_ACCESSOR_BOUNDARY_CHECK
		    auto oob_info =
		        prepare_accessor_boundary_check(htinstr.get_access_allocations(), htinstr.get_oob_task_id(), htinstr.get_oob_task_name(), target::host_task);
#endif
		    prepare_accessor_hydration(target::host_task, htinstr.get_access_allocations() CELERITY_DETAIL_IF_ACCESSOR_BOUNDARY_CHECK(, oob_info));

		    // TODO executor must not have any direct dependency on MPI!
		    MPI_Comm mpi_comm = MPI_COMM_NULL;
		    if(const auto cgid = htinstr.get_collective_group_id(); cgid != non_collective_group_id) {
			    const auto cg = m_collective_groups.at(htinstr.get_collective_group_id());
			    mpi_comm = dynamic_cast<mpi_communicator::collective_group&>(*cg).get_mpi_comm();
		    }

		    const auto& launch = htinstr.get_launcher();
		    auto evt = launch(m_host_queue, htinstr.get_execution_range(), mpi_comm);

#if CELERITY_ACCESSOR_BOUNDARY_CHECK
		    return make_async_event<boundary_checked_event>(this, std::move(evt), std::move(oob_info));
#else
		    return evt;
#endif
	    },
	    [&](const send_instruction& sinstr) {
		    CELERITY_DEBUG("[executor] I{}: send {}+{}, {}x{} bytes to N{} (MSG{})", sinstr.get_id(), sinstr.get_source_allocation_id(),
		        sinstr.get_offset_in_source_allocation(), sinstr.get_send_range(), sinstr.get_element_size(), sinstr.get_dest_node_id(),
		        sinstr.get_message_id());
		    CELERITY_DETAIL_TRACY_ASYNC_ZONE_BEGIN_SCOPED(active_instruction.tracy_lane, "cy-executor", "executor::send", Violet, "I{} send", sinstr.get_id());
		    CELERITY_DETAIL_TRACY_ASYNC_ZONE_METADATA()
		    CELERITY_DETAIL_TRACY_ASYNC_ZONE_TEXT(active_instruction.tracy_lane, "send {}+{}, {}x{} bytes to N{}\n{} bytes total",
		        sinstr.get_source_allocation_id(), sinstr.get_offset_in_source_allocation(), sinstr.get_send_range(), sinstr.get_element_size(),
		        sinstr.get_dest_node_id(), sinstr.get_send_range() * sinstr.get_element_size());

		    const auto allocation_base = m_allocations.at(sinstr.get_source_allocation_id());
		    const communicator::stride stride{
		        sinstr.get_source_allocation_range(),
		        subrange<3>{sinstr.get_offset_in_source_allocation(), sinstr.get_send_range()},
		        sinstr.get_element_size(),
		    };
		    return m_communicator->send_payload(sinstr.get_dest_node_id(), sinstr.get_message_id(), allocation_base, stride);
	    },
	    [&](const receive_instruction& rinstr) {
		    CELERITY_DEBUG("[executor] I{}: receive {} {} into {} ({}), x{} bytes\n{} bytes total", rinstr.get_id(), rinstr.get_transfer_id(),
		        rinstr.get_requested_region(), rinstr.get_dest_allocation_id(), rinstr.get_allocated_box(), rinstr.get_element_size(),
		        rinstr.get_requested_region().get_area() * rinstr.get_element_size());
		    CELERITY_DETAIL_TRACY_ASYNC_ZONE_BEGIN_SCOPED(
		        active_instruction.tracy_lane, "cy-executor", "executor::receive", DarkViolet, "I{} receive", rinstr.get_id());
		    CELERITY_DETAIL_TRACY_ASYNC_ZONE_METADATA()
		    CELERITY_DETAIL_TRACY_ASYNC_ZONE_TEXT(active_instruction.tracy_lane, "receive {} {} into {} ({}), x{} bytes", rinstr.get_transfer_id(),
		        rinstr.get_requested_region(), rinstr.get_dest_allocation_id(), rinstr.get_allocated_box(), rinstr.get_element_size());

		    const auto allocation = m_allocations.at(rinstr.get_dest_allocation_id());
		    return m_recv_arbiter.receive(
		        rinstr.get_transfer_id(), rinstr.get_requested_region(), allocation, rinstr.get_allocated_box(), rinstr.get_element_size());
	    },
	    [&](const split_receive_instruction& srinstr) {
		    CELERITY_DEBUG("[executor] I{}: split receive {} {} into {} ({}), x{} bytes\n{} bytes total", srinstr.get_id(), srinstr.get_transfer_id(),
		        srinstr.get_requested_region(), srinstr.get_dest_allocation_id(), srinstr.get_allocated_box(), srinstr.get_element_size(),
		        srinstr.get_requested_region().get_area() * srinstr.get_element_size());
		    CELERITY_DETAIL_TRACY_ASYNC_ZONE_BEGIN_SCOPED(
		        active_instruction.tracy_lane, "cy-executor", "executor::split_receive", DarkViolet, "I{} split receive", srinstr.get_id());
		    CELERITY_DETAIL_TRACY_ASYNC_ZONE_METADATA()
		    CELERITY_DETAIL_TRACY_ASYNC_ZONE_TEXT(active_instruction.tracy_lane, "split receive {} {} into {} ({}), x{} bytes", srinstr.get_transfer_id(),
		        srinstr.get_requested_region(), srinstr.get_dest_allocation_id(), srinstr.get_allocated_box(), srinstr.get_element_size());

		    const auto allocation = m_allocations.at(srinstr.get_dest_allocation_id());
		    m_recv_arbiter.begin_split_receive(
		        srinstr.get_transfer_id(), srinstr.get_requested_region(), allocation, srinstr.get_allocated_box(), srinstr.get_element_size());
		    return make_complete_event();
	    },
	    [&](const await_receive_instruction& arinstr) {
		    CELERITY_DEBUG("[executor] I{}: await receive {} {}", arinstr.get_id(), arinstr.get_transfer_id(), arinstr.get_received_region());
		    CELERITY_DETAIL_TRACY_ASYNC_ZONE_BEGIN_SCOPED(
		        active_instruction.tracy_lane, "cy-executor", "executor::await_receive", DarkViolet, "I{} await receive", arinstr.get_id());
		    CELERITY_DETAIL_TRACY_ASYNC_ZONE_METADATA()
		    CELERITY_DETAIL_TRACY_ASYNC_ZONE_TEXT(
		        active_instruction.tracy_lane, "await receive {} {}", arinstr.get_transfer_id(), arinstr.get_received_region());

		    return m_recv_arbiter.await_split_receive_subregion(arinstr.get_transfer_id(), arinstr.get_received_region());
	    },
	    [&](const gather_receive_instruction& grinstr) {
		    CELERITY_DEBUG("[executor] I{}: gather receive {} into {}, {} bytes per node", grinstr.get_id(), grinstr.get_transfer_id(),
		        grinstr.get_dest_allocation_id(), grinstr.get_node_chunk_size());
		    CELERITY_DETAIL_TRACY_ASYNC_ZONE_BEGIN_SCOPED(
		        active_instruction.tracy_lane, "cy-executor", "executor::gather_receive", DarkViolet, "I{} gather receive", grinstr.get_id());
		    CELERITY_DETAIL_TRACY_ASYNC_ZONE_METADATA()
		    CELERITY_DETAIL_TRACY_ASYNC_ZONE_TEXT(active_instruction.tracy_lane, "gather receive {} into {}, {} bytes per node", grinstr.get_transfer_id(),
		        grinstr.get_dest_allocation_id(), grinstr.get_node_chunk_size());

		    const auto allocation = m_allocations.at(grinstr.get_dest_allocation_id());
		    return m_recv_arbiter.gather_receive(grinstr.get_transfer_id(), allocation, grinstr.get_node_chunk_size());
	    },
	    [&](const fill_identity_instruction& fiinstr) {
		    CELERITY_DEBUG("[executor] I{}: fill identity {} x{} for R{}", fiinstr.get_id(), fiinstr.get_allocation_id(), fiinstr.get_num_values(),
		        fiinstr.get_reduction_id());
		    CELERITY_DETAIL_TRACY_SCOPED_ZONE("executor::fill_identity", Blue, "I{} fill identity", fiinstr.get_id());
		    CELERITY_DETAIL_TRACY_ZONE_METADATA()
		    CELERITY_DETAIL_TRACY_ZONE_TEXT("fill identity {} x{} for R{}", fiinstr.get_allocation_id(), fiinstr.get_num_values(), fiinstr.get_reduction_id());

		    const auto allocation = m_allocations.at(fiinstr.get_allocation_id());
		    const auto& reduction = *m_reductions.at(fiinstr.get_reduction_id());
		    reduction.fill_identity(allocation, fiinstr.get_num_values());
		    return make_complete_event();
	    },
	    [&](const reduce_instruction& rinstr) {
		    CELERITY_DEBUG("[executor] I{}: reduce {} x{} into {} as R{}", rinstr.get_id(), rinstr.get_source_allocation_id(), rinstr.get_num_source_values(),
		        rinstr.get_dest_allocation_id(), rinstr.get_reduction_id());
		    CELERITY_DETAIL_TRACY_SCOPED_ZONE("executor::reduce", Blue, "I{} reduce", rinstr.get_id());
		    CELERITY_DETAIL_TRACY_ZONE_METADATA()
		    CELERITY_DETAIL_TRACY_ZONE_TEXT("reduce {} x{} into {} as R{}", rinstr.get_source_allocation_id(), rinstr.get_num_source_values(),
		        rinstr.get_dest_allocation_id(), rinstr.get_reduction_id());

		    const auto gather_allocation = m_allocations.at(rinstr.get_source_allocation_id());
		    const auto dest_allocation = m_allocations.at(rinstr.get_dest_allocation_id());
		    const bool include_dest = false; // TODO
		    const auto& reduction = *m_reductions.at(rinstr.get_reduction_id());
		    reduction.reduce(dest_allocation, gather_allocation, rinstr.get_num_source_values(), include_dest);
		    // TODO GC runtime_reduction at some point
		    return make_complete_event();
	    },
	    [&](const fence_instruction& finstr) {
		    CELERITY_DEBUG("[executor] I{}: fence", finstr.get_id());
		    CELERITY_DETAIL_TRACY_SCOPED_ZONE("executor::fence", Blue, "fence");
		    CELERITY_DETAIL_TRACY_ZONE_METADATA()

		    finstr.get_promise()->fulfill();
		    return make_complete_event();
	    },
	    [&](const destroy_host_object_instruction& dhoinstr) {
		    assert(m_host_object_instances.count(dhoinstr.get_host_object_id()) != 0);
		    CELERITY_DEBUG("[executor] I{}: destroy H{}", dhoinstr.get_id(), dhoinstr.get_host_object_id());
		    CELERITY_DETAIL_TRACY_SCOPED_ZONE("executor::destroy_host_object", Gray, "destroy host object");
		    CELERITY_DETAIL_TRACY_ZONE_METADATA()
		    CELERITY_DETAIL_TRACY_ZONE_TEXT("destroy H{}", dhoinstr.get_host_object_id());

		    m_host_object_instances.erase(dhoinstr.get_host_object_id());
		    return make_complete_event();
	    },
	    [&](const horizon_instruction& hinstr) {
		    CELERITY_DEBUG("[executor] I{}: horizon", hinstr.get_id());
		    CELERITY_DETAIL_TRACY_SCOPED_ZONE("executor::horizon", Gray, "horizon");
		    CELERITY_DETAIL_TRACY_ZONE_METADATA()

		    if(m_delegate != nullptr) { m_delegate->horizon_reached(hinstr.get_horizon_task_id()); }
		    collect(hinstr.get_garbage());
		    return make_complete_event();
	    },
	    [&](const epoch_instruction& einstr) {
		    CELERITY_DETAIL_TRACY_SCOPED_ZONE("executor::epoch", Gray, "epoch");
		    CELERITY_DETAIL_TRACY_ZONE_METADATA()

		    switch(einstr.get_epoch_action()) {
		    case epoch_action::none: CELERITY_DEBUG("[executor] I{}: epoch", einstr.get_id()); break;
		    case epoch_action::barrier:
			    CELERITY_DEBUG("[executor] I{}: epoch (barrier)", einstr.get_id());
			    m_communicator->get_collective_root()->barrier();
			    break;
		    case epoch_action::shutdown:
			    CELERITY_DEBUG("[executor] I{}: epoch (shutdown)", einstr.get_id());
			    m_expecting_more_submissions = false;
			    break;
		    }
		    if(m_delegate != nullptr && einstr.get_epoch_task_id() != 0 /* TODO tm doesn't expect us to actually execute the init epoch */) {
			    m_delegate->epoch_reached(einstr.get_epoch_task_id());
		    }
		    collect(einstr.get_garbage());
		    return make_complete_event();
	    });
	return active_instruction;
}

} // namespace celerity::detail
