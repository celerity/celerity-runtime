#include "dry_run_executor.h"
#include "host_object.h"
#include "instruction_graph.h"
#include "log.h"
#include "task.h"


namespace celerity::detail {

void dry_run_executor::wait() {
	// wait() is called from application thread while submits happen from scheduler thread
	std::unique_lock lock(m_resume_mutex);
	m_resume.wait(lock, [&] { return m_has_shut_down; });
}

void dry_run_executor::submit_instruction(const instruction* instr) {
	matchbox::match(
	    *instr,                                                //
	    [&](const destroy_host_object_instruction& dhoinstr) { //
		    std::lock_guard lock(m_host_object_instances_mutex);
		    m_host_object_instances.erase(dhoinstr.get_host_object_id());
	    },
	    [&](const epoch_instruction& einstr) { //
		    if(m_delegate != nullptr && einstr.get_epoch_task_id() != 0 /* TODO tm doesn't expect us to actually execute the init epoch */) {
			    m_delegate->epoch_reached(einstr.get_epoch_task_id());
		    }
		    if(einstr.get_epoch_action() == epoch_action::shutdown) {
			    (std::lock_guard(m_resume_mutex), m_has_shut_down = true); // trust me I'm a software engineer
			    m_resume.notify_all();
		    }
	    },
	    [&](const fence_instruction& finstr) {
		    CELERITY_WARN("Encountered a \"fence\" command while \"CELERITY_DRY_RUN_NODES\" is set. "
		                  "The result of this operation will not match the expected output of an actual run.");
		    finstr.get_promise()->fulfill();
	    },
	    [&](const horizon_instruction& hinstr) { //
		    if(m_delegate != nullptr) { m_delegate->horizon_reached(hinstr.get_horizon_task_id()); }
	    },
	    [&](const auto& /* any other instr */) {
		    // ignore
	    });
}

void dry_run_executor::submit_pilot(const outbound_pilot& pilot) {
	(void)pilot; // ignore
}

void dry_run_executor::announce_user_allocation(allocation_id aid, void* ptr) {
	(void)aid, (void)ptr; // ignore
}

void dry_run_executor::announce_host_object_instance(host_object_id hoid, std::unique_ptr<host_object_instance> instance) {
	std::lock_guard lock(m_host_object_instances_mutex);
	assert(m_host_object_instances.count(hoid) == 0);
	m_host_object_instances.emplace(hoid, std::move(instance));
}

void dry_run_executor::announce_reduction(reduction_id rid, std::unique_ptr<runtime_reduction> reduction) {
	(void)rid, (void)reduction; // destroy reduction object immediately
}

} // namespace celerity::detail
