#include "dry_run_executor.h"
#include "host_object.h"
#include "instruction_graph.h"
#include "log.h"
#include "named_threads.h"


namespace celerity::detail {

dry_run_executor::dry_run_executor(delegate* const dlg) : m_thread(&dry_run_executor::thread_main, this, dlg) {
	set_thread_name(m_thread.native_handle(), "cy-executor");
}

dry_run_executor::~dry_run_executor() { m_thread.join(); }

void dry_run_executor::track_user_allocation(const allocation_id aid, void* const ptr) {
	(void)aid; // ignore
	(void)ptr; // ignore (allocation is owned by user)
}

void dry_run_executor::track_host_object_instance(const host_object_id hoid, std::unique_ptr<host_object_instance> instance) {
	m_submission_queue.push(std::pair{hoid, std::move(instance)});
}

void dry_run_executor::track_reducer(reduction_id rid, std::unique_ptr<reducer> reducer) {
	(void)rid;     // ignore
	(void)reducer; // destroy immediately
}

void dry_run_executor::submit(std::vector<const instruction*> instructions, std::vector<outbound_pilot> pilots) {
	m_submission_queue.push(std::move(instructions));
	(void)pilots; // ignore;
}

void dry_run_executor::thread_main(delegate* const dlg) {
	// For simplicity we keep all executor state within this function.
	std::unordered_map<host_object_id, std::unique_ptr<host_object_instance>> host_object_instances;
	bool shutdown = false;

	const auto issue = [&](const instruction* instr) {
		matchbox::match(
		    *instr, //
		    [&](const destroy_host_object_instruction& dhoinstr) {
			    assert(host_object_instances.count(dhoinstr.get_host_object_id()) != 0);
			    host_object_instances.erase(dhoinstr.get_host_object_id());
		    },
		    [&](const epoch_instruction& einstr) {
			    // task_manager doesn't expect us to actually execute the init epoch (TODO)
			    if(dlg != nullptr && einstr.get_epoch_task_id() != 0) { dlg->epoch_reached(einstr.get_epoch_task_id()); }
			    shutdown |= einstr.get_epoch_action() == epoch_action::shutdown;
		    },
		    [&](const fence_instruction& finstr) {
			    CELERITY_WARN("Encountered a \"fence\" command while \"CELERITY_DRY_RUN_NODES\" is set. "
			                  "The result of this operation will not match the expected output of an actual run.");
			    finstr.get_promise()->fulfill();
		    },
		    [&](const horizon_instruction& hinstr) { //
			    if(dlg != nullptr) { dlg->horizon_reached(hinstr.get_horizon_task_id()); }
		    },
		    [&](const auto& /* any other instr */) { /* ignore */ });
	};

	while(!shutdown) {
		m_submission_queue.wait_while_empty();

		for(auto& submission : m_submission_queue.pop_all()) {
			matchbox::match(
			    submission, //
			    [&](std::vector<const instruction*>& instrs) {
				    for(const auto instr : instrs) {
					    issue(instr);
				    }
			    },
			    [&](host_object_transfer& hot) { //
				    host_object_instances.insert(std::move(hot));
			    });
		}
	}

	assert(host_object_instances.empty());
}

} // namespace celerity::detail
