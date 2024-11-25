#pragma once

#include "scheduler.h"
#include "scheduler_testspy.h"

#include <cstddef>

// This file is tail-included by scheduler.cc; also make it parsable as a standalone file for clangd and clang-tidy
#ifndef CELERITY_DETAIL_TAIL_INCLUDE
#include "../scheduler.cc" // NOLINT(bugprone-suspicious-include)
#endif

namespace celerity::detail {

scheduler scheduler_testspy::make_threadless_scheduler(size_t num_nodes, node_id local_node_id, const system_info& system_info, scheduler::delegate* delegate,
    command_recorder* crec, instruction_recorder* irec, const scheduler::policy_set& policy) //
{
	scheduler schdlr; // default-constructible by testspy, keeps m_impl == nullptr
	schdlr.m_impl = std::make_unique<scheduler_impl>(false /* start_thread */, num_nodes, local_node_id, system_info, delegate, crec, irec, policy);
	return schdlr;
}

void scheduler_testspy::run_scheduling_loop(scheduler& schdlr) { schdlr.m_impl->scheduling_loop(); }

void scheduler_testspy::begin_inspect_thread(scheduler& schdlr, event_inspect inspect) { schdlr.m_impl->task_queue.push(std::move(inspect)); }

} // namespace celerity::detail
