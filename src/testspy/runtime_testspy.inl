#pragma once

#include "print_graph.h"
#include "runtime.h"
#include "runtime_testspy.h"
#include "scheduler_testspy.h"
#include "types.h"

#include <cstddef>

// This file is tail-included by runtime.cc; also make it parsable as a standalone file for clangd and clang-tidy
#ifndef CELERITY_DETAIL_TAIL_INCLUDE
#include "../runtime.cc" // NOLINT(bugprone-suspicious-include)
#endif


namespace celerity::detail {

node_id runtime_testspy::get_local_nid(const runtime& rt) { return rt.m_impl->m_local_nid; }

size_t runtime_testspy::get_num_nodes(const runtime& rt) { return rt.m_impl->m_num_nodes; }

size_t runtime_testspy::get_num_local_devices(const runtime& rt) { return rt.m_impl->m_num_local_devices; }

task_graph& runtime_testspy::get_task_graph(runtime& rt) { return rt.m_impl->m_tdag; }

task_manager& runtime_testspy::get_task_manager(runtime& rt) { return *rt.m_impl->m_task_mngr; }

scheduler& runtime_testspy::get_schdlr(runtime& rt) { return *rt.m_impl->m_schdlr; }

executor& runtime_testspy::get_exec(runtime& rt) { return *rt.m_impl->m_exec; }

task_id runtime_testspy::get_latest_epoch_reached(const runtime& rt) { return rt.m_impl->m_latest_epoch_reached.load(std::memory_order_relaxed); }

std::string runtime_testspy::print_task_graph(runtime& rt) {
	return detail::print_task_graph(*rt.m_impl->m_task_recorder); // task recorder is mutated by task manager (application / test thread)
}

std::string runtime_testspy::print_command_graph(const node_id local_nid, runtime& rt) {
	// command_recorder is mutated by scheduler thread
	return scheduler_testspy::inspect_thread(
	    get_schdlr(rt), [&](const auto&) { return detail::print_command_graph(local_nid, *rt.m_impl->m_command_recorder); });
}

std::string runtime_testspy::print_instruction_graph(runtime& rt) {
	// instruction recorder is mutated by scheduler thread
	return scheduler_testspy::inspect_thread(get_schdlr(rt), [&](const auto&) {
		return detail::print_instruction_graph(*rt.m_impl->m_instruction_recorder, *rt.m_impl->m_command_recorder, *rt.m_impl->m_task_recorder,
		    rt.m_impl->m_instruction_performance_recorder.get());
	});
}

void runtime_testspy::test_mode_enter() {
	assert(!runtime::s_mpi_initialized);
	runtime::s_test_mode = true;
}

void runtime_testspy::test_mode_exit() {
	assert(runtime::s_test_mode && !runtime::s_test_active && !runtime::s_mpi_finalized);
	if(runtime::s_mpi_initialized) { runtime::mpi_finalize_once(); }
}

void runtime_testspy::test_require_mpi() {
	assert(runtime::s_test_mode && !runtime::s_test_active);
	if(!runtime::s_mpi_initialized) { runtime::mpi_initialize_once(nullptr, nullptr); }
}

void runtime_testspy::test_case_enter() {
	assert(runtime::s_test_mode && !runtime::s_test_active && runtime::s_mpi_initialized && !runtime::has_instance());
	runtime::s_test_active = true;
	runtime::s_test_runtime_was_instantiated = false;
}

bool runtime_testspy::test_runtime_was_instantiated() {
	assert(runtime::s_test_mode);
	return runtime::s_test_runtime_was_instantiated;
}

void runtime_testspy::test_case_exit() {
	assert(runtime::s_test_mode && runtime::s_test_active);
	runtime::shutdown();
	runtime::s_test_active = false;
}

} // namespace celerity::detail
