#pragma once

#include "runtime.h"
#include "types.h"

#include <cstddef>
#include <string>


namespace celerity::detail {

class executor;
class scheduler;
class task_graph;
class task_manager;

struct runtime_testspy {
	static node_id get_local_nid(const runtime& rt);
	static size_t get_num_nodes(const runtime& rt);
	static size_t get_num_local_devices(const runtime& rt);

	static task_graph& get_task_graph(runtime& rt);
	static task_manager& get_task_manager(runtime& rt);
	static scheduler& get_schdlr(runtime& rt);
	static executor& get_exec(runtime& rt);

	static task_id get_latest_epoch_reached(const runtime& rt);

	static std::string print_task_graph(runtime& rt);
	static std::string print_command_graph(const node_id local_nid, runtime& rt);
	static std::string print_instruction_graph(runtime& rt);

	// We have to jump through some hoops to be able to re-initialize the runtime for unit testing.
	// MPI does not like being initialized more than once per process, so we have to skip that part for
	// re-initialization.

	/// Switches to test mode, where MPI will be initialized through test_case_enter() instead of runtime::runtime(). Called on Catch2 startup.
	static void test_mode_enter();

	/// Finalizes MPI if it was ever initialized in test mode. Called on Catch2 shutdown.
	static void test_mode_exit();

	/// Initializes MPI for tests, if it was not initialized before
	static void test_require_mpi();

	/// Allows the runtime to be transitively instantiated in tests. Called from runtime_fixture.
	static void test_case_enter();

	static bool test_runtime_was_instantiated();

	/// Deletes the runtime instance, which happens only in tests. Called from runtime_fixture.
	static void test_case_exit();
};

} // namespace celerity::detail
