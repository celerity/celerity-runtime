#include "runtime.h"

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <allscale/utils/string_utils.h>
#include <mpi.h>

#include "buffer.h"
#include "buffer_storage.h"
#include "command.h"
#include "graph_utils.h"
#include "grid.h"
#include "logger.h"
#include "task.h"

#define MAX_CONCURRENT_JOBS 20

namespace celerity {

std::unique_ptr<runtime> runtime::instance = nullptr;
bool runtime::test_skip_mpi_lifecycle = false;

void runtime::init(int* argc, char** argv[]) {
	instance = std::unique_ptr<runtime>(new runtime(argc, argv));
}

runtime& runtime::get_instance() {
	if(instance == nullptr) { throw std::runtime_error("Runtime has not been initialized"); }
	return *instance;
}

runtime::runtime(int* argc, char** argv[]) {
	if(!test_skip_mpi_lifecycle) {
		// We specify MPI_THREAD_FUNNELED even though we currently don't use multiple threads,
		// as we link with various multi-threaded libraries. This will likely not make any difference,
		// but we do it anyway, just in case. See here for more information:
		// http://users.open-mpi.narkive.com/T04C74T4/ompi-users-mpi-thread-single-vs-mpi-thread-funneled
		int provided;
		MPI_Init_thread(argc, argv, MPI_THREAD_FUNNELED, &provided);
		assert(provided == MPI_THREAD_FUNNELED);
	}

	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	num_nodes = world_size;

	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	is_master = world_rank == 0;

	default_logger = logger("default").create_context({{"rank", std::to_string(world_rank)}});
	graph_logger = logger("graph").create_context({{"rank", std::to_string(world_rank)}});

	btm = std::make_unique<buffer_transfer_manager>(default_logger);

	if(is_master) { ggen = std::make_unique<detail::graph_generator>(num_nodes); }
}

runtime::~runtime() {
	// Allow BTM to clean up MPI data types before we finalize
	btm.reset();
	if(!test_skip_mpi_lifecycle) { MPI_Finalize(); }
}

void send_command(node_id target, const command_pkg& pkg) {
	// Command packages are small enough to use a blocking send.
	// This way we don't have to ensure the data stays around long enough (until asynchronous send is complete).
	const auto result = MPI_Send(&pkg, sizeof(command_pkg), MPI_BYTE, static_cast<int>(target), CELERITY_MPI_TAG_CMD, MPI_COMM_WORLD);
	assert(result == MPI_SUCCESS);
}

void runtime::distribute_commands(std::queue<command_pkg>& master_command_queue) {
	assert(is_master);
	const auto& task_graph = queue->get_task_graph();
	std::queue<task_id> task_queue;
	std::unordered_set<task_id> queued_tasks;

	const auto& command_graph = ggen->get_command_graph();
	const auto& cmd_dag_task_vertices = GRAPH_PROP(command_graph, task_vertices);

	// Find all root tasks
	for(auto v : task_graph.vertex_set()) {
		if(boost::in_degree(v, task_graph) == 0) {
			task_queue.push(v);
			queued_tasks.insert(v);
		}
	}

	while(!task_queue.empty()) {
		const task_id tid = task_queue.front();
		task_queue.pop();
		// Verify that we didn't foget to generate commands for this task for some reason
		assert(cmd_dag_task_vertices.count(tid) != 0);
		const auto& tv = cmd_dag_task_vertices.at(tid);
		std::queue<cdag_vertex> cmd_queue;
		std::unordered_set<cdag_vertex> queued_cmds;
		cmd_queue.push(tv.first);
		queued_cmds.insert(tv.first);

		while(!cmd_queue.empty()) {
			const cdag_vertex v = cmd_queue.front();
			cmd_queue.pop();
			auto& cmd_v = command_graph[v];
			if(cmd_v.cmd != command::NOP) {
				const command_pkg pkg{cmd_v.tid, cmd_v.cid, cmd_v.cmd, cmd_v.data};
				const node_id target = cmd_v.nid;
				if(target != 0) {
					send_command(target, pkg);
				} else {
					master_command_queue.push(pkg);
				}
			}

			// NOTE: This assumes that we have no inter-task command dependencies!
			graph_utils::for_successors(command_graph, v, [tv, &queued_cmds, &cmd_queue](cdag_vertex s) {
				if(s != tv.second && queued_cmds.count(s) == 0) {
					cmd_queue.push(s);
					queued_cmds.insert(s);
				}
				return true;
			});
		}

		graph_utils::for_successors(task_graph, static_cast<tdag_vertex>(tid), [&queued_tasks, &task_queue](tdag_vertex v) {
			const auto t = static_cast<task_id>(v);
			if(queued_tasks.count(t) == 0) {
				task_queue.push(t);
				queued_tasks.insert(t);
			}
		});
	}
}

void runtime::TEST_do_work() {
	if(is_master) {
		const auto& task_graph = queue->get_task_graph();
		if(task_graph.m_vertices.size() < 200) {
			graph_utils::print_graph(task_graph, graph_logger);
		} else {
			default_logger->info("Task graph is very large ({} vertices). Skipping GraphViz output", task_graph.m_vertices.size());
		}

		while(ggen->has_unbuilt_tasks()) {
			ggen->build_task();
		}

		if(ggen->get_command_graph().m_vertices.size() < 200) {
			// FIXME: HACK: const_cast - we shouldn't print the DAG from here.
			graph_utils::print_graph(const_cast<command_dag&>(ggen->get_command_graph()), graph_logger);
		} else {
			default_logger->info("Command graph is very large ({} vertices). Skipping GraphViz output", ggen->get_command_graph().m_vertices.size());
		}
	}

	assert(queue != nullptr);

	bool done = false;
	std::queue<command_pkg> command_queue;

	if(is_master) {
		distribute_commands(command_queue);

		// Finally, send shutdown commands to all worker nodes
		// FIXME: Handle this through the graph_generator (::finalize() ?) as well (but when do we actually call this once we get rid of TEST_do_work?)
		command_id next_cmd_id = 10e4; // HACK
		for(auto n = 1u; n < num_nodes; ++n) {
			command_pkg pkg{0, next_cmd_id++, command::SHUTDOWN, command_data{}};
			send_command(n, pkg);
		}

		// Master can just exit after handling all open jobs
		command_queue.push(command_pkg{0, next_cmd_id++, command::SHUTDOWN, command_data{}});
	}

	while(!done || !jobs.empty()) {
		btm->poll();

		for(auto it = jobs.begin(); it != jobs.end();) {
			auto job = *it;
			job->update();
			if(job->is_done()) {
				it = jobs.erase(it);
			} else {
				++it;
			}
		}

		if(!is_master) {
			// Check for incoming commands
			// TODO: Move elswhere
			MPI_Status status;
			int flag;
			MPI_Iprobe(MPI_ANY_SOURCE, CELERITY_MPI_TAG_CMD, MPI_COMM_WORLD, &flag, &status);
			if(flag == 1) {
				// Commands should be small enough to block here
				command_pkg pkg;
				MPI_Recv(&pkg, sizeof(command_pkg), MPI_BYTE, status.MPI_SOURCE, CELERITY_MPI_TAG_CMD, MPI_COMM_WORLD, &status);
				command_queue.push(pkg);
			}
		}

		if(jobs.size() < MAX_CONCURRENT_JOBS && !command_queue.empty()) {
			const auto pkg = command_queue.front();
			command_queue.pop();
			if(pkg.cmd == command::SHUTDOWN) {
				// NOTE: This might fail if commands come in out-of-order
				assert(command_queue.empty());
				done = true;
			} else {
				handle_command_pkg(pkg);
			}
		}
	}
}

void runtime::handle_command_pkg(const command_pkg& pkg) {
	switch(pkg.cmd) {
	case command::PUSH: create_job<push_job>(pkg, *btm); break;
	case command::AWAIT_PUSH: create_job<await_push_job>(pkg, *btm); break;
	case command::COMPUTE: create_job<compute_job>(pkg, *queue); break;
	case command::MASTER_ACCESS: create_job<master_access_job>(pkg); break;
	default: { assert(false && "Unexpected command"); }
	}
}

void runtime::register_queue(distr_queue* queue) {
	if(this->queue != nullptr) { throw std::runtime_error("Only one celerity::distr_queue can be created per process"); }
	this->queue = queue;
	if(is_master) { ggen->set_queue(queue); }
}

distr_queue& runtime::get_queue() {
	assert(queue != nullptr);
	return *queue;
}

buffer_id runtime::register_buffer(cl::sycl::range<3> range, std::shared_ptr<detail::buffer_storage_base> buf_storage) {
	buf_storage->set_type(is_master ? detail::buffer_type::HOST_BUFFER : detail::buffer_type::DEVICE_BUFFER);
	const buffer_id bid = buffer_count++;
	buffer_ptrs[bid] = buf_storage;
	if(is_master) { ggen->add_buffer(bid, range); }
	return bid;
}

void runtime::free_buffers() {
	buffer_ptrs.clear();
}

void runtime::execute_master_access_task(task_id tid) const {
	const auto tsk = dynamic_cast<const master_access_task*>(queue->get_task(tid).get());
	master_access_livepass_handler handler;
	tsk->get_functor()(handler);
}

} // namespace celerity
