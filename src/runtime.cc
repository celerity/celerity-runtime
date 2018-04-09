#include "runtime.h"

#include <algorithm>
#include <limits>
#include <queue>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <allscale/utils/string_utils.h>
#include <boost/variant.hpp>
#include <mpi.h>

#include "command.h"
#include "graph_utils.h"
#include "grid.h"
#include "subrange.h"
#include "task.h"

namespace celerity {

std::unique_ptr<runtime> runtime::instance = nullptr;

void runtime::init(int* argc, char** argv[]) {
	instance = std::unique_ptr<runtime>(new runtime(argc, argv));
}

runtime& runtime::get_instance() {
	if(instance == nullptr) { throw std::runtime_error("Runtime has not been initialized"); }
	return *instance;
}

runtime::runtime(int* argc, char** argv[]) {
	MPI_Init(argc, argv);

	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	num_nodes = world_size;

	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	is_master = world_rank == 0;

	command_graph[boost::graph_bundle].name = "CommandGraph";
}

runtime::~runtime() {
	MPI_Finalize();
}

void send_command(node_id target, const command_pkg& pkg) {
	MPI_Request req;
	// TODO: Do we need the request object?
	const auto result = MPI_Isend(&pkg, sizeof(command_pkg), MPI_BYTE, target, CELERITY_MPI_TAG_CMD, MPI_COMM_WORLD, &req);
	assert(result == MPI_SUCCESS);
}

void runtime::TEST_do_work() {
	assert(queue != nullptr);

	bool done = false;
	// Instead of sending commands to itself, master stores them here
	std::queue<command_pkg> master_commands;

	if(is_master) {
		queue->debug_print_task_graph();
		build_command_graph();

		// TODO: Is a BFS walk sufficient or do we need to properly check for fulfilled dependencies?
		graph_utils::search_vertex_bf(0, command_graph, [&master_commands](vertex v, const command_dag& cdag) {
			auto& v_data = cdag[v];

			// Debug output
			switch(v_data.cmd) {
			case command::PULL: std::cout << "Sending PULL command for task " << v_data.tid << " to node " << v_data.nid << std::endl; break;
			case command::AWAIT_PULL: std::cout << "Sending AWAIT_PULL command for task " << v_data.tid << " to node " << v_data.nid << std::endl; break;
			case command::COMPUTE: std::cout << "Sending COMPUTE command for task " << v_data.tid << " to node " << v_data.nid << std::endl; break;
			case command::MASTER_ACCESS: std::cout << "Sending MASTER_ACCESS command for task " << v_data.tid << " to node " << v_data.nid << std::endl; break;
			default: return false;
			}

			const command_pkg pkg{v_data.tid, v_data.cmd, v_data.data};
			const node_id target = cdag[v].nid;
			if(target != 0) {
				send_command(target, pkg);
			} else {
				master_commands.push(pkg);
			}

			return false;
		});

		// Finally, send shutdown commands to all worker nodes
		command_pkg pkg{0, command::SHUTDOWN, command_data{}};
		for(auto n = 1; n < num_nodes; ++n) {
			send_command(n, pkg);
		}

		// Master can just exit after handling all open jobs
		master_commands.push(pkg);
	}

	while(!done || !jobs.empty()) {
		btm.poll();

		for(auto it = jobs.begin(); it != jobs.end();) {
			auto job = *it;
			job->update();
			if(job->is_done()) {
				it = jobs.erase(it);
			} else {
				++it;
			}
		}

		bool has_pkg = false;
		command_pkg pkg;
		if(!is_master) {
			// Check for incoming commands
			// TODO: Move elswhere
			MPI_Status status;
			int flag;
			MPI_Iprobe(MPI_ANY_SOURCE, CELERITY_MPI_TAG_CMD, MPI_COMM_WORLD, &flag, &status);
			if(flag == 1) {
				// Commands should be small enough to block here
				MPI_Recv(&pkg, sizeof(command_pkg), MPI_BYTE, status.MPI_SOURCE, CELERITY_MPI_TAG_CMD, MPI_COMM_WORLD, &status);
				has_pkg = true;
			}
		} else {
			if(!master_commands.empty()) {
				pkg = master_commands.front();
				master_commands.pop();
				has_pkg = true;
			}
		}

		if(has_pkg) {
			if(pkg.cmd == command::SHUTDOWN) {
				done = true;
			} else {
				handle_command_pkg(pkg);
			}
		}
	}
}

void runtime::handle_command_pkg(const command_pkg& pkg) {
	std::shared_ptr<worker_job> job = nullptr;
	switch(pkg.cmd) {
	case command::PULL: job = std::make_shared<pull_job>(pkg, btm); break;
	case command::AWAIT_PULL: job = std::make_shared<await_pull_job>(pkg, btm); break;
	case command::COMPUTE: job = std::make_shared<compute_job>(pkg, *queue); break;
	case command::MASTER_ACCESS: job = std::make_shared<master_access_job>(pkg); break;
	default: { assert(false && "Unexpected command"); }
	}
	job->initialize(*queue, jobs);
	jobs.insert(job);
}

void runtime::register_queue(distr_queue* queue) {
	if(this->queue != nullptr) { throw std::runtime_error("Only one celerity::distr_queue can be created per process"); }
	this->queue = queue;
}

distr_queue& runtime::get_queue() {
	assert(queue != nullptr);
	return *queue;
}

void runtime::schedule_buffer_send(node_id recipient, const command_pkg& pkg) {
	auto job = std::make_shared<send_job>(pkg, btm, recipient);
	job->initialize(*queue, jobs);
	jobs.insert(job);
}

std::vector<subrange<1>> split_equal(const subrange<1>& sr, size_t num_chunks) {
	assert(num_chunks > 0);
	subrange<1> chunk;
	chunk.global_size = sr.global_size;
	chunk.start = cl::sycl::range<1>(0);
	chunk.range = cl::sycl::range<1>(sr.range.size() / num_chunks);

	std::vector<subrange<1>> result;
	for(auto i = 0u; i < num_chunks; ++i) {
		result.push_back(chunk);
		chunk.start = chunk.start + chunk.range;
		if(i == num_chunks - 1) { result[i].range += sr.range.size() % num_chunks; }
	}
	return result;
}

/**
 * Computes a command graph from the task graph, in batches of sibling sets.
 *
 * This currently (= likely to change in the future!) works as follows:
 *
 * 1) Obtain a suitable satisfied sibling set from the task graph.
 * 2) For every task within that sibling set:
 *    a) Split the task into equally sized chunks.
 *    b) Obtain all range mappers for that task and iterate over them,
 *       determining the read and write regions for every chunk. Note that a
 *       task may contain several read/write accessors for the same buffer,
 *       which is why we first have to compute their union regions.
 *    c) (In assign_chunks_to_nodes): Iterate over all per-chunk read regions
 *		 and try to find the most suitable node to execute that chunk on, i.e.
 *		 the node that requires the least amount of data-transfer in order to
 *		 execute that chunk. Note that currently only the first read buffer is
 *		 considered, and nodes are assigned greedily.
 *    d) Insert compute commands for every node into the command graph.
 *       It is important to create these before pull-commands are inserted
 *       (see below).
 *    e) Iterate over per-chunk reads & writes to (i) store per-buffer per-node
 *       written regions and (ii) create pull / await-pull commands for
 *       all nodes, inserting them as requirements for their respective
 *       compute commands. If no task in the sibling set writes to a specific
 *       buffer, the await-pull command for that buffer will be inserted in
 *       the command subgraph for the current task (which is why it's important
 *       that all compute commands already exist).
 * 3) Finally, all per-buffer per-node written regions are used to update the
 *    data structure that keeps track of valid buffer regions.
 */
void runtime::build_command_graph() {
	// NOTE: We still need the ability to run the program on a single node (= master)
	// for easier debugging, so we create a single "split" instead of throwing
	// TODO: Remove this
	const size_t num_worker_nodes = std::max(num_nodes - 1, (size_t)1);
	const bool master_only = num_nodes == 1;

	const auto& task_graph = queue->get_task_graph();
	auto sibling_set = graph_utils::get_satisfied_sibling_set(task_graph);
	std::sort(sibling_set.begin(), sibling_set.end());

	std::unordered_map<task_id, graph_utils::task_vertices> taskvs;
	// FIXME: Dimensions
	std::unordered_map<buffer_id, std::unordered_map<node_id, std::vector<std::pair<task_id, GridRegion<1>>>>> buffer_writers;

	// Iterate over tasks in reverse order so we can determine kernels which
	// write to certain buffer ranges before generating the pull commands for
	// those ranges, which allows us to insert "await-pull"s before writes.
	for(auto it = sibling_set.crbegin(); it != sibling_set.crend(); ++it) {
		const task_id tid = *it;
		auto tsk = queue->get_task(tid);
		taskvs[tid] = graph_utils::add_task(tid, task_graph, command_graph);

		size_t num_chunks = 0;
		chunk_buffer_requirements_map chunk_reqs;
		chunk_buffer_source_map chunk_buffer_sources;
		std::unordered_map<chunk_id, node_id> chunk_nodes;
		// Vertices corresponding to the per-chunk compute / master access command
		std::vector<vertex> chunk_command_vertices;

		if(tsk->get_type() == task_type::COMPUTE) {
			const auto ctsk = dynamic_cast<const compute_task*>(tsk.get());

			// Split task into equal chunks for every worker node
			// TODO: In the future, we may want to adjust our split based on the range
			// mapper results and data location!
			num_chunks = num_worker_nodes;

			// FIXME: Dimensions
			auto sr = subrange<1>();
			sr.global_size = boost::get<cl::sycl::range<1>>(ctsk->get_global_size());
			sr.range = sr.global_size;
			auto chunks = split_equal(sr, num_chunks);

			const auto& rms = ctsk->get_range_mappers();
			for(auto& it : rms) {
				const buffer_id bid = it.first;

				for(auto& rm : it.second) {
					auto mode = rm->get_access_mode();
					assert(mode == cl::sycl::access::mode::read || mode == cl::sycl::access::mode::write);

					for(auto i = 0u; i < chunks.size(); ++i) {
						// FIXME: Dimensions
						subrange<1> req = (*rm)(chunks[i]);
						chunk_reqs[i][bid][mode] = GridRegion<1>::merge(chunk_reqs[i][bid][mode], detail::subrange_to_grid_region(req));
					}
				}
			}

			std::set<node_id> free_nodes;
			for(auto i = 0u; i < num_nodes; ++i) {
				if(!master_only && i == 0) { continue; }
				free_nodes.insert(i);
			}

			chunk_nodes = assign_chunks_to_nodes(chunks.size(), chunk_reqs, free_nodes, chunk_buffer_sources);

			// Create a compute command for every chunk
			for(chunk_id i = 0u; i < chunks.size(); ++i) {
				const node_id nid = chunk_nodes[i];
				const auto cv = graph_utils::add_compute_cmd(nid, taskvs[tid], chunks[i], command_graph);
				chunk_command_vertices.push_back(cv);
			}
		} else if(tsk->get_type() == task_type::MASTER_ACCESS) {
			const auto matsk = dynamic_cast<const master_access_task*>(tsk.get());
			const node_id master_node = 0;
			num_chunks = 1;
			chunk_nodes[master_node] = 0;

			const auto buffer_accesses = matsk->get_accesses();
			for(auto& it : buffer_accesses) {
				const buffer_id bid = it.first;

				for(auto& bacc : it.second) {
					// Note that subrange_to_grid_region clamps to the global size, which is why we set this to size_t max
					// TODO: Subrange is not ideal here, we don't need the global size
					// FIXME Dimensions
					auto req = subrange<1>{boost::get<cl::sycl::range<1>>(bacc.offset), boost::get<cl::sycl::range<1>>(bacc.range),
					    cl::sycl::range<1>(std::numeric_limits<size_t>::max())};
					chunk_reqs[master_node][bid][bacc.mode] =
					    GridRegion<1>::merge(chunk_reqs[master_node][bid][bacc.mode], detail::subrange_to_grid_region(req));
				}
			}

			for(auto& it : chunk_reqs.at(master_node)) {
				const buffer_id bid = it.first;
				if(it.second.count(cl::sycl::access::mode::read) == 0) continue;
				auto read_req = it.second.at(cl::sycl::access::mode::read);

				// FIXME Dimensions
				auto bs = dynamic_cast<detail::buffer_state<1>*>(valid_buffer_regions.at(bid).get());
				chunk_buffer_sources[master_node][bid] = bs->get_source_nodes(read_req);
			}

			const auto cv = graph_utils::add_master_access_cmd(taskvs[tid], command_graph);
			chunk_command_vertices.push_back(cv);
		}

		// Process writes and create pull / await-pull commands
		for(auto i = 0u; i < num_chunks; ++i) {
			const node_id nid = chunk_nodes[i];

			for(auto& it : chunk_reqs[i]) {
				const buffer_id bid = it.first;
				const vertex command_vertex = chunk_command_vertices[i];

				// Add read to command node label for debugging
				const auto& read_req = it.second[cl::sycl::access::mode::read];
				if(read_req.area() > 0) {
					command_graph[command_vertex].label =
					    (boost::format("%s\\nRead %d %s") % command_graph[command_vertex].label % bid % toString(read_req)).str();
				}

				// ==== Writes ====
				const auto& write_req = it.second[cl::sycl::access::mode::write];
				if(write_req.area() > 0) {
					buffer_writers[bid][nid].push_back(std::make_pair(tid, write_req));

					// Add to compute node label for debugging
					command_graph[command_vertex].label =
					    (boost::format("%s\\nWrite %d %s") % command_graph[command_vertex].label % bid % toString(write_req)).str();
				}

				// ==== Reads ====
				const auto buffer_sources = chunk_buffer_sources[i][bid];

				for(auto& box_sources : buffer_sources) {
					const auto& box = box_sources.first;
					const auto& box_src_nodes = box_sources.second;

					if(box_src_nodes.count(nid) == 1) {
						// No need to pull
						continue;
					}

					// We just pick the first source node for now
					const node_id source_nid = *box_src_nodes.cbegin();

					// Figure out where/when (and if) source node writes to that buffer
					// TODO: For now we just store the writer's task id since we assume
					// that every node has exactly one compute command per task. In the
					// future this may not be true.
					bool has_writer = false;
					task_id writer_tid = 0;
					for(const auto& bw : buffer_writers[bid][source_nid]) {
						if(GridRegion<1>::intersect(bw.second, GridRegion<1>(box)).area() > 0) {
#ifdef _DEBUG
							// We assume at most one sibling writes to that exact region
							// TODO: Is there any (useful) scenario where this isn't true?
							assert(!has_writer);
#endif
							has_writer = true;
							writer_tid = bw.first;
#ifndef _DEBUG
							break;
#endif
						}
					}

					// If we haven't found a writer, simply add the "await-pull" in
					// the current task
					const auto source_tv = has_writer ? taskvs[writer_tid] : taskvs[tid];

					// TODO: Update buffer regions since we copied some stuff!!
					graph_utils::add_pull_cmd(nid, source_nid, bid, taskvs[tid], source_tv, command_vertex, box, command_graph);
				}
			}
		}

		queue->mark_task_as_processed(tid);
	}

	// Update buffer regions
	// FIXME Dimensions
	for(auto it : buffer_writers) {
		const buffer_id bid = it.first;
		auto bs = static_cast<detail::buffer_state<1>*>(valid_buffer_regions[bid].get());

		for(auto jt : it.second) {
			const node_id nid = jt.first;
			GridRegion<1> region;

			for(const auto& kt : jt.second) {
				region = GridRegion<1>::merge(region, kt.second);
			}

			bs->update_region(region, {nid});
		}
	}

	// HACK: We recursively call this until all tasks have been processed
	// In the future, we may want to do this periodically in a worker thread
	if(graph_utils::get_satisfied_sibling_set(task_graph).size() > 0) {
		build_command_graph();
	} else {
		graph_utils::print_graph(command_graph);
	}
}

// FIXME: Dimensions
std::unordered_map<runtime::chunk_id, node_id> runtime::assign_chunks_to_nodes(
    size_t num_chunks, const chunk_buffer_requirements_map& chunk_reqs, std::set<node_id> free_nodes, chunk_buffer_source_map& chunk_buffer_sources) const {
	std::unordered_map<chunk_id, node_id> chunk_nodes;
	for(auto i = 0u; i < num_chunks; ++i) {
		bool node_assigned = false;
		node_id nid = std::numeric_limits<node_id>::max();

		for(auto& it : chunk_reqs.at(i)) {
			const buffer_id bid = it.first;

			// FIXME Dimensions
			GridRegion<1> read_req;
			if(it.second.count(cl::sycl::access::mode::read) != 0) { read_req = it.second.at(cl::sycl::access::mode::read); }

			// FIXME Dimensions
			const auto bs = dynamic_cast<detail::buffer_state<1>*>(valid_buffer_regions.at(bid).get());

			// TODO: Are these always sorted? Probably not. (requried for set_intersection)
			const auto sn = bs->get_source_nodes(read_req);
			chunk_buffer_sources[i][bid] = sn;

			if(!node_assigned) {
				assert(free_nodes.size() > 0);

				// If the chunk doesn't have any read requirements (for this buffer!),
				// we also won't get any source nodes
				if(sn.size() > 0) {
					const auto& source_nodes = sn[0].second;

					// We simply pick the first node that contains the largest chunk of
					// the first requested buffer, given it is still available.
					// Otherwise we simply pick the first available node.
					// TODO: We should probably consider all buffers, not just the first
					std::vector<node_id> intersection;
					std::set_intersection(free_nodes.cbegin(), free_nodes.cend(), source_nodes.cbegin(), source_nodes.cend(), std::back_inserter(intersection));
					if(!intersection.empty()) {
						nid = intersection[0];
					} else {
						nid = *free_nodes.cbegin();
					}
				} else {
					nid = *free_nodes.cbegin();
				}

				assert(nid != std::numeric_limits<node_id>::max());
				node_assigned = true;
				free_nodes.erase(nid);
				chunk_nodes[i] = nid;
			}
		}
	}

	return chunk_nodes;
}

void runtime::execute_master_access_task(task_id tid) const {
	const auto tsk = dynamic_cast<const master_access_task*>(queue->get_task(tid).get());
	const master_access_livepass_handler handler;
	tsk->get_functor()(handler);
}

} // namespace celerity
