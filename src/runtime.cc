#include "runtime.h"

#include <algorithm>
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

	if(is_master) {
		queue->debug_print_task_graph();
		build_command_graph();

		// TODO: Is a BFS walk sufficient or do we need to properly check for fulfilled dependencies?
		graph_utils::search_vertex_bf(0, command_graph, [](vertex v, const command_dag& cdag) {
			auto& v_data = cdag[v];

			// Debug output
			switch(v_data.cmd) {
			case command::PULL: std::cout << "Sending PULL command for task " << v_data.tid << " to node " << v_data.nid << std::endl; break;
			case command::AWAIT_PULL: std::cout << "Sending AWAIT_PULL command for task " << v_data.tid << " to node " << v_data.nid << std::endl; break;
			case command::COMPUTE: std::cout << "Sending COMPUTE command for task " << v_data.tid << " to node " << v_data.nid << std::endl; break;
			default: return false;
			}

			command_pkg pkg{v_data.tid, v_data.cmd, v_data.data};
			send_command(cdag[v].nid, pkg);

			return false;
		});

		// Pull in all the buffer parts so we can check if the result is correct

		// ================================================== HACK HACK HACK ===========================================

		{
			// Sleep a bit so the workers can complete writing their results into buffer 3
			std::this_thread::sleep_for(std::chrono::seconds(2));

			// TODO: This is specific to the current example code (i.e. which buffers) and should probably be done during command graph generation instead
			buffer_id bid = 3;
			auto bs = dynamic_cast<detail::buffer_state<1>*>(valid_buffer_regions[bid].get());
			auto sources = bs->get_source_nodes(GridRegion<1>(1024));
			for(auto& source : sources) {
				auto box = source.first;
				node_id nid = *source.second.begin();
				{
					// FIXME: This needs a proper task id
					command_data data{};
					data.await_pull = await_pull_data{bid, 0, 9999, detail::grid_box_to_subrange(box)};
					std::cout << "Sending final AWAIT PULL to node " << nid << std::endl;
					send_command(nid, command_pkg(9999, command::AWAIT_PULL, data));
				}
				{
					command_data data{};
					data.pull = pull_data{bid, nid, detail::grid_box_to_subrange(box)};
					command_pkg pkg{9999, command::PULL, data};
					auto job = std::make_shared<pull_job>(pkg, btm);
					job->initialize(*queue, jobs);
					jobs.insert(job);
				}
			}
		}

		// ================================================== HACK HACK HACK ===========================================

		// Finally, send shutdown commands to all worker nodes
		command_pkg pkg{0, command::SHUTDOWN, command_data{}};
		for(auto n = 1; n < num_nodes; ++n) {
			send_command(n, pkg);
		}

		// Master can just exit after handling all open jobs
		done = true;
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

		{
			// Check for incoming commands
			// TODO: Move elswhere
			MPI_Status status;
			int flag;
			MPI_Iprobe(MPI_ANY_SOURCE, CELERITY_MPI_TAG_CMD, MPI_COMM_WORLD, &flag, &status);
			if(flag == 1) {
				command_pkg pkg;
				// Commands should be small enough to block here
				MPI_Recv(&pkg, sizeof(command_pkg), MPI_BYTE, status.MPI_SOURCE, CELERITY_MPI_TAG_CMD, MPI_COMM_WORLD, &status);
				if(pkg.cmd == command::SHUTDOWN) {
					done = true;
				} else {
					std::shared_ptr<worker_job> job = nullptr;
					switch(pkg.cmd) {
					case command::PULL: job = std::make_shared<pull_job>(pkg, btm); break;
					case command::AWAIT_PULL: job = std::make_shared<await_pull_job>(pkg, btm); break;
					case command::COMPUTE: job = std::make_shared<compute_job>(pkg, *queue); break;
					default: { assert(false && "Unexpected command"); }
					}
					job->initialize(*queue, jobs);
					jobs.insert(job);
				}
			}
		}
	}
}

void runtime::register_queue(distr_queue* queue) {
	if(this->queue != nullptr) { throw std::runtime_error("Only one celerity::distr_queue can be created per process"); }
	this->queue = queue;
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
 *    c) Iterate over all per-chunk read regions and try to find the most
 *       suitable node to execute that chunk on, i.e. the node that requires
 *       the least amount of data-transfer in order to execute that chunk.
 *       Note that currently only the first read buffer is considered, and
 *       nodes are assigned greedily.
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
	using chunk_id = size_t;

	// NOTE: We still need the ability to run the program on a single node (= master)
	// for easier debugging, so we create a single "split" instead of throwing
	// TODO: Remove this
	const size_t num_worker_nodes = std::max(num_nodes - 1, 1ull);
	const bool master_only = num_nodes == 1;

	const auto& task_graph = queue->get_task_graph();
	const auto& task_range_mappers = queue->get_task_range_mappers();
	const auto& task_global_sizes = queue->get_task_global_sizes();

	auto sibling_set = graph_utils::get_satisfied_sibling_set(task_graph);
	std::sort(sibling_set.begin(), sibling_set.end());

	std::unordered_map<task_id, graph_utils::task_vertices> taskvs;

	// FIXME: Dimensions. Also, containers much??
	std::unordered_map<buffer_id, std::unordered_map<node_id, std::vector<std::pair<task_id, GridRegion<1>>>>> buffer_writers;

	// Iterate over tasks in reverse order so we can determine kernels which
	// write to certain buffer ranges before generating the pull commands for
	// those ranges, which allows us to insert "await-pull"s before writes.
	for(auto it = sibling_set.crbegin(); it != sibling_set.crend(); ++it) {
		const task_id tid = *it;
		const auto& rms = task_range_mappers.at(tid);

		taskvs[tid] = graph_utils::add_task(tid, task_graph, command_graph);

		// Split task into equal chunks for every worker node
		// TODO: In the future, we may want to adjust our split based on the range
		// mapper results and data location!
		auto sr = subrange<1>();
		// FIXME: We assume task dimensionality 1 here
		sr.global_size = boost::get<cl::sycl::range<1>>(task_global_sizes.at(tid));
		sr.range = sr.global_size;

		auto chunks = split_equal(sr, num_worker_nodes);

		// FIXME: Dimensions
		std::unordered_map<chunk_id, std::unordered_map<buffer_id, std::unordered_map<cl::sycl::access::mode, GridRegion<1>>>> chunk_reqs;

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

		std::unordered_map<chunk_id, node_id> chunk_nodes;
		std::set<node_id> free_nodes;
		for(auto i = 0u; i < num_nodes; ++i) {
			if(!master_only && i == 0) { continue; }
			free_nodes.insert(i);
		}

		// FIXME: Dimensions
		std::unordered_map<chunk_id, std::unordered_map<buffer_id, std::vector<std::pair<GridBox<1>, std::unordered_set<node_id>>>>> chunk_buffer_sources;

		// Find per-chunk per-buffer sources and assign nodes to chunks
		for(auto i = 0u; i < chunks.size(); ++i) {
			bool node_assigned = false;
			node_id nid = 0;

			for(auto& it : chunk_reqs[i]) {
				const buffer_id bid = it.first;
				const auto& read_req = it.second[cl::sycl::access::mode::read];

				// FIXME Dimensions
				auto bs = dynamic_cast<detail::buffer_state<1>*>(valid_buffer_regions[bid].get());

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
						std::set_intersection(
						    free_nodes.cbegin(), free_nodes.cend(), source_nodes.cbegin(), source_nodes.cend(), std::back_inserter(intersection));
						if(!intersection.empty()) {
							nid = intersection[0];
						} else {
							nid = *free_nodes.cbegin();
						}
					} else {
						nid = *free_nodes.cbegin();
					}

					assert(master_only || nid != 0);
					node_assigned = true;
					free_nodes.erase(nid);
					chunk_nodes[i] = nid;
				}
			}
		}

		// Create a compute command for every chunk
		std::vector<vertex> chunk_compute_vertices;
		for(chunk_id i = 0u; i < chunks.size(); ++i) {
			const node_id nid = chunk_nodes[i];
			const auto cv = graph_utils::add_compute_cmd(nid, taskvs[tid], chunks[i], command_graph);
			chunk_compute_vertices.push_back(cv);
		}

		// Process writes and create pull / await-pull commands
		for(auto i = 0u; i < chunks.size(); ++i) {
			const node_id nid = chunk_nodes[i];

			for(auto& it : chunk_reqs[i]) {
				const buffer_id bid = it.first;

				// Add read to compute node label for debugging
				const auto& read_req = it.second[cl::sycl::access::mode::read];
				if(read_req.area() > 0) {
					command_graph[chunk_compute_vertices[i]].label =
					    (boost::format("%s\\nRead %d %s") % command_graph[chunk_compute_vertices[i]].label % bid % toString(read_req)).str();
				}

				// ==== Writes ====
				const auto& write_req = it.second[cl::sycl::access::mode::write];
				if(write_req.area() > 0) {
					buffer_writers[bid][nid].push_back(std::make_pair(tid, write_req));

					// Add to compute node label for debugging
					command_graph[chunk_compute_vertices[i]].label =
					    (boost::format("%s\\nWrite %d %s") % command_graph[chunk_compute_vertices[i]].label % bid % toString(write_req)).str();
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
					graph_utils::add_pull_cmd(nid, source_nid, bid, taskvs[tid], source_tv, chunk_compute_vertices[i], box, command_graph);
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

} // namespace celerity
