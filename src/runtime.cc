#include "runtime.h"

#include <algorithm>

#include <allscale/utils/string_utils.h>
#include <mpi.h>

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

void runtime::TEST_do_work() {
	assert(queue != nullptr);
	if(is_master) {
		queue->debug_print_task_graph();
		queue->TEST_execute_deferred();
		build_command_graph();
	} else {
		std::cout << "Worker is idle" << std::endl;
	}
}

void runtime::register_queue(distr_queue* queue) {
	if(this->queue != nullptr) { throw std::runtime_error("Only one celerity::distr_queue can be created per process"); }
	this->queue = queue;
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
		std::unordered_set<node_id> free_nodes;
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
