#include "runtime.h"

#include <algorithm>
#include <limits>
#include <queue>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <allscale/utils/string_utils.h>
#include <boost/variant.hpp>
#include <mpi.h>
#include <spdlog/fmt/fmt.h>

#include "command.h"
#include "grid.h"
#include "logger.h"
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

	default_logger = logger("default").create_context({{"rank", std::to_string(world_rank)}});
	graph_logger = logger("graph").create_context({{"rank", std::to_string(world_rank)}});

	command_graph[boost::graph_bundle].name = "CommandGraph";

	btm = std::make_unique<buffer_transfer_manager>(default_logger);
}

runtime::~runtime() {
	// Allow BTM to clean up MPI data types before we finalize
	btm.release();
	MPI_Finalize();
}

void send_command(node_id target, const command_pkg& pkg) {
	// Command packages are small enough to use a blocking send.
	// This way we don't have to ensure the data stays around long enough (until asynchronous send is complete).
	const auto result = MPI_Send(&pkg, sizeof(command_pkg), MPI_BYTE, target, CELERITY_MPI_TAG_CMD, MPI_COMM_WORLD);
	assert(result == MPI_SUCCESS);
}

void runtime::TEST_do_work() {
	assert(queue != nullptr);

	bool done = false;
	// Instead of sending commands to itself, master stores them here
	std::queue<command_pkg> master_commands;

	if(is_master) {
		queue->debug_print_task_graph(graph_logger);
		build_command_graph();
		graph_utils::print_graph(command_graph, graph_logger);

		// TODO: Is a BFS walk sufficient or do we need to properly check for fulfilled dependencies?
		// FIXME: This doesn't support disconnected tasks (e.g. two kernels with no dependencies whatsoever)
		graph_utils::search_vertex_bf(0, command_graph, [&master_commands, this](vertex v, const command_dag& cdag) {
			auto& v_data = cdag[v];

			// Debug output
			switch(v_data.cmd) {
			case command::PULL: default_logger->info("Sending PULL command for task {} to node {}", v_data.tid, v_data.nid); break;
			case command::AWAIT_PULL: default_logger->info("Sending AWAIT_PULL command for task {} to node {}", v_data.tid, v_data.nid); break;
			case command::COMPUTE: default_logger->info("Sending COMPUTE command for task {} to node {}", v_data.tid, v_data.nid); break;
			case command::MASTER_ACCESS: default_logger->info("Sending MASTER_ACCESS command for task {} to node {}", v_data.tid, v_data.nid); break;
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
	switch(pkg.cmd) {
	case command::PULL: create_job<pull_job>(pkg, *btm); break;
	case command::AWAIT_PULL: create_job<await_pull_job>(pkg, *btm); break;
	case command::COMPUTE: create_job<compute_job>(pkg, *queue); break;
	case command::MASTER_ACCESS: create_job<master_access_job>(pkg); break;
	default: { assert(false && "Unexpected command"); }
	}
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
	create_job<send_job>(pkg, *btm, recipient);
}

// ---------------------------------------------------------------------------------------------------------------
// -----------------------------------------  COMMAND GRAPH GENERATION  ------------------------------------------
// ---------------------------------------------------------------------------------------------------------------

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

// We simply split by row for now
// TODO: There's other ways to split in 2D as well.
std::vector<subrange<2>> split_equal(const subrange<2>& sr, size_t num_chunks) {
	const auto rows =
	    split_equal(subrange<1>{cl::sycl::range<1>(sr.start[0]), cl::sycl::range<1>(sr.range[0]), cl::sycl::range<1>(sr.global_size[0])}, num_chunks);
	std::vector<subrange<2>> result;
	for(auto& row : rows) {
		result.push_back(subrange<2>{cl::sycl::range<2>(row.start[0], sr.start[1]), cl::sycl::range<2>(row.range[0], sr.range[1]), sr.global_size});
	}
	return result;
}

std::vector<subrange<3>> split_equal(const subrange<3>& sr, size_t num_chunks) {
	throw std::runtime_error("3D split_equal NYI");
}

template <int Dims>
std::vector<std::pair<any_grid_box, std::unordered_set<node_id>>> any_wrap_source_nodes_grid_boxes(
    const std::vector<std::pair<GridBox<Dims>, std::unordered_set<node_id>>>& source_nodes) {
	std::vector<std::pair<any_grid_box, std::unordered_set<node_id>>> wrapped(source_nodes.size());
	std::transform(source_nodes.cbegin(), source_nodes.cend(), wrapped.begin(), [](auto& sn) { return std::make_pair(any_grid_box(sn.first), sn.second); });
	return wrapped;
};

/**
 * Assigns a number of chunks to a given set of free nodes.
 * Additionally computes the source nodes for the buffers required by the individual chunks.
 */
std::unordered_map<chunk_id, node_id> assign_chunks_to_nodes(size_t num_chunks, const chunk_buffer_requirements_map& chunk_reqs,
    const buffer_state_map& valid_buffer_regions, std::set<node_id> free_nodes, chunk_buffer_source_map& chunk_buffer_sources) {
	std::unordered_map<chunk_id, node_id> chunk_nodes;
	for(auto i = 0u; i < num_chunks; ++i) {
		bool node_assigned = false;
		node_id nid = std::numeric_limits<node_id>::max();

		for(auto& it : chunk_reqs.at(i)) {
			const buffer_id bid = it.first;

			// TODO: Are these always sorted (return value of buffer_state::get_source_nodes)? Probably not. (requried for set_intersection)
			std::unordered_set<node_id> source_nodes;

			if(it.second.count(cl::sycl::access::mode::read) > 0) {
				const int dimensions = it.second.at(cl::sycl::access::mode::read).which() + 1;
				switch(dimensions) {
				default:
				case 1: {
					const auto& read_req = boost::get<GridRegion<1>>(it.second.at(cl::sycl::access::mode::read));
					const auto bs = dynamic_cast<detail::buffer_state<1>*>(valid_buffer_regions.at(bid).get());
					const auto sn = bs->get_source_nodes(read_req);
					chunk_buffer_sources[i][bid] = any_wrap_source_nodes_grid_boxes<1>(sn);
					if(sn.size() > 0) { source_nodes = sn[0].second; }
				} break;
				case 2: {
					const auto& read_req = boost::get<GridRegion<2>>(it.second.at(cl::sycl::access::mode::read));
					const auto bs = dynamic_cast<detail::buffer_state<2>*>(valid_buffer_regions.at(bid).get());
					const auto sn = bs->get_source_nodes(read_req);
					chunk_buffer_sources[i][bid] = any_wrap_source_nodes_grid_boxes<2>(sn);
					if(sn.size() > 0) { source_nodes = sn[0].second; }
				} break;
				case 3: {
					const auto& read_req = boost::get<GridRegion<3>>(it.second.at(cl::sycl::access::mode::read));
					const auto bs = dynamic_cast<detail::buffer_state<3>*>(valid_buffer_regions.at(bid).get());
					const auto sn = bs->get_source_nodes(read_req);
					chunk_buffer_sources[i][bid] = any_wrap_source_nodes_grid_boxes<3>(sn);
					if(sn.size() > 0) { source_nodes = sn[0].second; }
				} break;
				}
			}

			if(!node_assigned) {
				assert(free_nodes.size() > 0);

				// If the chunk doesn't have any read requirements (for this buffer!),
				// we also won't get any source nodes
				if(source_nodes.size() > 0) {
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

template <int Dims>
void process_compute_task(task_id tid, const compute_task* ctsk, size_t num_worker_nodes, bool master_only,
    const std::unordered_map<task_id, graph_utils::task_vertices>& taskvs, const buffer_state_map& valid_buffer_regions, size_t& num_chunks,
    std::unordered_map<chunk_id, node_id>& chunk_nodes, chunk_buffer_requirements_map& chunk_requirements, chunk_buffer_source_map& chunk_buffer_sources,
    std::vector<vertex>& chunk_command_vertices, command_dag& command_graph) {
	assert(ctsk->get_dimensions() == Dims);

	// Split task into equal chunks for every worker node
	// TODO: In the future, we may want to adjust our split based on the range
	// mapper results and data location!
	num_chunks = num_worker_nodes;

	// The chunks have the same dimensionality as the task
	auto sr = subrange<Dims>();
	sr.global_size = boost::get<cl::sycl::range<Dims>>(ctsk->get_global_size());
	sr.range = sr.global_size;
	auto chunks = split_equal(sr, num_chunks);

	const auto& rms = ctsk->get_range_mappers();
	for(auto& it : rms) {
		const buffer_id bid = it.first;

		for(auto& rm : it.second) {
			auto mode = rm->get_access_mode();
			assert(mode == cl::sycl::access::mode::read || mode == cl::sycl::access::mode::write);
			assert(rm->get_kernel_dimensions() == Dims);

			for(auto i = 0u; i < chunks.size(); ++i) {
				// The chunk requirements have the dimensionality of the corresponding buffer
				switch(rm->get_buffer_dimensions()) {
				default:
				case 1: {
					const subrange<1> req = (*rm).map_1(chunks[i]);
					const auto& reqs = boost::get<GridRegion<1>>(chunk_requirements[i][bid][mode]);
					chunk_requirements[i][bid][mode] = GridRegion<1>::merge(reqs, detail::subrange_to_grid_region(req));
				} break;
				case 2: {
					const subrange<2> req = (*rm).map_2(chunks[i]);
					if(chunk_requirements[i][bid][mode].which() == 0) {
						chunk_requirements[i][bid][mode] = detail::subrange_to_grid_region(req);
						continue;
					}
					const auto& reqs = boost::get<GridRegion<2>>(chunk_requirements[i][bid][mode]);
					chunk_requirements[i][bid][mode] = GridRegion<2>::merge(reqs, detail::subrange_to_grid_region(req));
				} break;
				case 3: {
					const subrange<3> req = (*rm).map_3(chunks[i]);
					if(chunk_requirements[i][bid][mode].which() == 0) {
						chunk_requirements[i][bid][mode] = detail::subrange_to_grid_region(req);
						continue;
					}
					const auto& reqs = boost::get<GridRegion<3>>(chunk_requirements[i][bid][mode]);
					chunk_requirements[i][bid][mode] = GridRegion<3>::merge(reqs, detail::subrange_to_grid_region(req));
				} break;
				}
			}
		}
	}

	std::set<node_id> free_nodes;
	for(auto i = 0u; i < num_worker_nodes + 1; ++i) {
		if(!master_only && i == 0) { continue; }
		free_nodes.insert(i);
	}

	chunk_nodes = assign_chunks_to_nodes(chunks.size(), chunk_requirements, valid_buffer_regions, free_nodes, chunk_buffer_sources);

	// Create a compute command for every chunk
	for(chunk_id i = 0u; i < chunks.size(); ++i) {
		const node_id nid = chunk_nodes[i];
		const auto cv = graph_utils::add_compute_cmd(nid, taskvs.at(tid), chunks[i], command_graph);
		chunk_command_vertices.push_back(cv);
	}
}

void process_master_access_task(task_id tid, const master_access_task* matsk, const std::unordered_map<task_id, graph_utils::task_vertices>& taskvs,
    const buffer_state_map& valid_buffer_regions, size_t& num_chunks, std::unordered_map<chunk_id, node_id>& chunk_nodes,
    chunk_buffer_requirements_map& chunk_requirements, chunk_buffer_source_map& chunk_buffer_sources, std::vector<vertex>& chunk_command_vertices,
    command_dag& command_graph) {
	const node_id master_node = 0;
	num_chunks = 1;
	chunk_nodes[master_node] = 0;

	const auto& buffer_accesses = matsk->get_accesses();
	for(auto& it : buffer_accesses) {
		const buffer_id bid = it.first;

		for(auto& bacc : it.second) {
			// Note that subrange_to_grid_region clamps to the global size, which is why we set this to size_t max
			// TODO: Subrange is not ideal here, we don't need the global size
			switch(bacc.get_dimensions()) {
			default:
			case 1: {
				const auto req = subrange<1>{boost::get<cl::sycl::range<1>>(bacc.offset), boost::get<cl::sycl::range<1>>(bacc.range),
				    cl::sycl::range<1>(std::numeric_limits<size_t>::max())};
				const auto& reqs = boost::get<GridRegion<1>>(chunk_requirements[master_node][bid][bacc.mode]);
				chunk_requirements[master_node][bid][bacc.mode] = GridRegion<1>::merge(reqs, detail::subrange_to_grid_region(req));
			} break;
			case 2: {
				const auto req = subrange<2>{boost::get<cl::sycl::range<2>>(bacc.offset), boost::get<cl::sycl::range<2>>(bacc.range),
				    cl::sycl::range<2>(std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max())};
				if(chunk_requirements[master_node][bid][bacc.mode].which() == 0) {
					chunk_requirements[master_node][bid][bacc.mode] = detail::subrange_to_grid_region(req);
					continue;
				}
				const auto& reqs = boost::get<GridRegion<2>>(chunk_requirements[master_node][bid][bacc.mode]);
				chunk_requirements[master_node][bid][bacc.mode] = GridRegion<2>::merge(reqs, detail::subrange_to_grid_region(req));
			} break;
			case 3: {
				const auto req = subrange<3>{boost::get<cl::sycl::range<3>>(bacc.offset), boost::get<cl::sycl::range<3>>(bacc.range),
				    cl::sycl::range<3>(std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max())};
				if(chunk_requirements[master_node][bid][bacc.mode].which() == 0) {
					chunk_requirements[master_node][bid][bacc.mode] = detail::subrange_to_grid_region(req);
					continue;
				}
				const auto& reqs = boost::get<GridRegion<3>>(chunk_requirements[master_node][bid][bacc.mode]);
				chunk_requirements[master_node][bid][bacc.mode] = GridRegion<3>::merge(reqs, detail::subrange_to_grid_region(req));
			} break;
			}
		}
	}

	for(auto& it : chunk_requirements.at(master_node)) {
		const buffer_id bid = it.first;
		if(it.second.count(cl::sycl::access::mode::read) == 0) continue;
		const int dimensions = it.second.at(cl::sycl::access::mode::read).which() + 1;

		switch(dimensions) {
		default:
		case 1: {
			const auto& read_req = boost::get<GridRegion<1>>(it.second.at(cl::sycl::access::mode::read));
			const auto bs = dynamic_cast<detail::buffer_state<1>*>(valid_buffer_regions.at(bid).get());
			const auto source_nodes = bs->get_source_nodes(read_req);
			chunk_buffer_sources[master_node][bid] = any_wrap_source_nodes_grid_boxes<1>(source_nodes);
		} break;
		case 2: {
			const auto& read_req = boost::get<GridRegion<2>>(it.second.at(cl::sycl::access::mode::read));
			const auto bs = dynamic_cast<detail::buffer_state<2>*>(valid_buffer_regions.at(bid).get());
			const auto source_nodes = bs->get_source_nodes(read_req);
			chunk_buffer_sources[master_node][bid] = any_wrap_source_nodes_grid_boxes<2>(source_nodes);
		} break;
		case 3: {
			const auto& read_req = boost::get<GridRegion<3>>(it.second.at(cl::sycl::access::mode::read));
			const auto bs = dynamic_cast<detail::buffer_state<3>*>(valid_buffer_regions.at(bid).get());
			const auto source_nodes = bs->get_source_nodes(read_req);
			chunk_buffer_sources[master_node][bid] = any_wrap_source_nodes_grid_boxes<3>(source_nodes);
		} break;
		}
	}

	const auto cv = graph_utils::add_master_access_cmd(taskvs.at(tid), command_graph);
	chunk_command_vertices.push_back(cv);
}


template <int Dims>
void update_buffer_state(const std::unordered_map<node_id, std::vector<std::pair<task_id, any_grid_region>>>& buffer_writers, detail::buffer_state<Dims>& bs) {
	for(const auto& w : buffer_writers) {
		const node_id nid = w.first;
		GridRegion<Dims> region;
		for(const auto& tr : w.second) {
			region = GridRegion<Dims>::merge(region, boost::get<GridRegion<Dims>>(tr.second));
		}
		bs.update_region(region, {nid});
	}
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
	buffer_writers_map buffer_writers;

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
		// TODO: Either use map <chunk_id, node_id> for this, or vector for chunk_nodes - not both
		std::vector<vertex> chunk_command_vertices;

		if(tsk->get_type() == task_type::COMPUTE) {
			const auto ctsk = dynamic_cast<const compute_task*>(tsk.get());
			switch(ctsk->get_dimensions()) {
			default:
			case 1:
				process_compute_task<1>(tid, ctsk, num_worker_nodes, master_only, taskvs, valid_buffer_regions, num_chunks, chunk_nodes, chunk_reqs,
				    chunk_buffer_sources, chunk_command_vertices, command_graph);
				break;
			case 2:
				process_compute_task<2>(tid, ctsk, num_worker_nodes, master_only, taskvs, valid_buffer_regions, num_chunks, chunk_nodes, chunk_reqs,
				    chunk_buffer_sources, chunk_command_vertices, command_graph);
				break;
			case 3:
				process_compute_task<3>(tid, ctsk, num_worker_nodes, master_only, taskvs, valid_buffer_regions, num_chunks, chunk_nodes, chunk_reqs,
				    chunk_buffer_sources, chunk_command_vertices, command_graph);
				break;
			}
		} else if(tsk->get_type() == task_type::MASTER_ACCESS) {
			const auto matsk = dynamic_cast<const master_access_task*>(tsk.get());
			process_master_access_task(
			    tid, matsk, taskvs, valid_buffer_regions, num_chunks, chunk_nodes, chunk_reqs, chunk_buffer_sources, chunk_command_vertices, command_graph);
		}

		process_task_data_requirements(tid, num_chunks, chunk_nodes, chunk_reqs, chunk_buffer_sources, taskvs, chunk_command_vertices, buffer_writers);
		queue->mark_task_as_processed(tid);
	}

	// Update buffer regions
	for(auto it : buffer_writers) {
		const buffer_id bid = it.first;
		switch(valid_buffer_regions[bid]->get_dimensions()) {
		default:
		case 1: update_buffer_state(it.second, *dynamic_cast<detail::buffer_state<1>*>(valid_buffer_regions[bid].get())); break;
		case 2: update_buffer_state(it.second, *dynamic_cast<detail::buffer_state<2>*>(valid_buffer_regions[bid].get())); break;
		case 3: update_buffer_state(it.second, *dynamic_cast<detail::buffer_state<3>*>(valid_buffer_regions[bid].get())); break;
		}
	}

	// HACK: We recursively call this until all tasks have been processed
	// In the future, we may want to do this periodically in a worker thread
	if(graph_utils::get_satisfied_sibling_set(task_graph).size() > 0) { build_command_graph(); }
}

// Process writes and create pull / await-pull commands
void runtime::process_task_data_requirements(task_id tid, size_t num_chunks, const std::unordered_map<chunk_id, node_id>& chunk_nodes,
    const chunk_buffer_requirements_map& chunk_requirements, const chunk_buffer_source_map& chunk_buffer_sources,
    const std::unordered_map<task_id, graph_utils::task_vertices>& taskvs, const std::vector<vertex>& chunk_command_vertices,
    buffer_writers_map& buffer_writers) {
	for(auto i = 0u; i < num_chunks; ++i) {
		const node_id nid = chunk_nodes.at(i);

		for(auto& it : chunk_requirements.at(i)) {
			const buffer_id bid = it.first;
			int buffer_dimensions = 0;
			const vertex command_vertex = chunk_command_vertices.at(i);

			// ==== Writes ====
			if(it.second.count(cl::sycl::access::mode::write) > 0) {
				buffer_dimensions = it.second.at(cl::sycl::access::mode::write).which() + 1;

				switch(buffer_dimensions) {
				default:
				case 1: {
					const auto& write_req = boost::get<GridRegion<1>>(it.second.at(cl::sycl::access::mode::write));
					assert(write_req.area() > 0);
					buffer_writers[bid][nid].push_back(std::make_pair(tid, write_req));
					// Add to compute node label for debugging
					command_graph[command_vertex].label = fmt::format("{}\\nWrite {} {}", command_graph[command_vertex].label, bid, toString(write_req));
				} break;
				case 2: {
					const auto& write_req = boost::get<GridRegion<2>>(it.second.at(cl::sycl::access::mode::write));
					assert(write_req.area() > 0);
					buffer_writers[bid][nid].push_back(std::make_pair(tid, write_req));
					// Add to compute node label for debugging
					command_graph[command_vertex].label = fmt::format("{}\\nWrite {} {}", command_graph[command_vertex].label, bid, toString(write_req));
				} break;
				case 3: {
					const auto& write_req = boost::get<GridRegion<3>>(it.second.at(cl::sycl::access::mode::write));
					assert(write_req.area() > 0);
					buffer_writers[bid][nid].push_back(std::make_pair(tid, write_req));
					// Add to compute node label for debugging
					command_graph[command_vertex].label = fmt::format("{}\\nWrite {} {}", command_graph[command_vertex].label, bid, toString(write_req));
				} break;
				}
			}

			// ==== Reads ====
			// Add read to command node label for debugging
			if(it.second.count(cl::sycl::access::mode::read) > 0) {
				assert(buffer_dimensions == 0 || buffer_dimensions == it.second.at(cl::sycl::access::mode::read).which() + 1);
				buffer_dimensions = it.second.at(cl::sycl::access::mode::read).which() + 1;

				switch(buffer_dimensions) {
				default:
				case 1: {
					const auto& read_req = boost::get<GridRegion<1>>(it.second.at(cl::sycl::access::mode::read));
					assert(read_req.area() > 0);
					command_graph[command_vertex].label = fmt::format("{}\\nRead {} {}", command_graph[command_vertex].label, bid, toString(read_req));
				} break;
				case 2: {
					const auto& read_req = boost::get<GridRegion<2>>(it.second.at(cl::sycl::access::mode::read));
					assert(read_req.area() > 0);
					command_graph[command_vertex].label = fmt::format("{}\\nRead {} {}", command_graph[command_vertex].label, bid, toString(read_req));
				} break;
				case 3: {
					const auto& read_req = boost::get<GridRegion<3>>(it.second.at(cl::sycl::access::mode::read));
					assert(read_req.area() > 0);
					command_graph[command_vertex].label = fmt::format("{}\\nRead {} {}", command_graph[command_vertex].label, bid, toString(read_req));
				} break;
				}
			} else {
				continue;
			}

			const std::vector<std::pair<any_grid_box, std::unordered_set<node_id>>>& buffer_sources = chunk_buffer_sources.at(i).at(bid);

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
					bool intersects = false;
					switch(buffer_dimensions) {
					default:
					case 1:
						intersects = GridRegion<1>::intersect(boost::get<GridRegion<1>>(bw.second), GridRegion<1>(boost::get<GridBox<1>>(box))).area() > 0;
						break;
					case 2:
						intersects = GridRegion<2>::intersect(boost::get<GridRegion<2>>(bw.second), GridRegion<2>(boost::get<GridBox<2>>(box))).area() > 0;
						break;
					case 3:
						intersects = GridRegion<3>::intersect(boost::get<GridRegion<3>>(bw.second), GridRegion<3>(boost::get<GridBox<3>>(box))).area() > 0;
						break;
					}

					if(intersects) {
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
				const auto source_tv = has_writer ? taskvs.at(writer_tid) : taskvs.at(tid);

				// TODO: Update buffer regions since we copied some stuff!!
				switch(buffer_dimensions) {
				default:
				case 1:
					graph_utils::add_pull_cmd(nid, source_nid, bid, taskvs.at(tid), source_tv, command_vertex, boost::get<GridBox<1>>(box), command_graph);
					break;
				case 2:
					graph_utils::add_pull_cmd(nid, source_nid, bid, taskvs.at(tid), source_tv, command_vertex, boost::get<GridBox<2>>(box), command_graph);
					break;
				case 3:
					graph_utils::add_pull_cmd(nid, source_nid, bid, taskvs.at(tid), source_tv, command_vertex, boost::get<GridBox<3>>(box), command_graph);
					break;
				}
			}
		}
	}
}

void runtime::execute_master_access_task(task_id tid) const {
	const auto tsk = dynamic_cast<const master_access_task*>(queue->get_task(tid).get());
	master_access_livepass_handler handler;
	tsk->get_functor()(handler);
}

} // namespace celerity
