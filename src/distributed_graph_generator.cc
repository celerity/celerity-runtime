#include "distributed_graph_generator.h"

#include "access_modes.h"
#include "command.h"
#include "command_graph.h"
#include "task.h"
#include "task_manager.h"

#include "print_utils.h" // NOCOMMIT

namespace celerity::detail {

constexpr inline command_id no_command = command_id(-1);

void distributed_graph_generator::add_buffer(const buffer_id bid, const range<3>& range) {
	m_buffer_last_writer.emplace(bid, range);
	m_buffer_last_writer.at(bid).update_region(subrange_to_grid_box({id<3>(), range}), no_command);
	m_buffer_last_writer_task.emplace(bid, range);
}

// We simply split in the first dimension for now
static std::vector<chunk<3>> split_equal(const chunk<3>& full_chunk, const range<3>& granularity, const size_t num_chunks, const int dims) {
#ifndef NDEBUG
	assert(num_chunks > 0);
	for(int d = 0; d < dims; ++d) {
		assert(granularity[d] > 0);
		assert(full_chunk.range[d] % granularity[d] == 0);
	}
#endif

	// Due to split granularity requirements or if num_workers > global_size[0],
	// we may not be able to create the requested number of chunks.
	const auto actual_num_chunks = std::min(num_chunks, full_chunk.range[0] / granularity[0]);

	// If global range is not divisible by (actual_num_chunks * granularity),
	// assign ceil(quotient) to the first few chunks and floor(quotient) to the remaining
	const auto small_chunk_size_dim0 = full_chunk.range[0] / (actual_num_chunks * granularity[0]) * granularity[0];
	const auto large_chunk_size_dim0 = small_chunk_size_dim0 + granularity[0];
	const auto num_large_chunks = (full_chunk.range[0] - small_chunk_size_dim0 * actual_num_chunks) / granularity[0];
	assert(num_large_chunks * large_chunk_size_dim0 + (actual_num_chunks - num_large_chunks) * small_chunk_size_dim0 == full_chunk.range[0]);

	std::vector<chunk<3>> result(actual_num_chunks, {full_chunk.offset, full_chunk.range, full_chunk.global_size});
	for(auto i = 0u; i < num_large_chunks; ++i) {
		result[i].range[0] = large_chunk_size_dim0;
		result[i].offset[0] += i * large_chunk_size_dim0;
	}
	for(auto i = num_large_chunks; i < actual_num_chunks; ++i) {
		result[i].range[0] = small_chunk_size_dim0;
		result[i].offset[0] += num_large_chunks * large_chunk_size_dim0 + (i - num_large_chunks) * small_chunk_size_dim0;
	}

#ifndef NDEBUG
	size_t total_range_dim0 = 0;
	for(size_t i = 0; i < result.size(); ++i) {
		total_range_dim0 += result[i].range[0];
		if(i == 0) {
			assert(result[i].offset[0] == full_chunk.offset[0]);
		} else {
			assert(result[i].offset[0] == result[i - 1].offset[0] + result[i - 1].range[0]);
		}
	}
	assert(total_range_dim0 == full_chunk.range[0]);
#endif

	return result;
}

using buffer_requirements_map = std::unordered_map<buffer_id, std::unordered_map<access_mode, GridRegion<3>>>;

static buffer_requirements_map get_buffer_requirements_for_mapped_access(const task& tsk, subrange<3> sr, const range<3> global_size) {
	buffer_requirements_map result;
	const auto& access_map = tsk.get_buffer_access_map();
	const auto buffers = access_map.get_accessed_buffers();
	for(const buffer_id bid : buffers) {
		const auto modes = access_map.get_access_modes(bid);
		for(auto m : modes) {
			result[bid][m] = access_map.get_mode_requirements(bid, m, tsk.get_dimensions(), sr, global_size);
		}
	}
	return result;
}

#define DEBUG_PRINT(...) fmt::print(stderr, __VA_ARGS__)

// Steps:
// 1. Compute local chunk(s)
// 2. Compute data sources
// 3. Resolve data dependencies
//    => Avoid generating the same transfers twice (both across chunks and tasks)
//    => Could become tricky in conjunction with anti-dependency use-counters (= how to properly anticipate number of requests?)
// ?. Generate anti-dependencies ("use-counters"? "semaphores"?)
//    => Consider these additional use cases:
//       - Facilitate conflict resolution through copying
//       - Facilitate partial execution ("sub splits")
void distributed_graph_generator::build_task(const task& tsk) {
	DEBUG_PRINT("Processing task {}\n", tsk.get_id());

	if(tsk.get_type() == task_type::epoch) {
		DEBUG_PRINT("Generating epoch\n");
		generate_epoch_command(tsk);
		return;
	}

	if(tsk.get_type() != task_type::device_compute && tsk.get_type() != task_type::host_compute) return; // NOCOMMIT
	assert(tsk.has_variable_split()); // NOCOMMIT Not true for master tasks (TODO: rename single-node tasks? or just make it the default?)

	const size_t num_chunks = m_num_nodes; // TODO Make configurable

	// TODO: Pieced together from naive_split_transformer. We can probably do without creating all chunks and discarding everything except our own.
	// TODO: Or - maybe - we actually want to store all chunks somewhere b/c we'll probably need them frequently for lookups later on?
	chunk<3> full_chunk{tsk.get_global_offset(), tsk.get_global_size(), tsk.get_global_size()};
	const auto chunks = split_equal(full_chunk, tsk.get_granularity(), num_chunks, tsk.get_dimensions());
	assert(chunks.size() <= num_chunks); // We may have created less than requested
	assert(!chunks.empty());

	// Assign each chunk to a node
	// We assign chunks next to each other to the same worker (if there is more chunks than workers), as this is likely to produce less
	// transfers between tasks than a round-robin assignment (for typical stencil codes).
	// FIXME: This only works if the number of chunks is an integer multiple of the number of workers, e.g. 3 chunks for 2 workers degrades to RR.
	const auto chunks_per_node = std::max<size_t>(1, chunks.size() / m_num_nodes);

	for(size_t i = 0; i < chunks.size(); ++i) {
		assert(chunks[i].range.size() != 0);
		const node_id nid = (i / chunks_per_node) % m_num_nodes;
		if(nid == m_local_nid) {
			DEBUG_PRINT("Creating cmd for local chunk {}\n", chunks[i]);
			m_cdag.create<execution_command>(nid, tsk.get_id(), subrange{chunks[i]});
		}
	}

	////////////////////

	// Resolve data dependencies
	// 1. Query region map to find out which up-to-date regions we have
	//   - Q: Store "oudated" or "up-to-date" regions?
	// 2. For all outdated regions
	//   - Find last writer tasks
	//   - Apply inverse range mappers to find corresponding nodes
	//   - Generate data transfers
	//   - Update local region map
	// 3. Update region map to reflect buffer state after task

	const task_id tid = tsk.get_id();
	const auto task_commands = m_cdag.task_commands(tid);
	std::unordered_map<buffer_id, GridRegion<3>> per_buffer_local_writes;
	for(auto* cmd : task_commands) {
		const command_id cid = cmd->get_cid();
		assert(isa<execution_command>(cmd));
		auto* ecmd = static_cast<execution_command*>(cmd);
		auto requirements = get_buffer_requirements_for_mapped_access(tsk, ecmd->get_execution_range(), tsk.get_global_size());

		for(auto& [bid, reqs_by_mode] : requirements) {
			auto& last_writer_cmd = m_buffer_last_writer.at(bid);
			auto& last_writer_task = m_buffer_last_writer_task.at(bid); // NOCOMMIT Naming

			std::vector<access_mode> required_modes;
			for(const auto mode : detail::access::all_modes) {
				if(auto req_it = reqs_by_mode.find(mode); req_it != reqs_by_mode.end()) {
					// While uncommon, we do support chunks that don't require access to a particular buffer at all.
					if(!req_it->second.empty()) { required_modes.push_back(mode); }
				}
			}

			for(const auto mode : required_modes) {
				const auto& req = reqs_by_mode.at(mode);
				if(detail::access::mode_traits::is_consumer(mode)) {
					const auto local_sources = last_writer_cmd.get_region_values(req);
					GridRegion<3> missing_parts;
					for(const auto& [box, cid] : local_sources) {
						if(cid == no_command) {
							missing_parts = GridRegion<3>::merge(missing_parts, box);
							continue;
						}
						m_cdag.add_dependency(cmd, m_cdag.get(cid), dependency_kind::true_dep, dependency_origin::dataflow);
					}

					const auto task_sources = last_writer_task.get_region_values(missing_parts);
					for(const auto& [box, tid] : task_sources) {
						assert(m_task_mngr.has_task(tid));
						const auto& tsk = *m_task_mngr.get_task(tid);

						///////////////////////// DRY THIS UP ////////////////////////////////

						chunk<3> full_chunk{tsk.get_global_offset(), tsk.get_global_size(), tsk.get_global_size()};
						const auto chunks = split_equal(full_chunk, tsk.get_granularity(), num_chunks, tsk.get_dimensions());
						const auto chunks_per_node = std::max<size_t>(1, chunks.size() / m_num_nodes);
						for(size_t i = 0; i < chunks.size(); ++i) {
							assert(chunks[i].range.size() != 0);
							[[maybe_unused]] const node_id nid = (i / chunks_per_node) % m_num_nodes;
							// NOCOMMIT: We conceptually assume a 1-to-1 access mode (which is required currently, but we should still codify this).
							const auto overlap =
							    GridBox<3>::intersect(subrange_to_grid_box(chunks[i]), box); // NOCOMMIT TODO: Wait, couldn't this be a region..?
							if(!overlap.empty()) {
								assert(nid != m_local_nid);
								auto dr_cmd = m_cdag.create<data_request_command>(m_local_nid, bid, nid, grid_box_to_subrange(overlap));
								m_cdag.add_dependency(cmd, dr_cmd, dependency_kind::true_dep, dependency_origin::dataflow);
								// Remember that we have this data now
								last_writer_cmd.update_region(overlap, dr_cmd->get_cid());
							}
						}

						///////////////////////// DRY THIS UP ////////////////////////////////
					}
				}

				if(detail::access::mode_traits::is_producer(mode)) {
					// NOCOMMIT Remember to not create intra-task anti-dependencies onto data requests for RW accesses
					// NOCOMMIT Use update list to make RW accesses work
					last_writer_cmd.update_region(req, cid);
					per_buffer_local_writes[bid] = GridRegion<3>::merge(per_buffer_local_writes[bid], req);
				}
			}
		}
	}

	// Update task-level buffer states
	auto requirements = get_buffer_requirements_for_mapped_access(tsk, subrange<3>(id<3>{}, tsk.get_global_size()), tsk.get_global_size());
	for(auto& [bid, reqs_by_mode] : requirements) {
		GridRegion<3> global_writes;
		for(const auto mode : access::producer_modes) {
			if(reqs_by_mode.count(mode) == 0) continue;
			global_writes = GridRegion<3>::merge(global_writes, reqs_by_mode.at(mode));
		}
		const auto& local_writes = per_buffer_local_writes[bid];
		const auto remote_writes = GridRegion<3>::difference(global_writes, local_writes);
		m_buffer_last_writer_task.at(bid).update_region(global_writes, tsk.get_id());
		m_buffer_last_writer.at(bid).update_region(remote_writes, no_command);
	}
}

void distributed_graph_generator::reduce_execution_front_to(abstract_command* const new_front) {
	const auto nid = new_front->get_nid();
	const auto previous_execution_front = m_cdag.get_execution_front(nid);
	for(const auto front_cmd : previous_execution_front) {
		if(front_cmd != new_front) { m_cdag.add_dependency(new_front, front_cmd, dependency_kind::true_dep, dependency_origin::execution_front); }
	}
	assert(m_cdag.get_execution_front(nid).size() == 1 && *m_cdag.get_execution_front(nid).begin() == new_front);
}

void distributed_graph_generator::generate_epoch_command(const task& tsk) {
	assert(tsk.get_type() == task_type::epoch);
	const auto epoch = m_cdag.create<epoch_command>(m_local_nid, tsk.get_id(), tsk.get_epoch_action());
	// Make the epoch depend on the previous execution front
	reduce_execution_front_to(epoch);
}

} // namespace celerity::detail