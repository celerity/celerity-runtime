#include "distributed_graph_generator.h"

#include "access_modes.h"
#include "command.h"
#include "command_graph.h"
#include "task.h"
#include "task_manager.h"

#include "print_utils.h" // NOCOMMIT

namespace celerity::detail {

distributed_graph_generator::distributed_graph_generator(const size_t num_nodes, const node_id local_nid, command_graph& cdag, const task_manager& tm)
    : m_num_nodes(num_nodes), m_local_nid(local_nid), m_cdag(cdag), m_task_mngr(tm) {
	if(m_num_nodes > max_num_nodes) {
		throw std::runtime_error(fmt::format("Number of nodes requested ({}) exceeds compile-time maximum of {}", m_num_nodes, max_num_nodes));
	}

	// Build initial epoch command (this is required to properly handle anti-dependencies on host-initialized buffers).
	// We manually generate the first command, this will be replaced by applied horizons or explicit epochs down the line (see
	// set_epoch_for_new_commands).
	const auto epoch_cmd = cdag.create<epoch_command>(m_local_nid, task_manager::initial_epoch_task, epoch_action::none);
	epoch_cmd->mark_as_flushed(); // there is no point in flushing the initial epoch command
	m_epoch_for_new_commands = epoch_cmd->get_cid();
}

void distributed_graph_generator::add_buffer(const buffer_id bid, const range<3>& range, int dims) {
#if USE_COOL_REGION_MAP
	m_buffer_states.emplace(
	    std::piecewise_construct, std::tuple{bid}, std::tuple{region_map_t<write_command_state>{range, dims}, region_map_t<node_bitset>{range, dims}});
#else
	m_buffer_states.try_emplace(bid, buffer_state{range, range});
#endif
	m_buffer_states.at(bid).local_last_writer.update_region(subrange_to_grid_box({id<3>(), range}), no_command);
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
	const auto epoch_to_prune_before = m_epoch_for_new_commands;

	if(tsk.get_type() == task_type::epoch) {
		generate_epoch_command(tsk);
	} else if(tsk.get_type() == task_type::horizon) {
		generate_horizon_command(tsk);
	} else if(tsk.get_type() == task_type::device_compute || tsk.get_type() == task_type::host_compute || tsk.get_type() == task_type::master_node) {
		generate_execution_commands(tsk);
	} else {
		throw std::runtime_error("Task type NYI");
	}

	// Commands without any other true-dependency must depend on the active epoch command to ensure they cannot be re-ordered before the epoch.
	// Need to check count b/c for some tasks we may not have generated any commands locally.
	if(m_cdag.task_command_count(tsk.get_id()) > 0) {
		for(const auto cmd : m_cdag.task_commands(tsk.get_id())) {
			generate_epoch_dependencies(cmd);
		}
	}

	// If a new epoch was completed in the CDAG before the current task, prune all predecessor commands of that epoch.
	prune_commands_before(epoch_to_prune_before);
}

void distributed_graph_generator::generate_execution_commands(const task& tsk) {
	// TODO: Pieced together from naive_split_transformer. We can probably do without creating all chunks and discarding everything except our own.
	// TODO: Or - maybe - we actually want to store all chunks somewhere b/c we'll probably need them frequently for lookups later on?
	chunk<3> full_chunk{tsk.get_global_offset(), tsk.get_global_size(), tsk.get_global_size()};
	const size_t num_chunks = m_num_nodes; // TODO Make configurable
	const auto chunks = ([&] {
		if(tsk.has_variable_split()) {
			return split_equal(full_chunk, tsk.get_granularity(), num_chunks, tsk.get_dimensions());
		} else {
			return std::vector<chunk<3>>{full_chunk};
		}
	})();
	assert(chunks.size() <= num_chunks); // We may have created less than requested
	assert(!chunks.empty());

	// Assign each chunk to a node
	// We assign chunks next to each other to the same worker (if there is more chunks than workers), as this is likely to produce less
	// transfers between tasks than a round-robin assignment (for typical stencil codes).
	// FIXME: This only works if the number of chunks is an integer multiple of the number of workers, e.g. 3 chunks for 2 workers degrades to RR.
	const auto chunks_per_node = std::max<size_t>(1, chunks.size() / m_num_nodes);

	// Distributed push model:
	// - Iterate over all remote chunks and find read requirements intersecting with owned buffer regions.
	// 	 Generate push commands for those regions.
	// - Iterate over all local chunks and find read requirements on remote data.
	//   Generate single await push command for each buffer that contains the entire region (will be fulfilled by one or more pushes).

	std::unordered_map<buffer_id, GridRegion<3>> per_buffer_local_writes;
	for(size_t i = 0; i < chunks.size(); ++i) {
		const node_id nid = (i / chunks_per_node) % m_num_nodes;
		const bool is_local_chunk = nid == m_local_nid;

		auto requirements = get_buffer_requirements_for_mapped_access(tsk, chunks[i], tsk.get_global_size());

		execution_command* cmd = nullptr;
		if(is_local_chunk) { cmd = m_cdag.create<execution_command>(nid, tsk.get_id(), subrange{chunks[i]}); }

		for(auto& [bid, reqs_by_mode] : requirements) {
			auto& buffer_state = m_buffer_states.at(bid);
			// For "true writes" (= not replicated) we have to wait with updating the last writer
			// map until all modes have been processed, as we'll otherwise end up with a cycle in
			// the graph if a command both writes and reads the same buffer region.
			GridRegion<3> written_region;

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
					if(is_local_chunk) {
						// Store the read access for determining anti-dependencies later on
						m_command_buffer_reads[cmd->get_cid()][bid] = GridRegion<3>::merge(m_command_buffer_reads[cmd->get_cid()][bid], req);

						const auto local_sources = buffer_state.local_last_writer.get_region_values(req);
						GridRegion<3> missing_parts;
						for(const auto& [box, wcs] : local_sources) {
							if(!wcs.is_fresh()) {
								missing_parts = GridRegion<3>::merge(missing_parts, box);
								continue;
							}
							// NEXT STEP: It seems like we are adding a dependency on the write access in same task.
							// => We'll probably need an update list after all...
							// Q: Does this even make sense? Why is the read access overlapping with the write? Check range mappers
							m_cdag.add_dependency(cmd, m_cdag.get(wcs), dependency_kind::true_dep, dependency_origin::dataflow);
						}

						// There is data we don't yet have locally. Generate an await push command for it.
						if(!missing_parts.empty()) {
							// We simply use the task ID to, together with the buffer id, uniquely identify all pushes corresponding to this await push
							const transfer_id trid = static_cast<transfer_id>(tsk.get_id());
							auto ap_cmd = m_cdag.create<await_push_command>(m_local_nid, bid, trid, missing_parts);
							m_cdag.add_dependency(cmd, ap_cmd, dependency_kind::true_dep, dependency_origin::dataflow);
							generate_anti_dependencies(tsk.get_id(), bid, buffer_state.local_last_writer, missing_parts, ap_cmd);
							generate_epoch_dependencies(ap_cmd);
							// Remember that we have this data now
							buffer_state.local_last_writer.update_region(missing_parts, {ap_cmd->get_cid(), true});
						}
					} else {
						const auto local_sources = buffer_state.local_last_writer.get_region_values(req);
						for(const auto& [local_box, wcs] : local_sources) {
							if(!wcs.is_fresh() || wcs.is_replicated()) { continue; }

							// Check if we've already pushed this box
							const auto replicated_boxes = buffer_state.replicated_regions.get_region_values(local_box);
							for(const auto& [replicated_box, nodes] : replicated_boxes) {
								if(nodes.test(nid)) continue;

								// Generate separate PUSH command for each last writer command for now,
								// possibly even multiple for partially already-replicated data
								// TODO: Can we consolidate?
								const transfer_id trid = static_cast<transfer_id>(tsk.get_id());
								auto push_cmd = m_cdag.create<push_command>(m_local_nid, bid, 0, nid, trid, grid_box_to_subrange(replicated_box));
								m_cdag.add_dependency(push_cmd, m_cdag.get(wcs), dependency_kind::true_dep, dependency_origin::dataflow);

								// Store the read access for determining anti-dependencies later on
								m_command_buffer_reads[push_cmd->get_cid()][bid] = replicated_box;

								// Remember that we've replicated this region
								buffer_state.replicated_regions.update_box(replicated_box, node_bitset{nodes}.set(nid));
							}
						}
					}
				}

				if(is_local_chunk && detail::access::mode_traits::is_producer(mode)) {
					generate_anti_dependencies(tsk.get_id(), bid, buffer_state.local_last_writer, req, cmd);

					// NOCOMMIT Remember to not create intra-task anti-dependencies onto data requests for RW accesses
					written_region = GridRegion<3>::merge(written_region, req);
					per_buffer_local_writes[bid] = GridRegion<3>::merge(per_buffer_local_writes[bid], req);
				}
			}

			if(!written_region.empty()) {
				buffer_state.local_last_writer.update_region(written_region, cmd->get_cid());
				buffer_state.replicated_regions.update_region(written_region, node_bitset{});
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
		auto& buffer_state = m_buffer_states.at(bid);

		// TODO: We need a way of updating regions in place!
		auto boxes_and_cids = buffer_state.local_last_writer.get_region_values(remote_writes);
		for(auto& [box, wcs] : boxes_and_cids) {
			if(wcs.is_fresh()) {
				wcs.mark_as_stale();
				buffer_state.local_last_writer.update_region(box, wcs);
			}
		}
	}
}

void distributed_graph_generator::generate_anti_dependencies(
    task_id tid, buffer_id bid, const region_map_t<write_command_state>& last_writers_map, const GridRegion<3>& write_req, abstract_command* write_cmd) {
	const auto last_writers = last_writers_map.get_region_values(write_req);
	for(auto& box_and_writers : last_writers) {
		// FIXME: This is ugly. Region maps should be able to store sparse entries.
		if(box_and_writers.second == no_command) continue;
		const command_id last_writer_cid = box_and_writers.second;
		const auto last_writer_cmd = m_cdag.get(last_writer_cid);
		assert(!isa<task_command>(last_writer_cmd) || static_cast<task_command*>(last_writer_cmd)->get_tid() != tid);

		// Add anti-dependencies onto all dependents of the writer
		bool has_dependents = false;
		for(auto d : last_writer_cmd->get_dependents()) {
			// Only consider true dependencies
			if(d.kind != dependency_kind::true_dep) continue;

			const auto cmd = d.node;

			// We might have already generated new commands within the same task that also depend on this; in that case, skip it
			if(isa<task_command>(cmd) && static_cast<task_command*>(cmd)->get_tid() == tid) continue;

			// So far we don't know whether the dependent actually intersects with the subrange we're writing
			if(const auto command_reads_it = m_command_buffer_reads.find(cmd->get_cid()); command_reads_it != m_command_buffer_reads.end()) {
				const auto& command_reads = command_reads_it->second;
				// The task might be a dependent because of another buffer
				if(const auto buffer_reads_it = command_reads.find(bid); buffer_reads_it != command_reads.end()) {
					if(!GridRegion<3>::intersect(write_req, buffer_reads_it->second).empty()) {
						has_dependents = true;
						m_cdag.add_dependency(write_cmd, cmd, dependency_kind::anti_dep, dependency_origin::dataflow);
					}
				}
			}
		}

		// In some cases (horizons, master node host task, weird discard_* constructs...)
		// the last writer might not have any dependents. Just add the anti-dependency onto the writer itself then.
		if(!has_dependents) {
			m_cdag.add_dependency(write_cmd, last_writer_cmd, dependency_kind::anti_dep, dependency_origin::dataflow);

			// This is a good time to validate our assumption that every await_push command has a dependent
			assert(!isa<await_push_command>(last_writer_cmd));
		}
	}
}

void distributed_graph_generator::set_epoch_for_new_commands(const abstract_command* const epoch_or_horizon) {
	// both an explicit epoch command and an applied horizon can be effective epochs
	assert(isa<epoch_command>(epoch_or_horizon) || isa<horizon_command>(epoch_or_horizon));

	for(auto& [bid, bs] : m_buffer_states) {
		// TODO this could be optimized to something like cdag.apply_horizon(node_id, horizon_cmd) with much fewer internal operations
		bs.local_last_writer.apply_to_values([epoch_or_horizon](const write_command_state& wcs) {
			if(wcs == no_command) return wcs;
			auto new_wcs = write_command_state(std::max(epoch_or_horizon->get_cid(), static_cast<command_id>(wcs)), wcs.is_replicated());
			if(!wcs.is_fresh()) new_wcs.mark_as_stale();
			return new_wcs;
		});
	}

	m_epoch_for_new_commands = epoch_or_horizon->get_cid();
}

void distributed_graph_generator::reduce_execution_front_to(abstract_command* const new_front) {
	const auto nid = new_front->get_nid();
	const auto previous_execution_front = m_cdag.get_execution_front(nid);
	for(const auto front_cmd : previous_execution_front) {
		if(front_cmd != new_front) { m_cdag.add_dependency(new_front, front_cmd, dependency_kind::true_dep, dependency_origin::execution_front); }
	}
	assert(m_cdag.get_execution_front(nid).size() == 1 && *m_cdag.get_execution_front(nid).begin() == new_front);
}

// NOCOMMIT TODO: Apply epoch to data structures
void distributed_graph_generator::generate_epoch_command(const task& tsk) {
	assert(tsk.get_type() == task_type::epoch);
	const auto epoch = m_cdag.create<epoch_command>(m_local_nid, tsk.get_id(), tsk.get_epoch_action());
	set_epoch_for_new_commands(epoch);
	m_current_horizon = no_command;
	// Make the epoch depend on the previous execution front
	reduce_execution_front_to(epoch);
}

// NOCOMMIT TODO: Apply previous horizon to data structures
void distributed_graph_generator::generate_horizon_command(const task& tsk) {
	assert(tsk.get_type() == task_type::horizon);
	const auto horizon = m_cdag.create<horizon_command>(m_local_nid, tsk.get_id());

	if(m_current_horizon != static_cast<command_id>(no_command)) {
		// Apply the previous horizon
		set_epoch_for_new_commands(m_cdag.get(m_current_horizon));
	}
	m_current_horizon = horizon->get_cid();

	// Make the horizon depend on the previous execution front
	reduce_execution_front_to(horizon);
}

void distributed_graph_generator::generate_epoch_dependencies(abstract_command* cmd) {
	// No command must be re-ordered before its last preceding epoch to enforce the barrier semantics of epochs.
	// To guarantee that each node has a transitive true dependency (=temporal dependency) on the epoch, it is enough to add an epoch -> command dependency
	// to any command that has no other true dependencies itself and no graph traversal is necessary. This can be verified by a simple induction proof.

	// As long the first epoch is present in the graph, all transitive dependencies will be visible and the initial epoch commands (tid 0) are the only
	// commands with no true predecessor. As soon as the first epoch is pruned through the horizon mechanism however, more than one node with no true
	// predecessor can appear (when visualizing the graph). This does not violate the ordering constraint however, because all "free-floating" nodes
	// in that snapshot had a true-dependency chain to their predecessor epoch at the point they were flushed, which is sufficient for following the
	// dependency chain from the executor perspective.

	if(const auto deps = cmd->get_dependencies();
	    std::none_of(deps.begin(), deps.end(), [](const abstract_command::dependency d) { return d.kind == dependency_kind::true_dep; })) {
		assert(cmd->get_cid() != m_epoch_for_new_commands);
		m_cdag.add_dependency(cmd, m_cdag.get(m_epoch_for_new_commands), dependency_kind::true_dep, dependency_origin::last_epoch);
	}
}

void distributed_graph_generator::prune_commands_before(const command_id epoch) {
	if(epoch > m_epoch_last_pruned_before) {
		m_cdag.erase_if([&](abstract_command* cmd) {
			if(cmd->get_cid() < epoch) {
				m_command_buffer_reads.erase(cmd->get_cid());
				return true;
			}
			return false;
		});
		m_epoch_last_pruned_before = epoch;
	}
}

} // namespace celerity::detail