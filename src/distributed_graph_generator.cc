#include "distributed_graph_generator.h"

#include "access_modes.h"
#include "command.h"
#include "command_graph.h"
#include "recorders.h"
#include "task.h"
#include "task_manager.h"

namespace celerity::detail {

distributed_graph_generator::distributed_graph_generator(
    const size_t num_nodes, const node_id local_nid, command_graph& cdag, const task_manager& tm, detail::command_recorder* recorder)
    : m_num_nodes(num_nodes), m_local_nid(local_nid), m_cdag(cdag), m_task_mngr(tm), m_recorder(recorder) {
	if(m_num_nodes > max_num_nodes) {
		throw std::runtime_error(fmt::format("Number of nodes requested ({}) exceeds compile-time maximum of {}", m_num_nodes, max_num_nodes));
	}

	// Build initial epoch command (this is required to properly handle anti-dependencies on host-initialized buffers).
	// We manually generate the first command, this will be replaced by applied horizons or explicit epochs down the line (see
	// set_epoch_for_new_commands).
	auto* const epoch_cmd = cdag.create<epoch_command>(task_manager::initial_epoch_task, epoch_action::none);
	epoch_cmd->mark_as_flushed(); // there is no point in flushing the initial epoch command
	if(m_recorder != nullptr) m_recorder->record_command(*epoch_cmd);
	m_epoch_for_new_commands = epoch_cmd->get_cid();
}

void distributed_graph_generator::add_buffer(const buffer_id bid, const int dims, const range<3>& range) {
	m_buffer_states.emplace(
	    std::piecewise_construct, std::tuple{bid}, std::tuple{region_map<write_command_state>{range, dims}, region_map<node_bitset>{range, dims}});
	// Mark contents as available locally (= don't generate await push commands) and fully replicated (= don't generate push commands).
	// This is required when tasks access host-initialized or uninitialized buffers.
	m_buffer_states.at(bid).local_last_writer.update_region(subrange<3>({}, range), m_epoch_for_new_commands);
	m_buffer_states.at(bid).replicated_regions.update_region(subrange<3>({}, range), node_bitset{}.set());
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

using buffer_requirements_map = std::unordered_map<buffer_id, std::unordered_map<access_mode, region<3>>>;

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

std::unordered_set<abstract_command*> distributed_graph_generator::build_task(const task& tsk) {
	assert(m_current_cmd_batch.empty());
	[[maybe_unused]] const auto cmd_count_before = m_cdag.command_count();

	const auto epoch_to_prune_before = m_epoch_for_new_commands;

	switch(tsk.get_type()) {
	case task_type::epoch: generate_epoch_command(tsk); break;
	case task_type::horizon: generate_horizon_command(tsk); break;
	case task_type::device_compute:
	case task_type::host_compute:
	case task_type::master_node:
	case task_type::collective:
	case task_type::fence: generate_distributed_commands(tsk); break;
	default: throw std::runtime_error("Task type NYI");
	}

	// It is currently undefined to split reduction-producer tasks into multiple chunks on the same node:
	//   - Per-node reduction intermediate results are stored with fixed access to a single backing buffer,
	//     so multiple chunks on the same node will race on this buffer access
	//   - Inputs to the final reduction command are ordered by origin node ids to guarantee bit-identical results. It is not possible to distinguish
	//     more than one chunk per node in the serialized commands, so different nodes can produce different final reduction results for non-associative
	//     or non-commutative operations
	if(!tsk.get_reductions().empty()) { assert(m_cdag.task_command_count(tsk.get_id()) <= 1); }

	// Commands without any other true-dependency must depend on the active epoch command to ensure they cannot be re-ordered before the epoch.
	// Need to check count b/c for some tasks we may not have generated any commands locally.
	if(m_cdag.task_command_count(tsk.get_id()) > 0) {
		for(auto* const cmd : m_cdag.task_commands(tsk.get_id())) {
			generate_epoch_dependencies(cmd);
		}
	}

	// Check that all commands have been created through create_command
	assert(m_cdag.command_count() - cmd_count_before == m_current_cmd_batch.size());

	// If a new epoch was completed in the CDAG before the current task, prune all predecessor commands of that epoch.
	prune_commands_before(epoch_to_prune_before);

	// If we have a command_recorder, record the current batch of commands
	if(m_recorder != nullptr) {
		for(const auto& cmd : m_current_cmd_batch) {
			m_recorder->record_command(*cmd);
		}
	}

	return std::move(m_current_cmd_batch);
}

void distributed_graph_generator::generate_distributed_commands(const task& tsk) {
	chunk<3> full_chunk{tsk.get_global_offset(), tsk.get_global_size(), tsk.get_global_size()};
	const size_t num_chunks = m_num_nodes * 1; // TODO Make configurable
	const auto chunks = ([&] {
		if(tsk.get_type() == task_type::collective || tsk.get_type() == task_type::fence) {
			std::vector<chunk<3>> chunks;
			for(size_t nid = 0; nid < m_num_nodes; ++nid) {
				chunks.push_back(chunk_cast<3>(chunk<1>{id<1>{tsk.get_type() == task_type::collective ? nid : 0}, ones, {m_num_nodes}}));
			}
			return chunks;
		}
		if(tsk.has_variable_split()) { return split_equal(full_chunk, tsk.get_granularity(), num_chunks, tsk.get_dimensions()); }
		return std::vector<chunk<3>>{full_chunk};
	})();
	assert(chunks.size() <= num_chunks); // We may have created less than requested
	assert(!chunks.empty());

	// Assign each chunk to a node
	// We assign chunks next to each other to the same worker (if there is more chunks than workers), as this is likely to produce less
	// transfers between tasks than a round-robin assignment (for typical stencil codes).
	// FIXME: This only works if the number of chunks is an integer multiple of the number of workers, e.g. 3 chunks for 2 workers degrades to RR.
	const auto chunks_per_node = std::max<size_t>(1, chunks.size() / m_num_nodes);

	// Union of all per-buffer writes on this node, used to determine which parts of a buffer are fresh/stale later on.
	std::unordered_map<buffer_id, region<3>> per_buffer_local_writes;
	// In case we need to push a region that is overwritten in the same task, we have to defer updating the last writer.
	std::unordered_map<buffer_id, std::vector<std::pair<region<3>, command_id>>> per_buffer_last_writer_update_list;
	// Buffers that currently are in a pending reduction state will receive a new buffer state after a reduction has been generated.
	std::unordered_map<buffer_id, buffer_state> post_reduction_buffer_states;

	// Remember all generated pushes for determining intra-task anti-dependencies.
	std::vector<push_command*> generated_pushes;

	// In the master/worker model, we used to try and find the node best suited for initializing multiple
	// reductions that do not initialize_to_identity based on current data distribution.
	// This is more difficult in a distributed setting, so for now we just hard code it to node 0.
	// TODO: Revisit this at some point.
	const node_id reduction_initializer_nid = 0;

	const box<3> empty_box({0, 0, 0}, {0, 0, 0});
	const box<3> scalar_box({0, 0, 0}, {1, 1, 1});

	// Iterate over all chunks, distinguish between local / remote chunks and normal / reduction access.
	//
	// Normal buffer access:
	// - For local chunks, find read requirements on remote data.
	//   Generate a single await push command for each buffer that awaits the entire required region.
	//   This will then be fulfilled by one or more incoming pushes.
	// - For remote chunks, find read requirements intersecting with owned buffer regions.
	// 	 Generate push commands for those regions.
	//
	// Reduction buffer access:
	// - For local chunks, create a reduction command and a single await_push command that receives the
	//   partial reduction results from all other nodes.
	// - For remote chunks, always create a push command, regardless of whether we have relevant data or not.
	//   This is required because the remote node does not know how many partial reduction results there are.
	for(size_t i = 0; i < chunks.size(); ++i) {
		const node_id nid = (i / chunks_per_node) % m_num_nodes;
		const bool is_local_chunk = nid == m_local_nid;

		auto requirements = get_buffer_requirements_for_mapped_access(tsk, chunks[i], tsk.get_global_size());

		// Add requirements for reductions
		for(const auto& reduction : tsk.get_reductions()) {
			auto rmode = access_mode::discard_write;
			if(nid == reduction_initializer_nid && reduction.init_from_buffer) { rmode = access_mode::read_write; }
#ifndef NDEBUG
			for(auto pmode : access::producer_modes) {
				assert(requirements[reduction.bid].count(pmode) == 0); // task_manager verifies that there are no reduction <-> write-access conflicts
			}
#endif
			requirements[reduction.bid][rmode] = scalar_box;
		}

		abstract_command* cmd = nullptr;
		if(is_local_chunk) {
			if(tsk.get_type() == task_type::fence) {
				cmd = create_command<fence_command>(tsk.get_id());
			} else {
				cmd = create_command<execution_command>(tsk.get_id(), subrange{chunks[i]});

				// Go over all reductions that are to be performed *during* the execution of this chunk,
				// not to be confused with any pending reductions that need to be finalized *before* the
				// execution of this chunk.
				// If a reduction reads the previous value of the buffer (i.e. w/o property::reduction::initialize_to_identity),
				// we have to include it in exactly one of the per-node intermediate reductions.
				for(const auto& reduction : tsk.get_reductions()) {
					if(nid == reduction_initializer_nid && reduction.init_from_buffer) {
						utils::as<execution_command>(cmd)->set_is_reduction_initializer(true);
						break;
					}
				}
			}

			if(tsk.get_type() == task_type::collective) {
				// Collective host tasks have an implicit dependency on the previous task in the same collective group,
				// which is required in order to guarantee they are executed in the same order on every node.
				auto cgid = tsk.get_collective_group_id();
				if(auto prev = m_last_collective_commands.find(cgid); prev != m_last_collective_commands.end()) {
					m_cdag.add_dependency(cmd, m_cdag.get(prev->second), dependency_kind::true_dep, dependency_origin::collective_group_serialization);
					m_last_collective_commands.erase(prev);
				}
				m_last_collective_commands.emplace(cgid, cmd->get_cid());
			}
		}

		// We use the task id, together with the "chunk id" and the buffer id (stored separately) to match pushes against their corresponding await pushes
		const transfer_id trid = static_cast<transfer_id>((tsk.get_id() << 32) | i);
		for(auto& [bid, reqs_by_mode] : requirements) {
			auto& buffer_state = m_buffer_states.at(bid);
			std::vector<access_mode> required_modes;
			for(const auto mode : detail::access::all_modes) {
				if(auto req_it = reqs_by_mode.find(mode); req_it != reqs_by_mode.end()) {
					// While uncommon, we do support chunks that don't require access to a particular buffer at all.
					if(!req_it->second.empty()) { required_modes.push_back(mode); }
				}
			}

			// Don't add reduction commands within the loop to make sure there is at most one reduction command
			// even in the presence of multiple consumer requirements.
			const bool is_pending_reduction = buffer_state.pending_reduction.has_value();
			const bool generate_reduction =
			    is_pending_reduction && std::any_of(required_modes.begin(), required_modes.end(), detail::access::mode_traits::is_consumer);
			if(generate_reduction) {
				// Prepare the buffer state for after the reduction has been performed:
				// Set the current epoch as last writer and mark it as stale so that if we don't generate a reduction command,
				// we'll know to get the data from elsewhere. If we generate a reduction command, this will be overwritten by its command id.
				write_command_state wcs{m_epoch_for_new_commands};
				wcs.mark_as_stale();
				// We just treat this buffer as 1-dimensional, regardless of its actual dimensionality (as it must be unit-sized anyway)
				post_reduction_buffer_states.emplace(std::piecewise_construct, std::tuple{bid},
				    std::tuple{region_map<write_command_state>{ones, 1, wcs}, region_map<node_bitset>{ones, 1, node_bitset{}}});
			}

			if(is_pending_reduction && !generate_reduction) {
				// TODO the per-node reduction result is discarded - warn user about dead store
			}

			for(const auto mode : required_modes) {
				const auto& req = reqs_by_mode.at(mode);
				if(detail::access::mode_traits::is_consumer(mode)) {
					if(is_local_chunk) {
						// Store the read access for determining anti-dependencies later on
						m_command_buffer_reads[cmd->get_cid()][bid] = region_union(m_command_buffer_reads[cmd->get_cid()][bid], req);
					}

					if(is_local_chunk && !is_pending_reduction) {
						const auto local_sources = buffer_state.local_last_writer.get_region_values(req);
						std::vector<box<3>> missing_part_boxes;
						for(const auto& [box, wcs] : local_sources) {
							if(box.empty()) continue;
							if(!wcs.is_fresh()) {
								missing_part_boxes.push_back(box);
								continue;
							}
							m_cdag.add_dependency(cmd, m_cdag.get(wcs), dependency_kind::true_dep, dependency_origin::dataflow);
						}

						// There is data we don't yet have locally. Generate an await push command for it.
						if(!missing_part_boxes.empty()) {
							const region missing_parts(std::move(missing_part_boxes));
							assert(m_num_nodes > 1);
							auto* const ap_cmd = create_command<await_push_command>(bid, 0, trid, missing_parts);
							m_cdag.add_dependency(cmd, ap_cmd, dependency_kind::true_dep, dependency_origin::dataflow);
							generate_anti_dependencies(tsk.get_id(), bid, buffer_state.local_last_writer, missing_parts, ap_cmd);
							generate_epoch_dependencies(ap_cmd);
							// Remember that we have this data now
							buffer_state.local_last_writer.update_region(missing_parts, {ap_cmd->get_cid(), true /* is_replicated */});
						}
					} else if(!is_pending_reduction) {
						const auto local_sources = buffer_state.local_last_writer.get_region_values(req);
						for(const auto& [local_box, wcs] : local_sources) {
							if(!wcs.is_fresh() || wcs.is_replicated()) { continue; }

							// Check if we've already pushed this box
							const auto replicated_boxes = buffer_state.replicated_regions.get_region_values(local_box);
							for(const auto& [replicated_box, nodes] : replicated_boxes) {
								if(nodes.test(nid)) continue;

								// Generate separate push command for each last writer command for now,
								// possibly even multiple for partially already-replicated data.
								// TODO: Can and/or should we consolidate?
								auto* const push_cmd = create_command<push_command>(bid, 0, nid, trid, replicated_box.get_subrange());
								assert(!utils::isa<await_push_command>(m_cdag.get(wcs)) && "Attempting to push non-owned data?!");
								m_cdag.add_dependency(push_cmd, m_cdag.get(wcs), dependency_kind::true_dep, dependency_origin::dataflow);
								generated_pushes.push_back(push_cmd);

								// Store the read access for determining anti-dependencies later on
								m_command_buffer_reads[push_cmd->get_cid()][bid] = replicated_box;

								// Remember that we've replicated this region
								buffer_state.replicated_regions.update_box(replicated_box, node_bitset{nodes}.set(nid));
							}
						}
					}
				}

				if(is_local_chunk && detail::access::mode_traits::is_producer(mode)) {
					// If we are going to insert a reduction command, we will also create a true-dependency chain to the last writer. The new last writer
					// cid however is not known at this point because the the reduction command has not been generated yet. Instead, we simply skip
					// generating anti-dependencies around this requirement. This might not be valid if (multivariate) reductions ever operate on regions.
					if(!generate_reduction) { generate_anti_dependencies(tsk.get_id(), bid, buffer_state.local_last_writer, req, cmd); }

					per_buffer_local_writes[bid] = region_union(per_buffer_local_writes[bid], req);
					per_buffer_last_writer_update_list[bid].push_back({req, cmd->get_cid()});
				}
			}

			if(generate_reduction) {
				const auto& reduction = *buffer_state.pending_reduction;

				const auto local_last_writer = buffer_state.local_last_writer.get_region_values(scalar_box);
				assert(local_last_writer.size() == 1);

				if(is_local_chunk) {
					auto* const reduce_cmd = create_command<reduction_command>(reduction);

					// Only generate a true dependency on the last writer if this node participated in the intermediate result computation.
					if(local_last_writer[0].second.is_fresh()) {
						m_cdag.add_dependency(reduce_cmd, m_cdag.get(local_last_writer[0].second), dependency_kind::true_dep, dependency_origin::dataflow);
					}

					auto* const ap_cmd = create_command<await_push_command>(bid, reduction.rid, trid, scalar_box.get_subrange());
					m_cdag.add_dependency(reduce_cmd, ap_cmd, dependency_kind::true_dep, dependency_origin::dataflow);
					generate_epoch_dependencies(ap_cmd);

					m_cdag.add_dependency(cmd, reduce_cmd, dependency_kind::true_dep, dependency_origin::dataflow);

					// Reduction command becomes the last writer (this may be overriden if this task also writes to the reduction buffer)
					post_reduction_buffer_states.at(bid).local_last_writer.update_box(scalar_box, reduce_cmd->get_cid());
				} else {
					// Push an empty range if we don't have any fresh data on this node
					const bool notification_only = !local_last_writer[0].second.is_fresh();
					const auto push_box = notification_only ? empty_box : scalar_box;

					auto* const push_cmd = create_command<push_command>(bid, reduction.rid, nid, trid, push_box.get_subrange());
					generated_pushes.push_back(push_cmd);

					if(notification_only) {
						generate_epoch_dependencies(push_cmd);
					} else {
						m_command_buffer_reads[push_cmd->get_cid()][bid] = region_union(m_command_buffer_reads[push_cmd->get_cid()][bid], scalar_box);
						m_cdag.add_dependency(push_cmd, m_cdag.get(local_last_writer[0].second), dependency_kind::true_dep, dependency_origin::dataflow);
					}

					// Mark the reduction result as replicated so we don't generate data transfers to this node
					// TODO: We need a way of updating regions in place! E.g. apply_to_values(box, callback)
					const auto replicated_box = post_reduction_buffer_states.at(bid).replicated_regions.get_region_values(scalar_box);
					assert(replicated_box.size() == 1);
					for(const auto& [_, nodes] : replicated_box) {
						post_reduction_buffer_states.at(bid).replicated_regions.update_box(scalar_box, node_bitset{nodes}.set(nid));
					}
				}
			}
		}
	}

	// For buffers that were in a pending reduction state and a reduction was generated
	// (i.e., the result was not discarded), set their new state.
	for(auto& [bid, new_state] : post_reduction_buffer_states) {
		m_buffer_states.insert_or_assign(bid, std::move(new_state));
	}

	// Update per-buffer last writers
	// This has to happen after applying post_reduction_buffer_states to properly support chained reductions.
	for(auto& [bid, updates] : per_buffer_last_writer_update_list) {
		auto& buffer_state = m_buffer_states.at(bid);
		for(auto& [req, cid] : updates) {
			buffer_state.local_last_writer.update_region(req, cid);
			buffer_state.replicated_regions.update_region(req, node_bitset{});
		}
		// In case this buffer was in a pending reduction state but the result was discarded, remove the pending reduction.
		buffer_state.pending_reduction = std::nullopt;
	}

	// Mark any buffers that now are in a pending reduction state as such.
	// This has to happen after applying post_reduction_buffer_states and per_buffer_last_writer_update_list
	// to properly support chained reductions.
	// If there is only one chunk/command, it already implicitly generates the final reduced value
	// and the buffer does not need to be flagged as a pending reduction.
	if(chunks.size() > 1) {
		for(const auto& reduction : tsk.get_reductions()) {
			m_buffer_states.at(reduction.bid).pending_reduction = reduction;

			// In some cases this node may not actually participate in the computation of the
			// intermediate reduction result (because there was no chunk). If so, mark the
			// reduction buffer as stale so we do not use it as input for the final reduction command.
			if(per_buffer_local_writes.count(reduction.bid) == 0) {
				[[maybe_unused]] size_t num_entries = 0;
				m_buffer_states.at(reduction.bid).local_last_writer.apply_to_values([&num_entries](const write_command_state& wcs) {
					num_entries++;
					write_command_state stale_state{wcs};
					stale_state.mark_as_stale();
					return stale_state;
				});
				assert(num_entries == 1);
			}
		}
	}

	// Determine potential "intra-task" race conditions.
	// These can happen in rare cases, when the node that pushes a buffer range also writes to that range within the same task.
	// We cannot do this while generating the push command, as we may not have the writing command recorded at that point.
	for(auto* push_cmd : generated_pushes) {
		const auto last_writers = m_buffer_states.at(push_cmd->get_bid()).local_last_writer.get_region_values(region(push_cmd->get_range()));

		for(const auto& [box, wcs] : last_writers) {
			assert(!box.empty()); // If we want to push it it cannot be empty
			// In general we should only be pushing fresh data.
			// If the push is for a reduction and the data no longer is fresh, it means
			// that we did not generate a reduction command on this node and the data becomes
			// stale after the remote reduction command has been executed.
			assert(wcs.is_fresh() || push_cmd->get_reduction_id() != 0);
			auto* const writer_cmd = m_cdag.get(wcs);
			assert(writer_cmd != nullptr);

			// We're only interested in writes that happen within the same task as the push
			if(utils::isa<task_command>(writer_cmd) && utils::as<task_command>(writer_cmd)->get_tid() == tsk.get_id()) {
				// In certain situations the push might have a true dependency on the last writer,
				// in that case don't add an anti-dependency (as that would cause a cycle).
				// TODO: Is this still possible? We don't have a unit test exercising this branch...
				if(push_cmd->has_dependency(writer_cmd, dependency_kind::true_dep)) {
					// This can currently only happen for await_push commands.
					assert(utils::isa<await_push_command>(writer_cmd));
					continue;
				}
				m_cdag.add_dependency(writer_cmd, push_cmd, dependency_kind::anti_dep, dependency_origin::dataflow);
			}
		}
	}

	// Determine which local data is fresh/stale based on task-level writes.
	auto requirements = get_buffer_requirements_for_mapped_access(tsk, subrange<3>(tsk.get_global_offset(), tsk.get_global_size()), tsk.get_global_size());
	for(auto& [bid, reqs_by_mode] : requirements) {
		std::vector<box<3>> global_write_boxes;
		for(const auto mode : access::producer_modes) {
			if(reqs_by_mode.count(mode) == 0) continue;
			const auto& by_mode = reqs_by_mode.at(mode);
			global_write_boxes.insert(global_write_boxes.end(), by_mode.get_boxes().begin(), by_mode.get_boxes().end());
		}
		const region global_writes(std::move(global_write_boxes));
		const auto& local_writes = per_buffer_local_writes[bid];
		assert(region_difference(local_writes, global_writes).empty()); // Local writes have to be a subset of global writes
		const auto remote_writes = region_difference(global_writes, local_writes);
		auto& buffer_state = m_buffer_states.at(bid);

		// TODO: We need a way of updating regions in place! E.g. apply_to_values(box, callback)
		auto boxes_and_cids = buffer_state.local_last_writer.get_region_values(remote_writes);
		for(auto& [box, wcs] : boxes_and_cids) {
			if(wcs.is_fresh()) {
				wcs.mark_as_stale();
				buffer_state.local_last_writer.update_region(box, wcs);
			}
		}
	}

	process_task_side_effect_requirements(tsk);
}

void distributed_graph_generator::generate_anti_dependencies(
    task_id tid, buffer_id bid, const region_map<write_command_state>& last_writers_map, const region<3>& write_req, abstract_command* write_cmd) {
	const auto last_writers = last_writers_map.get_region_values(write_req);
	for(const auto& [box, wcs] : last_writers) {
		auto* const last_writer_cmd = m_cdag.get(static_cast<command_id>(wcs));
		assert(!utils::isa<task_command>(last_writer_cmd) || utils::as<task_command>(last_writer_cmd)->get_tid() != tid);

		// Add anti-dependencies onto all successors of the writer
		bool has_successors = false;
		for(auto d : last_writer_cmd->get_dependents()) {
			// Only consider true dependencies
			if(d.kind != dependency_kind::true_dep) continue;

			auto* const cmd = d.node;

			// We might have already generated new commands within the same task that also depend on this; in that case, skip it
			if(utils::isa<task_command>(cmd) && utils::as<task_command>(cmd)->get_tid() == tid) continue;

			// So far we don't know whether the dependent actually intersects with the subrange we're writing
			if(const auto command_reads_it = m_command_buffer_reads.find(cmd->get_cid()); command_reads_it != m_command_buffer_reads.end()) {
				const auto& command_reads = command_reads_it->second;
				// The task might be a dependent because of another buffer
				if(const auto buffer_reads_it = command_reads.find(bid); buffer_reads_it != command_reads.end()) {
					if(!region_intersection(write_req, buffer_reads_it->second).empty()) {
						has_successors = true;
						m_cdag.add_dependency(write_cmd, cmd, dependency_kind::anti_dep, dependency_origin::dataflow);
					}
				}
			}
		}

		// In some cases (horizons, master node host task, weird discard_* constructs...)
		// the last writer might not have any successors. Just add the anti-dependency onto the writer itself then.
		if(!has_successors) { m_cdag.add_dependency(write_cmd, last_writer_cmd, dependency_kind::anti_dep, dependency_origin::dataflow); }
	}
}

void distributed_graph_generator::process_task_side_effect_requirements(const task& tsk) {
	const task_id tid = tsk.get_id();
	if(tsk.get_side_effect_map().empty()) return; // skip the loop in the common case
	if(m_cdag.task_command_count(tid) == 0) return;

	for(auto* const cmd : m_cdag.task_commands(tid)) {
		for(const auto& side_effect : tsk.get_side_effect_map()) {
			const auto [hoid, order] = side_effect;
			if(const auto last_effect = m_host_object_last_effects.find(hoid); last_effect != m_host_object_last_effects.end()) {
				// TODO once we have different side_effect_orders, their interaction will determine the dependency kind
				m_cdag.add_dependency(cmd, m_cdag.get(last_effect->second), dependency_kind::true_dep, dependency_origin::dataflow);
			}

			// Simplification: If there are multiple chunks per node, we generate true-dependencies between them in an arbitrary order, when all we really
			// need is mutual exclusion (i.e. a bi-directional pseudo-dependency).
			m_host_object_last_effects.insert_or_assign(hoid, cmd->get_cid());
		}
	}
}

void distributed_graph_generator::set_epoch_for_new_commands(const abstract_command* const epoch_or_horizon) {
	// both an explicit epoch command and an applied horizon can be effective epochs
	assert(utils::isa<epoch_command>(epoch_or_horizon) || utils::isa<horizon_command>(epoch_or_horizon));

	for(auto& [bid, bs] : m_buffer_states) {
		bs.local_last_writer.apply_to_values([epoch_or_horizon](const write_command_state& wcs) {
			auto new_wcs = write_command_state(std::max(epoch_or_horizon->get_cid(), static_cast<command_id>(wcs)), wcs.is_replicated());
			if(!wcs.is_fresh()) new_wcs.mark_as_stale();
			return new_wcs;
		});
	}
	for(auto& [cgid, cid] : m_last_collective_commands) {
		cid = std::max(epoch_or_horizon->get_cid(), cid);
	}
	for(auto& [cgid, cid] : m_host_object_last_effects) {
		cid = std::max(epoch_or_horizon->get_cid(), cid);
	}

	m_epoch_for_new_commands = epoch_or_horizon->get_cid();
}

void distributed_graph_generator::reduce_execution_front_to(abstract_command* const new_front) {
	const auto previous_execution_front = m_cdag.get_execution_front();
	for(auto* const front_cmd : previous_execution_front) {
		if(front_cmd != new_front) { m_cdag.add_dependency(new_front, front_cmd, dependency_kind::true_dep, dependency_origin::execution_front); }
	}
	assert(m_cdag.get_execution_front().size() == 1 && *m_cdag.get_execution_front().begin() == new_front);
}

void distributed_graph_generator::generate_epoch_command(const task& tsk) {
	assert(tsk.get_type() == task_type::epoch);
	auto* const epoch = create_command<epoch_command>(tsk.get_id(), tsk.get_epoch_action());
	set_epoch_for_new_commands(epoch);
	m_current_horizon = no_command;
	// Make the epoch depend on the previous execution front
	reduce_execution_front_to(epoch);
}

void distributed_graph_generator::generate_horizon_command(const task& tsk) {
	assert(tsk.get_type() == task_type::horizon);
	auto* const horizon = create_command<horizon_command>(tsk.get_id());

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