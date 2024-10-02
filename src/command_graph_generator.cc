#include "command_graph_generator.h"

#include "access_modes.h"
#include "command.h"
#include "command_graph.h"
#include "recorders.h"
#include "split.h"
#include "task.h"
#include "task_manager.h"

namespace celerity::detail {

command_graph_generator::command_graph_generator(
    const size_t num_nodes, const node_id local_nid, command_graph& cdag, const task_manager& tm, detail::command_recorder* recorder, const policy_set& policy)
    : m_num_nodes(num_nodes), m_local_nid(local_nid), m_policy(policy), m_cdag(cdag), m_task_mngr(tm), m_recorder(recorder) {
	if(m_num_nodes > max_num_nodes) {
		throw std::runtime_error(fmt::format("Number of nodes requested ({}) exceeds compile-time maximum of {}", m_num_nodes, max_num_nodes));
	}

	// Build initial epoch command (this is required to properly handle anti-dependencies on host-initialized buffers).
	// We manually generate the first command so it doesn't get added to the current batch; it will be replaced by applied horizons
	// or explicit epochs down the line (see set_epoch_for_new_commands).
	auto* const epoch_cmd = cdag.create<epoch_command>(task_manager::initial_epoch_task, epoch_action::none, std::vector<reduction_id>{});
	if(is_recording()) { m_recorder->record_command(std::make_unique<epoch_command_record>(*epoch_cmd, *tm.get_task(task_manager::initial_epoch_task))); }
	m_epoch_for_new_commands = epoch_cmd->get_cid();
}

void command_graph_generator::notify_buffer_created(const buffer_id bid, const range<3>& range, bool host_initialized) {
	m_buffers.emplace(std::piecewise_construct, std::tuple{bid}, std::tuple{range, range});
	if(host_initialized && m_policy.uninitialized_read_error != error_policy::ignore) { m_buffers.at(bid).initialized_region = box(subrange({}, range)); }
	// Mark contents as available locally (= don't generate await push commands) and fully replicated (= don't generate push commands).
	// This is required when tasks access host-initialized or uninitialized buffers.
	m_buffers.at(bid).local_last_writer.update_region(subrange<3>({}, range), m_epoch_for_new_commands);
	m_buffers.at(bid).replicated_regions.update_region(subrange<3>({}, range), node_bitset{}.set());
}

void command_graph_generator::notify_buffer_debug_name_changed(const buffer_id bid, const std::string& debug_name) {
	m_buffers.at(bid).debug_name = debug_name;
}

void command_graph_generator::notify_buffer_destroyed(const buffer_id bid) {
	assert(m_buffers.count(bid) != 0);
	m_buffers.erase(bid);
}

void command_graph_generator::notify_host_object_created(const host_object_id hoid) {
	assert(m_host_objects.count(hoid) == 0);
	m_host_objects.emplace(hoid, host_object_state{m_epoch_for_new_commands});
}

void command_graph_generator::notify_host_object_destroyed(const host_object_id hoid) {
	assert(m_host_objects.count(hoid) != 0);
	m_host_objects.erase(hoid);
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

// According to Wikipedia https://en.wikipedia.org/wiki/Topological_sorting#Depth-first_search
// TODO: This may no longer be necessary since different command types are now generated in pre-determined order - revisit
std::vector<abstract_command*> sort_topologically(command_set unmarked) {
	command_set temporary_marked;
	command_set permanent_marked;
	std::vector<abstract_command*> sorted(unmarked.size());
	auto sorted_front = sorted.rbegin();

	const auto visit = [&](abstract_command* const cmd, auto& visit /* to allow recursion in lambda */) {
		if(permanent_marked.count(cmd) != 0) return;
		assert(temporary_marked.count(cmd) == 0 && "cyclic command graph");
		unmarked.erase(cmd);
		temporary_marked.insert(cmd);
		for(const auto dep : cmd->get_dependents()) {
			visit(dep.node, visit);
		}
		temporary_marked.erase(cmd);
		permanent_marked.insert(cmd);
		*sorted_front++ = cmd;
	};

	while(!unmarked.empty()) {
		visit(*unmarked.begin(), visit);
	}

	return sorted;
}

command_set command_graph_generator::build_task(const task& tsk) {
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

	// Check that all commands have been recorded
	if(is_recording()) {
		assert(std::all_of(m_current_cmd_batch.begin(), m_current_cmd_batch.end(), [this](const abstract_command* cmd) {
			return m_recorder->get_commands().end()
			       != std::find_if(m_recorder->get_commands().begin(), m_recorder->get_commands().end(),
			           [cmd](const std::unique_ptr<command_record>& rec) { return rec->cid == cmd->get_cid(); });
		}));
	}

	return std::move(m_current_cmd_batch);
}

void command_graph_generator::report_overlapping_writes(const task& tsk, const box_vector<3>& local_chunks) const {
	const chunk<3> full_chunk{tsk.get_global_offset(), tsk.get_global_size(), tsk.get_global_size()};

	// Since this check is run distributed on every node, we avoid quadratic behavior by only checking for conflicts between all local chunks and the
	// region-union of remote chunks. This way, every conflict will be reported by at least one node.
	const box<3> global_chunk(subrange(full_chunk.offset, full_chunk.range));
	auto remote_chunks = region_difference(global_chunk, region(box_vector<3>(local_chunks))).into_boxes();

	// detect_overlapping_writes takes a single box_vector, so we concatenate local and global chunks (the order does not matter)
	auto distributed_chunks = std::move(remote_chunks);
	distributed_chunks.insert(distributed_chunks.end(), local_chunks.begin(), local_chunks.end());

	if(const auto overlapping_writes = detect_overlapping_writes(tsk, distributed_chunks); !overlapping_writes.empty()) {
		auto error = fmt::format("{} has overlapping writes between multiple nodes in", print_task_debug_label(tsk, true /* title case */));
		for(const auto& [bid, overlap] : overlapping_writes) {
			fmt::format_to(std::back_inserter(error), " {} {}", print_buffer_debug_label(bid), overlap);
		}
		error += ". Choose a non-overlapping range mapper for this write access or constrain the split via experimental::constrain_split to make the access "
		         "non-overlapping.";
		utils::report_error(m_policy.overlapping_write_error, "{}", error);
	}
}

std::vector<command_graph_generator::assigned_chunk> command_graph_generator::split_task_and_assign_chunks(const task& tsk) const {
	const chunk<3> full_chunk{tsk.get_global_offset(), tsk.get_global_size(), tsk.get_global_size()};
	const size_t num_chunks = m_num_nodes * 1; // TODO Make configurable
	const auto chunks = ([&] {
		if(tsk.get_type() == task_type::collective || tsk.get_type() == task_type::fence) {
			std::vector<chunk<3>> chunks;
			for(size_t nid = 0; nid < m_num_nodes; ++nid) {
				chunks.push_back(chunk_cast<3>(chunk<1>{id<1>{tsk.get_type() == task_type::collective ? nid : 0}, ones, {m_num_nodes}}));
			}
			return chunks;
		}
		if(tsk.has_variable_split()) {
			if(tsk.get_hint<experimental::hints::split_1d>() != nullptr) {
				// no-op, keeping this for documentation purposes
			}
			if(tsk.get_hint<experimental::hints::split_2d>() != nullptr) { return split_2d(full_chunk, tsk.get_granularity(), num_chunks); }
			return split_1d(full_chunk, tsk.get_granularity(), num_chunks);
		}
		return std::vector<chunk<3>>{full_chunk};
	})();
	assert(chunks.size() <= num_chunks); // We may have created less than requested
	assert(!chunks.empty());

	// Assign each chunk to a node
	// We assign chunks next to each other to the same worker (if there is more chunks than workers), as this is likely to produce less
	// transfers between tasks than a round-robin assignment (for typical stencil codes).
	// FIXME: This only works if the number of chunks is an integer multiple of the number of workers, e.g. 3 chunks for 2 workers degrades to RR.
	const auto chunks_per_node = std::max<size_t>(1, chunks.size() / m_num_nodes);

	std::vector<assigned_chunk> assigned_chunks;
	for(size_t i = 0; i < chunks.size(); ++i) {
		const node_id nid = (i / chunks_per_node) % m_num_nodes;
		assigned_chunks.push_back({nid, chunks[i]});
	}
	return assigned_chunks;
}

const box<3> empty_reduction_box({0, 0, 0}, {0, 0, 0});
const box<3> scalar_reduction_box({0, 0, 0}, {1, 1, 1});

command_graph_generator::assigned_chunks_with_requirements command_graph_generator::compute_per_chunk_requirements(
    const task& tsk, const std::vector<assigned_chunk>& assigned_chunks) const {
	assigned_chunks_with_requirements result;

	for(const auto& a_chunk : assigned_chunks) {
		const node_id nid = a_chunk.executed_on;
		auto requirements = get_buffer_requirements_for_mapped_access(tsk, a_chunk.chnk, tsk.get_global_size());

		// Add read/write requirements for reductions performed in this task.
		for(const auto& reduction : tsk.get_reductions()) {
			auto rmode = access_mode::discard_write;
			if(nid == reduction_initializer_nid && reduction.init_from_buffer) { rmode = access_mode::read_write; }
#ifndef NDEBUG
			for(auto pmode : access::producer_modes) {
				assert(requirements[reduction.bid].count(pmode) == 0); // task_manager verifies that there are no reduction <-> write-access conflicts
			}
#endif
			requirements[reduction.bid][rmode] = scalar_reduction_box;
		}

		if(nid == m_local_nid) {
			result.local_chunks.push_back({a_chunk, requirements});
		} else {
			result.remote_chunks.push_back({a_chunk, requirements});
		}
	}

	return result;
}

void command_graph_generator::resolve_pending_reductions(const task& tsk, const assigned_chunks_with_requirements& chunks_with_requirements) {
	// Buffers that currently are in a pending reduction state will receive a new buffer state after a reduction has been generated.
	std::unordered_map<buffer_id, buffer_state> post_reduction_buffers;

	std::unordered_map<buffer_id, size_t> number_of_participating_nodes;
	const auto process_chunks = [&](auto& chunks) {
		for(auto& [a_chunk, requirements] : chunks) {
			const node_id nid = a_chunk.executed_on;
			const bool is_local_chunk = nid == m_local_nid;

			for(const auto& [bid, reqs_by_mode] : requirements) {
				auto& buffer = m_buffers.at(bid);
				if(!buffer.pending_reduction.has_value()) continue;
				bool has_consumer = false;
				for(const auto mode : detail::access::consumer_modes) {
					if(auto req_it = reqs_by_mode.find(mode); req_it != reqs_by_mode.end()) {
						// While uncommon, we do support chunks that don't require access to a particular buffer at all.
						if(!req_it->second.empty()) {
							has_consumer = true;
							break;
						}
					}
				}

				if(!has_consumer) {
					// TODO the per-node reduction result is discarded - warn user about dead store
					continue;
				}

				const auto& reduction = *buffer.pending_reduction;

				const auto local_last_writer = buffer.local_last_writer.get_region_values(scalar_reduction_box);
				assert(local_last_writer.size() == 1);

				// Prepare the buffer state for after the reduction has been performed:
				// Keep the current last writer, but mark it as stale, so that if we don't generate a reduction command locally,
				// we'll know to get the data from elsewhere. If we generate a reduction command, this will be overwritten by its command id.
				write_command_state wcs{static_cast<command_id>(local_last_writer[0].second)};
				wcs.mark_as_stale();

				auto [it, _] = post_reduction_buffers.emplace(std::piecewise_construct, std::tuple{bid},
				    std::tuple{region_map<write_command_state>{ones, wcs}, region_map<node_bitset>{ones, node_bitset{}}});
				auto& post_reduction_buffer = it->second;

				if(m_policy.uninitialized_read_error != error_policy::ignore) { post_reduction_buffers.at(bid).initialized_region = scalar_reduction_box; }

				if(is_local_chunk) {
					// We currently don't support multiple chunks on a single node for reductions (there is also -- for now -- no way to create multiple chunks,
					// as oversubscription is handled by the instruction graph).
					// NOTE: The number_of_participating_nodes check below relies on this being true
					assert(chunks_with_requirements.local_chunks.size() == 1);

					auto* const reduce_cmd = create_command<reduction_command>(reduction, local_last_writer[0].second.is_fresh() /* has_local_contribution */,
					    [&](const auto& record_debug_info) { record_debug_info(buffer.debug_name); });

					// Only generate a true dependency on the last writer if this node participated in the intermediate result computation.
					if(local_last_writer[0].second.is_fresh()) {
						add_dependency(reduce_cmd, m_cdag.get(local_last_writer[0].second), dependency_kind::true_dep, dependency_origin::dataflow);
					}

					auto* const ap_cmd = create_command<await_push_command>(transfer_id(tsk.get_id(), bid, reduction.rid), scalar_reduction_box.get_subrange(),
					    [&](const auto& record_debug_info) { record_debug_info(buffer.debug_name); });
					add_dependency(reduce_cmd, ap_cmd, dependency_kind::true_dep, dependency_origin::dataflow);
					generate_epoch_dependencies(ap_cmd);

					generate_anti_dependencies(tsk.get_id(), bid, buffer.local_last_writer, scalar_reduction_box, reduce_cmd);

					post_reduction_buffer.local_last_writer.update_box(scalar_reduction_box, reduce_cmd->get_cid());
					number_of_participating_nodes[bid]++; // We are participating
				} else {
					const bool notification_only = !local_last_writer[0].second.is_fresh();

					// Push an empty range if we don't have any fresh data on this node. This will then generate an empty pilot that tells the
					// other node's receive_arbiter to not expect a send.
					const auto push_box = notification_only ? empty_reduction_box : scalar_reduction_box;

					auto* const push_cmd = create_command<push_command>(nid, transfer_id(tsk.get_id(), bid, reduction.rid), push_box.get_subrange(),
					    [&](const auto& record_debug_info) { record_debug_info(buffer.debug_name); });

					if(notification_only) {
						generate_epoch_dependencies(push_cmd);
					} else {
						m_command_buffer_reads[push_cmd->get_cid()][bid] = region_union(m_command_buffer_reads[push_cmd->get_cid()][bid], scalar_reduction_box);
						add_dependency(push_cmd, m_cdag.get(local_last_writer[0].second), dependency_kind::true_dep, dependency_origin::dataflow);
					}

					// Mark the reduction result as replicated so we don't generate data transfers to this node
					// TODO: We need a way of updating regions in place! E.g. apply_to_values(box, callback)
					const auto replicated_box = post_reduction_buffer.replicated_regions.get_region_values(scalar_reduction_box);
					assert(replicated_box.size() == 1);
					for(const auto& [_, nodes] : replicated_box) {
						post_reduction_buffer.replicated_regions.update_box(scalar_reduction_box, node_bitset{nodes}.set(nid));
					}
					number_of_participating_nodes[bid]++; // This node is participating
				}
			}
		}
	};

	// Since the local reduction command overwrites the buffer contents that need to be pushed to other nodes, we need to process remote chunks first.
	// TODO: Replace with a C++20 join view once we have upgraded
	process_chunks(chunks_with_requirements.remote_chunks);
	process_chunks(chunks_with_requirements.local_chunks);

	// We currently do not support generating reduction commands on only a subset of nodes, except for the special case of a single command.
	// This is because it is unclear who owns the final result in this case (normally all nodes "own" the result).
	//   => I.e., reducing and using the result on the participating nodes is actually not the problem (this works as intended); the issue only arises
	//      if the result is subsequently required in other tasks. Since we don't have a good way of detecting this condition however, we currently
	//      disallow partial reductions altogether.
	for(auto& [bid, number_of_participating_nodes] : number_of_participating_nodes) {
		// NOTE: This check relies on the fact that we currently only generate a single chunk per node for reductions (see assertion above).
		if(number_of_participating_nodes > 1 && number_of_participating_nodes != m_num_nodes) {
			utils::report_error(error_policy::panic,
			    "{} requires a reduction on {} that is not performed on all nodes. This is currently not supported. Either "
			    "ensure that all nodes receive a chunk that reads from the buffer, or reduce the data on a single node.",
			    print_task_debug_label(tsk, true /* title case */), print_buffer_debug_label(bid));
		}
	}

	// For buffers that were in a pending reduction state and a reduction was generated
	// (i.e., the result was not discarded), set their new state.
	for(auto& [bid, new_state] : post_reduction_buffers) {
		auto& buffer = m_buffers.at(bid);
		if(buffer.pending_reduction.has_value()) { m_completed_reductions.push_back(buffer.pending_reduction->rid); }
		buffer = std::move(new_state);
	}
}

void command_graph_generator::generate_pushes(const task& tsk, const assigned_chunks_with_requirements& chunks_with_requirements) {
	for(auto& [a_chunk, requirements] : chunks_with_requirements.remote_chunks) {
		const node_id nid = a_chunk.executed_on;

		for(const auto& [bid, reqs_by_mode] : requirements) {
			auto& buffer = m_buffers.at(bid);

			for(const auto& [mode, req] : reqs_by_mode) {
				if(!detail::access::mode_traits::is_consumer(mode)) continue;
				// We generate separate push command for each last writer command for now, possibly even multiple for partially already-replicated data.
				// TODO: Can and/or should we consolidate?
				const auto local_sources = buffer.local_last_writer.get_region_values(req);
				for(const auto& [local_box, wcs] : local_sources) {
					if(!wcs.is_fresh() || wcs.is_replicated()) { continue; }

					// Make sure we don't push anything we've already pushed to this node before
					box_vector<3> non_replicated_boxes;
					for(const auto& [replicated_box, nodes] : buffer.replicated_regions.get_region_values(local_box)) {
						if(nodes.test(nid)) continue;
						non_replicated_boxes.push_back(replicated_box);
					}

					// Merge all connected boxes to determine final set of pushes
					const auto push_region = region<3>(std::move(non_replicated_boxes));
					for(const auto& push_box : push_region.get_boxes()) {
						auto* const push_cmd = create_command<push_command>(nid, transfer_id(tsk.get_id(), bid, no_reduction_id), push_box.get_subrange(),
						    [&](const auto& record_debug_info) { record_debug_info(buffer.debug_name); });
						assert(!utils::isa<await_push_command>(m_cdag.get(wcs)) && "Attempting to push non-owned data?!");
						add_dependency(push_cmd, m_cdag.get(wcs), dependency_kind::true_dep, dependency_origin::dataflow);

						// Store the read access for determining anti-dependencies later on
						m_command_buffer_reads[push_cmd->get_cid()][bid] = push_box;
					}

					// Remember that we've replicated this region
					for(const auto& [replicated_box, nodes] : buffer.replicated_regions.get_region_values(push_region)) {
						buffer.replicated_regions.update_box(replicated_box, node_bitset{nodes}.set(nid));
					}
				}
			}
		}
	}
}

void command_graph_generator::generate_await_pushes(const task& tsk, const assigned_chunks_with_requirements& chunks_with_requirements) {
	for(auto& [a_chunk, requirements] : chunks_with_requirements.local_chunks) {
		for(auto& [bid, reqs_by_mode] : requirements) {
			auto& buffer = m_buffers.at(bid);

			for(const auto& [mode, req] : reqs_by_mode) {
				if(!detail::access::mode_traits::is_consumer(mode)) continue;

				const auto local_sources = buffer.local_last_writer.get_region_values(req);
				box_vector<3> missing_part_boxes;
				for(const auto& [box, wcs] : local_sources) {
					if(!box.empty() && !wcs.is_fresh()) { missing_part_boxes.push_back(box); }
				}

				// There is data we don't yet have locally. Generate an await push command for it.
				if(!missing_part_boxes.empty()) {
					const region missing_parts(std::move(missing_part_boxes));
					assert(m_num_nodes > 1);
					auto* const ap_cmd = create_command<await_push_command>(transfer_id(tsk.get_id(), bid, no_reduction_id), missing_parts,
					    [&](const auto& record_debug_info) { record_debug_info(buffer.debug_name); });
					generate_anti_dependencies(tsk.get_id(), bid, buffer.local_last_writer, missing_parts, ap_cmd);
					generate_epoch_dependencies(ap_cmd);
					// Remember that we have this data now
					buffer.local_last_writer.update_region(missing_parts, {ap_cmd->get_cid(), true /* is_replicated */});
				}
			}
		}
	}
}

void command_graph_generator::update_local_buffer_fresh_regions(const task& tsk, const std::unordered_map<buffer_id, region<3>>& per_buffer_local_writes) {
	auto requirements = get_buffer_requirements_for_mapped_access(tsk, subrange<3>(tsk.get_global_offset(), tsk.get_global_size()), tsk.get_global_size());
	// Add requirements for reductions
	for(const auto& reduction : tsk.get_reductions()) {
		// the actual mode is irrelevant as long as it's a producer - TODO have a better query API for task buffer requirements
		requirements[reduction.bid][access_mode::write] = scalar_reduction_box;
	}
	for(auto& [bid, reqs_by_mode] : requirements) {
		box_vector<3> global_write_boxes;
		for(const auto mode : access::producer_modes) {
			if(reqs_by_mode.count(mode) == 0) continue;
			const auto& by_mode = reqs_by_mode.at(mode);
			global_write_boxes.insert(global_write_boxes.end(), by_mode.get_boxes().begin(), by_mode.get_boxes().end());
		}

		region global_writes(std::move(global_write_boxes));
		auto& buffer = m_buffers.at(bid);
		if(m_policy.uninitialized_read_error != error_policy::ignore) { buffer.initialized_region = region_union(buffer.initialized_region, global_writes); }

		const auto remote_writes = ([&, bid = bid] {
			if(auto it = per_buffer_local_writes.find(bid); it != per_buffer_local_writes.end()) {
				const auto& local_writes = it->second;
				assert(region_difference(local_writes, global_writes).empty()); // Local writes have to be a subset of global writes
				return region_difference(global_writes, local_writes);
			}
			return std::move(global_writes);
		})(); // IIFE

		// TODO: We need a way of updating regions in place! E.g. apply_to_values(box, callback)
		auto boxes_and_cids = buffer.local_last_writer.get_region_values(remote_writes);
		for(auto& [box, wcs] : boxes_and_cids) {
			if(wcs.is_fresh()) {
				wcs.mark_as_stale();
				buffer.local_last_writer.update_region(box, wcs);
			}
		}
	}
}

void command_graph_generator::generate_distributed_commands(const task& tsk) {
	const auto chunks = split_task_and_assign_chunks(tsk);
	const auto chunks_with_requirements = compute_per_chunk_requirements(tsk, chunks);

	// Check for and report overlapping writes between local chunks, and between local and remote chunks.
	if(m_policy.overlapping_write_error != error_policy::ignore) {
		box_vector<3> local_chunks;
		for(const auto& [a_chunk, _] : chunks_with_requirements.local_chunks) {
			local_chunks.push_back(box<3>{a_chunk.chnk});
		}
		report_overlapping_writes(tsk, local_chunks);
	}

	resolve_pending_reductions(tsk, chunks_with_requirements);
	generate_pushes(tsk, chunks_with_requirements);
	generate_await_pushes(tsk, chunks_with_requirements);

	// Union of all per-buffer writes on this node, used to determine which parts of a buffer are fresh/stale later on.
	std::unordered_map<buffer_id, region<3>> per_buffer_local_writes;

	// Create command for each local chunk and resolve local data dependencies.
	for(const auto& [a_chunk, requirements] : chunks_with_requirements.local_chunks) {
		abstract_command* cmd = nullptr;
		if(tsk.get_type() == task_type::fence) {
			cmd = create_command<fence_command>(tsk.get_id(),
			    [&](const auto& record_debug_info) { record_debug_info(tsk, [this](const buffer_id bid) { return m_buffers.at(bid).debug_name; }); });
		} else {
			cmd = create_command<execution_command>(tsk.get_id(), subrange{a_chunk.chnk},
			    [&](const auto& record_debug_info) { record_debug_info(tsk, [this](const buffer_id bid) { return m_buffers.at(bid).debug_name; }); });

			// Go over all reductions that are to be performed *during* the execution of this chunk,
			// not to be confused with any pending reductions that need to be finalized *before* the
			// execution of this chunk (those have already been handled by resolve_pending_reductions).
			// If a reduction reads the previous value of the buffer (i.e. w/o property::reduction::initialize_to_identity),
			// we have to include it in exactly one of the per-node intermediate reductions.
			for(const auto& reduction : tsk.get_reductions()) {
				if(m_local_nid == reduction_initializer_nid && reduction.init_from_buffer) {
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
				add_dependency(cmd, m_cdag.get(prev->second), dependency_kind::true_dep, dependency_origin::collective_group_serialization);
				m_last_collective_commands.erase(prev);
			}
			m_last_collective_commands.emplace(cgid, cmd->get_cid());
		}

		for(const auto& [bid, reqs_by_mode] : requirements) {
			auto& buffer = m_buffers.at(bid);

			// Process consuming accesses first, so we don't add dependencies onto our own writes
			region<3> uninitialized_reads;
			region<3> all_reads;
			for(const auto& [mode, req] : reqs_by_mode) {
				if(!detail::access::mode_traits::is_consumer(mode)) continue;
				all_reads = region_union(all_reads, req);
				if(m_policy.uninitialized_read_error != error_policy::ignore
				    && !bounding_box(buffer.initialized_region).covers(bounding_box(req.get_boxes()))) {
					uninitialized_reads = region_union(uninitialized_reads, region_difference(req, buffer.initialized_region));
				}
			}

			if(!all_reads.empty()) {
				for(const auto& [box, wcs] : buffer.local_last_writer.get_region_values(all_reads)) {
					if(box.empty()) continue;
					assert(wcs.is_fresh() && "Unresolved remote data dependency");
					add_dependency(cmd, m_cdag.get(wcs), dependency_kind::true_dep, dependency_origin::dataflow);
				}

				// Store the read access for determining anti-dependencies later on
				m_command_buffer_reads[cmd->get_cid()].emplace(bid, std::move(all_reads));
			}

			region<3> all_writes;
			for(const auto& [mode, req] : reqs_by_mode) {
				if(!detail::access::mode_traits::is_producer(mode)) continue;
				all_writes = region_union(all_writes, req);
			}

			if(!all_writes.empty()) {
				generate_anti_dependencies(tsk.get_id(), bid, buffer.local_last_writer, all_writes, cmd);

				// Update last writer
				buffer.local_last_writer.update_region(all_writes, cmd->get_cid());
				buffer.replicated_regions.update_region(all_writes, node_bitset{});

				// In case this buffer was in a pending reduction state we discarded the result and need to remove the pending reduction.
				if(buffer.pending_reduction.has_value()) {
					m_completed_reductions.push_back(buffer.pending_reduction->rid);
					buffer.pending_reduction = std::nullopt;
				}

				per_buffer_local_writes.emplace(bid, std::move(all_writes));
			}

			if(!uninitialized_reads.empty()) {
				utils::report_error(m_policy.uninitialized_read_error,
				    "Command C{} on N{}, which executes {} of {}, reads {} {}, which has not been written by any node.", cmd->get_cid(), m_local_nid,
				    box(subrange(a_chunk.chnk.offset, a_chunk.chnk.range)), print_task_debug_label(tsk), print_buffer_debug_label(bid),
				    detail::region(std::move(uninitialized_reads)));
			}
		}
	}

	// Mark any buffers that now are in a pending reduction state as such.
	// If there is only one chunk/command, it already implicitly generates the final reduced value
	// and the buffer does not need to be flagged as a pending reduction.
	for(const auto& reduction : tsk.get_reductions()) {
		if(chunks.size() > 1) {
			m_buffers.at(reduction.bid).pending_reduction = reduction;
		} else {
			m_completed_reductions.push_back(reduction.rid);
		}
	}

	update_local_buffer_fresh_regions(tsk, per_buffer_local_writes);
	process_task_side_effect_requirements(tsk);
}

void command_graph_generator::generate_anti_dependencies(
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
						add_dependency(write_cmd, cmd, dependency_kind::anti_dep, dependency_origin::dataflow);
					}
				}
			}
		}

		// In some cases (horizons, master node host task, weird discard_* constructs...)
		// the last writer might not have any successors. Just add the anti-dependency onto the writer itself then.
		if(!has_successors) { add_dependency(write_cmd, last_writer_cmd, dependency_kind::anti_dep, dependency_origin::dataflow); }
	}
}

void command_graph_generator::process_task_side_effect_requirements(const task& tsk) {
	const task_id tid = tsk.get_id();
	if(tsk.get_side_effect_map().empty()) return; // skip the loop in the common case
	if(m_cdag.task_command_count(tid) == 0) return;

	for(auto* const cmd : m_cdag.task_commands(tid)) {
		for(const auto& side_effect : tsk.get_side_effect_map()) {
			const auto [hoid, order] = side_effect;
			auto& host_object = m_host_objects.at(hoid);

			if(host_object.last_side_effect.has_value()) {
				// TODO once we have different side_effect_orders, their interaction will determine the dependency kind
				add_dependency(cmd, m_cdag.get(*host_object.last_side_effect), dependency_kind::true_dep, dependency_origin::dataflow);
			}

			// Simplification: If there are multiple chunks per node, we generate true-dependencies between them in an arbitrary order, when all we really
			// need is mutual exclusion (i.e. a bi-directional pseudo-dependency).
			host_object.last_side_effect = cmd->get_cid();
		}
	}
}

void command_graph_generator::set_epoch_for_new_commands(const abstract_command* const epoch_or_horizon) {
	// both an explicit epoch command and an applied horizon can be effective epochs
	assert(utils::isa<epoch_command>(epoch_or_horizon) || utils::isa<horizon_command>(epoch_or_horizon));

	for(auto& [bid, bs] : m_buffers) {
		bs.local_last_writer.apply_to_values([epoch_or_horizon](const write_command_state& wcs) {
			auto new_wcs = write_command_state(std::max(epoch_or_horizon->get_cid(), static_cast<command_id>(wcs)), wcs.is_replicated());
			if(!wcs.is_fresh()) new_wcs.mark_as_stale();
			return new_wcs;
		});
	}
	for(auto& [cgid, cid] : m_last_collective_commands) {
		cid = std::max(epoch_or_horizon->get_cid(), cid);
	}
	for(auto& [_, host_object] : m_host_objects) {
		if(host_object.last_side_effect.has_value()) { host_object.last_side_effect = std::max(epoch_or_horizon->get_cid(), *host_object.last_side_effect); }
	}

	m_epoch_for_new_commands = epoch_or_horizon->get_cid();
}

void command_graph_generator::reduce_execution_front_to(abstract_command* const new_front) {
	const auto previous_execution_front = m_cdag.get_execution_front();
	for(auto* const front_cmd : previous_execution_front) {
		if(front_cmd != new_front) { add_dependency(new_front, front_cmd, dependency_kind::true_dep, dependency_origin::execution_front); }
	}
	assert(m_cdag.get_execution_front().size() == 1 && *m_cdag.get_execution_front().begin() == new_front);
}

void command_graph_generator::generate_epoch_command(const task& tsk) {
	assert(tsk.get_type() == task_type::epoch);
	auto* const epoch = create_command<epoch_command>(
	    tsk.get_id(), tsk.get_epoch_action(), std::move(m_completed_reductions), [&](const auto& record_debug_info) { record_debug_info(tsk); });
	set_epoch_for_new_commands(epoch);
	m_current_horizon = no_command;
	// Make the epoch depend on the previous execution front
	reduce_execution_front_to(epoch);
}

void command_graph_generator::generate_horizon_command(const task& tsk) {
	assert(tsk.get_type() == task_type::horizon);
	auto* const horizon =
	    create_command<horizon_command>(tsk.get_id(), std::move(m_completed_reductions), [&](const auto& record_debug_info) { record_debug_info(tsk); });

	if(m_current_horizon != static_cast<command_id>(no_command)) {
		// Apply the previous horizon
		set_epoch_for_new_commands(m_cdag.get(m_current_horizon));
	}
	m_current_horizon = horizon->get_cid();

	// Make the horizon depend on the previous execution front
	reduce_execution_front_to(horizon);
}

void command_graph_generator::generate_epoch_dependencies(abstract_command* cmd) {
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
		add_dependency(cmd, m_cdag.get(m_epoch_for_new_commands), dependency_kind::true_dep, dependency_origin::last_epoch);
	}
}

void command_graph_generator::prune_commands_before(const command_id epoch) {
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

std::string command_graph_generator::print_buffer_debug_label(const buffer_id bid) const {
	return utils::make_buffer_debug_label(bid, m_buffers.at(bid).debug_name);
}

} // namespace celerity::detail
