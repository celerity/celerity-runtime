#include "command_graph_generator.h"

#include "command_graph.h"
#include "grid.h"
#include "intrusive_graph.h"
#include "log.h"
#include "loop_template.h"
#include "print_utils.h"
#include "print_utils_internal.h"
#include "ranges.h"
#include "recorders.h"
#include "region_map.h"
#include "split.h"
#include "task.h"
#include "tracy.h"
#include "types.h"
#include "utils.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <memory>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

#include <matchbox.hh>


namespace celerity::detail {

command_graph_generator::command_graph_generator(
    const size_t num_nodes, const node_id local_nid, command_graph& cdag, detail::command_recorder* const recorder, const policy_set& policy)
    : m_num_nodes(num_nodes), m_local_nid(local_nid), m_policy(policy), m_cdag(&cdag), m_recorder(recorder) {
	if(m_num_nodes > max_num_nodes) {
		throw std::runtime_error(fmt::format("Number of nodes requested ({}) exceeds compile-time maximum of {}", m_num_nodes, max_num_nodes));
	}
}

void command_graph_generator::notify_buffer_created(const buffer_id bid, const range<3>& range, bool host_initialized) {
	assert(m_epoch_for_new_commands != nullptr);
	assert(!m_buffers.contains(bid));
	// Mark contents as available locally (= don't generate await push commands) and fully replicated (= don't generate push commands).
	// This is required when tasks access host-initialized or uninitialized buffers.
	auto& buffer = m_buffers.emplace(std::piecewise_construct, std::tuple(bid), std::tuple(range, m_epoch_for_new_commands, node_bitset().set())).first->second;
	if(host_initialized && m_policy.uninitialized_read_error != error_policy::ignore) { buffer.initialized_region = box(subrange({}, range)); }
}

void command_graph_generator::notify_buffer_debug_name_changed(const buffer_id bid, const std::string& debug_name) {
	m_buffers.at(bid).debug_name = debug_name;
}

void command_graph_generator::notify_buffer_destroyed(const buffer_id bid) {
	assert(m_buffers.contains(bid));
	m_buffers.erase(bid);
}

void command_graph_generator::notify_host_object_created(const host_object_id hoid) {
	assert(m_epoch_for_new_commands != nullptr);
	assert(!m_host_objects.contains(hoid));
	m_host_objects.emplace(hoid, m_epoch_for_new_commands);
}

void command_graph_generator::notify_host_object_destroyed(const host_object_id hoid) {
	assert(m_host_objects.contains(hoid));
	m_host_objects.erase(hoid);
}

/// Returns whether an iterator range of commands is topologically sorted, i.e. sequential execution would satisfy all internal dependencies.
template <typename Iterator>
bool is_topologically_sorted(Iterator begin, Iterator end) {
	for(auto check = begin; check != end; ++check) {
		for(const auto dep : (*check)->get_dependencies()) {
			if(std::find_if(std::next(check), end, [dep](const auto& node) { return node == dep.node; }) != end) return false;
		}
	}
	return true;
}

std::vector<const command*> command_graph_generator::build_task(const task& tsk, loop_template* const templ) {
	const auto epoch_to_prune_before = m_epoch_for_new_commands;
	batch current_batch;

	if(templ != nullptr && templ->cdag.is_verified) {
		const auto clone = [&](const command& cmd) { return clone_command(current_batch, &cmd, tsk); };
		const auto add_dependency = [&](command* from, command* to, dependency_kind kind, dependency_origin origin) {
			this->add_dependency(from, to, kind, origin);
		};

		// CELERITY_CRITICAL("CDAG: Applying loop template to task {}!", tsk.get_id()); //
		templ->cdag.instantiate(clone, add_dependency);
		return current_batch;
	}

	switch(tsk.get_type()) {
	case task_type::epoch: generate_epoch_command(current_batch, tsk); break;
	case task_type::horizon: generate_horizon_command(current_batch, tsk); break;
	case task_type::device_compute:
	case task_type::host_compute:
	case task_type::master_node:
	case task_type::collective:
	case task_type::fence: generate_distributed_commands(current_batch, tsk); break;
	default: throw std::runtime_error("Task type NYI");
	}

	// It is currently undefined to split reduction-producer tasks into multiple chunks on the same node:
	//   - Per-node reduction intermediate results are stored with fixed access to a single backing buffer,
	//     so multiple chunks on the same node will race on this buffer access
	//   - Inputs to the final reduction command are ordered by origin node ids to guarantee bit-identical results. It is not possible to distinguish
	//     more than one chunk per node in the serialized commands, so different nodes can produce different final reduction results for non-associative
	//     or non-commutative operations
	if(!tsk.get_reductions().empty()) {
		assert(std::count_if(current_batch.begin(), current_batch.end(), [](const command* cmd) { return utils::isa<task_command>(cmd); }) <= 1);
	}

	// If a new epoch was completed in the CDAG before the current task, we can erase all tracking information from earlier commands.
	// After the epoch (or horizon) command has been executed, the scheduler will then delete all obsolete commands from the CDAG.
	if(epoch_to_prune_before != nullptr) {
		std::erase_if(m_command_buffer_reads, [=](const auto& cid_reads) { return cid_reads.first < epoch_to_prune_before->get_id(); });
	}

	// Check that all commands have been recorded
	if(is_recording()) {
		assert(std::all_of(current_batch.begin(), current_batch.end(), [this](const command* cmd) {
			return std::any_of(m_recorder->get_graph_nodes().begin(), m_recorder->get_graph_nodes().end(),
			    [cmd](const std::unique_ptr<command_record>& rec) { return rec->id == cmd->get_id(); });
		}));
	}

	if(templ != nullptr) {
		if(!templ->cdag.is_primed) {
			templ->cdag.prime(current_batch);
		} else {
			assert(!templ->cdag.is_verified);
			templ->cdag.verify(current_batch);
			if(templ->cdag.is_verified && m_recorder != nullptr) { m_recorder->begin_loop_template(); }
		}
	}

	assert(is_topologically_sorted(current_batch.begin(), current_batch.end()));
	return current_batch;
}

void command_graph_generator::finalize_loop_template(loop_template& templ) {
	// CELERITY_CRITICAL("CDAG: Finalizing template!"); //

	const auto repl_map = templ.cdag.get_replacement_map();

	// TODO: Could optimize this to only update touched buffers
	for(auto& [bid, buffer] : m_buffers) {
		buffer.local_last_writer.apply_to_values([&repl_map](const write_command_state& wcs) {
			if(wcs.get_command() == nullptr) return wcs;
			const auto it = repl_map.find(wcs.get_command());
			if(it == repl_map.end()) return wcs;
			write_command_state new_wcs{it->second, wcs.is_replicated()};
			if(!wcs.is_fresh()) { new_wcs.mark_as_stale(); }
			return new_wcs;
		});
	}

	for(const auto& [from, to] : repl_map) {
		if(m_execution_front.contains(from)) {
			m_execution_front.erase(from);
			m_execution_front.insert(to);
		}

		if(m_command_buffer_reads.contains(from->get_id())) {
			m_command_buffer_reads[to->get_id()] = m_command_buffer_reads.at(from->get_id());
			m_command_buffer_reads.erase(from->get_id());
		}
	}

	m_epoch_for_new_commands = repl_map.at(m_epoch_for_new_commands);
	m_current_horizon = repl_map.at(m_current_horizon);

	if(m_recorder != nullptr) { m_recorder->end_loop_template(); }
}

command* command_graph_generator::clone_command(batch& current_batch, const command* const cmd, const task& tsk) {
	return matchbox::match<command*>(
	    *cmd,
	    [&](const execution_command& ecmd) {
		    return create_command<execution_command>(current_batch, &tsk, ecmd.get_execution_spec(), ecmd.is_reduction_initializer(),
		        [&](const auto& record_debug_info) { record_debug_info(tsk, [this](const buffer_id bid) { return m_buffers.at(bid).debug_name; }); });
	    },
	    [&](const push_command& pcmd) {
		    const auto trid = pcmd.get_transfer_id();
		    if(trid.rid != no_reduction_id) { throw std::runtime_error("Reduction push commands are not supported (yet) in loop templates"); }
		    return create_command<push_command>(current_batch, transfer_id(tsk.get_id(), trid.bid, trid.rid), pcmd.get_target_regions(),
		        [&](const auto& record_debug_info) { record_debug_info(m_buffers.at(trid.bid).debug_name); });
	    },
	    [&](const await_push_command& apcmd) {
		    const auto trid = apcmd.get_transfer_id();
		    if(trid.rid != no_reduction_id) { throw std::runtime_error("Reduction await-push commands are not supported (yet) in loop templates"); }
		    return create_command<await_push_command>(current_batch, transfer_id(tsk.get_id(), trid.bid, trid.rid), apcmd.get_region(),
		        [&](const auto& record_debug_info) { record_debug_info(m_buffers.at(trid.bid).debug_name); });
	    },
	    [&](const horizon_command& hcmd) {
		    if(!hcmd.get_completed_reductions().empty()) {
			    // NOCOMMIT Test this
			    throw std::runtime_error("Horizon commands with completed reductions are not supported (yet) in loop templates");
		    }
		    m_cdag->begin_epoch(tsk.get_id()); // NOCOMMIT Test this!
		    return create_command<horizon_command>(
		        current_batch, &tsk, std::vector<reduction_id>{}, [&](const auto& record_debug_info) { record_debug_info(tsk); });
	    },
	    [&](const auto& cmd) {
		    throw std::runtime_error("Command type NYI!");
		    return nullptr;
	    });
}

void command_graph_generator::report_overlapping_writes(const task& tsk, const std::vector<assigned_chunk>& chunks) const {
	box_vector<3> all_chunks;
	std::transform(chunks.begin(), chunks.end(), std::back_inserter(all_chunks), [](const auto& ac) { return box<3>(ac.chnk); });
	if(const auto overlapping_writes = detect_overlapping_writes(tsk, all_chunks); !overlapping_writes.empty()) {
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
	return matchbox::match(
	    tsk.get_geometry(),
	    [&](const basic_task_geometry& geo) {
		    const box<3> full_chunk{subrange<3>(geo.global_offset, geo.global_size)};
		    const size_t num_chunks = m_num_nodes * m_test_chunk_multiplier;
		    const auto chunks = ([&] {
			    if(tsk.get_type() == task_type::collective || tsk.get_type() == task_type::fence) {
				    std::vector<box<3>> chunks;
				    for(size_t nid = 0; nid < m_num_nodes; ++nid) {
					    const id<1> min = tsk.get_type() == task_type::collective ? nid : 0;
					    const id<1> max = min + 1;
					    chunks.push_back(box_cast<3>(box<1>{min, max}));
				    }
				    return chunks;
			    }
			    if(tsk.has_variable_split()) {
				    if(tsk.get_hint<experimental::hints::split_1d>() != nullptr) {
					    // no-op, keeping this for documentation purposes
				    }
				    if(tsk.get_hint<experimental::hints::split_2d>() != nullptr) { return split_2d(full_chunk, geo.granularity, num_chunks); }
				    return split_1d(full_chunk, geo.granularity, num_chunks);
			    }
			    return std::vector<box<3>>{full_chunk};
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
			    assigned_chunks.push_back({nid, chunk<3>(chunks[i].get_min(), chunks[i].get_range(), geo.global_size), std::nullopt});
		    }
		    return assigned_chunks;
	    },
	    [&](const custom_task_geometry_desc& geo) {
		    std::vector<assigned_chunk> result;
		    for(auto& [sr, nid, did] : geo.assigned_chunks) {
			    result.emplace_back(nid, chunk<3>{sr.offset, sr.range, geo.global_size}, did);
		    }
		    return result;
	    });
}

command_graph_generator::buffer_requirements_list command_graph_generator::get_buffer_requirements_for_mapped_access(
    const task& tsk, const subrange<3>& sr) const {
	buffer_requirements_list result;
	const auto& access_map = tsk.get_buffer_access_map();
	for(const buffer_id bid : access_map.get_accessed_buffers()) {
		result.push_back(buffer_requirements{bid, access_map.compute_consumed_region(bid, box<3>(sr)), access_map.compute_produced_region(bid, box<3>(sr))});
	}
	return result;
}

const box<3> empty_reduction_box({0, 0, 0}, {0, 0, 0});
const box<3> scalar_reduction_box({0, 0, 0}, {1, 1, 1});

command_graph_generator::assigned_chunks_with_requirements command_graph_generator::compute_per_chunk_requirements(
    const task& tsk, const std::vector<assigned_chunk>& assigned_chunks) const {
	assigned_chunks_with_requirements result;

	for(const auto& a_chunk : assigned_chunks) {
		const node_id nid = a_chunk.executed_on;
		auto requirements = get_buffer_requirements_for_mapped_access(tsk, a_chunk.chnk);

		// Add read/write requirements for reductions performed in this task.
		for(const auto& reduction : tsk.get_reductions()) {
			// task_manager verifies that there are no reduction <-> write-access conflicts
			assert(std::none_of(
			    requirements.begin(), requirements.end(), [&](const buffer_requirements& br) { return br.bid == reduction.bid && !br.produced.empty(); }));
			auto it = std::find_if(requirements.begin(), requirements.end(), [&](const buffer_requirements& br) { return br.bid == reduction.bid; });
			if(it == requirements.end()) { it = requirements.insert(requirements.end(), buffer_requirements{reduction.bid, {}, {}}); }
			it->produced = scalar_reduction_box;
			if(nid == reduction_initializer_nid && reduction.init_from_buffer) { it->consumed = scalar_reduction_box; }
		}

		if(nid == m_local_nid) {
			result.local_chunks.emplace_back(a_chunk, std::move(requirements));
		} else {
			result.remote_chunks.emplace_back(a_chunk, std::move(requirements));
		}
	}

	return result;
}

void command_graph_generator::resolve_pending_reductions(
    batch& current_batch, const task& tsk, const assigned_chunks_with_requirements& chunks_with_requirements) {
	CELERITY_DETAIL_TRACY_ZONE_SCOPED("cggen::resolve_pending_reductions", generic_red);
	auto accessed_buffers = tsk.get_buffer_access_map().get_accessed_buffers();
	// Handle chained reductions (i.e., reductions that combine into a buffer that currently is in a pending reduction state)
	for(const auto& reduction : tsk.get_reductions()) {
		accessed_buffers.insert(reduction.bid);
	}

	for(const auto bid : accessed_buffers) {
		auto& buffer = m_buffers.at(bid);
		if(!buffer.pending_reduction.has_value()) { continue; }
		const auto& reduction = *buffer.pending_reduction;

		const auto local_last_writer_set = buffer.local_last_writer.get_region_values(scalar_reduction_box);
		assert(local_last_writer_set.size() == 1);
		const auto local_last_writer = local_last_writer_set[0].second;

		// Prepare the buffer state for after the reduction has been performed:
		// Keep the current last writer, but mark it as stale, so that if we don't generate a reduction command locally,
		// we'll know to get the data from elsewhere. If we generate a reduction command, this will be overwritten by its command id.
		auto wcs = local_last_writer;
		wcs.mark_as_stale();
		buffer_state post_reduction_state(ones, wcs, node_bitset());
		if(m_policy.uninitialized_read_error != error_policy::ignore) { post_reduction_state.initialized_region = scalar_reduction_box; }

		node_bitset participating_nodes;

		// Since the local reduction command overwrites the buffer contents that need to be pushed to other nodes, we need to process remote chunks first.
		for(const auto& [a_chunk, requirements] : chunks_with_requirements.remote_chunks) {
			if(std::none_of(requirements.begin(), requirements.end(), [&](const buffer_requirements& br) { return br.bid == bid && !br.consumed.empty(); })) {
				// This chunk doesn't read from the buffer
				continue;
			}
			participating_nodes.set(a_chunk.executed_on);
		}

		// Generate push command to all participating nodes
		if(participating_nodes.any()) {
			// Push an empty range if we don't have any fresh data on this node. This will then generate an empty pilot that tells the
			// other node's receive_arbiter to not expect a send.
			const bool notification_only = !local_last_writer.is_fresh();
			const auto push_box = notification_only ? empty_reduction_box : scalar_reduction_box;
			assert(participating_nodes.count() == m_num_nodes - 1 || participating_nodes.count() == 1);
			std::vector<std::pair<node_id, region<3>>> regions;
			for(node_id nid = 0; nid < m_num_nodes; ++nid) {
				if(!participating_nodes.test(nid)) continue;
				regions.push_back({nid, push_box});
			}
			auto* const cmd = create_command<push_command>(current_batch, transfer_id(tsk.get_id(), bid, reduction.rid), std::move(regions),
			    [&, bid = bid](const auto& record_debug_info) { record_debug_info(m_buffers.at(bid).debug_name); });
			if(notification_only) {
				generate_epoch_dependencies(cmd);
			} else {
				m_command_buffer_reads[cmd->get_id()][bid] = region_union(m_command_buffer_reads[cmd->get_id()][bid], scalar_reduction_box);
				add_dependency(cmd, local_last_writer, dependency_kind::true_dep, dependency_origin::dataflow);
			}

			// Mark the reduction result as replicated so we don't generate data transfers to any of the participating nodes
			post_reduction_state.replicated_regions.update_box(scalar_reduction_box, participating_nodes);
		}

		// We currently don't support multiple chunks on a single node for reductions (there is also -- for now -- no way to create multiple chunks,
		// as oversubscription is handled by the instruction graph).
		// NOTE: The participating_nodes.count() check below relies on this being true
		assert(chunks_with_requirements.local_chunks.size() <= 1);
		for(const auto& [a_chunk, requirements] : chunks_with_requirements.local_chunks) {
			if(std::none_of(requirements.begin(), requirements.end(), [&](const buffer_requirements& br) { return br.bid == bid && !br.consumed.empty(); })) {
				// This chunk doesn't read from the buffer
				continue;
			}

			auto* const ap_cmd = create_command<await_push_command>(current_batch, transfer_id(tsk.get_id(), bid, reduction.rid),
			    scalar_reduction_box.get_subrange(), [&](const auto& record_debug_info) { record_debug_info(buffer.debug_name); });
			generate_epoch_dependencies(ap_cmd);

			auto* const reduce_cmd = create_command<reduction_command>(current_batch, reduction, local_last_writer.is_fresh() /* has_local_contribution */,
			    [&](const auto& record_debug_info) { record_debug_info(buffer.debug_name); });

			// Only generate a true dependency on the last writer if this node participated in the intermediate result computation.
			if(local_last_writer.is_fresh()) { add_dependency(reduce_cmd, local_last_writer, dependency_kind::true_dep, dependency_origin::dataflow); }
			add_dependency(reduce_cmd, ap_cmd, dependency_kind::true_dep, dependency_origin::dataflow);
			generate_anti_dependencies(tsk, bid, buffer.local_last_writer, scalar_reduction_box, reduce_cmd);

			post_reduction_state.local_last_writer.update_box(scalar_reduction_box, reduce_cmd);
			participating_nodes.set(m_local_nid); // We are participating
		}

		// We currently do not support generating reduction commands on only a subset of nodes, except for the special case of a single command.
		// This is because it is unclear who owns the final result in this case (normally all nodes "own" the result).
		//   => I.e., reducing and using the result on the participating nodes is actually not the problem (this works as intended); the issue only arises
		//      if the result is subsequently required in other tasks. Since we don't have a good way of detecting this condition however, we currently
		//      disallow partial reductions altogether.
		// NOTE: This check relies on the fact that we currently only generate a single chunk per node for reductions (see assertion above).
		if(participating_nodes.count() > 1 && participating_nodes.count() != m_num_nodes) {
			utils::report_error(error_policy::panic,
			    "{} requires a reduction on {} that is not performed on all nodes. This is currently not supported. Either "
			    "ensure that all nodes receive a chunk that reads from the buffer, or reduce the data on a single node.",
			    print_task_debug_label(tsk, true /* title case */), print_buffer_debug_label(bid));
		}

		// For buffers that were in a pending reduction state and a reduction was generated
		// (i.e., the result was not discarded), set their new state.
		if(participating_nodes.count() > 0) {
			m_completed_reductions.push_back(reduction.rid);
			buffer = std::move(post_reduction_state);
		}
	}
}

void command_graph_generator::generate_pushes(batch& current_batch, const task& tsk, const assigned_chunks_with_requirements& chunks_with_requirements) {
	CELERITY_DETAIL_TRACY_ZONE_SCOPED("cggen::generate_pushes", generic_red);
	struct push_scratch {
		std::unordered_map<node_id, region_builder<3>> target_regions;
		std::unordered_set<command*> depends_on;
	};
	std::unordered_map<buffer_id, push_scratch> per_buffer_pushes;

	for(auto& [a_chunk, requirements] : chunks_with_requirements.remote_chunks) {
		const node_id nid = a_chunk.executed_on;

		for(const auto& [bid, consumed, _] : requirements) {
			if(consumed.empty()) continue;
			auto& buffer = m_buffers.at(bid);

			const auto local_sources = buffer.local_last_writer.get_region_values(consumed);
			for(const auto& [local_box, wcs] : local_sources) {
				if(!wcs.is_fresh() || wcs.is_replicated()) { continue; }

				// Make sure we don't push anything we've already pushed to this node before
				region_builder<3> non_replicated_boxes;
				for(const auto& [replicated_box, nodes] : buffer.replicated_regions.get_region_values(local_box)) {
					if(nodes.test(nid)) continue;
					non_replicated_boxes.add(replicated_box);
				}

				if(!non_replicated_boxes.empty()) {
					assert(!utils::isa<await_push_command>(wcs.get_command()) && "Attempting to push non-owned data?!");
					auto push_region = std::move(non_replicated_boxes).into_region();
					// Remember that we've replicated this region
					for(const auto& [replicated_box, nodes] : buffer.replicated_regions.get_region_values(push_region)) {
						buffer.replicated_regions.update_box(replicated_box, node_bitset{nodes}.set(nid));
					}
					auto& scratch = per_buffer_pushes[bid]; // allow default-insert
					scratch.target_regions[nid /* allow default-insert */].add(push_region);
					scratch.depends_on.insert(wcs);
				}
			}
		}
	}

	// Generate push command for each buffer
	for(auto& [bid, scratch] : per_buffer_pushes) {
		region_builder<3> combined_region;
		std::vector<std::pair<node_id, region<3>>> target_regions;
		for(auto& [nid, boxes] : scratch.target_regions) {
			auto region = std::move(boxes).into_region();
			combined_region.add(region);
			target_regions.push_back({nid, std::move(region)});
		}

		auto* const cmd = create_command<push_command>(current_batch, transfer_id(tsk.get_id(), bid, no_reduction_id), std::move(target_regions),
		    [&, bid = bid](const auto& record_debug_info) { record_debug_info(m_buffers.at(bid).debug_name); });
		for(const auto dep : scratch.depends_on) {
			add_dependency(cmd, dep, dependency_kind::true_dep, dependency_origin::dataflow);
		}

		// Store the read access for determining anti-dependencies
		m_command_buffer_reads[cmd->get_id()].emplace(bid, std::move(combined_region).into_region());
	}
}

void command_graph_generator::generate_await_pushes(batch& current_batch, const task& tsk, const assigned_chunks_with_requirements& chunks_with_requirements) {
	CELERITY_DETAIL_TRACY_ZONE_SCOPED("cggen::generate_await_pushes", generic_red);
	std::unordered_map<buffer_id, region_builder<3>> per_buffer_required_boxes;

	for(auto& [a_chunk, requirements] : chunks_with_requirements.local_chunks) {
		for(auto& [bid, consumed, _] : requirements) {
			if(consumed.empty()) continue;
			auto& buffer = m_buffers.at(bid);

			const auto local_sources = buffer.local_last_writer.get_region_values(consumed);
			box_vector<3> missing_parts_boxes;
			for(const auto& [box, wcs] : local_sources) {
				// Note that we initialize all buffers as fresh, so this doesn't trigger for uninitialized reads
				if(!box.empty() && !wcs.is_fresh()) { missing_parts_boxes.push_back(box); }
			}

			// There is data we don't yet have locally. Generate an await push command for it.
			if(!missing_parts_boxes.empty()) {
				if(tsk.perf_assertions.assert_no_data_movement) {
					CELERITY_ERROR("Performance assertion at {}:{} failed: Chunk {} of task {} requires region {} of buffer {} from remote nodes",
					    tsk.perf_assertions.assert_no_data_movement_source_loc.file_name(), tsk.perf_assertions.assert_no_data_movement_source_loc.line(),
					    a_chunk.chnk, tsk.get_id(), region<3>(box_vector<3>{missing_parts_boxes}), bid);
				}

				assert(m_num_nodes > 1);
				auto& required_boxes = per_buffer_required_boxes[bid]; // allow default-insert
				required_boxes.add(missing_parts_boxes);
			}
		}
	}

	for(auto& [bid, boxes] : per_buffer_required_boxes) {
		auto& buffer = m_buffers.at(bid);
		auto region = std::move(boxes).into_region(); // moved-from after next line!
		auto* const ap_cmd = create_command<await_push_command>(current_batch, transfer_id(tsk.get_id(), bid, no_reduction_id), std::move(region),
		    [&](const auto& record_debug_info) { record_debug_info(buffer.debug_name); });
		generate_anti_dependencies(tsk, bid, buffer.local_last_writer, ap_cmd->get_region(), ap_cmd);
		generate_epoch_dependencies(ap_cmd);
		// Remember that we have this data now
		buffer.local_last_writer.update_region(ap_cmd->get_region(), {ap_cmd, true /* is_replicated */});
	}
}

void command_graph_generator::generate_local_execution_command(batch& current_batch, const task& tsk,
    const assigned_chunks_with_requirements& chunks_with_requirements, std::unordered_map<buffer_id, region<3>>& per_buffer_local_writes) {
	CELERITY_DETAIL_TRACY_ZONE_SCOPED("cggen::generate_local_execution_command", generic_blue);
	if(chunks_with_requirements.local_chunks.empty()) return;

	command* cmd = ([&]() -> command* {
		if(tsk.get_type() == task_type::fence) {
			return create_command<fence_command>(current_batch, &tsk,
			    [&](const auto& record_debug_info) { record_debug_info(tsk, [this](const buffer_id bid) { return m_buffers.at(bid).debug_name; }); });
		}

		// Go over all reductions that are to be performed *during* the execution of this chunk,
		// not to be confused with any pending reductions that need to be finalized *before* the
		// execution of this chunk (those have already been handled by resolve_pending_reductions).
		// If a reduction reads the previous value of the buffer (i.e. w/o property::reduction::initialize_to_identity),
		// we have to include it in exactly one of the per-node intermediate reductions.
		const bool is_reduction_initializer = std::any_of(tsk.get_reductions().begin(), tsk.get_reductions().end(),
		    [&](const auto& reduction) { return m_local_nid == reduction_initializer_nid && reduction.init_from_buffer; });
		execution_spec spec;
		if(chunks_with_requirements.local_chunks.size() == 1) {
			spec = chunks_with_requirements.local_chunks[0].first.chnk;
		} else {
			std::vector<device_execution_range> exec_ranges(chunks_with_requirements.local_chunks.size());
			for(size_t i = 0; i < chunks_with_requirements.local_chunks.size(); ++i) {
				const auto& [a_chunk, _] = chunks_with_requirements.local_chunks[i];
				// NOCOMMIT HACK: We produce multiple chunks if m_test_chunk_multiplier > 1, but those don't have device id.
				//                We should really just get rid of that multiplier, and change those tests to use custom task geometries.
				const auto did = a_chunk.target_device.value_or(-1);
				exec_ranges[i] = {a_chunk.chnk, did};
			}
			spec = std::move(exec_ranges);
		}
		return create_command<execution_command>(current_batch, &tsk, spec, is_reduction_initializer,
		    [&](const auto& record_debug_info) { record_debug_info(tsk, [this](const buffer_id bid) { return m_buffers.at(bid).debug_name; }); });
	})(); // IIFE

	if(tsk.get_type() == task_type::collective) {
		// Collective host tasks have an implicit dependency on the previous task in the same collective group,
		// which is required in order to guarantee they are executed in the same order on every node.
		auto cgid = tsk.get_collective_group_id();
		if(const auto cg = m_collective_groups.find(cgid); cg != m_collective_groups.end()) {
			add_dependency(cmd, cg->second.last_collective_command, dependency_kind::true_dep, dependency_origin::collective_group_serialization);
			cg->second.last_collective_command = cmd;
		} else {
			m_collective_groups.emplace(cgid, cmd);
		}
	}

	// Resolve local data dependencies
	// NOCOMMIT TODO: Factor this into a separate function?
	// Process consuming accesses first, so we don't add dependencies onto our own writes
	std::unordered_map<buffer_id, region_builder<3>> per_buffer_consumed;
	for(const auto& [a_chunk, requirements] : chunks_with_requirements.local_chunks) {
		for(const auto& [bid, consumed, _] : requirements) {
			if(consumed.empty()) continue;
			auto& buffer = m_buffers.at(bid);

			for(const auto& [box, wcs] : buffer.local_last_writer.get_region_values(consumed)) {
				if(box.empty()) continue;
				assert(wcs.is_fresh() && "Unresolved remote data dependency");
				add_dependency(cmd, wcs, dependency_kind::true_dep, dependency_origin::dataflow);
			}

			per_buffer_consumed[bid].add(consumed); // allow default-insert
			if(m_policy.uninitialized_read_error != error_policy::ignore) {
				if(const auto uninitialized_reads = region_difference(consumed, buffer.initialized_region); !uninitialized_reads.empty()) {
					// TODO: Print target device if available
					utils::report_error(m_policy.uninitialized_read_error,
					    "Command C{} on N{}, which executes {} of {}, reads {} {}, which has not been written by any node.", cmd->get_id(), m_local_nid,
					    box(subrange(a_chunk.chnk.offset, a_chunk.chnk.range)), print_task_debug_label(tsk), print_buffer_debug_label(bid),
					    uninitialized_reads);
				}
			}
		}
	}
	for(auto& [bid, builder] : per_buffer_consumed) {
		// Store the read access for determining anti-dependencies later on
		m_command_buffer_reads[cmd->get_id()].emplace(bid, std::move(builder).into_region());
	}

	std::unordered_map<buffer_id, region_builder<3>> per_buffer_produced;
	for(const auto& [a_chunk, requirements] : chunks_with_requirements.local_chunks) {
		for(const auto& [bid, _, produced] : requirements) {
			if(produced.empty()) continue;
			per_buffer_produced[bid].add(produced); // allow default-insert
		}
	}
	for(auto& [bid, builder] : per_buffer_produced) {
		auto& buffer = m_buffers.at(bid);
		auto produced = std::move(builder).into_region();
		generate_anti_dependencies(tsk, bid, buffer.local_last_writer, produced, cmd);

		// Update last writer
		buffer.local_last_writer.update_region(produced, cmd);
		buffer.replicated_regions.update_region(produced, node_bitset{});

		// In case this buffer was in a pending reduction state we discarded the result and need to remove the pending reduction.
		if(buffer.pending_reduction.has_value()) {
			m_completed_reductions.push_back(buffer.pending_reduction->rid);
			buffer.pending_reduction = std::nullopt;
		}

		per_buffer_local_writes[bid] = std::move(produced);
	}

	for(const auto& side_effect : tsk.get_side_effect_map()) {
		const auto [hoid, order] = side_effect;
		auto& host_object = m_host_objects.at(hoid);

		if(host_object.last_side_effect != nullptr) {
			// TODO once we have different side_effect_orders, their interaction will determine the dependency kind
			add_dependency(cmd, host_object.last_side_effect, dependency_kind::true_dep, dependency_origin::dataflow);
		}

		// Simplification: If there are multiple chunks per node, we generate true-dependencies between them in an arbitrary order, when all we really
		// need is mutual exclusion (i.e. a bi-directional pseudo-dependency).
		host_object.last_side_effect = cmd;
	}

	generate_epoch_dependencies(cmd);
}

void command_graph_generator::update_local_buffer_fresh_regions(
    const task& tsk, const std::unordered_map<buffer_id, region<3>>& per_buffer_local_writes, const std::vector<assigned_chunk>& chunks) {
	CELERITY_DETAIL_TRACY_ZONE_SCOPED("cggen::update_local_buffer_fresh_regions", generic_green);
	buffer_requirements_list requirements;
	for(const auto bid : tsk.get_buffer_access_map().get_accessed_buffers()) {
		const auto& bam = tsk.get_buffer_access_map();
		requirements.push_back({bid, bam.get_task_consumed_region(bid), bam.get_task_produced_region(bid)});
	}
	// Add requirements for reductions
	for(const auto& reduction : tsk.get_reductions()) {
		auto it = std::find_if(requirements.begin(), requirements.end(), [&](const buffer_requirements& br) { return br.bid == reduction.bid; });
		if(it == requirements.end()) { it = requirements.insert(requirements.end(), buffer_requirements{reduction.bid, {}, {}}); }
		it->produced = scalar_reduction_box;
	}
	for(auto& [bid, _, produced] : requirements) {
		region global_writes = produced;
		auto& buffer = m_buffers.at(bid);
		if(m_policy.uninitialized_read_error != error_policy::ignore) { buffer.initialized_region = region_union(buffer.initialized_region, global_writes); }

		const auto remote_writes = ([&] {
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
				buffer.local_last_writer.update_box(box, wcs);
			}
		}
	}

	// Handle replicated writes
	// NOCOMMIT TODO: General question - should writes above even include replicated writes? Or do we just override everything again here anyway..?
	// TODO: What happens if there are multiple replicated writes to the same buffer in a single task? Can/should we allow that?
	// NOCOMMIT TODO: Add benchmark
	// TODO: Add some trace logging that heuristic is being used b/c no canonical last writers were provided

	for(size_t i = 0; i < tsk.get_buffer_access_map().get_num_accesses(); ++i) {
		const auto& [bid, mode] = tsk.get_buffer_access_map().get_nth_access(i);
		if(!tsk.get_buffer_access_map().is_replicated(i)) continue;
		assert(is_producer_mode(mode));
		auto& buffer = m_buffers.at(bid);

		// Compute locally produced region
		region_builder<3> locally_produced_builder;
		for(const auto& a_chunk : chunks) {
			if(a_chunk.executed_on != m_local_nid) continue;
			locally_produced_builder.add(tsk.get_buffer_access_map().get_requirements_for_nth_access(i, box<3>(a_chunk.chnk)));
		}
		const auto locally_produced = std::move(locally_produced_builder).into_region();

		std::unordered_map<node_id, region<3>> produced_by_node;  // TODO: Use builder?
		region_map<node_bitset> overlapping_regions(buffer.size); // TODO: Can we avoid region map?

		for(const auto& a_chunk : chunks) {
			if(a_chunk.executed_on == m_local_nid) continue;
			const auto remote_produced = tsk.get_buffer_access_map().get_requirements_for_nth_access(i, box<3>(a_chunk.chnk));
			// NOCOMMIT TODO: This might be expensive if we have a lot of chunks. Do broad-phase on bounding boxes first..?
			const auto intersection = region_intersection(locally_produced, remote_produced);
			if(!intersection.empty()) {
				produced_by_node[a_chunk.executed_on] = remote_produced;
				for(auto& [box, nodes] : overlapping_regions.get_region_values(intersection)) {
					nodes.set(a_chunk.executed_on);
					overlapping_regions.update_box(box, nodes);
				}
			}
		}

		const auto for_all_bits = [](const node_bitset& nodes, const auto& f) {
			uint64_t bitset = nodes.to_ullong();
			while(bitset != 0) {
				const uint64_t t = bitset & -bitset;
				const int r = __builtin_ctzl(bitset);
				f(r);
				bitset ^= t;
			}
		};

		const auto first_bit = [](const node_bitset& nodes) {
			const uint64_t bitset = nodes.to_ullong();
			return __builtin_ctzl(bitset);
		};

		// TODO: Use float type for this..?
		const auto center = [](const box<3>& b) { return b.get_min() + (b.get_range() / 2); };

		struct vec {
			int64_t coords[3];

			vec(const int64_t dim0, const int64_t dim1, const int64_t dim2) : coords{dim0, dim1, dim2} {}
			explicit(false) vec(const id<3>& id) : coords{static_cast<int64_t>(id[0]), static_cast<int64_t>(id[1]), static_cast<int64_t>(id[2])} {}

			vec operator-(const vec& other) const { return {coords[0] - other.coords[0], coords[1] - other.coords[1], coords[2] - other.coords[2]}; }
			int64_t operator[](const size_t idx) const {
				assert(idx < 3);
				return coords[idx];
			}
			int64_t length_squared() const { return coords[0] * coords[0] + coords[1] * coords[1] + coords[2] * coords[2]; }
		};

		for(const auto& [overlap_box, nodes] : overlapping_regions.get_region_values(box<3>::full_range(buffer.size))) {
			if(nodes.count() == 0) continue;
			assert(nodes.count() <= m_num_nodes);
			// Mark as replicated so we don't generate data transfers to any of the participating nodes
			// when they read from the region in subsequent tasks.
			// TODO: We only need to do this if we are the canonical last writer. Otherwise we won't attempt to push anyway.
			buffer.replicated_regions.update_box(overlap_box, nodes);

			// TODO: If all nodes are set, we don't need to find a canonical last writer

			if(nodes.count() == 1) {
				const node_id other_nid = first_bit(nodes);
				// TODO: Compute centroid instead (also O(N))? Should we only consider boxes that touch the overlapping region?
				const vec other_center = center(bounding_box(produced_by_node.at(other_nid)));
				const vec my_center = center(bounding_box(locally_produced));

				const auto dist = my_center - other_center;
				// Choose largest component as split dimension; prefer splitting along lower dimensions (>=).
				const auto split_dimension = std::abs(dist[0]) >= std::abs(dist[1]) ? (std::abs(dist[0]) >= std::abs(dist[2]) ? 0 : 2)
				                                                                    : (std::abs(dist[1]) >= std::abs(dist[2]) ? 1 : 2);

				// NOCOMMIT TODO Handle case where box cannot be split along this dimension (extent 1)

				// NOCOMMIT TODO: Do we need to consider rounding?
				auto split_point = overlap_box.get_min()[split_dimension] + (overlap_box.get_range()[split_dimension] / 2);
				auto box1max = overlap_box.get_max();
				box1max[split_dimension] = split_point;
				auto box2min = overlap_box.get_min();
				box2min[split_dimension] = split_point;
				auto box1 = box<3>{overlap_box.get_min(), box1max};
				auto box2 = box<3>{box2min, overlap_box.get_max()};

				// TODO: Can we infer this from sign of dist[split_dimension]..?
				if((vec{center(box1)} - my_center).length_squared() > (vec{center(box2)} - my_center).length_squared()) { std::swap(box1, box2); }
				for(auto& [box, wcs] : buffer.local_last_writer.get_region_values(box2)) {
					assert(wcs.is_fresh() && !wcs.is_replicated());
					// fmt::print("SPLIT CASE! I am node {} and I am marking {} as replicated\n", m_local_nid, box);
					buffer.local_last_writer.update_region(box, write_command_state(wcs.get_command(), true /* is_replicated */));
				}
			} else {
				// If there's more than two participating nodes, we fall back to a simple distance heuristic.
				// TODO: In case of ties we might want to do a round robin or randomized assignment for better load balancing.
				// NOCOMMIT: This is wrong, actually. Need have deterministic order in which we process nodes.
				const vec box_center = center(overlap_box);
				int64_t min_distance = (vec{center(bounding_box(locally_produced))} - box_center).length_squared();
				node_id min_distance_nid = m_local_nid;
				for_all_bits(nodes, [&](const node_id nid) {
					const vec node_center = center(bounding_box(produced_by_node[nid]));
					const auto distance = (node_center - box_center).length_squared();
					if(distance < min_distance) {
						min_distance = distance;
						min_distance_nid = nid;
					}
				});
				if(min_distance_nid != m_local_nid) {
					for(auto& [box, wcs] : buffer.local_last_writer.get_region_values(overlap_box)) {
						assert(wcs.is_fresh() && !wcs.is_replicated());
						// fmt::print("DISTANCE CASE! I am node {} and I am marking {} as replicated because canonical last writer is {}\n", m_local_nid, box,
						//     min_distance_nid);
						buffer.local_last_writer.update_region(box, write_command_state(wcs.get_command(), true /* is_replicated */));
					}
				}
			}
		}
	}
}

void command_graph_generator::generate_distributed_commands(batch& current_batch, const task& tsk) {
	CELERITY_DETAIL_TRACY_ZONE_SCOPED("cggen::generate_distributed_commands", generic_yellow);
	const auto chunks = split_task_and_assign_chunks(tsk);
	const auto chunks_with_requirements = compute_per_chunk_requirements(tsk, chunks);

	// Check for and report overlapping writes between local chunks, and between local and remote chunks.
	if(m_policy.overlapping_write_error != error_policy::ignore) { report_overlapping_writes(tsk, chunks); }

	resolve_pending_reductions(current_batch, tsk, chunks_with_requirements);
	generate_pushes(current_batch, tsk, chunks_with_requirements);
	generate_await_pushes(current_batch, tsk, chunks_with_requirements);
	// Union of all per-buffer writes on this node, used to determine which parts of a buffer are fresh/stale later on.
	std::unordered_map<buffer_id, region<3>> per_buffer_local_writes;
	generate_local_execution_command(current_batch, tsk, chunks_with_requirements, per_buffer_local_writes);

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
	update_local_buffer_fresh_regions(tsk, per_buffer_local_writes, chunks);
}

void command_graph_generator::generate_anti_dependencies(
    const task& tsk, const buffer_id bid, const region_map<write_command_state>& last_writers_map, const region<3>& write_req, command* const write_cmd) //
{
	const auto last_writers = last_writers_map.get_region_values(write_req);
	for(const auto& [box, wcs] : last_writers) {
		auto* const last_writer_cmd = wcs.get_command();
		// NOCOMMIT This breaks for replicated writes with multiple chunks per node
		// assert(!utils::isa<task_command>(last_writer_cmd) || utils::as<task_command>(last_writer_cmd)->get_task() != &tsk);

		// Add anti-dependencies onto all successors of the writer
		bool has_successors = false;
		for(auto d : last_writer_cmd->get_dependents()) {
			// Only consider true dependencies
			if(d.kind != dependency_kind::true_dep) continue;

			auto* const cmd = d.node;

			// We might have already generated new commands within the same task that also depend on this; in that case, skip it
			if(utils::isa<task_command>(cmd) && utils::as<task_command>(cmd)->get_task() == &tsk) continue;

			// So far we don't know whether the dependent actually intersects with the subrange we're writing
			if(const auto command_reads_it = m_command_buffer_reads.find(cmd->get_id()); command_reads_it != m_command_buffer_reads.end()) {
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

void command_graph_generator::set_epoch_for_new_commands(command* const epoch_or_horizon) {
	// both an explicit epoch command and an applied horizon can be effective epochs
	assert(utils::isa<epoch_command>(epoch_or_horizon) || utils::isa<horizon_command>(epoch_or_horizon));

	for(auto& [_, buffer] : m_buffers) {
		buffer.local_last_writer.apply_to_values([epoch_or_horizon](write_command_state wcs) {
			if(epoch_or_horizon->get_id() <= wcs.get_command()->get_id()) return wcs;
			write_command_state new_wcs(epoch_or_horizon, wcs.is_replicated());
			if(!wcs.is_fresh()) new_wcs.mark_as_stale();
			return new_wcs;
		});
	}
	for(auto& [_, host_object] : m_host_objects) {
		if(host_object.last_side_effect != nullptr && host_object.last_side_effect->get_id() < epoch_or_horizon->get_id()) {
			host_object.last_side_effect = epoch_or_horizon;
		}
	}
	for(auto& [cgid, collective_group] : m_collective_groups) {
		if(collective_group.last_collective_command->get_id() < epoch_or_horizon->get_id()) { collective_group.last_collective_command = epoch_or_horizon; }
	}

	m_epoch_for_new_commands = epoch_or_horizon;
}

void command_graph_generator::reduce_execution_front_to(command* const new_front) {
	const auto previous_execution_front = m_execution_front; // modified inside loop through add_dependency
	for(auto* const front_cmd : previous_execution_front) {
		if(front_cmd != new_front) { add_dependency(new_front, front_cmd, dependency_kind::true_dep, dependency_origin::execution_front); }
	}
	assert(m_execution_front.size() == 1 && *m_execution_front.begin() == new_front);
}

void command_graph_generator::generate_epoch_command(batch& current_batch, const task& tsk) {
	assert(tsk.get_type() == task_type::epoch);
	m_cdag->begin_epoch(tsk.get_id());
	auto* const epoch = create_command<epoch_command>(
	    current_batch, &tsk, tsk.get_epoch_action(), std::move(m_completed_reductions), [&](const auto& record_debug_info) { record_debug_info(tsk); });
	set_epoch_for_new_commands(epoch);
	m_current_horizon = no_command;
	// Make the epoch depend on the previous execution front
	reduce_execution_front_to(epoch);
}

void command_graph_generator::generate_horizon_command(batch& current_batch, const task& tsk) {
	assert(tsk.get_type() == task_type::horizon);
	m_cdag->begin_epoch(tsk.get_id());
	auto* const new_horizon =
	    create_command<horizon_command>(current_batch, &tsk, std::move(m_completed_reductions), [&](const auto& record_debug_info) { record_debug_info(tsk); });

	if(m_current_horizon != nullptr) {
		// Apply the previous horizon
		set_epoch_for_new_commands(m_current_horizon);
	}
	m_current_horizon = new_horizon;

	// Make the horizon depend on the previous execution front
	reduce_execution_front_to(new_horizon);
}

void command_graph_generator::generate_epoch_dependencies(command* cmd) {
	// No command must be re-ordered before its last preceding epoch to enforce the barrier semantics of epochs.
	// To guarantee that each node has a transitive true dependency (=temporal dependency) on the epoch, it is enough to add an epoch -> command dependency
	// to any command that has no other true dependencies itself and no graph traversal is necessary. This can be verified by a simple induction proof.

	// As long the first epoch is present in the graph, all transitive dependencies will be visible and the initial epoch commands (tid 0) are the only
	// commands with no true predecessor. As soon as the first epoch is pruned through the horizon mechanism however, more than one node with no true
	// predecessor can appear (when visualizing the graph). This does not violate the ordering constraint however, because all "free-floating" nodes
	// in that snapshot had a true-dependency chain to their predecessor epoch at the point they were flushed, which is sufficient for following the
	// dependency chain from the executor perspective.

	if(const auto deps = cmd->get_dependencies();
	    std::none_of(deps.begin(), deps.end(), [](const command::dependency d) { return d.kind == dependency_kind::true_dep; })) {
		if(!utils::isa<epoch_command>(cmd) || utils::as<epoch_command>(cmd)->get_epoch_action() != epoch_action::init) {
			assert(cmd != m_epoch_for_new_commands);
			add_dependency(cmd, m_epoch_for_new_commands, dependency_kind::true_dep, dependency_origin::last_epoch);
		}
	}
}

std::string command_graph_generator::print_buffer_debug_label(const buffer_id bid) const {
	return utils::make_buffer_debug_label(bid, m_buffers.at(bid).debug_name);
}

} // namespace celerity::detail
