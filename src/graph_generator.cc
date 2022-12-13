#include "graph_generator.h"

#include "access_modes.h"
#include "command.h"
#include "command_graph.h"
#include "graph_transformer.h"
#include "task.h"
#include "task_manager.h"

namespace celerity {
namespace detail {

	graph_generator::graph_generator(size_t num_nodes, command_graph& cdag) : m_num_nodes(num_nodes), m_cdag(cdag) {
		// Build initial epoch command for each node (these are required to properly handle anti-dependencies on host-initialized buffers).
		// We manually generate the first set of commands, these will be replaced by applied horizons or explicit epochs down the line (see
		// set_epoch_for_new_commands).
		for(node_id nid = 0; nid < num_nodes; ++nid) {
			const auto epoch_cmd = cdag.create<epoch_command>(nid, task_manager::initial_epoch_task, epoch_action::none);
			epoch_cmd->mark_as_flushed(); // there is no point in flushing the initial epoch command
			m_node_data[nid].epoch_for_new_commands = epoch_cmd->get_cid();
		}
	}

	void graph_generator::add_buffer(buffer_id bid, const cl::sycl::range<3>& range) {
		// Initialize the whole range to all nodes, so that we always use local buffer ranges when they haven't been written to (on any node) yet.
		// TODO: Consider better handling for when buffers are not host initialized
		std::vector<node_id> all_nodes(m_num_nodes);
		for(auto i = 0u; i < m_num_nodes; ++i) {
			all_nodes[i] = i;
			m_node_data[i].buffer_last_writer.emplace(bid, range);
			m_node_data[i].buffer_last_writer.at(bid).update_region(subrange_to_grid_box({cl::sycl::id<3>(), range}), m_node_data[i].epoch_for_new_commands);
		}

		m_buffer_states.emplace(bid, distributed_state{{range, std::move(all_nodes)}});
	}

	void graph_generator::build_task(const task& tsk, const std::vector<graph_transformer*>& transformers) {
		// TODO: Maybe assert that this task hasn't been processed before

		const auto tid = tsk.get_id();
		const auto min_epoch_to_prune_before = m_min_epoch_for_new_commands;

		switch(tsk.get_type()) {
		case task_type::epoch: generate_epoch_commands(tsk); break;
		case task_type::horizon: generate_horizon_commands(tsk); break;
		case task_type::collective: generate_collective_execution_commands(tsk); break;
		case task_type::host_compute:
		case task_type::device_compute:
		case task_type::master_node: generate_independent_execution_commands(tsk); break;
		case task_type::fence: generate_fence_commands(tsk); break;
		}

		for(auto& t : transformers) {
			t->transform_task(tsk, m_cdag);
		}

#ifndef NDEBUG
		// It is currently undefined to split reduction-producer tasks into multiple chunks on the same node:
		//   - Per-node reduction intermediate results are stored with fixed access to a single SYCL buffer, so multiple chunks on the same node will race
		//   on
		//     this buffer access
		//   - Inputs to the final reduction command are ordered by origin node ids to guarantee bit-identical results. It is not possible to distinguish
		//     more than one chunk per node in the serialized commands, so different nodes can produce different final reduction results for non-associative
		//     or non-commutative operations
		if(!tsk.get_reductions().empty()) {
			std::unordered_set<node_id> producer_nids;
			for(auto& cmd : m_cdag.task_commands(tid)) {
				assert(producer_nids.insert(cmd->get_nid()).second);
			}
		}
#endif

		// TODO: At some point we might want to do this also before calling transformers
		// --> So that more advanced transformations can also take data transfers into account
		process_task_data_requirements(tsk);
		process_task_side_effect_requirements(tsk);

		// Commands without any other true-dependency must depend on the active epoch command to ensure they cannot be re-ordered before the epoch
		for(const auto cmd : m_cdag.task_commands(tid)) {
			generate_epoch_dependencies(cmd);
		}

		// If a new epoch was completed in the CDAG before the current task, prune all predecessor commands of that epoch.
		// Also removes these commands from command_buffer_reads (if it exists)
		prune_commands_before(min_epoch_to_prune_before);
	}

	void graph_generator::set_epoch_for_new_commands(per_node_data& node, const command_id epoch) { // NOLINT(readability-convert-member-functions-to-static)
		// both an explicit epoch command and an applied horizon can be effective epochs
		assert(isa<epoch_command>(m_cdag.get(epoch)) || isa<horizon_command>(m_cdag.get(epoch)));

		// update "buffer_last_writer" and "last_collective_commands" structures to subsume pre-epoch commands
		for(auto& blw_pair : node.buffer_last_writer) {
			// TODO this could be optimized to something like cdag.apply_horizon(node_id, horizon_cmd) with much fewer internal operations
			blw_pair.second.apply_to_values([epoch](const std::optional<command_id> cid) -> std::optional<command_id> {
				if(!cid) return cid;
				return {std::max(epoch, *cid)};
			});
		}
		for(auto& [cgid, cid] : node.last_collective_commands) {
			cid = std::max(epoch, cid);
		}
		for(auto& [cgid, cid] : node.host_object_last_effects) {
			cid = std::max(epoch, cid);
		}

		node.epoch_for_new_commands = epoch;
	}

	void graph_generator::reduce_execution_front_to(abstract_command* const new_front) {
		const auto nid = new_front->get_nid();
		const auto previous_execution_front = m_cdag.get_execution_front(nid);
		for(const auto front_cmd : previous_execution_front) {
			if(front_cmd != new_front) { m_cdag.add_dependency(new_front, front_cmd, dependency_kind::true_dep, dependency_origin::execution_front); }
		}
		assert(m_cdag.get_execution_front(nid).size() == 1 && *m_cdag.get_execution_front(nid).begin() == new_front);
	}

	void graph_generator::generate_epoch_commands(const task& tsk) {
		assert(tsk.get_type() == task_type::epoch);

		command_id min_new_epoch;
		for(node_id nid = 0; nid < m_num_nodes; ++nid) {
			auto& node = m_node_data.at(nid);

			const auto epoch = m_cdag.create<epoch_command>(nid, tsk.get_id(), tsk.get_epoch_action());
			const auto cid = epoch->get_cid();
			if(nid == 0) { min_new_epoch = cid; }

			set_epoch_for_new_commands(node, cid);
			node.current_horizon = std::nullopt;

			// Make the epoch depend on the previous execution front
			reduce_execution_front_to(epoch);
		}

		m_min_epoch_for_new_commands = min_new_epoch;
	}

	void graph_generator::generate_horizon_commands(const task& tsk) {
		assert(tsk.get_type() == task_type::horizon);

		std::optional<command_id> min_new_epoch;
		for(node_id nid = 0; nid < m_num_nodes; ++nid) {
			auto& node = m_node_data.at(nid);

			const auto horizon = m_cdag.create<horizon_command>(nid, tsk.get_id());
			const auto cid = horizon->get_cid();

			if(node.current_horizon) {
				if(min_new_epoch) {
					min_new_epoch = std::min(*min_new_epoch, *node.current_horizon);
				} else {
					min_new_epoch = node.current_horizon;
				}
				set_epoch_for_new_commands(node, *node.current_horizon);
			}
			node.current_horizon = cid;

			// Make the horizon depend on the previous execution front
			reduce_execution_front_to(horizon);
		}

		if(min_new_epoch) {
			assert(!min_new_epoch.has_value() || m_min_epoch_for_new_commands < *min_new_epoch);
			m_min_epoch_for_new_commands = *min_new_epoch;
		}
	}

	void graph_generator::generate_collective_execution_commands(const task& tsk) {
		assert(tsk.get_type() == task_type::collective);

		for(size_t nid = 0; nid < m_num_nodes; ++nid) {
			auto offset = cl::sycl::id<1>{nid};
			auto range = cl::sycl::range<1>{1};
			const auto sr = subrange_cast<3>(subrange<1>{offset, range});
			auto* cmd = m_cdag.create<execution_command>(nid, tsk.get_id(), sr);

			// Collective host tasks have an implicit dependency on the previous task in the same collective group, which is required in order to guarantee
			// they are executed in the same order on every node.
			auto cgid = tsk.get_collective_group_id();
			auto& last_collective_commands = m_node_data.at(nid).last_collective_commands;
			if(auto prev = last_collective_commands.find(cgid); prev != last_collective_commands.end()) {
				m_cdag.add_dependency(cmd, m_cdag.get(prev->second), dependency_kind::true_dep, dependency_origin::collective_group_serialization);
				last_collective_commands.erase(prev);
			}
			last_collective_commands.emplace(cgid, cmd->get_cid());
		}
	}

	void graph_generator::generate_independent_execution_commands(const task& tsk) {
		assert(tsk.get_type() == task_type::host_compute || tsk.get_type() == task_type::device_compute || tsk.get_type() == task_type::master_node);

		const auto sr = subrange<3>{tsk.get_global_offset(), tsk.get_global_size()};
		m_cdag.create<execution_command>(0, tsk.get_id(), sr);
	}

	void graph_generator::generate_fence_commands(const task& tsk) {
		assert(tsk.get_type() == task_type::fence);
		for(size_t nid = 0; nid < m_num_nodes; ++nid) {
			m_cdag.create<fence_command>(nid, tsk.get_id());
		}
	}

	using buffer_requirements_map = std::unordered_map<buffer_id, std::unordered_map<cl::sycl::access::mode, GridRegion<3>>>;

	buffer_requirements_map get_buffer_requirements_for_mapped_access(const task& tsk, subrange<3> sr, const cl::sycl::range<3> global_size) {
		buffer_requirements_map result;
		const auto& access_map = tsk.get_buffer_access_map();
		const auto buffers = access_map.get_accessed_buffers();
		for(const buffer_id bid : buffers) {
			const auto modes = access_map.get_access_modes(bid);
			for(auto m : modes) {
				result[bid][m] = access_map.get_requirements_for_access(bid, m, tsk.get_dimensions(), sr, global_size);
			}
		}
		return result;
	}

	void graph_generator::generate_anti_dependencies(task_id tid, buffer_id bid, const region_map<std::optional<command_id>>& last_writers_map,
	    const GridRegion<3>& write_req, abstract_command* write_cmd) {
		const auto last_writers = last_writers_map.get_region_values(write_req);
		for(auto& box_and_writers : last_writers) {
			if(box_and_writers.second == std::nullopt) continue;
			const command_id last_writer_cid = *box_and_writers.second;
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

	namespace {
		template <typename RegionMap>
		inline void add_dependencies_for_box(command_graph& cdag, abstract_command* cmd, const RegionMap& map, const GridBox<3>& box) {
			auto sources = map.get_region_values(box);
			for(const auto& source : sources) {
				auto source_cmd = cdag.get(*source.second);
				cdag.add_dependency(cmd, source_cmd, dependency_kind::true_dep, dependency_origin::dataflow);
			}
		}
	} // namespace

	void graph_generator::process_task_data_requirements(const task& tsk) {
		const task_id tid = tsk.get_id();

		// Copy the list of task commands so we can safely modify the command graph in the loop below
		// NOTE: We assume that none of these commands are deleted
		const auto task_commands = m_cdag.task_commands(tid);

		// In reductions including the current buffer value, we need to designate a single command that will read from the output buffer. To avoid unnecessary
		// data transfers to that reader, we choose a command on a node where that buffer is already present. We cannot in general guarantee an optimal choice
		// for all reductions of a task, since communicating this choice to workers must happen through the fixed-size command_data. Instead, we try to find a
		// single command_id that can initialize all reductions without requiring transfers and settle for one that requires a small-ish number of transfers if
		// that is not possible.
		command_id reduction_initializer_cid = 0;
		{
			std::unordered_set<command_id> optimal_reduction_initializer_cids;
			for(const auto& reduction : tsk.get_reductions()) {
				if(reduction.init_from_buffer) {
					if(optimal_reduction_initializer_cids.empty()) {
						// lazy-initialize (there will be no buffer-initialized reductions most of the time)
						optimal_reduction_initializer_cids.reserve(task_commands.size());
						for(auto* cmd : task_commands) {
							optimal_reduction_initializer_cids.emplace(cmd->get_cid());
						}
					}

					// If there is a command that does not need a transfer for *some* of the reductions, that's better than nothing
					reduction_initializer_cid = *optimal_reduction_initializer_cids.begin();

					auto& buffer_state = m_buffer_states.at(reduction.bid);
					if(auto* distr_state = std::get_if<distributed_state>(&buffer_state)) {
						// set intersection: remove all commands on a node where the initial value is not present
						const GridBox<3> box{GridPoint<3>{1, 1, 1}};
						for(auto& [box, nids] : distr_state->region_sources.get_region_values(box)) {
							for(auto* cmd : task_commands) {
								if(std::find(nids.begin(), nids.end(), cmd->get_nid()) == nids.end()) {
									optimal_reduction_initializer_cids.erase(cmd->get_cid());
								}
							}
						}
					} // else in pending_reduction_state, we will have transfers anyway

					if(optimal_reduction_initializer_cids.empty()) break;
				}
			}

			if(!optimal_reduction_initializer_cids.empty()) {
				// There actually is a command that can initialize all reductions without a transfer
				reduction_initializer_cid = *optimal_reduction_initializer_cids.begin();
			}
		}

		// Store a list of writes (directly done by computation) while processing commands,
		// so that we can update the state with them at the end, while using only the state
		// before this task while resolving its dependencies.
		std::vector<std::tuple<node_id, buffer_id, GridRegion<3>>> buffer_state_write_list;
		std::vector<std::tuple<node_id, buffer_id, GridRegion<3>, command_id>> per_node_last_writer_update_list;
		std::unordered_map<buffer_id, std::vector<node_id>> buffer_reduction_resolve_list;

		// Remember all generated pushes for determining intra-task anti-dependencies.
		std::vector<push_command*> generated_pushes;

		for(auto* cmd : task_commands) {
			const command_id cid = cmd->get_cid();
			const node_id nid = cmd->get_nid();

			buffer_requirements_map requirements;
			if(auto* ecmd = dynamic_cast<execution_command*>(cmd)) {
				ecmd->set_is_reduction_initializer(cid == reduction_initializer_cid);

				requirements = get_buffer_requirements_for_mapped_access(tsk, ecmd->get_execution_range(), tsk.get_global_size());

				// Any reduction that includes the value previously found in the buffer (i.e. the absence of sycl::property::reduction::initialize_to_identity)
				// must read that original value in the eventual reduction_command generated by a future buffer requirement. Since whenever a buffer is used as
				// a reduction output, we replace its state with a pending_reduction_state, that original value would be lost. To avoid duplicating the buffer,
				// we simply include it in the pre-reduced state of a single execution_command.
				for(const auto& reduction : tsk.get_reductions()) {
					auto rmode = cl::sycl::access::mode::discard_write;
					if(ecmd->is_reduction_initializer() && reduction.init_from_buffer) { rmode = cl::sycl::access::mode::read_write; }

#ifndef NDEBUG
					for(auto pmode : detail::access::producer_modes) {
						assert(requirements[reduction.bid].count(pmode) == 0); // task_manager verifies that there are no reduction <-> write-access conflicts
					}
#endif

					// We need to add a proper requirement here because bid might itself be in pending_reduction_state
					requirements[reduction.bid][rmode] = GridRegion<3>{{1, 1, 1}};
				}
			} else {
				// tasks without an execution range (e.g. fences) can still have fixed or all-accesses.
				requirements = get_buffer_requirements_for_mapped_access(tsk, {}, {});
			}

			for(auto& it : requirements) {
				const buffer_id bid = it.first;
				const auto& reqs_by_mode = it.second;

				auto& buffer_state = m_buffer_states.at(bid);
				const auto& node_buffer_last_writer = m_node_data.at(nid).buffer_last_writer.at(bid);

				std::vector<cl::sycl::access::mode> required_modes;
				for(const auto mode : detail::access::all_modes) {
					if(auto req_it = reqs_by_mode.find(mode); req_it != reqs_by_mode.end()) {
						// While uncommon, we do support chunks that don't require access to a particular buffer at all.
						if(!req_it->second.empty()) { required_modes.push_back(mode); }
					}
				}

				// Don't add reduction commands within the loop to make sure there is at most one reduction command even in the presence of multiple
				// consumer requirements
				const bool is_pending_reduction = std::holds_alternative<pending_reduction_state>(buffer_state);
				const bool generate_reduction =
				    is_pending_reduction && std::any_of(required_modes.begin(), required_modes.end(), detail::access::mode_traits::is_consumer);

				if(is_pending_reduction && !generate_reduction) {
					// TODO the per-node reduction result is discarded - warn user about dead store
				}

				for(const auto mode : required_modes) {
					const auto& req = reqs_by_mode.at(mode);
					if(detail::access::mode_traits::is_consumer(mode)) {
						// Store the read access for determining anti-dependencies later on
						m_command_buffer_reads[cid][bid] = GridRegion<3>::merge(m_command_buffer_reads[cid][bid], req);

						if(auto* distributed = std::get_if<distributed_state>(&buffer_state)) {
							// Determine whether data transfers are required to fulfill the read requirements
							const auto buffer_source_nodes = distributed->region_sources.get_region_values(req);
							assert(!buffer_source_nodes.empty());

							for(const auto& [box, box_sources] : buffer_source_nodes) {
								assert(!box_sources.empty());

								if(std::find(box_sources.cbegin(), box_sources.cend(), nid) != box_sources.cend()) {
									// No need to push if found locally, but make sure to add dependencies
									add_dependencies_for_box(m_cdag, cmd, node_buffer_last_writer, box);
									continue;
								}

								// If not local, the original producer is the primary source (i.e., an execution_command, as opposed to an await_push)
								// it is used as the transfer source to avoid creating long dependency chains across nodes.
								// TODO: For larger numbers of nodes this might become a bottleneck.
								auto source_nid = box_sources[0];

								// Generate push command
								push_command* push_cmd;
								{
									push_cmd = m_cdag.create<push_command>(source_nid, bid, 0, nid, grid_box_to_subrange(box));
									generated_pushes.push_back(push_cmd);

									// Store the read access on the pushing node
									m_command_buffer_reads[push_cmd->get_cid()][bid] =
									    GridRegion<3>::merge(m_command_buffer_reads[push_cmd->get_cid()][bid], box);

									// Add dependencies on the source node between the push and the commands that last wrote that box
									add_dependencies_for_box(m_cdag, push_cmd, m_node_data.at(source_nid).buffer_last_writer.at(bid), box);
								}

								// Generate await_push command
								{
									auto await_push_cmd = m_cdag.create<await_push_command>(nid, push_cmd);

									m_cdag.add_dependency(cmd, await_push_cmd, dependency_kind::true_dep, dependency_origin::dataflow);
									generate_anti_dependencies(tid, bid, node_buffer_last_writer, box, await_push_cmd);
									generate_epoch_dependencies(await_push_cmd);

									// Remember the fact that we now have this valid buffer range on this node.
									auto new_box_sources = box_sources;
									new_box_sources.push_back(nid);
									distributed->region_sources.update_region(box, new_box_sources);

									per_node_last_writer_update_list.emplace_back(nid, bid, box, await_push_cmd->get_cid());
								}
							}
						}
					}

					if(detail::access::mode_traits::is_producer(mode)) {
						// If we are going to insert a reduction command, we will also create a true-dependency chain to the last writer. The new last writer
						// cid however is not known at this point because the the reduction command has not been generated yet. Instead, we simply skip
						// generating anti-dependencies around this requirement. This might not be valid if (multivariate) reductions ever operate on regions.
						if(!generate_reduction) { generate_anti_dependencies(tid, bid, node_buffer_last_writer, req, cmd); }

						// After this task is completed, this node and command are the last writer of this region
						buffer_state_write_list.emplace_back(nid, bid, req);
						per_node_last_writer_update_list.emplace_back(nid, bid, req, cid);
					}
				}

				if(generate_reduction) {
					auto& pending_reduction = std::get<pending_reduction_state>(buffer_state);
					const auto& reduction = pending_reduction.reduction;

					const GridBox<3> box{GridPoint<3>{1, 1, 1}};
					const subrange<3> sr{{}, {1, 1, 1}};

					auto reduce_cmd = m_cdag.create<reduction_command>(nid, reduction);

					// pending_reduction_state with 1 operand is equivalent to a distributed_state with 1 operand. We make sure to always generate the
					// latter since it allows us to avoid generating a no-op reduction command (and also gets rid of an edge case)
					assert(pending_reduction.operand_sources.size() > 1);

					for(auto source_nid : pending_reduction.operand_sources) {
						if(source_nid == nid) {
							add_dependencies_for_box(m_cdag, reduce_cmd, m_node_data.at(source_nid).buffer_last_writer.at(bid), box);
						} else {
							auto push_cmd = m_cdag.create<push_command>(source_nid, bid, reduction.rid, nid, sr);
							generated_pushes.push_back(push_cmd);

							m_command_buffer_reads[push_cmd->get_cid()][bid] = GridRegion<3>::merge(m_command_buffer_reads[push_cmd->get_cid()][bid], box);
							add_dependencies_for_box(m_cdag, push_cmd, m_node_data.at(source_nid).buffer_last_writer.at(bid), box);

							auto await_push_cmd = m_cdag.create<await_push_command>(nid, push_cmd);
							m_cdag.add_dependency(reduce_cmd, await_push_cmd, dependency_kind::true_dep, dependency_origin::dataflow);
							generate_epoch_dependencies(await_push_cmd);
						}
					}

					m_cdag.add_dependency(cmd, reduce_cmd, dependency_kind::true_dep, dependency_origin::dataflow);

					// Unless this task also writes the reduction buffer, the reduction command becomes the last writer
					if(!std::any_of(required_modes.begin(), required_modes.end(), detail::access::mode_traits::is_producer)) {
						per_node_last_writer_update_list.emplace_back(nid, bid, box, reduce_cmd->get_cid());
					}
				}

				if(is_pending_reduction) {
					// More general than generate_reduction: A producer-only access on a pending-reduction buffer will not generate a reduction command, but
					// still change the buffer state to distributed
					buffer_reduction_resolve_list[bid].push_back(nid);
				}
			}
		}

		for(auto [bid, nids] : buffer_reduction_resolve_list) {
			GridBox<3> box{GridPoint<3>{1, 1, 1}};
			distributed_state state_after_reduction{cl::sycl::range<3>{1, 1, 1}};
			state_after_reduction.region_sources.update_region(box, nids);
			m_buffer_states.at(bid) = distributed_state{std::move(state_after_reduction)};
		}

		// If there is only one chunk/command, it already implicitly generates the final reduced value and the buffer does not need to be flagged as
		// a pending reduction.
		if(task_commands.size() > 1) {
			for(const auto& reduction : tsk.get_reductions()) {
				m_buffer_states.at(reduction.bid) = pending_reduction_state{reduction, {}};
			}
		}

		// Update global buffer_states with writes from task_commands after handling is complete.
		for(const auto& write : buffer_state_write_list) {
			const auto& nid = std::get<0>(write);
			const auto& bid = std::get<1>(write);
			const auto& region = std::get<2>(write);

			auto& state = m_buffer_states.at(bid);
			if(auto* distributed = std::get_if<distributed_state>(&state)) {
				distributed->region_sources.update_region(region, {nid});
			} else if(auto* pending = std::get_if<pending_reduction_state>(&state)) {
				assert(std::find(pending->operand_sources.begin(), pending->operand_sources.end(), nid) == pending->operand_sources.end());
				pending->operand_sources.push_back(nid);
			}
		}

		// Update per-node last writer information to take into account writes from await_pushes generated for this task.
		for(const auto& write : per_node_last_writer_update_list) {
			const auto& nid = std::get<0>(write);
			const auto& bid = std::get<1>(write);
			const auto& region = std::get<2>(write);
			const auto& cid = std::get<3>(write);
			m_node_data.at(nid).buffer_last_writer.at(bid).update_region(region, {cid});
		}

		// As the last step, we determine potential "intra-task" race conditions.
		// These can happen in rare cases, when the node that pushes a buffer range also writes to that range within the same task.
		// We cannot do this while generating the push command, as we may not have the writing command recorded at that point.
		for(auto cmd : generated_pushes) {
			const auto push_cmd = static_cast<push_command*>(cmd);
			const auto last_writers =
			    m_node_data.at(push_cmd->get_nid()).buffer_last_writer.at(push_cmd->get_bid()).get_region_values(subrange_to_grid_box(push_cmd->get_range()));
			for(auto& box_and_writer : last_writers) {
				assert(!box_and_writer.first.empty());         // If we want to push it it cannot be empty
				assert(box_and_writer.second != std::nullopt); // Exactly one command last wrote to that box
				const auto writer_cmd = m_cdag.get(*box_and_writer.second);
				assert(writer_cmd != nullptr);

				// We're only interested in writes that happen within the same task as the push
				if(isa<task_command>(writer_cmd) && static_cast<task_command*>(writer_cmd)->get_tid() == tid) {
					// In certain situations the push might have a true dependency on the last writer,
					// in that case don't add an anti-dependency (as that would cause a cycle).
					if(cmd->has_dependency(writer_cmd, dependency_kind::true_dep)) {
						// This can currently only happen for await_push commands.
						assert(isa<await_push_command>(writer_cmd));
						continue;
					}
					m_cdag.add_dependency(writer_cmd, push_cmd, dependency_kind::anti_dep, dependency_origin::dataflow);
				}
			}
		}
	}

	void graph_generator::process_task_side_effect_requirements(const task& tsk) {
		const task_id tid = tsk.get_id();
		if(tsk.get_side_effect_map().empty()) return; // skip the loop in the common case

		for(const auto cmd : m_cdag.task_commands(tid)) {
			auto& nd = m_node_data.at(cmd->get_nid());

			for(const auto& side_effect : tsk.get_side_effect_map()) {
				const auto [hoid, order] = side_effect;
				if(const auto last_effect = nd.host_object_last_effects.find(hoid); last_effect != nd.host_object_last_effects.end()) {
					// TODO once we have different side_effect_orders, their interaction will determine the dependency kind
					m_cdag.add_dependency(cmd, m_cdag.get(last_effect->second), dependency_kind::true_dep, dependency_origin::dataflow);
				}

				// Simplification: If there are multiple chunks per node, we generate true-dependencies between them in an arbitrary order, when all we really
				// need is mutual exclusion (i.e. a bi-directional pseudo-dependency).
				nd.host_object_last_effects.insert_or_assign(hoid, cmd->get_cid());
			}
		}
	}

	void graph_generator::generate_epoch_dependencies(abstract_command* cmd) {
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
			auto last_epoch = m_node_data.at(cmd->get_nid()).epoch_for_new_commands;
			assert(cmd->get_cid() != last_epoch);
			m_cdag.add_dependency(cmd, m_cdag.get(last_epoch), dependency_kind::true_dep, dependency_origin::last_epoch);
		}
	}

	void graph_generator::prune_commands_before(const command_id min_epoch) {
		if(min_epoch > m_min_epoch_last_pruned_before) {
			m_cdag.erase_if([&](abstract_command* cmd) {
				if(cmd->get_cid() < min_epoch) {
					assert(cmd->is_flushed() && "Cannot prune unflushed command");
					m_command_buffer_reads.erase(cmd->get_cid());
					return true;
				}
				return false;
			});
			m_min_epoch_last_pruned_before = min_epoch;
		}
	}

} // namespace detail
} // namespace celerity
