#pragma once

#include "command_graph.h"
#include "instruction_graph.h"
#include "pilot.h"
#include "task.h"
#include "tracy.h"

#include <cstddef>
#include <unordered_map>
#include <utility>
#include <vector>


// NOCOMMIT TODO: Move most of this to impl file
namespace celerity::detail {

namespace loop_template_detail {

	// Both command- and instruction-graph generators have the notion of a "batch", the set of commands/instructions
	// generated for a given task / set of commands, respectively. Since a loop template can contain more than one
	// task, we need to keep track of all commands / instructions generated across all batches within one iteration
	// of the loop.
	template <typename T>
	using superbatch = std::vector<T>;

	// NOCOMMIT TODO: Rename current_superbatch, previous_superbatch, ...?
	struct intra_batch_dependency {
		size_t offset;
	};
	struct inter_batch_dependency {
		size_t offset;
	};
	template <typename NodeT>
	struct external_dependency {
		NodeT* ptr;
	};
	template <typename NodeT>
	using dependency_distance = std::variant<intra_batch_dependency, inter_batch_dependency, external_dependency<NodeT>>;

	// NOCOMMIT Rename dependency_template?
	template <typename NodeT>
	struct dependency {
		dependency_distance<NodeT> distance;
		dependency_kind kind;
		dependency_origin origin;
	};

}; // namespace loop_template_detail

class tdag_loop_template {
  public:
	size_t prime_count = 0;
	bool is_primed = false;
	bool is_verified = false;

	// TODO: Naming
	// NOCOMMIT: Here and for others: Simply use horizon as indicator of completed iteration?
	//		=> But then what do we do at the end?
	void complete_iteration() {
		CELERITY_DETAIL_TRACY_MESSAGE("tdag_lt::complete_iteration");
		if(!is_primed && !m_working_batch.empty()) {
			// NOCOMMIT Why do we need prime count for wave sim but not our unit tests? Figure out!
			// (What happened was that the first iteration only depended on epoch, but second depended on both ping and pong tasks)
			is_primed = (++prime_count == 2);
		}
		// NOCOMMIT TODO: Verify that we've completed the superbatch
		m_task_index = 0;
		std::swap(m_working_batch, m_previous_batch);
		m_working_batch.clear();
	}

	// TODO: Here we could verify that there are no epochs in the batch
	//  => Could we support epochs? We'd have to delay graph pruning b/c we need to reference older tasks
	void prime(task& tsk) {
		m_working_batch.push_back(&tsk);
		m_task_index++;
	}

	// NOCOMMIT TODO Naming? It also computes something
	// NOCOMMIT => Why even have two separate methods? That's an implementation detail
	//          => How about this: There's a single entry point that receives task/batch while template is not ready
	//             internally it will try to create the template after the first call, and up to 3 calls. If it fails
	//             after the 3rd call, it will throw.
	// TODO: We should also check that the tasks actually do the same thing (in terms of data access)
	void verify(task& tsk) {
		assert(is_primed && !is_verified);
		assert(tsk.get_id() > m_previous_batch[m_task_index]->get_id());
		// NOCOMMIT TODO: Also assert that task ID is 1 higher than current working batch. ALSO IN CDAG / IDAG!
		// NOCOMMIT TODO: Check that this superbatch isn't larger then the previous one

		// Verify task geometry hasn't changed
		// NOCOMMIT TODO: Only do this in debug builds?
		// NOCOMMIT TODO: Test this. Also do the same for commands in idag template
		if(tsk.get_type() != m_previous_batch[m_task_index]->get_type()) { throw std::runtime_error("Task type mismatch"); }
		if(tsk.get_geometry() != m_previous_batch[m_task_index]->get_geometry()) { throw std::runtime_error("Task geometry mismatch"); }
		if(tsk.get_buffer_access_map().get_accessed_buffers() != m_previous_batch[m_task_index]->get_buffer_access_map().get_accessed_buffers()) {
			throw std::runtime_error("Task buffer access mismatch");
		}

		const auto& deps = tsk.get_dependencies();
		const auto& prime_deps = m_previous_batch[m_task_index]->get_dependencies();
		if(deps.size() != prime_deps.size()) { throw std::runtime_error("Different number of dependencies in task"); }
		std::vector<dependency> dep_templ;
		dep_templ.reserve(deps.size());
		for(size_t i = 0; i < deps.size(); ++i) {
			const auto dep_node = deps[i].node;
			if(dep_node != prime_deps[i].node) {
				bool found = false;
				// NOOCMMIT This used to be k < i and was not caught by our tests (only wavesim)
				for(size_t k = 0; !found && k < m_working_batch.size(); ++k) {
					if(dep_node == m_working_batch[k]) {
						dep_templ.push_back({.distance = intra_batch_dependency{.offset = k}, .kind = deps[i].kind, .origin = deps[i].origin});
						found = true;
					}
				}
				for(size_t k = 0; !found && k < m_previous_batch.size(); ++k) {
					if(dep_node == m_previous_batch[k]) {
						dep_templ.push_back({.distance = inter_batch_dependency{.offset = k}, .kind = deps[i].kind, .origin = deps[i].origin});
						found = true;
					}
				}
				if(!found) { throw std::runtime_error("TDAG dependency mismatch"); }
			} else {
				dep_templ.push_back({.distance = external_dependency{.ptr = dep_node}, .kind = deps[i].kind, .origin = deps[i].origin});
			}
		}

		m_dependency_template.push_back(std::move(dep_templ));
		m_working_batch.push_back(&tsk);
		m_task_index++;

		if(m_task_index == m_previous_batch.size()) {
			is_verified = true;
			m_replace_batch = m_working_batch;
		}
	}

	// NOCOMMIT TODO: Fix dependency kind, also include dependency origins
	void apply(task& tsk, const std::function<void(task*, task*, dependency_kind, dependency_origin)>& add_dependency) {
		CELERITY_DETAIL_TRACY_ZONE_SCOPED("tdag_lt::apply", generic_red);
		// CELERITY_CRITICAL("TDAG: Applying loop template to task {}", tsk.get_id());
		for(const auto& dep : m_dependency_template[m_task_index]) {
			matchbox::match(
			    dep.distance, [&](const intra_batch_dependency& dist) { add_dependency(&tsk, m_working_batch[dist.offset], dep.kind, dep.origin); },
			    [&](const inter_batch_dependency& dist) { add_dependency(&tsk, m_previous_batch[dist.offset], dep.kind, dep.origin); },
			    [&](const external_dependency& dist) { add_dependency(&tsk, dist.ptr, dep.kind, dep.origin); });
		}

		m_working_batch.push_back(&tsk);
		m_task_index++;
	}

	// NOCOMMIT Naming
	bool is_at_end_of_iteration() const { return m_task_index == m_previous_batch.size(); }

	std::unordered_map<task*, task*> get_replacement_map() {
		assert(is_verified);
		std::unordered_map<task*, task*> map;
		for(size_t i = 0; i < m_replace_batch.size(); ++i) {
			map[m_replace_batch[i]] = m_previous_batch[i];
		}
		return map;
	}

  private:
	// TODO: Need non-const pointers here because add_dependency also modifies the dependee
	using superbatch = loop_template_detail::superbatch<task*>;

	// TODO: Move these to shared template base class instead?
	using external_dependency = loop_template_detail::external_dependency<task>;
	using inter_batch_dependency = loop_template_detail::inter_batch_dependency;
	using intra_batch_dependency = loop_template_detail::intra_batch_dependency;
	using dependency = loop_template_detail::dependency<task>;

	superbatch m_replace_batch;
	superbatch m_previous_batch;
	superbatch m_working_batch;
	size_t m_task_index = 0; // NOCOMMIT Use iterator instead?

	std::vector<std::vector<dependency>> m_dependency_template;
};

// TODO: Is there any point in having all the DAGs in the same class? Or should we just have 3 different contexts that are held together by some struct?
class cdag_loop_template {
	friend struct cdag_loop_template_testspy;

  public:
	size_t prime_count = 0;
	bool is_primed = false;
	bool is_verified = false;

	size_t loop_instantiations = 0; // For testing only

	void complete_iteration() {
		CELERITY_DETAIL_TRACY_MESSAGE("cdag_lt::complete_iteration");
		if(!is_primed && !m_working_batch.empty()) {
			// NOCOMMIT TODO: Consider this: First iteration gets some data from remote node and reads it to compute something.
			//                Second iteration does the same, but now the await push has a new anti-dependency onto the previous iterations computation.
			//                => It seems like that should be allowed, but for now we go the safe route and require two rounds of priming.
			// UPDATE: Actually need 3 rounds: First iteration doesn't need data transfer, second does, and third also does but now has the anti-dep
			is_primed = (++prime_count == 3);
		}
		// NOCOMMIT TODO: Verify that we've completed the superbatch
		m_batch_index = 0;
		std::swap(m_working_batch, m_previous_batch);
		m_working_batch.clear();
	}

	// TODO: For IDAG we need to fail priming if batch contains allocations
	void prime(std::vector<const command*>& batch) {
		for(const auto* cmd : batch) {
			if(auto hcmd = dynamic_cast<const horizon_command*>(cmd)) { handle_horizon(hcmd); }
		}

		if(prime_count == 0) {
			// m_individual_batch_offsets.push_back(m_individual_batch_offsets.back() + batch.size());
		} else {
			// NOCOMMIT This actually happens - second iteration requires data transfers, while first did not
			// if(batch.size() != get_individual_batch_size(m_batch_index)) {
			// 	throw std::runtime_error("Different number of commands in batch during priming?!");
			// }
			if(prime_count == 2) { m_individual_batch_offsets.push_back(m_individual_batch_offsets.back() + batch.size()); }
		}
		m_working_batch.insert(m_working_batch.end(), batch.begin(), batch.end());
		m_batch_index++;
	}

	void verify(std::vector<const command*>& batch) {
		assert(is_primed && !is_verified);
		if(batch.size() != get_individual_batch_size(m_batch_index)) { throw std::runtime_error("Different number of commands in batch"); }
		std::vector<std::vector<dependency>> dep_templ(batch.size());
		for(size_t i = 0; i < batch.size(); ++i) {
			const size_t superbatch_offset = m_individual_batch_offsets[m_batch_index] + i;
			if(batch[i]->get_dependencies().size() != m_previous_batch[superbatch_offset]->get_dependencies().size()) {
				throw std::runtime_error("Different number of dependencies in command");
			}
			dep_templ[i].reserve(batch[i]->get_dependencies().size());
			for(size_t j = 0; j < batch[i]->get_dependencies().size(); ++j) {
				const auto& dep = batch[i]->get_dependencies()[j];
				const auto dep_node = dep.node;
				if(dep_node != m_previous_batch[superbatch_offset]->get_dependencies()[j].node) {
					bool found = false;
					for(size_t k = 0; !found && k < m_working_batch.size(); ++k) {
						if(dep_node == m_working_batch[k]) {
							dep_templ[i].push_back({.distance = intra_batch_dependency{.offset = k}, .kind = dep.kind, .origin = dep.origin});
							found = true;
						}
					}
					for(size_t k = 0; !found && k < m_previous_batch.size(); ++k) {
						if(dep_node == m_previous_batch[k]) {
							dep_templ[i].push_back({.distance = inter_batch_dependency{.offset = k}, .kind = dep.kind, .origin = dep.origin});
							found = true;
						}
					}
					if(!found) { throw std::runtime_error("CDAG dependency mismatch"); }
				} else {
					dep_templ[i].push_back({.distance = external_dependency{.ptr = dep_node}, .kind = dep.kind, .origin = dep.origin});
				}
			}
			m_working_batch.push_back(batch[i]);
			if(auto hcmd = dynamic_cast<const horizon_command*>(batch[i])) { handle_horizon(hcmd); }
		}

		m_dependency_template.insert(m_dependency_template.end(), std::make_move_iterator(dep_templ.begin()), std::make_move_iterator(dep_templ.end()));
		m_batch_index++;

		if(m_batch_index == m_individual_batch_offsets.size() - 1) {
			is_verified = true;
			m_replace_batch = m_working_batch;
			m_replace_horizon = m_applied_horizon;
		}
	}

	void instantiate(const std::function<command*(const command&)>& clone_command,
	    const std::function<void(command*, command*, dependency_kind, dependency_origin)>& add_dependency) {
		CELERITY_DETAIL_TRACY_ZONE_SCOPED("cdag_lt::instantiate", generic_blue);
		for(size_t i = 0; i < get_individual_batch_size(m_batch_index); ++i) {
			const size_t superbatch_offset = m_individual_batch_offsets[m_batch_index] + i;
			auto cmd = clone_command(*m_previous_batch[superbatch_offset]);

			for(const auto& dep : m_dependency_template[superbatch_offset]) {
				matchbox::match(
				    dep.distance, //
				                  // FIXME: Get rid of const_cast
				    [&](const intra_batch_dependency& dist) { add_dependency(cmd, const_cast<command*>(m_working_batch[dist.offset]), dep.kind, dep.origin); },
				    [&](const inter_batch_dependency& dist) { add_dependency(cmd, const_cast<command*>(m_previous_batch[dist.offset]), dep.kind, dep.origin); },
				    [&](const external_dependency& dist) { add_dependency(cmd, dist.ptr, dep.kind, dep.origin); });
			}
			m_working_batch.push_back(cmd);

			if(auto hcmd = dynamic_cast<const horizon_command*>(cmd)) { handle_horizon(hcmd); }
		}
		m_batch_index++;
		if(m_batch_index == m_individual_batch_offsets.size() - 1) { ++loop_instantiations; }
	}

	std::unordered_map<command*, command*> get_replacement_map() {
		assert(is_verified);
		std::unordered_map<command*, command*> map;
		for(size_t i = 0; i < m_replace_batch.size(); ++i) {
			// NOCOMMIT Uuuh - why do we need working batch here? Should't the last call to complete_iteration have swapped it..?
			// => NOCOMMIT UUH it was wrong - WTF ! Why do we not have tests that catch this?
			map[const_cast<command*>(m_replace_batch[i])] = const_cast<command*>(m_previous_batch[i]); // FIXME: Get rid of const_cast
		}
		if(m_replace_horizon != nullptr) {
			map[const_cast<command*>(m_replace_horizon)] = const_cast<command*>(m_applied_horizon); // FIXME: Get rid of const_cast
		}
		return map;
	}

	// TODO: Why do we need this in IDAG? Isn't it just the same as get_replacement_map()[current_epoch]?
	// command* get_applied_horizon() const { return const_cast<command*>(m_applied_horizon); }

  private:
	// TODO: Need non-const pointers here because add_dependency also modifies the dependee
	using superbatch = loop_template_detail::superbatch<const command*>;

	superbatch m_replace_batch;
	const command* m_replace_horizon = nullptr;

	const command* m_applied_horizon = nullptr;
	const command* m_current_horizon = nullptr;
	superbatch m_previous_batch;
	superbatch m_working_batch;
	std::vector<size_t> m_individual_batch_offsets = {0};
	size_t m_batch_index = 0; // NOCOMMIT Use iterator instead?

	// TODO: Move these to shared template base class instead?
	using external_dependency = loop_template_detail::external_dependency<command>;
	using inter_batch_dependency = loop_template_detail::inter_batch_dependency;
	using intra_batch_dependency = loop_template_detail::intra_batch_dependency;
	using dependency = loop_template_detail::dependency<command>;

	std::vector<std::vector<dependency>> m_dependency_template;

	size_t get_individual_batch_size(const size_t index) {
		assert(index < m_individual_batch_offsets.size() - 1);
		return m_individual_batch_offsets[index + 1] - m_individual_batch_offsets[index];
	}

	void handle_horizon(const horizon_command* const hcmd) {
		m_applied_horizon = m_current_horizon;
		m_current_horizon = hcmd;
	}
};

class idag_loop_template {
	friend struct cdag_loop_template_testspy;

  public:
	bool need_more_priming = false;
	size_t total_prime_count = 0;
	size_t prime_count = 0;
	bool is_primed = false;
	bool is_verified = false;

	// TODO: Add trace messages for how often template was instantiated. Maybe warn if not instantiated at all..?
	size_t loop_instantiations = 0; // For testing only

	void complete_iteration() {
		CELERITY_DETAIL_TRACY_MESSAGE("idag_lt::complete_iteration");
		if(!is_primed && !m_working_batch.empty()) {
			// NOCOMMIT TODO: Consider this: First iteration gets some data from remote node and reads it to compute something.
			//                Second iteration does the same, but now the await push has a new anti-dependency onto the previous iterations computation.
			//                => It seems like that should be allowed, but for now we go the safe route and require two rounds of priming.

			// TODO: What is the maximum number of priming rounds that make sense?
			// => In tests we don't seem to exceed 4, but real wave sim with --outset 4 needs 5?!
			if(total_prime_count++ > 5) { throw std::runtime_error("Too many priming rounds - something went wrong"); }

			if(need_more_priming) {
				need_more_priming = false;
				prime_count = 0;
				// NOCOMMIT TODO: Had a bug where the last priming round had an allocation (for some reason?!), so the wrong batch sizes were stored
				// and not reset. => Add test for that.
				m_individual_batch_offsets.resize(1); // Keep first element (0)
			} else {
				// NOCOMMIT: For wavesim we need 3 priming rounds in IDAG. WHY?? investigate
				is_primed = (++prime_count == 3);
			}
		}

		// TODO: Also do this in TDAG/CDAG
		if(is_verified && m_batch_index != m_individual_batch_offsets.size() - 1) {
			// NOCOMMIT TODO: Add test for this
			throw std::runtime_error("IDAG template not fully instantiated");
		}

		m_batch_index = 0;
		std::swap(m_working_batch, m_previous_batch);
		m_working_batch.clear();
	}

	void prime(std::vector<const instruction*>& batch) {
		for(const auto instr : batch) {
			if(auto ainstr = dynamic_cast<const alloc_instruction*>(instr)) {
				// CELERITY_CRITICAL("Found allocation - skippedyskoop");
				need_more_priming = true; // NOCOMMIT Add a test for this - allocations reset prime count
				return;
			}
			if(auto hinstr = dynamic_cast<const horizon_instruction*>(instr)) { handle_horizon(hinstr); }
		}

		// NOCOMMIT TODO: Do we even need the count thing for IDAG?
		if(prime_count == 0) {
			// m_individual_batch_offsets.push_back(m_individual_batch_offsets.back() + batch.size());
		} else {
			// NOCOMMIT This actually happens - second iteration requires data transfers, while first did not
			// if(batch.size() != get_individual_batch_size(m_batch_index)) {
			// 	throw std::runtime_error("Different number of commands in batch during priming?!");
			// }

			// NOCOMMIT This is kinda brittle. Better solution?
			if(prime_count == 2) { m_individual_batch_offsets.push_back(m_individual_batch_offsets.back() + batch.size()); }
		}
		m_working_batch.insert(m_working_batch.end(), batch.begin(), batch.end());
		m_batch_index++;
	}

	// NOCOMMIT TODO: We're passing all components of the batch as parameters, could be a single parameter instead. Also pass priority into priming for
	// verification.
	void verify(std::vector<const instruction*>& batch, const std::vector<outbound_pilot>& pilot_batch, const int base_priority) {
		assert(is_primed && !is_verified);
		if(batch.size() != get_individual_batch_size(m_batch_index)) {
			throw std::runtime_error(fmt::format("Different number of instructions in batch {} (current: {}, previous: {})", m_batch_index, batch.size(),
			    get_individual_batch_size(m_batch_index)));
		}
		std::vector<std::vector<dependency>> dep_templ(batch.size());
		for(size_t i = 0; i < batch.size(); ++i) {
			const size_t superbatch_offset = m_individual_batch_offsets[m_batch_index] + i;
			if(batch[i]->get_dependencies().size() != m_previous_batch[superbatch_offset]->get_dependencies().size()) {
				throw std::runtime_error("Different number of dependencies in instructions");
			}
			dep_templ[i].reserve(batch[i]->get_dependencies().size());
			for(size_t j = 0; j < batch[i]->get_dependencies().size(); ++j) {
				const auto dep = batch[i]->get_dependencies()[j];
				if(dep != m_previous_batch[superbatch_offset]->get_dependencies()[j]) {
					bool found = false;
					for(size_t k = 0; !found && k < m_working_batch.size(); ++k) {
						if(dep == m_working_batch[k]->get_id()) {
							// NOCOMMIT FIXME No dependency kind / origin
							dep_templ[i].push_back(
							    {.distance = intra_batch_dependency{.offset = k}, .kind = dependency_kind::true_dep, .origin = dependency_origin::dataflow});
							found = true;
						}
					}
					for(size_t k = 0; !found && k < m_previous_batch.size(); ++k) {
						if(dep == m_previous_batch[k]->get_id()) {
							// NOCOMMIT FIXME No dependency kind / origin
							dep_templ[i].push_back(
							    {.distance = inter_batch_dependency{.offset = k}, .kind = dependency_kind::true_dep, .origin = dependency_origin::dataflow});
							found = true;
						}
					}
					if(!found) { throw std::runtime_error("IDAG dependency mismatch"); }
				} else {
					// NOCOMMIT Hacky ptr storage
					dep_templ[i].push_back({.distance = external_dependency{.ptr = reinterpret_cast<instruction*>(dep.value)},
					    .kind = dependency_kind::true_dep,
					    .origin = dependency_origin::dataflow});
				}
			}
			m_working_batch.push_back(batch[i]);
			if(auto hinstr = dynamic_cast<const horizon_instruction*>(batch[i])) { handle_horizon(hinstr); }
		}

		m_dependency_template.insert(m_dependency_template.end(), std::make_move_iterator(dep_templ.begin()), std::make_move_iterator(dep_templ.end()));
		m_pilot_template.push_back(pilot_batch);
		m_priority_template.push_back(base_priority);
		m_batch_index++;

		if(m_batch_index == m_individual_batch_offsets.size() - 1) {
			is_verified = true;
			m_replace_batch = m_working_batch;
			m_read_replace_batch = m_previous_batch;
			m_replace_horizon = m_applied_horizon;
		}
	}

	// This has to be called BEFORE instantiate (NOCOMMIT TODO Ugh)
	int get_batch_priority() const {
		assert(is_verified);
		return m_priority_template[m_batch_index];
	}

	void instantiate(const std::function<instruction*(const instruction*)>& clone_instruction,
	    const std::function<void(instruction*, instruction*, dependency_kind, dependency_origin)>& add_dependency,
	    const std::function<void(const outbound_pilot&)>& clone_pilot) {
		CELERITY_DETAIL_TRACY_ZONE_SCOPED("idag_lt::instantiate", generic_green);
		CELERITY_DETAIL_TRACY_ZONE_TEXT("Batch size: {}", get_individual_batch_size(m_batch_index));
		// CELERITY_CRITICAL("IDAG - INSTANTIATING!");

		// Clone pilots first because IGGEN stores message ids (NOCOMMIT HACK!)
		for(auto pilot : m_pilot_template[m_batch_index]) {
			clone_pilot(pilot);
		}

		for(size_t i = 0; i < get_individual_batch_size(m_batch_index); ++i) {
			const size_t superbatch_offset = m_individual_batch_offsets[m_batch_index] + i;
			auto instr = clone_instruction(m_previous_batch[superbatch_offset]);

			for(const auto& dep : m_dependency_template[superbatch_offset]) {
				matchbox::match(
				    dep.distance,
				    // FIXME: Get rid of const_cast
				    [&](const intra_batch_dependency& dist) {
					    add_dependency(instr, const_cast<instruction*>(m_working_batch[dist.offset]), dep.kind, dep.origin);
				    },
				    [&](const inter_batch_dependency& dist) {
					    add_dependency(instr, const_cast<instruction*>(m_previous_batch[dist.offset]), dep.kind, dep.origin);
				    },
				    [&](const external_dependency& dist) { add_dependency(instr, dist.ptr, dep.kind, dep.origin); });
			}
			m_working_batch.push_back(instr);

			if(auto hinstr = dynamic_cast<const horizon_instruction*>(instr)) { handle_horizon(hinstr); }
		}

		m_batch_index++;
		if(m_batch_index == m_individual_batch_offsets.size() - 1) { ++loop_instantiations; }
	}

	// TODO: Combine with other? Rename other to write_replacement_map?
	std::unordered_map<instruction*, instruction*> get_read_replacement_map() const {
		std::unordered_map<instruction*, instruction*> read_replace_map;
		for(size_t i = 0; i < m_read_replace_batch.size(); ++i) {
			// NOCOMMIT Ugh confusing that we use current batch here
			read_replace_map[const_cast<instruction*>(m_read_replace_batch[i])] = const_cast<instruction*>(m_working_batch[i]); // FIXME: Get rid of const_cast
		}
		// NOCOMMIT TODO: Do we need to do anything about horizons here..?
		return read_replace_map;
	}

	std::unordered_map<instruction*, instruction*> get_replacement_map() const {
		assert(is_verified);
		std::unordered_map<instruction*, instruction*> map;
		for(size_t i = 0; i < m_replace_batch.size(); ++i) {
			map[const_cast<instruction*>(m_replace_batch[i])] = const_cast<instruction*>(m_previous_batch[i]); // FIXME: Get rid of const_cast
		}
		if(m_replace_horizon != nullptr) {
			map[const_cast<instruction*>(m_replace_horizon)] = const_cast<instruction*>(m_applied_horizon); // FIXME: Get rid of const_cast
		}
		return map;
	}

	instruction* get_applied_horizon() const { return const_cast<instruction*>(m_applied_horizon); }

  private:
	// TODO: Need non-const pointers here because add_dependency also modifies the dependee
	using superbatch = loop_template_detail::superbatch<const instruction*>;

	superbatch m_replace_batch;
	superbatch m_read_replace_batch; // NOCOMMIT Hmm this is getting a bit convoluted
	const instruction* m_replace_horizon = nullptr;

	const instruction* m_applied_horizon = nullptr;
	const instruction* m_current_horizon = nullptr;
	superbatch m_previous_batch;
	superbatch m_working_batch;
	std::vector<size_t> m_individual_batch_offsets = {0};
	size_t m_batch_index = 0; // NOCOMMIT Use iterator instead? Here and in CDAG: This is still just the "task index"! Less confusing than batch

	// TODO: Move these to shared template base class instead?
	using external_dependency = loop_template_detail::external_dependency<instruction>;
	using inter_batch_dependency = loop_template_detail::inter_batch_dependency;
	using intra_batch_dependency = loop_template_detail::intra_batch_dependency;
	using dependency = loop_template_detail::dependency<instruction>;

	std::vector<std::vector<dependency>> m_dependency_template; // These are per instruction (TODO: Why?)
	std::vector<std::vector<outbound_pilot>> m_pilot_template;  // These are per batch
	std::vector<int> m_priority_template;                       // These are per batch

	size_t get_individual_batch_size(const size_t index) {
		assert(index < m_individual_batch_offsets.size() - 1);
		return m_individual_batch_offsets[index + 1] - m_individual_batch_offsets[index];
	}

	// NOCOMMIT TODO: Also need this in CDAG (and TDAG?). Test this: W/ initial horizon in each iteration, but should also work for arbitrary horizons
	void handle_horizon(const horizon_instruction* const hinstr) {
		m_applied_horizon = m_current_horizon;
		m_current_horizon = hinstr;
	}
};

// Each task within a celerity loop gets associated with a unique template for that loop
class loop_template {
  public:
	tdag_loop_template tdag;
	cdag_loop_template cdag;
	idag_loop_template idag;
};

} // namespace celerity::detail
