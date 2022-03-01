#pragma once

#include <cassert>
#include <unordered_map>
#include <unordered_set>

#include "types.h"


namespace celerity::detail {

class conflict_graph {
  public:
	using command_set = std::unordered_set<command_id>;

	void add_conflict(const command_id a, const command_id b) {
		assert(a != b);
		conflicts.emplace(a, b);
		conflicts.emplace(b, a);
	}

	bool has_conflict(const command_id a, const command_id b) const {
		const auto [first, last] = conflicts.equal_range(a);
		return std::find(first, last, conflict_pair{a, b}) != last;
	}

	template <typename... CommandSets>
	bool has_conflict_with_any_of(const command_id cid, const CommandSets&... sets) const {
		static_assert(sizeof...(sets) >= 1);
		static_assert((std::is_same_v<CommandSets, command_set> && ...));
		const auto [first, last] = conflicts.equal_range(cid);
		return std::find_if(first, last, [&](const conflict_pair& cf) { return ((sets.find(cf.second) != sets.end()) || ...); }) != last;
	}

	void forget_command(const command_id cid) {
		if(conflicts.erase(cid)) {
			for(auto it = conflicts.begin(); it != conflicts.end();) {
				if(it->first == cid || it->second == cid) {
					it = conflicts.erase(it);
				} else {
					++it;
				}
			}
		}
	}

	command_set largest_conflict_free_subset(command_set pending_commands, const command_set& active_commands = {}) const {
		// This method will not allocate unless there are conflicts.

		assert(std::none_of(pending_commands.begin(), pending_commands.end(), [&](const command_id cid) {
			return active_commands.find(cid) != active_commands.end();
		}) && "active and pending command sets intersect");
		assert(std::none_of(active_commands.begin(), active_commands.end(), [&](const command_id active_cid) {
			return has_conflict_with_any_of(active_cid, active_commands);
		}) && "there are conflicts within the set of active commands");

		// Separating conflicting and non-conflicting sets ahead of time reduces the number of candidates for backtracking
		command_set potentially_conflicting_commands;
		for(auto it = pending_commands.begin(); it != pending_commands.end();) {
			if(has_conflict_with_any_of(*it, active_commands, pending_commands, potentially_conflicting_commands)) {
				if(potentially_conflicting_commands.empty()) { potentially_conflicting_commands.reserve(pending_commands.size()); }
				potentially_conflicting_commands.insert(*it);
				it = pending_commands.erase(it);
			} else {
				++it;
			}
		}
		command_set known_conflict_free_commands = std::move(pending_commands);

		// We can give optimal solutions for the common case without backtracking:
		// 	- no conflicting commands => accept all pending
		//	- exactly one conflicting command: conflict is known to be with active_commands (otherwise we would see two conflicts), exclude it
		if(potentially_conflicting_commands.size() <= 1) {
			assert(std::all_of(potentially_conflicting_commands.begin(), potentially_conflicting_commands.end(),
			    [&](const command_id cid) { return has_conflict_with_any_of(cid, active_commands); }));
			return known_conflict_free_commands;
		}

		// Finding the largest conflict-free subset (i.e. the maximum independent set) is NP-hard, but we assume few conflicts in practice. In order to avoid
		// exponential runtimes for degenerate command graphs, we cap the number of backtracking steps and accept suboptimal (but correct) schedules.
		constexpr static size_t backtracking_max_abandoned_candidates = 100;

		class backtracker {
		  public:
			backtracker(command_set known_conflict_free_pending_commands, command_set potentially_conflicting_pending_commands,
			    const command_set& active_commands, const conflict_graph& cg)
			    : cg(cg), active_commands(active_commands) {
				const auto n_pending_commands = known_conflict_free_pending_commands.size() + potentially_conflicting_pending_commands.size();
				pending_commands = std::move(potentially_conflicting_pending_commands);
				best_candidate = std::move(known_conflict_free_pending_commands);
				best_candidate.reserve(n_pending_commands);
				candidate.reserve(n_pending_commands);
				candidate.insert(best_candidate.begin(), best_candidate.end());
			}

			command_set backtrack() && {
				backtrack(pending_commands.begin());
				return std::move(best_candidate);
			}

		  private:
			const conflict_graph& cg;
			const command_set& active_commands;
			command_set pending_commands;
			command_set best_candidate, candidate;
			size_t abandoned_candidates = 0;

			void backtrack(command_set::iterator it) noexcept { // NOLINT(misc-no-recursion)
				while(it != pending_commands.end() && abandoned_candidates < backtracking_max_abandoned_candidates) {
					const auto cid = *it++;
					if(!cg.has_conflict_with_any_of(cid, active_commands, candidate)) {
						candidate.insert(cid);
						backtrack(it);
						if(candidate.size() > best_candidate.size()) {
							best_candidate.clear();
							best_candidate.insert(candidate.begin(), candidate.end()); // don't re-allocate
						}
						candidate.erase(cid);
					} else {
						++abandoned_candidates;
					}
				}
			}
		};

		return backtracker{std::move(known_conflict_free_commands), std::move(potentially_conflicting_commands), active_commands, *this}.backtrack();
	}

  private:
	using conflict_map = std::unordered_multimap<command_id, command_id>;
	using conflict_pair = conflict_map::value_type;

	conflict_map conflicts;
};

} // namespace celerity::detail
