#pragma once

#include <cassert>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "types.h"


namespace celerity::detail {

class conflict_graph {
  public:
	using command_set = std::unordered_set<command_id>;

	void add_command(const command_id cid) { commands.emplace(cid); }

	void add_conflict(const command_id a, const command_id b) {
		assert(a != b);
		assert(commands.find(a) != commands.end());
		assert(commands.find(b) != commands.end());
		conflicts.emplace(a, b);
		conflicts.emplace(b, a);
	}

	bool has_any_conflict(const command_id cid) const {
		const auto [first, last] = conflicts.equal_range(cid);
		return first != last;
	}

	bool has_conflict(const command_id a, const command_id b) const {
		const auto [first, last] = conflicts.equal_range(a);
		return std::find(first, last, conflict_pair{a, b}) != last;
	}

	template <typename Predicate>
	bool has_conflict_if(const command_id cid, Predicate&& predicate) const {
		const auto [first, last] = conflicts.equal_range(cid);
		return std::find_if(first, last, [=](const conflict_pair& cf) { return predicate(cf.second); }) != last;
	}

	void erase_command(const command_id cid) {
		conflicts.erase(cid);
		for(auto it = conflicts.begin(); it != conflicts.end();) {
			if(it->first == cid || it->second == cid) {
				it = conflicts.erase(it);
			} else {
				++it;
			}
		}
	}

	const command_set& get_commands() const { return commands; }

	command_set largest_conflict_free_subset(command_set pending_commands, const command_set& active_commands = {}) const {
		class backtracker {
		  public:
			backtracker(command_set commands, const command_set& active_commands, const conflict_graph& cg)
			    : cg(cg), active_commands(active_commands), commands(std::move(commands)) {
				candidate.reserve(this->commands.size());
				best_candidate.reserve(this->commands.size());
			}

			command_set backtrack() && {
				backtrack(commands.begin());
				return std::move(best_candidate);
			}

		  private:
			const conflict_graph& cg;
			const command_set& active_commands;
			command_set commands;
			command_set best_candidate, candidate;

			void backtrack(command_set::iterator it) noexcept { // NOLINT(misc-no-recursion)
				while(it != commands.end()) {
					const auto cid = *it++;
					const auto has_conflict = cg.has_conflict_if(cid, [this](const command_id other_cid) {
						return active_commands.find(other_cid) != active_commands.end() || candidate.find(other_cid) != candidate.end();
					});
					if(!has_conflict) {
						candidate.insert(cid);
						backtrack(it);
						if(candidate.size() > best_candidate.size()) {
							best_candidate.clear();
							best_candidate.insert(candidate.begin(), candidate.end()); // don't re-allocate
						}
						candidate.erase(cid);
					}
				}
			}
		};

		assert(std::none_of(pending_commands.begin(), pending_commands.end(), [&](const command_id cid) {
			return active_commands.find(cid) != active_commands.end();
		}) && "active and pending command sets intersect");
		assert(std::none_of(active_commands.begin(), active_commands.end(), [&](const command_id active_cid) {
			return has_conflict_if(active_cid, [&](const command_id other_cid) { return active_commands.find(other_cid) != active_commands.end(); });
		}) && "there are conflicts within the set of active commands");

		return backtracker{pending_commands, active_commands, *this}.backtrack();
	}

  private:
	using conflict_map = std::unordered_multimap<command_id, command_id>;
	using conflict_pair = conflict_map::value_type;

	command_set commands;
	conflict_map conflicts;
};

} // namespace celerity::detail
