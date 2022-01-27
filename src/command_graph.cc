#include "command_graph.h"

#include "print_graph.h"

namespace celerity {
namespace detail {

	void command_graph::erase(abstract_command* cmd) {
		if(auto tcmd = dynamic_cast<task_command*>(cmd)) {
			// TODO: If the number of commands per task gets large, this could become problematic. Maybe use an unordered_set instead?
			by_task[tcmd->get_tid()].erase(std::find(by_task[tcmd->get_tid()].begin(), by_task[tcmd->get_tid()].end(), cmd));
		}
		execution_fronts[cmd->get_nid()].erase(cmd);
		commands.erase(cmd->get_cid());
	}


	void command_graph::erase_if(std::function<bool(abstract_command*)> condition) {
		for(auto it = commands.begin(); it != commands.end();) {
			if(condition(it->second.get())) {
				it = commands.erase(it);
			} else {
				++it;
			}
		}
	}

	std::optional<std::string> command_graph::print_graph(size_t max_nodes, const task_manager& tm) const {
		if(command_count() <= max_nodes) { return detail::print_command_graph(*this, tm); }
		return std::nullopt;
	}

} // namespace detail
} // namespace celerity
