#include "command_graph.h"

#include "print_graph.h"

namespace celerity {
namespace detail {

	void command_graph::erase(abstract_command* cmd) {
		if(auto tcmd = dynamic_cast<task_command*>(cmd)) {
			// TODO: If the number of commands per task gets large, this could become problematic. Maybe use an unordered_set instead?
			m_by_task[tcmd->get_tid()].erase(std::find(m_by_task[tcmd->get_tid()].begin(), m_by_task[tcmd->get_tid()].end(), cmd));
		}
		m_execution_fronts[cmd->get_nid()].erase(cmd);
		m_commands.erase(cmd->get_cid());
	}


	void command_graph::erase_if(std::function<bool(abstract_command*)> condition) {
		for(auto it = m_commands.begin(); it != m_commands.end();) {
			if(condition(it->second.get())) {
				it = m_commands.erase(it);
			} else {
				++it;
			}
		}
	}

	std::optional<std::string> command_graph::print_graph(
	    const node_id local_nid, const size_t max_nodes, const task_manager& tm, const buffer_manager* const bm) const {
		if(command_count() <= max_nodes) { return detail::print_command_graph(local_nid, *this, tm, bm); }
		return std::nullopt;
	}

} // namespace detail
} // namespace celerity
