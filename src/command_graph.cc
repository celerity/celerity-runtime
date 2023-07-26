#include "command_graph.h"

namespace celerity {
namespace detail {

	void command_graph::erase(abstract_command* cmd) {
		if(auto tcmd = dynamic_cast<task_command*>(cmd)) {
			// TODO: If the number of commands per task gets large, this could become problematic. Maybe use an unordered_set instead?
			m_by_task[tcmd->get_tid()].erase(std::find(m_by_task[tcmd->get_tid()].begin(), m_by_task[tcmd->get_tid()].end(), cmd));
		}
		m_execution_front.erase(cmd);
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

} // namespace detail
} // namespace celerity
