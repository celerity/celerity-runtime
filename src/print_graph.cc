#include "print_graph.h"

#include <sstream>

#include <spdlog/fmt/fmt.h>
#include <spdlog/fmt/ostr.h>

#include "command.h"
#include "command_graph.h"
#include "grid.h"
#include "task_manager.h"

namespace celerity {
namespace detail {

	template <typename Dependency>
	const char* dependency_style(const Dependency& dep) {
		if(dep.kind == dependency_kind::ANTI_DEP) return "color=limegreen";
		switch(dep.origin) {
		case dependency_origin::collective_group_serialization: return "color=blue";
		case dependency_origin::execution_front: return "color=orange";
		case dependency_origin::current_epoch: return "color=orchid";
		default: return "";
		}
	}

	std::string get_task_label(const task* tsk) {
		switch(tsk->get_type()) {
		case task_type::EPOCH: return fmt::format("Task {} (epoch)", tsk->get_id());
		case task_type::HOST_COMPUTE: return fmt::format("Task {} (host-compute)", tsk->get_id());
		case task_type::DEVICE_COMPUTE: return fmt::format("Task {} ({})", tsk->get_id(), tsk->get_debug_name());
		case task_type::COLLECTIVE: return fmt::format("Task {} (collective #{})", tsk->get_id(), static_cast<size_t>(tsk->get_collective_group_id()));
		case task_type::MASTER_NODE: return fmt::format("Task {} (master-node)", tsk->get_id());
		case task_type::HORIZON: return fmt::format("Task {} (horizon)", tsk->get_id());
		default: assert(false); return fmt::format("Task {} (unknown)", tsk->get_id());
		}
	}

	std::string print_task_graph(const std::unordered_map<task_id, std::unique_ptr<task>>& tdag) {
		std::ostringstream ss;
		ss << "digraph G { label=\"Task Graph\" ";

		for(auto& it : tdag) {
			const auto tsk = it.second.get();

			std::unordered_map<std::string, std::string> props;
			props["label"] = "\"" + get_task_label(tsk) + "\"";

			ss << tsk->get_id();
			ss << "[";
			for(const auto& it : props) {
				ss << " " << it.first << "=" << it.second;
			}
			ss << "];";

			for(auto d : tsk->get_dependencies()) {
				ss << fmt::format("{} -> {} [{}];", d.node->get_id(), tsk->get_id(), dependency_style(d));
			}
		}

		ss << "}";
		return ss.str();
	}

	std::string get_command_label(const abstract_command* cmd) {
		std::string label = fmt::format("[{}] Node {}:\\n", cmd->get_cid(), cmd->get_nid());
		if(const auto ecmd = dynamic_cast<const epoch_command*>(cmd)) {
			label += "EPOCH";
		} else if(const auto xcmd = dynamic_cast<const execution_command*>(cmd)) {
			label += fmt::format("EXECUTION {}\\n{}", subrange_to_grid_box(xcmd->get_execution_range()), cmd->debug_label);
		} else if(const auto pcmd = dynamic_cast<const push_command*>(cmd)) {
			if(pcmd->get_rid()) { label += fmt::format("(R{}) ", pcmd->get_rid()); }
			label += fmt::format("PUSH {} to {}\\n {}", pcmd->get_bid(), pcmd->get_target(), subrange_to_grid_box(pcmd->get_range()));
		} else if(const auto apcmd = dynamic_cast<const await_push_command*>(cmd)) {
			if(apcmd->get_source()->get_rid()) { label += fmt::format("(R{}) ", apcmd->get_source()->get_rid()); }
			label += fmt::format("AWAIT PUSH {} from {}\\n {}", apcmd->get_source()->get_bid(), apcmd->get_source()->get_nid(),
			    subrange_to_grid_box(apcmd->get_source()->get_range()));
		} else if(const auto rrcmd = dynamic_cast<const reduction_command*>(cmd)) {
			label += fmt::format("REDUCTION {}", rrcmd->get_rid());
		} else if(const auto hcmd = dynamic_cast<const horizon_command*>(cmd)) {
			label += "HORIZON";
		} else {
			assert(!"Unkown command");
			return fmt::format("[{}] UNKNOWN\\n{}", cmd->get_cid(), cmd->debug_label);
		}
		return label;
	}

	std::string print_command_graph(const command_graph& cdag, const task_manager& tm) {
		std::ostringstream main_ss;
		std::unordered_map<task_id, std::ostringstream> task_subgraph_ss;

		const auto write_vertex = [&](std::ostream& out, abstract_command* cmd) {
			const char* colors[] = {"black", "crimson", "dodgerblue4", "goldenrod", "maroon4", "springgreen2", "tan1", "chartreuse2"};

			std::unordered_map<std::string, std::string> props;
			props["label"] = "\"" + get_command_label(cmd) + "\"";
			props["fontcolor"] = colors[cmd->get_nid() % (sizeof(colors) / sizeof(char*))];
			if(isa<task_command>(cmd)) { props["shape"] = "box"; }

			out << cmd->get_cid();
			out << "[";
			for(const auto& it : props) {
				out << " " << it.first << "=" << it.second;
			}
			out << "];";
		};

		const auto write_command = [&](auto* cmd) {
			if(const auto tcmd = dynamic_cast<task_command*>(cmd)) {
				// Add to subgraph as well
				if(task_subgraph_ss.find(tcmd->get_tid()) == task_subgraph_ss.end()) {
					std::string task_label;
					if(const auto tsk = tm.find_task(tcmd->get_tid())) {
						task_label = get_task_label(tsk);
					} else {
						task_label = fmt::format("Task {} (deleted)", tcmd->get_tid());
					}
					task_subgraph_ss[tcmd->get_tid()] << fmt::format("subgraph cluster_{} {{ label=\"{}\"; color=gray;", tcmd->get_tid(), task_label);
				}
				write_vertex(task_subgraph_ss[tcmd->get_tid()], cmd);
			} else {
				write_vertex(main_ss, cmd);
			}

			for(auto d : cmd->get_dependencies()) {
				main_ss << fmt::format("{} -> {} [{}];", d.node->get_cid(), cmd->get_cid(), dependency_style(d));
			}

			// Add a dashed line to the corresponding PUSH
			if(isa<await_push_command>(cmd)) {
				auto await_push = static_cast<await_push_command*>(cmd);
				main_ss << fmt::format("{} -> {} [style=dashed color=gray40];", await_push->get_source()->get_cid(), cmd->get_cid());
			}
		};

		for(auto cmd : cdag.all_commands()) {
			write_command(cmd);
		}

		// Close all subgraphs
		for(auto& sg : task_subgraph_ss) {
			sg.second << "}";
		}

		std::ostringstream result_ss;
		result_ss << "digraph G { label=\"Command Graph\" ";
		for(auto& sg : task_subgraph_ss) {
			result_ss << sg.second.str();
		}
		result_ss << main_ss.str();
		result_ss << "}";
		return result_ss.str();
	}

} // namespace detail
} // namespace celerity
