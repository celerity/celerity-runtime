#include "print_graph.h"

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
		if(dep.kind == dependency_kind::anti_dep) return "color=limegreen";
		switch(dep.origin) {
		case dependency_origin::collective_group_serialization: return "color=blue";
		case dependency_origin::execution_front: return "color=orange";
		case dependency_origin::last_epoch: return "color=orchid";
		default: return "";
		}
	}

	std::string get_task_label(const task* tsk) {
		switch(tsk->get_type()) {
		case task_type::epoch: return fmt::format("Task {} (epoch)", tsk->get_id());
		case task_type::host_compute: return fmt::format("Task {} (host-compute)", tsk->get_id());
		case task_type::device_compute: return fmt::format("Task {} ({})", tsk->get_id(), tsk->get_debug_name());
		case task_type::collective: return fmt::format("Task {} (collective #{})", tsk->get_id(), static_cast<size_t>(tsk->get_collective_group_id()));
		case task_type::master_node: return fmt::format("Task {} (master-node)", tsk->get_id());
		case task_type::horizon: return fmt::format("Task {} (horizon)", tsk->get_id());
		default: assert(false); return fmt::format("Task {} (unknown)", tsk->get_id());
		}
	}

	std::string print_task_graph(const task_ring_buffer& tdag) {
		std::string dot = "digraph G {label=\"Task Graph\" ";
		auto dot_append = std::back_inserter(dot);

		for(auto tsk : tdag) {
			fmt::format_to(dot_append, "{}[label=\"{}\"];", tsk->get_id(), get_task_label(tsk));
			for(auto d : tsk->get_dependencies()) {
				fmt::format_to(dot_append, "{}->{}[{}];", d.node->get_id(), tsk->get_id(), dependency_style(d));
			}
		}

		dot += "}";
		return dot;
	}

	std::string get_command_label(const abstract_command& cmd, const task_manager& tm, const reduction_manager& rm) {
		const command_id cid = cmd.get_cid();
		const node_id nid = cmd.get_nid();

		std::string label = fmt::format("[{}] Node {}:\\n", cid, nid);
		auto label_append = std::back_inserter(label);

		if(const auto ecmd = dynamic_cast<const epoch_command*>(&cmd)) {
			label += "epoch";
			if(ecmd->get_epoch_action() == epoch_action::barrier) { label += " (barrier)"; }
			if(ecmd->get_epoch_action() == epoch_action::shutdown) { label += " (shutdown)"; }
		} else if(const auto xcmd = dynamic_cast<const execution_command*>(&cmd)) {
			fmt::format_to(label_append, "execution {}", subrange_to_grid_box(xcmd->get_execution_range()));
		} else if(const auto pcmd = dynamic_cast<const push_command*>(&cmd)) {
			if(pcmd->get_rid()) { fmt::format_to(label_append, "(R{}) ", pcmd->get_rid()); }
			fmt::format_to(label_append, "push {} to {}\\n{}", pcmd->get_bid(), pcmd->get_target(), subrange_to_grid_box(pcmd->get_range()));
		} else if(const auto apcmd = dynamic_cast<const await_push_command*>(&cmd)) {
			if(apcmd->get_source()->get_rid()) { label += fmt::format("(R{}) ", apcmd->get_source()->get_rid()); }
			fmt::format_to(label_append, "await push {} from {}\\n{}", apcmd->get_source()->get_bid(), apcmd->get_source()->get_nid(),
			    subrange_to_grid_box(apcmd->get_source()->get_range()));
		} else if(const auto rrcmd = dynamic_cast<const reduction_command*>(&cmd)) {
			fmt::format_to(label_append, "reduction R{}", rrcmd->get_rid());
		} else if(const auto hcmd = dynamic_cast<const horizon_command*>(&cmd)) {
			label += "horizon";
		} else {
			assert(!"Unkown command");
			label += "unknown";
		}

		if(const auto tcmd = dynamic_cast<const task_command*>(&cmd)) {
			const auto& tsk = *tm.get_task(tcmd->get_tid());

			auto reduction_init_mode = access_mode::discard_write;
			auto execution_range = subrange<3>{tsk.get_global_offset(), tsk.get_global_size()};
			if(const auto ecmd = dynamic_cast<const execution_command*>(&cmd)) {
				if(ecmd->is_reduction_initializer()) { reduction_init_mode = cl::sycl::access::mode::read_write; }
				execution_range = ecmd->get_execution_range();
			}

			for(auto rid : tsk.get_reductions()) {
				auto reduction = rm.get_reduction(rid);

				auto rmode = cl::sycl::access::mode::discard_write;
				if(reduction.initialize_from_buffer) { rmode = reduction_init_mode; }

				const auto bid = reduction.output_buffer_id;
				const auto req = GridRegion<3>{{1, 1, 1}};
				fmt::format_to(label_append, "\\n(R{}) {} {} {}", rid, detail::access::mode_traits::name(rmode), bid, req);
			}

			const auto& bam = tsk.get_buffer_access_map();
			for(const auto bid : bam.get_accessed_buffers()) {
				for(const auto mode : bam.get_access_modes(bid)) {
					const auto req = bam.get_requirements_for_access(bid, mode, tsk.get_dimensions(), execution_range, tsk.get_global_size());
					// While uncommon, we do support chunks that don't require access to a particular buffer at all.
					if(!req.empty()) { fmt::format_to(label_append, "\\n{} {} {}", detail::access::mode_traits::name(mode), bid, req); }
				}
			}

			for(const auto& [hoid, order] : tsk.get_side_effect_map()) {
				fmt::format_to(label_append, "\\naffect host-object {}", hoid);
			}
		}

		return label;
	}

	std::string print_command_graph(const command_graph& cdag, const task_manager& tm, const reduction_manager& rm) {
		std::string main_dot;
		std::unordered_map<task_id, std::string> task_subgraph_dot;

		const auto print_vertex = [&](const abstract_command& cmd) {
			static const char* const colors[] = {"black", "crimson", "dodgerblue4", "goldenrod", "maroon4", "springgreen2", "tan1", "chartreuse2"};

			const auto name = cmd.get_cid();
			const auto label = get_command_label(cmd, tm, rm);
			const auto fontcolor = colors[cmd.get_nid() % (sizeof(colors) / sizeof(char*))];
			const auto shape = isa<task_command>(&cmd) ? "box" : "ellipse";
			return fmt::format("{}[label=\"{}\" fontcolor={} shape={}];", name, label, fontcolor, shape);
		};

		for(const auto cmd : cdag.all_commands()) {
			if(const auto tcmd = dynamic_cast<const task_command*>(cmd)) {
				// Add to subgraph as well
				if(task_subgraph_dot.count(tcmd->get_tid()) == 0) {
					std::string task_label;
					if(const auto tsk = tm.find_task(tcmd->get_tid())) {
						task_label = get_task_label(tsk);
					} else {
						task_label = fmt::format("Task {} (deleted)", tcmd->get_tid());
					}
					task_subgraph_dot.emplace(tcmd->get_tid(), fmt::format("subgraph cluster_{}{{label=\"{}\";color=gray;", tcmd->get_tid(), task_label));
				}
				task_subgraph_dot[tcmd->get_tid()] += print_vertex(*cmd);
			} else {
				main_dot += print_vertex(*cmd);
			}

			for(const auto& d : cmd->get_dependencies()) {
				fmt::format_to(std::back_inserter(main_dot), "{}->{}[{}];", d.node->get_cid(), cmd->get_cid(), dependency_style(d));
			}

			// Add a dashed line to the corresponding push
			if(const auto apcmd = dynamic_cast<const await_push_command*>(cmd)) {
				fmt::format_to(std::back_inserter(main_dot), "{}->{}[style=dashed color=gray40];", apcmd->get_source()->get_cid(), cmd->get_cid());
			}
		};

		std::string result_dot = "digraph G{label=\"Command Graph\" ";
		for(auto& [sg_tid, sg_dot] : task_subgraph_dot) {
			result_dot += sg_dot;
			result_dot += "}";
		}
		result_dot += main_dot;
		result_dot += "}";
		return result_dot;
	}

} // namespace detail
} // namespace celerity
