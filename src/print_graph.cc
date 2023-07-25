#include "print_graph.h"

#include <spdlog/fmt/fmt.h>
#include <spdlog/fmt/ostr.h>

#include "command.h"
#include "command_graph.h"
#include "grid.h"
#include "task_manager.h"

namespace celerity {
namespace detail {

	namespace {
		std::unordered_map<buffer_id, std::string> g_buffer_names;

		task_printing_information::access_list build_task_access_list(const task& tsk) {
			task_printing_information::access_list ret;

			const auto execution_range = subrange<3>{tsk.get_global_offset(), tsk.get_global_size()};
			const auto& bam = tsk.get_buffer_access_map();
			for(const auto bid : bam.get_accessed_buffers()) {
				for(const auto mode : bam.get_access_modes(bid)) {
					const auto req = bam.get_mode_requirements(bid, mode, tsk.get_dimensions(), execution_range, tsk.get_global_size());
					ret.push_back({bid, mode, req});
				}
			}
			return ret;
		}
	} // namespace

	task_printing_information::task_printing_information(const task& from)
	    : m_tid(from.get_id()), m_debug_name(from.get_debug_name()), m_cgid(from.get_collective_group_id()), m_type(from.get_type()),
	      m_geometry(from.get_geometry()), m_reductions(from.get_reductions()), m_accesses(build_task_access_list(from)),
	      m_side_effect_map(from.get_side_effect_map()), m_dependencies(from.get_dependencies().begin(), from.get_dependencies().end()) {}

	void task_recorder::record_task(const task& from) {
		CELERITY_TRACE("Recording task {}", from.get_id());
		m_recorded_tasks.emplace_back(from);
	}

	void record_buffer_name(const buffer_id bid, const std::string& name) { g_buffer_names.emplace(bid, name); }

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

	const char* task_type_string(const task_type tt) {
		switch(tt) {
		case task_type::epoch: return "epoch";
		case task_type::host_compute: return "host-compute";
		case task_type::device_compute: return "device-compute";
		case task_type::collective: return "collective host";
		case task_type::master_node: return "master-node host";
		case task_type::horizon: return "horizon";
		case task_type::fence: return "fence";
		default: return "unknown";
		}
	}

	std::string get_buffer_label(const buffer_id bid) {
		// if there is no name defined, the name will be the buffer id.
		// if there is a name we want "id name"
		std::string name;
		if(g_buffer_names.find(bid) != g_buffer_names.end()) { name = g_buffer_names.at(bid); }
		return !name.empty() ? fmt::format("B{} \"{}\"", bid, name) : fmt::format("B{}", bid);
	}

	void format_requirements(std::string& label, const task_printing_information& tsk, subrange<3> execution_range, access_mode reduction_init_mode) {
		for(const auto& reduction : tsk.m_reductions) {
			auto rmode = cl::sycl::access::mode::discard_write;
			if(reduction.init_from_buffer) { rmode = reduction_init_mode; }

			const auto req = GridRegion<3>{{1, 1, 1}};
			const std::string bl = get_buffer_label(reduction.bid);
			fmt::format_to(std::back_inserter(label), "<br/>(R{}) <i>{}</i> {} {}", reduction.rid, detail::access::mode_traits::name(rmode), bl, req);
		}

		for(const auto& [bid, mode, req] : tsk.m_accesses) {
			const std::string bl = get_buffer_label(bid);
			// While uncommon, we do support chunks that don't require access to a particular buffer at all.
			if(!req.empty()) { fmt::format_to(std::back_inserter(label), "<br/><i>{}</i> {} {}", detail::access::mode_traits::name(mode), bl, req); }
		}

		for(const auto& [hoid, order] : tsk.m_side_effect_map) {
			fmt::format_to(std::back_inserter(label), "<br/><i>affect</i> H{}", hoid);
		}
	}

	std::string get_task_label(const task_printing_information& tsk) {
		std::string label;
		fmt::format_to(std::back_inserter(label), "T{}", tsk.m_tid);
		if(!tsk.m_debug_name.empty()) { fmt::format_to(std::back_inserter(label), " \"{}\" ", tsk.m_debug_name); }

		const auto execution_range = subrange<3>{tsk.m_geometry.global_offset, tsk.m_geometry.global_size};

		fmt::format_to(std::back_inserter(label), "<br/><b>{}</b>", task_type_string(tsk.m_type));
		if(tsk.m_type == task_type::host_compute || tsk.m_type == task_type::device_compute) {
			fmt::format_to(std::back_inserter(label), " {}", execution_range);
		} else if(tsk.m_type == task_type::collective) {
			fmt::format_to(std::back_inserter(label), " in CG{}", tsk.m_cgid);
		}

		format_requirements(label, tsk, execution_range, access_mode::read_write);

		return label;
	}

	std::string print_task_graph(const task_recorder& recorder) {
		std::string dot = "digraph G {label=\"Task Graph\" ";

		CELERITY_DEBUG("print_task_graph, {} entries", recorder.get_tasks().size());

		for(auto tsk : recorder.get_tasks()) {
			const auto shape = tsk.m_type == task_type::epoch || tsk.m_type == task_type::horizon ? "ellipse" : "box style=rounded";
			fmt::format_to(std::back_inserter(dot), "{}[shape={} label=<{}>];", tsk.m_tid, shape, get_task_label(tsk));
			for(auto d : tsk.m_dependencies) {
				fmt::format_to(std::back_inserter(dot), "{}->{}[{}];", d.node->get_id(), tsk.m_tid, dependency_style(d));
			}
		}

		dot += "}";
		return dot;
	}

	std::string get_command_label(const node_id nid, const abstract_command& cmd, const task_manager& tm, const buffer_manager* const bm) {
		const command_id cid = cmd.get_cid();

		std::string label = fmt::format("C{} on N{}<br/>", cid, nid);

		if(const auto ecmd = dynamic_cast<const epoch_command*>(&cmd)) {
			label += "<b>epoch</b>";
			if(ecmd->get_epoch_action() == epoch_action::barrier) { label += " (barrier)"; }
			if(ecmd->get_epoch_action() == epoch_action::shutdown) { label += " (shutdown)"; }
		} else if(const auto xcmd = dynamic_cast<const execution_command*>(&cmd)) {
			fmt::format_to(std::back_inserter(label), "<b>execution</b> {}", subrange_to_grid_box(xcmd->get_execution_range()));
		} else if(const auto pcmd = dynamic_cast<const push_command*>(&cmd)) {
			if(pcmd->get_reduction_id() != 0) { fmt::format_to(std::back_inserter(label), "(R{}) ", pcmd->get_reduction_id()); }
			const std::string bl = get_buffer_label(pcmd->get_bid());
			fmt::format_to(std::back_inserter(label), "<b>push</b> transfer {} to N{}<br/>B{} {}", pcmd->get_transfer_id(), pcmd->get_target(), bl,
			    subrange_to_grid_box(pcmd->get_range()));
		} else if(const auto apcmd = dynamic_cast<const await_push_command*>(&cmd)) {
			if(apcmd->get_reduction_id() != 0) { label += fmt::format("(R{}) ", apcmd->get_reduction_id()); }
			const std::string bl = get_buffer_label(apcmd->get_bid());
			fmt::format_to(std::back_inserter(label), "<b>await push</b> transfer {} <br/>B{} {}", apcmd->get_transfer_id(), bl, apcmd->get_region());
		} else if(const auto rrcmd = dynamic_cast<const reduction_command*>(&cmd)) {
			const auto& reduction = rrcmd->get_reduction_info();
			const auto req = GridRegion<3>{{1, 1, 1}};
			const auto bl = get_buffer_label(reduction.bid);
			fmt::format_to(std::back_inserter(label), "<b>reduction</b> R{}<br/> {} {}", reduction.rid, bl, req);
		} else if(const auto hcmd = dynamic_cast<const horizon_command*>(&cmd)) {
			label += "<b>horizon</b>";
		} else if(const auto fcmd = dynamic_cast<const fence_command*>(&cmd)) {
			label += "<b>fence</b>";
		} else {
			assert(!"Unkown command");
			label += "<b>unknown</b>";
		}

		if(const auto tcmd = dynamic_cast<const task_command*>(&cmd)) {
			assert(tm.has_task(tcmd->get_tid()));

			const auto& tsk = *tm.get_task(tcmd->get_tid());

			auto reduction_init_mode = access_mode::discard_write;
			auto execution_range = subrange<3>{tsk.get_global_offset(), tsk.get_global_size()};
			if(const auto ecmd = dynamic_cast<const execution_command*>(&cmd)) {
				if(ecmd->is_reduction_initializer()) { reduction_init_mode = cl::sycl::access::mode::read_write; }
				execution_range = ecmd->get_execution_range();
			}

			format_requirements(label, tsk, execution_range, reduction_init_mode);
		}

		return label;
	}

	std::string print_command_graph(const node_id local_nid, const command_graph& cdag, const task_manager& tm, const buffer_manager* const bm) {
		std::string main_dot;
		std::unordered_map<task_id, std::string> task_subgraph_dot;

		const auto local_to_global_id = [local_nid](uint64_t id) {
			// IDs in the DOT language may not start with a digit (unless the whole thing is a numeral)
			return fmt::format("id_{}_{}", local_nid, id);
		};

		const auto print_vertex = [&](const abstract_command& cmd) {
			static const char* const colors[] = {"black", "crimson", "dodgerblue4", "goldenrod", "maroon4", "springgreen2", "tan1", "chartreuse2"};

			const auto id = local_to_global_id(cmd.get_cid());
			const auto label = get_command_label(local_nid, cmd, tm, bm);
			const auto* const fontcolor = colors[local_nid % (sizeof(colors) / sizeof(char*))];
			const auto* const shape = utils::isa<task_command>(&cmd) ? "box" : "ellipse";
			return fmt::format("{}[label=<{}> fontcolor={} shape={}];", id, label, fontcolor, shape);
		};

		for(const auto cmd : cdag.all_commands()) {
			if(const auto tcmd = dynamic_cast<const task_command*>(cmd)) {
				const auto tid = tcmd->get_tid();
				// Add to subgraph as well
				if(task_subgraph_dot.count(tid) == 0) {
					std::string task_label;
					fmt::format_to(std::back_inserter(task_label), "T{} ", tid);
					if(const auto tsk = tm.find_task(tid)) {
						if(!tsk->get_debug_name().empty()) { fmt::format_to(std::back_inserter(task_label), "\"{}\" ", tsk->get_debug_name()); }
						task_label += "(";
						task_label += task_type_string(tsk->get_type());
						if(tsk->get_type() == task_type::collective) {
							fmt::format_to(std::back_inserter(task_label), " on CG{}", tsk->get_collective_group_id());
						}
						task_label += ")";
					} else {
						task_label += "(deleted)";
					}
					task_subgraph_dot.emplace(tid,
					    fmt::format("subgraph cluster_{}{{label=<<font color=\"#606060\">{}</font>>;color=darkgray;", local_to_global_id(tid), task_label));
				}
				task_subgraph_dot[tid] += print_vertex(*cmd);
			} else {
				main_dot += print_vertex(*cmd);
			}

			for(const auto& d : cmd->get_dependencies()) {
				fmt::format_to(std::back_inserter(main_dot), "{}->{}[{}];", local_to_global_id(d.node->get_cid()), local_to_global_id(cmd->get_cid()),
				    dependency_style(d));
			}
		};

		std::string result_dot = "digraph G{label=\"Command Graph\" "; // If this changes, also change in combine_command_graphs
		for(auto& [sg_tid, sg_dot] : task_subgraph_dot) {
			result_dot += sg_dot;
			result_dot += "}";
		}
		result_dot += main_dot;
		result_dot += "}";
		return result_dot;
	}

	std::string combine_command_graphs(const std::vector<std::string>& graphs) {
		const std::string preamble = "digraph G{label=\"Command Graph\" ";
		std::string result_dot = preamble;
		for(const auto& g : graphs) {
			result_dot += g.substr(preamble.size(), g.size() - preamble.size() - 1);
		}
		result_dot += "}";
		return result_dot;
	}

} // namespace detail
} // namespace celerity
