#include "print_graph.h"

#include <spdlog/fmt/fmt.h>
#include <spdlog/fmt/ostr.h>

#include "command.h"
#include "command_graph.h"
#include "grid.h"
#include "recorders.h"
#include "task_manager.h"

namespace celerity::detail {

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

std::string get_buffer_label(const buffer_id bid, const std::string& name = "") {
	// if there is no name defined, the name will be the buffer id.
	// if there is a name we want "id name"
	return !name.empty() ? fmt::format("B{} \"{}\"", bid, name) : fmt::format("B{}", bid);
}

void format_requirements(std::string& label, const reduction_list& reductions, const access_list& accesses, const side_effect_map& side_effects,
    const access_mode reduction_init_mode, const buffer_recorder& brec) {
	for(const auto& [rid, bid, init_from_buffer] : reductions) {
		auto rmode = init_from_buffer ? reduction_init_mode : cl::sycl::access::mode::discard_write;
		const auto req = region(box<3>{{0, 0, 0}, {1, 1, 1}});
		const std::string bl = get_buffer_label(bid, brec.get_buffer(bid).debug_name);
		fmt::format_to(std::back_inserter(label), "<br/>(R{}) <i>{}</i> {} {}", rid, detail::access::mode_traits::name(rmode), bl, req);
	}

	for(const auto& [bid, mode, req] : accesses) {
		const std::string bl = get_buffer_label(bid, brec.get_buffer(bid).debug_name);
		// While uncommon, we do support chunks that don't require access to a particular buffer at all.
		if(!req.empty()) { fmt::format_to(std::back_inserter(label), "<br/><i>{}</i> {} {}", detail::access::mode_traits::name(mode), bl, req); }
	}

	for(const auto& [hoid, order] : side_effects) {
		fmt::format_to(std::back_inserter(label), "<br/><i>affect</i> H{}", hoid);
	}
}

std::string get_task_label(const task_record& tsk, const buffer_recorder& brec) {
	std::string label;
	fmt::format_to(std::back_inserter(label), "T{}", tsk.tid);
	if(!tsk.debug_name.empty()) { fmt::format_to(std::back_inserter(label), " \"{}\" ", utils::escape_for_dot_label(tsk.debug_name)); }

	fmt::format_to(std::back_inserter(label), "<br/><b>{}</b>", task_type_string(tsk.type));
	if(tsk.type == task_type::host_compute || tsk.type == task_type::device_compute) {
		fmt::format_to(std::back_inserter(label), " {}", subrange<3>{tsk.geometry.global_offset, tsk.geometry.global_size});
	} else if(tsk.type == task_type::collective) {
		fmt::format_to(std::back_inserter(label), " in CG{}", tsk.cgid);
	}

	format_requirements(label, tsk.reductions, tsk.accesses, tsk.side_effect_map, access_mode::read_write, brec);

	return label;
}

std::string print_task_graph(const task_recorder& trec, const buffer_recorder& brec) {
	std::string dot = "digraph G {label=\"Task Graph\" ";

	CELERITY_DEBUG("print_task_graph, {} entries", trec.get_tasks().size());

	for(const auto& tsk : trec.get_tasks()) {
		const char* shape = tsk.type == task_type::epoch || tsk.type == task_type::horizon ? "ellipse" : "box style=rounded";
		fmt::format_to(std::back_inserter(dot), "{}[shape={} label=<{}>];", tsk.tid, shape, get_task_label(tsk, brec));
		for(auto d : tsk.dependencies) {
			fmt::format_to(std::back_inserter(dot), "{}->{}[{}];", d.node, tsk.tid, dependency_style(d));
		}
	}

	dot += "}";
	return dot;
}

std::string get_command_label(const node_id local_nid, const command_record& cmd, const task_recorder& trec, const buffer_recorder& brec) {
	const command_id cid = cmd.cid;

	std::string label = fmt::format("C{} on N{}<br/>", cid, local_nid);

	auto add_reduction_id_if_reduction = [&]() {
		if(cmd.reduction_id.has_value() && cmd.reduction_id != 0) { fmt::format_to(std::back_inserter(label), "(R{}) ", cmd.reduction_id.value()); }
	};
	const std::string buffer_label = cmd.buffer_id.has_value() ? get_buffer_label(*cmd.buffer_id, brec.get_buffer(*cmd.buffer_id).debug_name) : "";

	switch(cmd.type) {
	case command_type::epoch: {
		label += "<b>epoch</b>";
		if(cmd.epoch_action == epoch_action::barrier) { label += " (barrier)"; }
		if(cmd.epoch_action == epoch_action::shutdown) { label += " (shutdown)"; }
	} break;
	case command_type::execution: {
		fmt::format_to(std::back_inserter(label), "<b>execution</b> {}", cmd.execution_range.value());
	} break;
	case command_type::push: {
		add_reduction_id_if_reduction();
		fmt::format_to(std::back_inserter(label), "<b>push</b> transfer {} to N{}<br/>B{} {}", cmd.transfer_id.value(), cmd.target.value(), buffer_label,
		    cmd.push_range.value());
	} break;
	case command_type::await_push: {
		add_reduction_id_if_reduction();
		fmt::format_to(std::back_inserter(label), "<b>await push</b> transfer {} <br/>B{} {}", //
		    cmd.transfer_id.value(), buffer_label, cmd.await_region.value());
	} break;
	case command_type::reduction: {
		const region scalar_region(box<3>({0, 0, 0}, {1, 1, 1}));
		fmt::format_to(std::back_inserter(label), "<b>reduction</b> R{}<br/> {} {}", cmd.reduction_id.value(), buffer_label, scalar_region);
	} break;
	case command_type::horizon: {
		label += "<b>horizon</b>";
	} break;
	case command_type::fence: {
		label += "<b>fence</b>";
	} break;
	default: assert(!"Unkown command"); label += "<b>unknown</b>";
	}

	if(cmd.task_id.has_value()) {
		const auto& tsk = trec.get_task(*cmd.task_id);
		auto reduction_init_mode = cmd.is_reduction_initializer ? cl::sycl::access::mode::read_write : access_mode::discard_write;
		format_requirements(label, tsk.reductions, cmd.accesses.value_or(access_list()), tsk.side_effect_map, reduction_init_mode, brec);
	}

	return label;
}

const std::string command_graph_preamble = "digraph G{label=\"Command Graph\" ";

std::string print_command_graph(const node_id local_nid, const command_recorder& crec, const task_recorder& trec, const buffer_recorder& brec) {
	std::string main_dot;
	std::map<task_id, std::string> task_subgraph_dot; // this map must be ordered!

	const auto local_to_global_id = [local_nid](uint64_t id) {
		// IDs in the DOT language may not start with a digit (unless the whole thing is a numeral)
		return fmt::format("id_{}_{}", local_nid, id);
	};

	const auto print_vertex = [&](const command_record& cmd) {
		static const char* const colors[] = {"black", "crimson", "dodgerblue4", "goldenrod", "maroon4", "springgreen2", "tan1", "chartreuse2"};

		const auto id = local_to_global_id(cmd.cid);
		const auto label = get_command_label(local_nid, cmd, trec, brec);
		const auto* const fontcolor = colors[local_nid % (sizeof(colors) / sizeof(char*))];
		const auto* const shape = cmd.task_id.has_value() ? "box" : "ellipse";
		return fmt::format("{}[label=<{}> fontcolor={} shape={}];", id, label, fontcolor, shape);
	};

	// we want to iterate over our command records in a sorted order, without moving everything around, and we aren't in C++20 (yet)
	std::vector<const command_record*> sorted_cmd_pointers;
	for(const auto& cmd : crec.get_commands()) {
		sorted_cmd_pointers.push_back(&cmd);
	}
	std::sort(sorted_cmd_pointers.begin(), sorted_cmd_pointers.end(), [](const auto* a, const auto* b) { return a->cid < b->cid; });

	for(const auto& cmd : sorted_cmd_pointers) {
		if(cmd->task_id.has_value()) {
			const auto tid = *cmd->task_id;
			const auto& tsk = trec.get_task(tid);

			// Add to subgraph as well
			if(task_subgraph_dot.count(tid) == 0) {
				std::string task_label;
				fmt::format_to(std::back_inserter(task_label), "T{} ", tid);
				if(!tsk.debug_name.empty()) { fmt::format_to(std::back_inserter(task_label), "\"{}\" ", tsk.debug_name); }
				task_label += "(";
				task_label += task_type_string(tsk.type);
				if(tsk.type == task_type::collective) { fmt::format_to(std::back_inserter(task_label), " on CG{}", tsk.cgid); }
				task_label += ")";

				task_subgraph_dot.emplace(
				    tid, fmt::format("subgraph cluster_{}{{label=<<font color=\"#606060\">{}</font>>;color=darkgray;", local_to_global_id(tid), task_label));
			}
			task_subgraph_dot[tid] += print_vertex(*cmd);
		} else {
			main_dot += print_vertex(*cmd);
		}

		for(const auto& d : cmd->dependencies) {
			fmt::format_to(std::back_inserter(main_dot), "{}->{}[{}];", local_to_global_id(d.node), local_to_global_id(cmd->cid), dependency_style(d));
		}
	};

	std::string result_dot = command_graph_preamble;
	for(auto& [_, sg_dot] : task_subgraph_dot) {
		result_dot += sg_dot;
		result_dot += "}";
	}
	result_dot += main_dot;
	result_dot += "}";
	return result_dot;
}

std::string combine_command_graphs(const std::vector<std::string>& graphs) {
	std::string result_dot = command_graph_preamble;
	for(const auto& g : graphs) {
		result_dot += g.substr(command_graph_preamble.size(), g.size() - command_graph_preamble.size() - 1);
	}
	result_dot += "}";
	return result_dot;
}

} // namespace celerity::detail
