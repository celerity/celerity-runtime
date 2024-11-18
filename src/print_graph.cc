#include "print_graph.h"

#include "cgf.h"
#include "grid.h"
#include "instruction_graph.h"
#include "intrusive_graph.h"
#include "log.h"
#include "print_utils.h"
#include "print_utils_internal.h"
#include "ranges.h"
#include "recorders.h"
#include "task.h"
#include "types.h"
#include "utils.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <matchbox.hh>


namespace celerity::detail {

const char* dependency_style(const dependency_kind kind, const dependency_origin origin) {
	if(kind == dependency_kind::anti_dep) return "color=limegreen";
	switch(origin) {
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
	default: utils::unreachable(); // LCOV_EXCL_LINE
	}
}

const char* access_mode_string(const access_mode m) {
	switch(m) {
	case access_mode::read: return "read";
	case access_mode::write: return "write";
	case access_mode::read_write: return "read_write";
	case access_mode::discard_write: return "discard_write";
	case access_mode::discard_read_write: return "discard_read_write";
	default: utils::unreachable(); // LCOV_EXCL_LINE
	}
}

// NOCOMMIT TODO: Add a graph printing test
// TODO: What is a good cutoff? Use number of nodes-1 (i.e., at most one box per peer?)? (still need an additional hard cutoff probably though, maybe 8)
std::string format_region(const region<3>& reg) {
	if(reg.get_boxes().size() <= 5) { return fmt::format("{}", reg); }
	box_vector<3> excerpt{reg.get_boxes().begin(), reg.get_boxes().begin() + 5};
	return fmt::format("{} [+ {} boxes]", region<3>(std::move(excerpt)), reg.get_boxes().size() - 5);
}

void format_requirements(std::string& label, const reduction_list& reductions, const access_list& accesses, const side_effect_map& side_effects,
    const access_mode reduction_init_mode) {
	for(const auto& [rid, bid, buffer_name, init_from_buffer] : reductions) {
		auto rmode = init_from_buffer ? reduction_init_mode : access_mode::discard_write;
		const region scalar_region(box<3>({0, 0, 0}, {1, 1, 1}));
		const std::string bl = utils::escape_for_dot_label(utils::make_buffer_debug_label(bid, buffer_name));
		fmt::format_to(std::back_inserter(label), "<br/>(R{}) <i>{}</i> {} {}", rid, access_mode_string(rmode), bl, scalar_region);
	}

	for(const auto& [bid, buffer_name, mode, req] : accesses) {
		const std::string bl = utils::escape_for_dot_label(utils::make_buffer_debug_label(bid, buffer_name));
		// While uncommon, we do support chunks that don't require access to a particular buffer at all.
		if(!req.empty()) { fmt::format_to(std::back_inserter(label), "<br/><i>{}</i> {} {}", access_mode_string(mode), bl, format_region(req)); }
	}

	for(const auto& [hoid, order] : side_effects) {
		fmt::format_to(std::back_inserter(label), "<br/><i>affect</i> H{}", hoid);
	}
}

std::string get_task_label(const task_record& tsk) {
	std::string label;
	fmt::format_to(std::back_inserter(label), "T{}", tsk.tid);
	if(!tsk.debug_name.empty()) { fmt::format_to(std::back_inserter(label), " \"{}\"", utils::escape_for_dot_label(tsk.debug_name)); }

	fmt::format_to(std::back_inserter(label), "<br/><b>{}</b>", task_type_string(tsk.type));
	if(tsk.type == task_type::host_compute || tsk.type == task_type::device_compute) {
		fmt::format_to(std::back_inserter(label), " {}", subrange<3>{get_global_offset(tsk.geometry), get_global_size(tsk.geometry)});
	} else if(tsk.type == task_type::collective) {
		fmt::format_to(std::back_inserter(label), " in CG{}", tsk.cgid);
	}

	format_requirements(label, tsk.reductions, tsk.accesses, tsk.side_effect_map, access_mode::read_write);

	return label;
}

std::string make_graph_preamble(const std::string& title) { return fmt::format("digraph G{{label=<{}>;pad=0.2;", title); }

std::string print_task_graph(const task_recorder& recorder, const std::string& title) {
	std::string dot = make_graph_preamble(title);

	CELERITY_DEBUG("print_task_graph, {} entries", recorder.get_graph_nodes().size());

	for(const auto& tsk : recorder.get_graph_nodes()) {
		const char* shape = tsk->type == task_type::epoch || tsk->type == task_type::horizon ? "ellipse" : "box style=rounded";
		fmt::format_to(std::back_inserter(dot), "{}[shape={} label=<{}>];", tsk->tid, shape, get_task_label(*tsk));
		for(auto d : tsk->dependencies) {
			fmt::format_to(std::back_inserter(dot), "{}->{}[{}];", d.node, tsk->tid, dependency_style(d.kind, d.origin));
		}
	}

	dot += "}";
	return dot;
}

const char* print_epoch_label(epoch_action action) {
	switch(action) {
	case epoch_action::none: return "<b>epoch</b>";
	case epoch_action::init: return "<b>epoch</b> (init)";
	case epoch_action::barrier: return "<b>epoch</b> (barrier)";
	case epoch_action::shutdown: return "<b>epoch</b> (shutdown)";
	default: utils::unreachable(); // LCOV_EXCL_LINE
	}
}

std::string print_command_graph(const node_id local_nid, const command_recorder& recorder, const std::string& title) {
	std::string main_dot;
	std::map<task_id, std::string> task_subgraph_dot; // this map must be ordered!

	const auto local_to_global_id = [local_nid](uint64_t id) {
		// IDs in the DOT language may not start with a digit (unless the whole thing is a numeral)
		return fmt::format("id_{}_{}", local_nid, id);
	};

	const auto get_subgraph = [&](const task_command_record& task_cmd) {
		if(!task_subgraph_dot.contains(task_cmd.tid)) {
			std::string task_label;
			fmt::format_to(std::back_inserter(task_label), "T{} ", task_cmd.tid);
			if(!task_cmd.debug_name.empty()) { fmt::format_to(std::back_inserter(task_label), "\"{}\" ", utils::escape_for_dot_label(task_cmd.debug_name)); }
			task_label += "(";
			task_label += task_type_string(task_cmd.type);
			if(task_cmd.type == task_type::collective) { fmt::format_to(std::back_inserter(task_label), " on CG{}", task_cmd.cgid); }
			task_label += ")";

			task_subgraph_dot.emplace(task_cmd.tid,
			    fmt::format("subgraph cluster_{}{{label=<<font color=\"#606060\">{}</font>>;color=darkgray;", local_to_global_id(task_cmd.tid), task_label));
		}
		return &task_subgraph_dot[task_cmd.tid];
	};

	const auto get_buffer_label = [](const buffer_id bid, const std::string& debug_name) {
		return utils::escape_for_dot_label(utils::make_buffer_debug_label(bid, debug_name));
	};

	// we want to iterate over our command records in a sorted order, without moving everything around, and we aren't in C++20 (yet)
	std::vector<const command_record*> sorted_cmd_pointers;
	for(const auto& cmd : recorder.get_graph_nodes()) {
		sorted_cmd_pointers.push_back(cmd.get());
	}
	std::sort(sorted_cmd_pointers.begin(), sorted_cmd_pointers.end(), [](const command_record* a, const command_record* b) { return a->id < b->id; });

	for(const auto& cmd : sorted_cmd_pointers) {
		std::string* output = &main_dot;
		if(utils::isa<task_command_record>(cmd)) { output = get_subgraph(dynamic_cast<const task_command_record&>(*cmd)); }
		auto back = std::back_inserter(*output);

		const auto begin_node = [&](const command_record& cmd, const std::string_view& shape, const std::string_view& color) {
			const auto style = cmd.is_cloned ? "style=dashed," : "";
			fmt::format_to(back, "{}[color={},shape={},{}label=<C{} on N{}<br/>", local_to_global_id(cmd.id), color, shape, style, cmd.id, local_nid);
		};
		const auto end_node = [&] { fmt::format_to(back, ">];"); };

		const auto add_reduction_id_if_reduction = [&](const transfer_id trid) {
			if(trid.rid != 0) { fmt::format_to(back, "(R{}) ", trid.rid); }
		};
		const auto list_completed_reductions = [&](const std::vector<reduction_id>& completed_reductions) {
			for(const auto rid : completed_reductions) {
				fmt::format_to(back, "<br/>completed R{}", rid);
			}
		};

		matchbox::match(
		    *cmd,
		    [&](const push_command_record& pcmd) {
			    begin_node(pcmd, "ellipse", "deeppink2");
			    add_reduction_id_if_reduction(pcmd.trid);
			    fmt::format_to(back, "<b>push</b> {}", pcmd.trid);
			    if(!pcmd.buffer_name.empty()) { fmt::format_to(back, " {}", utils::escape_for_dot_label(pcmd.buffer_name)); }
			    fmt::format_to(back, "<br/>");
			    for(size_t i = 0; i < pcmd.target_regions.size(); ++i) {
				    const auto& [nid, region] = pcmd.target_regions[i];
				    fmt::format_to(back, "{} to N{}", format_region(region), nid);
				    if(i < pcmd.target_regions.size() + 1) { *output += "<br/>"; }
			    }
			    end_node();
		    },
		    [&](const await_push_command_record& apcmd) {
			    begin_node(apcmd, "ellipse", "deeppink2");
			    add_reduction_id_if_reduction(apcmd.trid);
			    fmt::format_to(back, "<b>await push</b> {} <br/>{} {}", apcmd.trid, get_buffer_label(apcmd.trid.bid, apcmd.buffer_name),
			        format_region(apcmd.await_region));
			    end_node();
		    },
		    [&](const reduction_command_record& rcmd) {
			    begin_node(rcmd, "ellipse", "blue");
			    const region scalar_region(box<3>({0, 0, 0}, {1, 1, 1}));
			    fmt::format_to(back, "<b>reduction</b> R{}<br/> {} {}", rcmd.rid, get_buffer_label(rcmd.bid, rcmd.buffer_name), scalar_region);
			    if(!rcmd.has_local_contribution) { *output += "<br/>(no local contribution)"; }
			    end_node();
		    },
		    [&](const epoch_command_record& ecmd) {
			    begin_node(ecmd, "box", "black");
			    *output += print_epoch_label(ecmd.action);
			    list_completed_reductions(ecmd.completed_reductions);
			    end_node();
		    },
		    [&](const horizon_command_record& hcmd) {
			    begin_node(hcmd, "box", "black");
			    *output += "<b>horizon</b>";
			    list_completed_reductions(hcmd.completed_reductions);
			    end_node();
		    },
		    [&](const execution_command_record& ecmd) {
			    begin_node(ecmd, "box", "darkorange2");
			    fmt::format_to(back, "<b>execution</b>");
			    matchbox::match(
			        ecmd.exec_spec, //
			        [&](const subrange<3>& sr) { fmt::format_to(back, " {}", sr); },
			        [&](const std::vector<device_execution_range>& exec_ranges) {
				        for(const auto& er : exec_ranges) {
					        fmt::format_to(back, "<br/>{} on D{}", er.range, er.target_device);
				        }
				        fmt::format_to(back, "<br/>");
			        });
			    auto reduction_init_mode = ecmd.is_reduction_initializer ? access_mode::read_write : access_mode::discard_write;
			    format_requirements(*output, ecmd.reductions, ecmd.accesses, ecmd.side_effects, reduction_init_mode);
			    end_node();
		    },
		    [&](const fence_command_record& fcmd) {
			    begin_node(fcmd, "box", "darkorange");
			    *output += "<b>fence</b>";
			    format_requirements(*output, reduction_list{}, fcmd.accesses, fcmd.side_effects, access_mode::discard_write);
			    end_node();
		    });
	};

	// Sort and deduplicate edges
	struct dependency_edge {
		command_id predecessor;
		command_id successor;
	};
	struct dependency_edge_order {
		bool operator()(const dependency_edge& lhs, const dependency_edge& rhs) const {
			if(lhs.predecessor < rhs.predecessor) return true;
			if(lhs.predecessor > rhs.predecessor) return false;
			return lhs.successor < rhs.successor;
		}
	};
	struct dependency_kind_order {
		bool operator()(const std::pair<dependency_kind, dependency_origin>& lhs, const std::pair<dependency_kind, dependency_origin>& rhs) const {
			return (lhs.first == dependency_kind::true_dep && rhs.first != dependency_kind::true_dep);
		}
	};
	std::map<dependency_edge, std::set<std::pair<dependency_kind, dependency_origin>, dependency_kind_order>, dependency_edge_order>
	    dependencies_by_edge; // ordered and unique
	for(const auto& dep : recorder.get_dependencies()) {
		dependencies_by_edge[{dep.predecessor, dep.successor}].insert(std::pair{dep.kind, dep.origin});
	}
	for(const auto& [edge, meta] : dependencies_by_edge) {
		// If there's at most two edges, take the first one (likely a true dependency followed by an anti-dependency). If there's more, bail (don't style).
		const auto style = meta.size() <= 2 ? dependency_style(meta.begin()->first, meta.begin()->second) : std::string{};
		fmt::format_to(std::back_inserter(main_dot), "{}->{}[{}];", local_to_global_id(edge.predecessor), local_to_global_id(edge.successor), style);
	}

	std::string result_dot = make_graph_preamble(title);
	for(auto& [_, sg_dot] : task_subgraph_dot) {
		result_dot += sg_dot;
		result_dot += "}";
	}
	result_dot += main_dot;
	result_dot += "}";
	return result_dot;
} // namespace celerity::detail

std::string combine_command_graphs(const std::vector<std::string>& graphs, const std::string& title) {
	const auto preamble = make_graph_preamble(title);
	std::string result_dot = make_graph_preamble(title);
	for(const auto& g : graphs) {
		result_dot += g.substr(preamble.size(), g.size() - preamble.size() - 1);
	}
	result_dot += "}";
	return result_dot;
}

std::string print_buffer_label(const buffer_id bid, const std::string& buffer_name = {}) {
	return utils::escape_for_dot_label(utils::make_buffer_debug_label(bid, buffer_name));
}

std::string instruction_dependency_style(const instruction_dependency_origin origin) {
	switch(origin) {
	case instruction_dependency_origin::allocation_lifetime: return "color=cyan3";
	case instruction_dependency_origin::write_to_allocation: return "color=limegreen";
	case instruction_dependency_origin::read_from_allocation: return {};
	case instruction_dependency_origin::side_effect: return {};
	case instruction_dependency_origin::collective_group_order: return "color=blue";
	case instruction_dependency_origin::last_epoch: return "color=orchid";
	case instruction_dependency_origin::execution_front: return "color=orange";
	case instruction_dependency_origin::split_receive: return "color=gray";
	default: utils::unreachable(); // LCOV_EXCL_LINE
	}
}

std::string print_instruction_graph(const instruction_recorder& irec, const command_recorder& crec, const task_recorder& trec,
    const instruction_performance_recorder* const iprec, const std::string& title) {
	std::string dot = make_graph_preamble(title);
	const auto back = std::back_inserter(dot);

	const auto begin_node = [&](const instruction_record& instr, const std::string_view& shape, const std::string_view& color) {
		// Note that this overrides the rounded corners for task instructions, as there unfortunately is no "rounded-dashed" style
		const auto style = instr.is_cloned ? "style=dashed," : "";
		fmt::format_to(back, "I{}[color={},shape={},{}label=<", instr.id, color, shape, style);
	};

	const auto end_node = [&] { fmt::format_to(back, ">];"); };

	const auto print_instruction_graph_garbage = [&](const instruction_garbage& garbage) {
		for(const auto rid : garbage.reductions) {
			fmt::format_to(back, "<br/>collect R{}", rid);
		}
		for(const auto aid : garbage.user_allocations) {
			fmt::format_to(back, "<br/>collect {}", aid);
		}
	};

	std::unordered_map<message_id, instruction_id> send_instructions_by_message_id; // for connecting pilot messages to send instructions
	for(const auto& instr : irec.get_graph_nodes()) {
		matchbox::match(
		    *instr,
		    [&](const clone_collective_group_instruction_record& ccginstr) {
			    begin_node(ccginstr, "ellipse", "darkred");
			    fmt::format_to(back, "I{}<br/><b>clone collective group</b><br/>CG{} -&gt; CG{}", ccginstr.id, ccginstr.original_collective_group_id,
			        ccginstr.new_collective_group_id);
			    end_node();
		    },
		    [&](const alloc_instruction_record& ainstr) {
			    begin_node(ainstr, "ellipse", "cyan3");
			    fmt::format_to(back, "I{}<br/>", ainstr.id);
			    switch(ainstr.origin) {
			    case alloc_instruction_record::alloc_origin::buffer: dot += "buffer "; break;
			    case alloc_instruction_record::alloc_origin::gather: dot += "gather "; break;
			    case alloc_instruction_record::alloc_origin::staging: dot += "staging "; break;
			    }
			    fmt::format_to(back, "<b>alloc</b> {}", ainstr.allocation_id);
			    if(ainstr.buffer_allocation.has_value()) {
				    fmt::format_to(back, "<br/>for {} {}", print_buffer_label(ainstr.buffer_allocation->buffer_id, ainstr.buffer_allocation->buffer_name),
				        ainstr.buffer_allocation->box);
				    if(ainstr.num_chunks.has_value()) { fmt::format_to(back, " x{}", *ainstr.num_chunks); }
			    }
			    fmt::format_to(back, "<br/>{} % {} bytes", fmt::group_digits(ainstr.size_bytes), ainstr.alignment_bytes);
			    end_node();
		    },
		    [&](const free_instruction_record& finstr) {
			    begin_node(finstr, "ellipse", "cyan3");
			    fmt::format_to(back, "I{}<br/>", finstr.id);
			    fmt::format_to(back, "<b>free</b> {}", finstr.allocation_id);
			    if(finstr.buffer_allocation.has_value()) {
				    fmt::format_to(back, "<br/>{} {}", print_buffer_label(finstr.buffer_allocation->buffer_id, finstr.buffer_allocation->buffer_name),
				        finstr.buffer_allocation->box);
			    }
			    fmt::format_to(back, " <br/>{} bytes", fmt::group_digits(finstr.size));
			    end_node();
		    },
		    [&](const copy_instruction_record& cinstr) {
			    begin_node(cinstr, "ellipse,margin=0", "green3");
			    fmt::format_to(back, "I{}<br/>", cinstr.id);
			    switch(cinstr.origin) {
			    case copy_instruction_record::copy_origin::resize: dot += "resize "; break;
			    case copy_instruction_record::copy_origin::coherence: dot += "coherence "; break;
			    case copy_instruction_record::copy_origin::gather: dot += "gather "; break;
			    case copy_instruction_record::copy_origin::fence: dot += "fence "; break;
			    case copy_instruction_record::copy_origin::staging: dot += "staging "; break;
			    case copy_instruction_record::copy_origin::linearizing: dot += "linearizing "; break;
			    case copy_instruction_record::copy_origin::delinearizing: dot += "delinearizing "; break;
			    }
			    fmt::format_to(back, "<b>copy</b><br/>from {} {}<br/>to {} {}<br/>{} {} x{} bytes<br/>{} bytes total", cinstr.source_allocation_id,
			        cinstr.source_layout, cinstr.dest_allocation_id, cinstr.dest_layout, print_buffer_label(cinstr.buffer_id, cinstr.buffer_name),
			        cinstr.copy_region, cinstr.element_size, fmt::group_digits(cinstr.copy_region.get_area() * cinstr.element_size));
			    end_node();
		    },
		    [&](const device_kernel_instruction_record& dkinstr) {
			    begin_node(dkinstr, "box,margin=0.2,style=rounded", "darkorange2");
			    fmt::format_to(back, "I{}", dkinstr.id);
			    fmt::format_to(
			        back, " (device-compute T{}, execution C{})<br/><b>device kernel</b>", dkinstr.command_group_task_id, dkinstr.execution_command_id);
			    if(!dkinstr.debug_name.empty()) { fmt::format_to(back, " {}", utils::escape_for_dot_label(dkinstr.debug_name)); }
			    fmt::format_to(back, "<br/>on D{} {}", dkinstr.device_id, dkinstr.execution_range);

			    for(const auto& access : dkinstr.access_map) {
				    const auto accessed_bounding_box_in_allocation = box( //
				        access.accessed_bounding_box_in_buffer.get_min() - access.allocated_box_in_buffer.get_min(),
				        access.accessed_bounding_box_in_buffer.get_max() - access.allocated_box_in_buffer.get_min());
				    fmt::format_to(back, "<br/>+ access {} {}", print_buffer_label(access.buffer_id, access.buffer_name), access.accessed_region_in_buffer);
				    fmt::format_to(back, "<br/>via {} {}", access.allocation_id, accessed_bounding_box_in_allocation);
			    }
			    for(const auto& access : dkinstr.reduction_map) {
				    const auto accessed_box_in_allocation = box( //
				        access.accessed_bounding_box_in_buffer.get_min() - access.allocated_box_in_buffer.get_min(),
				        access.accessed_bounding_box_in_buffer.get_max() - access.allocated_box_in_buffer.get_min());
				    fmt::format_to(back, "<br/>+ (R{}) reduce into {} {}", access.reduction_id, print_buffer_label(access.buffer_id, access.buffer_name),
				        access.accessed_bounding_box_in_buffer);
				    fmt::format_to(back, "<br/>via {} {}", access.allocation_id, accessed_box_in_allocation);
			    }
			    if(iprec != nullptr) {
				    // TODO: Print this for all
				    fmt::format_to(back, "<br/>{:.1f}", as_sub_second(iprec->get_execution_time(dkinstr.id)));
			    }
			    end_node();
		    },
		    [&](const host_task_instruction_record& htinstr) {
			    begin_node(htinstr, "box,margin=0.2,style=rounded", "darkorange2");
			    fmt::format_to(back, "I{}", htinstr.id);
			    // TODO does not correctly label master-node host tasks
			    fmt::format_to(back, " ({} T{}, execution C{})<br/><b>host task</b>",
			        htinstr.collective_group_id != non_collective_group_id ? fmt::format("CG{} collective-host", htinstr.collective_group_id) : "host-compute",
			        htinstr.command_group_task_id, htinstr.execution_command_id);
			    if(!htinstr.debug_name.empty()) { fmt::format_to(back, " {}", utils::escape_for_dot_label(htinstr.debug_name)); }
			    fmt::format_to(back, "<br/>on host {}", htinstr.execution_range);

			    for(const auto& access : htinstr.access_map) {
				    const auto accessed_bounding_box_in_allocation = box( //
				        access.accessed_bounding_box_in_buffer.get_min() - access.allocated_box_in_buffer.get_min(),
				        access.accessed_bounding_box_in_buffer.get_max() - access.allocated_box_in_buffer.get_min());
				    fmt::format_to(back, "<br/>+ access {} {}", print_buffer_label(access.buffer_id, access.buffer_name), access.accessed_region_in_buffer);
				    fmt::format_to(back, "<br/>via {} {}", access.allocation_id, accessed_bounding_box_in_allocation);
			    }
			    end_node();
		    },
		    [&](const send_instruction_record& sinstr) {
			    begin_node(sinstr, "box,margin=0.2,style=rounded", "deeppink2");
			    fmt::format_to(back, "I{} (push C{})", sinstr.id, sinstr.push_cid);
			    fmt::format_to(back, "<br/><b>send</b> {}", sinstr.transfer_id);
			    fmt::format_to(back, "<br/>to N{} MSG{}", sinstr.dest_node_id, sinstr.message_id);
			    fmt::format_to(back, "<br/>{} {}", print_buffer_label(sinstr.transfer_id.bid, sinstr.buffer_name),
			        box(subrange(sinstr.offset_in_buffer, sinstr.send_range)));
			    fmt::format_to(back, "<br/>via {} {}", sinstr.source_allocation_id, box(subrange(sinstr.offset_in_source_allocation, sinstr.send_range)));
			    fmt::format_to(back, "<br/>{}x{} bytes", sinstr.send_range, sinstr.element_size);
			    fmt::format_to(back, "<br/>{} bytes total", fmt::group_digits(sinstr.send_range.size() * sinstr.element_size));
			    send_instructions_by_message_id.emplace(sinstr.message_id, sinstr.id);
			    end_node();
		    },
		    [&](const receive_instruction_record& rinstr) {
			    begin_node(rinstr, "box,margin=0.2,style=rounded", "deeppink2");
			    fmt::format_to(back, "I{} (await-push C{})", rinstr.id, irec.get_await_push_command_id(rinstr.transfer_id));
			    fmt::format_to(back, "<br/><b>receive</b> {}", rinstr.transfer_id);
			    fmt::format_to(back, "<br/>{} {}", print_buffer_label(rinstr.transfer_id.bid, rinstr.buffer_name), rinstr.requested_region);
			    fmt::format_to(back, "<br/>into {} (B{} {})", rinstr.dest_allocation_id, rinstr.transfer_id.bid, rinstr.allocated_box);
			    fmt::format_to(back, "<br/>x{} bytes", rinstr.element_size);
			    fmt::format_to(back, "<br/>{} bytes total", fmt::group_digits(rinstr.requested_region.get_area() * rinstr.element_size));
			    end_node();
		    },
		    [&](const split_receive_instruction_record& srinstr) {
			    begin_node(srinstr, "box,margin=0.2,style=rounded", "deeppink2");
			    fmt::format_to(back, "I{} (await-push C{})", srinstr.id, irec.get_await_push_command_id(srinstr.transfer_id));
			    fmt::format_to(back, "<br/><b>split receive</b> {}", srinstr.transfer_id);
			    fmt::format_to(back, "<br/>{} {}", print_buffer_label(srinstr.transfer_id.bid, srinstr.buffer_name), srinstr.requested_region);
			    fmt::format_to(back, "<br/>into {} (B{} {})", srinstr.dest_allocation_id, srinstr.transfer_id.bid, srinstr.allocated_box);
			    fmt::format_to(back, "<br/>x{} bytes", srinstr.element_size);
			    fmt::format_to(back, "<br/>{} bytes total", fmt::group_digits(srinstr.requested_region.get_area() * srinstr.element_size));
			    end_node();
		    },
		    [&](const await_receive_instruction_record& arinstr) {
			    begin_node(arinstr, "box,margin=0.2,style=rounded", "deeppink2");
			    fmt::format_to(back, "I{} (await-push C{})", arinstr.id, irec.get_await_push_command_id(arinstr.transfer_id));
			    fmt::format_to(back, "<br/><b>await receive</b> {}", arinstr.transfer_id);
			    fmt::format_to(back, "<br/>{} {}", print_buffer_label(arinstr.transfer_id.bid, arinstr.buffer_name), arinstr.received_region);
			    end_node();
		    },
		    [&](const gather_receive_instruction_record& grinstr) {
			    begin_node(grinstr, "box,margin=0.2,style=rounded", "deeppink2");
			    fmt::format_to(back, "I{} (await-push C{})", grinstr.id, irec.get_await_push_command_id(grinstr.transfer_id));
			    fmt::format_to(back, "<br/><b>gather receive</b> {}", grinstr.transfer_id);
			    fmt::format_to(back, "<br/>{} {} x{}", print_buffer_label(grinstr.transfer_id.bid, grinstr.buffer_name), grinstr.gather_box, grinstr.num_nodes);
			    fmt::format_to(back, "<br/>into {}", grinstr.allocation_id);
			    end_node();
		    },
		    [&](const fill_identity_instruction_record& fiinstr) {
			    begin_node(fiinstr, "ellipse", "blue");
			    fmt::format_to(back, "I{}", fiinstr.id);
			    fmt::format_to(back, "<br/><b>fill identity</b> for R{}", fiinstr.reduction_id);
			    fmt::format_to(back, "<br/>{} x{}", fiinstr.allocation_id, fiinstr.num_values);
			    end_node();
		    },
		    [&](const reduce_instruction_record& rinstr) {
			    begin_node(rinstr, rinstr.reduction_command_id.has_value() ? "box,margin=0.2,style=rounded" : "ellipse", "blue");
			    fmt::format_to(back, "I{}", rinstr.id);
			    if(rinstr.reduction_command_id.has_value()) { fmt::format_to(back, " (reduction C{})", *rinstr.reduction_command_id); }
			    fmt::format_to(back, "<br/>{} <b>reduce</b> B{}.R{}", rinstr.scope == reduce_instruction_record::reduction_scope::global ? "global" : "local",
			        rinstr.buffer_id, rinstr.reduction_id);
			    fmt::format_to(back, "<br/>{} {}", print_buffer_label(rinstr.buffer_id, rinstr.buffer_name), rinstr.box);
			    fmt::format_to(back, "<br/>from {} x{}", rinstr.source_allocation_id, rinstr.num_source_values);
			    fmt::format_to(back, "<br/>to {} x1", rinstr.dest_allocation_id);
			    end_node();
		    },
		    [&](const fence_instruction_record& finstr) {
			    begin_node(finstr, "box,margin=0.2,style=rounded", "darkorange");
			    fmt::format_to(back, "I{} (T{}, C{})<br/><b>fence</b><br/>", finstr.id, finstr.tid, finstr.cid);
			    matchbox::match(
			        finstr.variant, //
			        [&](const fence_instruction_record::buffer_variant& buffer) {
				        fmt::format_to(back, "{} {}", print_buffer_label(buffer.bid, buffer.name), buffer.box);
			        },
			        [&](const fence_instruction_record::host_object_variant& obj) { fmt::format_to(back, "H{}", obj.hoid); });
			    end_node();
		    },
		    [&](const destroy_host_object_instruction_record& dhoinstr) {
			    begin_node(dhoinstr, "ellipse", "black");
			    fmt::format_to(back, "I{}<br/><b>destroy</b> H{}", dhoinstr.id, dhoinstr.host_object_id);
			    end_node();
		    },
		    [&](const horizon_instruction_record& hinstr) {
			    begin_node(hinstr, "box,margin=0.2,style=rounded", "black");
			    fmt::format_to(back, "I{} (T{}, C{})<br/><b>horizon</b>", hinstr.id, hinstr.horizon_task_id, hinstr.horizon_command_id);
			    print_instruction_graph_garbage(hinstr.garbage);
			    end_node();
		    },
		    [&](const epoch_instruction_record& einstr) {
			    begin_node(einstr, "box,margin=0.2,style=rounded", "black");
			    fmt::format_to(back, "I{} (T{}, C{})<br/>{}", einstr.id, einstr.epoch_task_id, einstr.epoch_command_id, print_epoch_label(einstr.epoch_action));
			    print_instruction_graph_garbage(einstr.garbage);
			    end_node();
		    });
	}

	struct dependency_edge {
		instruction_id predecessor;
		instruction_id successor;
	};
	struct dependency_edge_order {
		bool operator()(const dependency_edge& lhs, const dependency_edge& rhs) const {
			if(lhs.predecessor < rhs.predecessor) return true;
			if(lhs.predecessor > rhs.predecessor) return false;
			return lhs.successor < rhs.successor;
		}
	};
	std::map<dependency_edge, std::set<instruction_dependency_origin>, dependency_edge_order> dependencies_by_edge; // ordered and unique
	for(const auto& dep : irec.get_dependencies()) {
		dependencies_by_edge[{dep.predecessor, dep.successor}].insert(dep.origin);
	}
	for(const auto& [edge, origins] : dependencies_by_edge) {
		const auto style = origins.size() == 1 ? instruction_dependency_style(*origins.begin()) : std::string{};
		fmt::format_to(back, "I{}->I{}[{}];", edge.predecessor, edge.successor, style);
	}

	for(const auto& pilot : irec.get_outbound_pilots()) {
		fmt::format_to(back,
		    "P{}[margin=0.25,shape=cds,color=\"#606060\",label=<<font color=\"#606060\"><b>pilot</b> to N{} MSG{}<br/>{}<br/>for {} {}</font>>];",
		    pilot.message.id, pilot.to, pilot.message.id, pilot.message.transfer_id, print_buffer_label(pilot.message.transfer_id.bid), pilot.message.box);
		if(auto it = send_instructions_by_message_id.find(pilot.message.id); it != send_instructions_by_message_id.end()) {
			fmt::format_to(back, "P{}->I{}[dir=none,style=dashed,color=\"#606060\"];", pilot.message.id, it->second);
		}
	}

	dot += "}";
	return dot;
}


} // namespace celerity::detail
