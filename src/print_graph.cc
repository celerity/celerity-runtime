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
		std::string get_buffer_name(const buffer_id bid, const buffer_manager* buff_man) {
			return buff_man != nullptr ? buff_man->get_buffer_info(bid).debug_name : "";
		}

		access_list build_access_list(const task& tsk, const buffer_manager* buff_man, const std::optional<subrange<3>> execution_range = {}) {
			access_list ret;
			const auto exec_range = execution_range.value_or(subrange<3>{tsk.get_global_offset(), tsk.get_global_size()});
			const auto& bam = tsk.get_buffer_access_map();
			for(const auto bid : bam.get_accessed_buffers()) {
				for(const auto mode : bam.get_access_modes(bid)) {
					const auto req = bam.get_mode_requirements(bid, mode, tsk.get_dimensions(), exec_range, tsk.get_global_size());
					ret.push_back({bid, get_buffer_name(bid, buff_man), mode, req});
				}
			}
			return ret;
		}

		reduction_list build_reduction_list(const task& tsk, const buffer_manager* buff_man) {
			reduction_list ret;
			for(const auto& reduction : tsk.get_reductions()) {
				ret.push_back({reduction.rid, reduction.bid, get_buffer_name(reduction.bid, buff_man), reduction.init_from_buffer});
			}
			return ret;
		}

		task_dependency_list build_task_dependency_list(const task& tsk) {
			task_dependency_list ret;
			for(const auto& dep : tsk.get_dependencies()) {
				ret.push_back({dep.node->get_id(), dep.kind, dep.origin});
			}
			return ret;
		}

		// removes initial template qualifiers to simplify, and escapes '<' and '>' in the given name,
		// so that it can be successfully used in a dot graph label that uses HTML, and is hopefully readable
		std::string simplify_and_escape_name(const std::string& name) {
			// simplify
			auto first_opening_pos = name.find('<');
			auto namespace_qual_end_pos = name.rfind(':', first_opening_pos);
			auto simplified = namespace_qual_end_pos != std::string::npos ? name.substr(namespace_qual_end_pos + 1) : name;
			// escape
			simplified = std::regex_replace(simplified, std::regex("<"), "&lt;");
			return std::regex_replace(simplified, std::regex(">"), "&gt;");
		}
	} // namespace

	task_printing_information::task_printing_information(const task& from, const buffer_manager* buff_man)
	    : m_tid(from.get_id()), m_debug_name(simplify_and_escape_name(from.get_debug_name())), m_cgid(from.get_collective_group_id()), m_type(from.get_type()),
	      m_geometry(from.get_geometry()), m_reductions(build_reduction_list(from, buff_man)), m_accesses(build_access_list(from, buff_man)),
	      m_side_effect_map(from.get_side_effect_map()), m_dependencies(build_task_dependency_list(from)) {}

	void task_recorder::record_task(const task& tsk) {
		CELERITY_TRACE("Recording task {}", tsk.get_id());
		m_recorded_tasks.emplace_back(tsk, m_buff_man);
	}

	namespace {
		command_type get_command_type(const abstract_command& cmd) {
			if(utils::isa<epoch_command>(&cmd)) return command_type::epoch;
			if(utils::isa<horizon_command>(&cmd)) return command_type::horizon;
			if(utils::isa<execution_command>(&cmd)) return command_type::execution;
			if(utils::isa<push_command>(&cmd)) return command_type::push;
			if(utils::isa<await_push_command>(&cmd)) return command_type::await_push;
			if(utils::isa<reduction_command>(&cmd)) return command_type::reduction;
			if(utils::isa<fence_command>(&cmd)) return command_type::fence;
			CELERITY_CRITICAL("Unexpected command type");
			std::terminate();
		}

		std::optional<epoch_action> get_epoch_action(const abstract_command& cmd) {
			const auto* epoch_cmd = dynamic_cast<const epoch_command*>(&cmd);
			return epoch_cmd != nullptr ? epoch_cmd->get_epoch_action() : std::optional<epoch_action>{};
		}

		std::optional<subrange<3>> get_execution_range(const abstract_command& cmd) {
			const auto* execution_cmd = dynamic_cast<const execution_command*>(&cmd);
			return execution_cmd != nullptr ? execution_cmd->get_execution_range() : std::optional<subrange<3>>{};
		}

		std::optional<reduction_id> get_reduction_id(const abstract_command& cmd) {
			if(const auto* push_cmd = dynamic_cast<const push_command*>(&cmd)) return push_cmd->get_reduction_id();
			if(const auto* await_push_cmd = dynamic_cast<const await_push_command*>(&cmd)) return await_push_cmd->get_reduction_id();
			if(const auto* reduction_cmd = dynamic_cast<const reduction_command*>(&cmd)) return reduction_cmd->get_reduction_info().rid;
			return {};
		}

		std::optional<buffer_id> get_buffer_id(const abstract_command& cmd) {
			if(const auto* push_cmd = dynamic_cast<const push_command*>(&cmd)) return push_cmd->get_bid();
			if(const auto* await_push_cmd = dynamic_cast<const await_push_command*>(&cmd)) return await_push_cmd->get_bid();
			if(const auto* reduction_cmd = dynamic_cast<const reduction_command*>(&cmd)) return reduction_cmd->get_reduction_info().bid;
			return {};
		}

		std::string get_cmd_buffer_name(const std::optional<buffer_id>& bid, const buffer_manager* buff_man) {
			if(buff_man == nullptr || !bid.has_value()) return "";
			return get_buffer_name(bid.value(), buff_man);
		}

		std::optional<node_id> get_target(const abstract_command& cmd) {
			if(const auto* push_cmd = dynamic_cast<const push_command*>(&cmd)) return push_cmd->get_target();
			return {};
		}

		std::optional<GridRegion<3>> get_await_region(const abstract_command& cmd) {
			if(const auto* await_push_cmd = dynamic_cast<const await_push_command*>(&cmd)) return await_push_cmd->get_region();
			return {};
		}

		std::optional<subrange<3>> get_push_range(const abstract_command& cmd) {
			if(const auto* push_cmd = dynamic_cast<const push_command*>(&cmd)) return push_cmd->get_range();
			return {};
		}

		std::optional<transfer_id> get_transfer_id(const abstract_command& cmd) {
			if(const auto* push_cmd = dynamic_cast<const push_command*>(&cmd)) return push_cmd->get_transfer_id();
			if(const auto* await_push_cmd = dynamic_cast<const await_push_command*>(&cmd)) return await_push_cmd->get_transfer_id();
			return {};
		}

		std::optional<task_id> get_task_id(const abstract_command& cmd) {
			if(const auto* task_cmd = dynamic_cast<const task_command*>(&cmd)) return task_cmd->get_tid();
			return {};
		}

		const task* get_task_for(const abstract_command& cmd, const task_manager* task_man) {
			if(const auto* task_cmd = dynamic_cast<const task_command*>(&cmd)) {
				if(task_man != nullptr) {
					assert(task_man->has_task(task_cmd->get_tid()));
					return task_man->get_task(task_cmd->get_tid());
				}
			}
			return nullptr;
		}

		std::optional<task_geometry> get_task_geometry(const abstract_command& cmd, const task_manager* task_man) {
			if(const auto* tsk = get_task_for(cmd, task_man)) return tsk->get_geometry();
			return {};
		}

		bool get_is_reduction_initializer(const abstract_command& cmd) {
			if(const auto* execution_cmd = dynamic_cast<const execution_command*>(&cmd)) return execution_cmd->is_reduction_initializer();
			return false;
		}

		access_list build_cmd_access_list(const abstract_command& cmd, const task_manager* task_man, const buffer_manager* buff_man) {
			if(const auto* tsk = get_task_for(cmd, task_man)) {
				const auto execution_range = get_execution_range(cmd).value_or(subrange<3>{tsk->get_global_offset(), tsk->get_global_size()});
				return build_access_list(*tsk, buff_man, execution_range);
			}
			return {};
		}

		reduction_list build_cmd_reduction_list(const abstract_command& cmd, const task_manager* task_man, const buffer_manager* buff_man) {
			if(const auto* tsk = get_task_for(cmd, task_man)) return build_reduction_list(*tsk, buff_man);
			return {};
		}

		side_effect_map get_side_effects(const abstract_command& cmd, const task_manager* task_man) {
			if(const auto* tsk = get_task_for(cmd, task_man)) return tsk->get_side_effect_map();
			return {};
		}

		command_dependency_list build_command_dependency_list(const abstract_command& cmd) {
			command_dependency_list ret;
			for(const auto& dep : cmd.get_dependencies()) {
				ret.push_back({dep.node->get_cid(), dep.kind, dep.origin});
			}
			return ret;
		}

		std::string get_task_name(const abstract_command& cmd, const task_manager* task_man) {
			if(const auto* tsk = get_task_for(cmd, task_man)) return simplify_and_escape_name(tsk->get_debug_name());
			return {};
		}

		std::optional<task_type> get_task_type(const abstract_command& cmd, const task_manager* task_man) {
			if(const auto* tsk = get_task_for(cmd, task_man)) return tsk->get_type();
			return {};
		}

		std::optional<collective_group_id> get_collective_group_id(const abstract_command& cmd, const task_manager* task_man) {
			if(const auto* tsk = get_task_for(cmd, task_man)) return tsk->get_collective_group_id();
			return {};
		}
	} // namespace

	command_printing_information::command_printing_information(const abstract_command& cmd, const task_manager* task_man, const buffer_manager* buff_man)
	    : m_cid(cmd.get_cid()), m_type(get_command_type(cmd)), m_epoch_action(get_epoch_action(cmd)), m_execution_range(get_execution_range(cmd)),
	      m_reduction_id(get_reduction_id(cmd)), m_buffer_id(get_buffer_id(cmd)), m_buffer_name(get_cmd_buffer_name(m_buffer_id, buff_man)),
	      m_target(get_target(cmd)), m_await_region(get_await_region(cmd)), m_push_range(get_push_range(cmd)), m_transfer_id(get_transfer_id(cmd)),
	      m_task_id(get_task_id(cmd)), m_task_geometry(get_task_geometry(cmd, task_man)), m_is_reduction_initializer(get_is_reduction_initializer(cmd)),
	      m_accesses(build_cmd_access_list(cmd, task_man, buff_man)), m_reductions(build_cmd_reduction_list(cmd, task_man, buff_man)),
	      m_side_effects(get_side_effects(cmd, task_man)), m_dependencies(build_command_dependency_list(cmd)), m_task_name(get_task_name(cmd, task_man)),
	      m_task_type(get_task_type(cmd, task_man)), m_collective_group_id(get_collective_group_id(cmd, task_man)) {}

	void command_recorder::record_command(const abstract_command& com) {
		CELERITY_TRACE("Recording command {}", com.get_cid());
		m_recorded_commands.emplace_back(com, m_task_man, m_buff_man);
	}


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
	    const access_mode reduction_init_mode) {
		for(const auto& [rid, bid, buffer_name, init_from_buffer] : reductions) {
			auto rmode = init_from_buffer ? reduction_init_mode : cl::sycl::access::mode::discard_write;
			const auto req = GridRegion<3>{{1, 1, 1}};
			const std::string bl = get_buffer_label(bid, buffer_name);
			fmt::format_to(std::back_inserter(label), "<br/>(R{}) <i>{}</i> {} {}", rid, detail::access::mode_traits::name(rmode), bl, req);
		}

		for(const auto& [bid, buffer_name, mode, req] : accesses) {
			const std::string bl = get_buffer_label(bid, buffer_name);
			// While uncommon, we do support chunks that don't require access to a particular buffer at all.
			if(!req.empty()) { fmt::format_to(std::back_inserter(label), "<br/><i>{}</i> {} {}", detail::access::mode_traits::name(mode), bl, req); }
		}

		for(const auto& [hoid, order] : side_effects) {
			fmt::format_to(std::back_inserter(label), "<br/><i>affect</i> H{}", hoid);
		}
	}

	std::string get_task_label(const task_printing_information& tsk) {
		std::string label;
		fmt::format_to(std::back_inserter(label), "T{}", tsk.m_tid);
		if(!tsk.m_debug_name.empty()) { fmt::format_to(std::back_inserter(label), " \"{}\" ", tsk.m_debug_name); }

		fmt::format_to(std::back_inserter(label), "<br/><b>{}</b>", task_type_string(tsk.m_type));
		if(tsk.m_type == task_type::host_compute || tsk.m_type == task_type::device_compute) {
			fmt::format_to(std::back_inserter(label), " {}", subrange<3>{tsk.m_geometry.global_offset, tsk.m_geometry.global_size});
		} else if(tsk.m_type == task_type::collective) {
			fmt::format_to(std::back_inserter(label), " in CG{}", tsk.m_cgid);
		}

		format_requirements(label, tsk.m_reductions, tsk.m_accesses, tsk.m_side_effect_map, access_mode::read_write);

		return label;
	}

	std::string print_task_graph(const task_recorder& recorder) {
		std::string dot = "digraph G {label=\"Task Graph\" ";

		CELERITY_DEBUG("print_task_graph, {} entries", recorder.get_tasks().size());

		for(const auto& tsk : recorder.get_tasks()) {
			const char* shape = tsk.m_type == task_type::epoch || tsk.m_type == task_type::horizon ? "ellipse" : "box style=rounded";
			fmt::format_to(std::back_inserter(dot), "{}[shape={} label=<{}>];", tsk.m_tid, shape, get_task_label(tsk));
			for(auto d : tsk.m_dependencies) {
				fmt::format_to(std::back_inserter(dot), "{}->{}[{}];", d.node, tsk.m_tid, dependency_style(d));
			}
		}

		dot += "}";
		return dot;
	}

	std::string get_command_label(const node_id local_nid, const command_printing_information& cmd) {
		const command_id cid = cmd.m_cid;

		std::string label = fmt::format("C{} on N{}<br/>", cid, local_nid);

		auto add_reduction_id_if_reduction = [&]() {
			if(cmd.m_reduction_id.has_value() && cmd.m_reduction_id != 0) { fmt::format_to(std::back_inserter(label), "(R{}) ", cmd.m_reduction_id.value()); }
		};
		const std::string buffer_label = cmd.m_buffer_id.has_value() ? get_buffer_label(cmd.m_buffer_id.value(), cmd.m_buffer_name) : "";

		switch(cmd.m_type) {
		case command_type::epoch: {
			label += "<b>epoch</b>";
			if(cmd.m_epoch_action == epoch_action::barrier) { label += " (barrier)"; }
			if(cmd.m_epoch_action == epoch_action::shutdown) { label += " (shutdown)"; }
		} break;
		case command_type::execution: {
			fmt::format_to(std::back_inserter(label), "<b>execution</b> {}", subrange_to_grid_box(cmd.m_execution_range.value()));
		} break;
		case command_type::push: {
			add_reduction_id_if_reduction();
			fmt::format_to(std::back_inserter(label), "<b>push</b> transfer {} to N{}<br/>B{} {}", //
			    cmd.m_transfer_id.value(), cmd.m_target.value(), buffer_label, subrange_to_grid_box(cmd.m_push_range.value()));
		} break;
		case command_type::await_push: {
			add_reduction_id_if_reduction();
			fmt::format_to(std::back_inserter(label), "<b>await push</b> transfer {} <br/>B{} {}", //
			    cmd.m_transfer_id.value(), buffer_label, cmd.m_await_region.value());
		} break;
		case command_type::reduction: {
			fmt::format_to(std::back_inserter(label), "<b>reduction</b> R{}<br/> {} {}", cmd.m_reduction_id.value(), buffer_label, GridRegion<3>{{1, 1, 1}});
		} break;
		case command_type::horizon: {
			label += "<b>horizon</b>";
		} break;
		case command_type::fence: {
			label += "<b>fence</b>";
		} break;
		default: assert(!"Unkown command"); label += "<b>unknown</b>";
		}

		if(cmd.m_task_id.has_value() && cmd.m_task_geometry.has_value()) {
			auto reduction_init_mode = cmd.m_is_reduction_initializer ? cl::sycl::access::mode::read_write : access_mode::discard_write;

			format_requirements(label, cmd.m_reductions.value_or(reduction_list{}), cmd.m_accesses.value_or(access_list{}),
			    cmd.m_side_effects.value_or(side_effect_map{}), reduction_init_mode);
		}

		return label;
	}

	const std::string command_graph_preamble = "digraph G{label=\"Command Graph\" ";

	std::string print_command_graph(const node_id local_nid, const command_recorder& recorder) {
		std::string main_dot;
		std::map<task_id, std::string> task_subgraph_dot; // this map must be ordered!

		const auto local_to_global_id = [local_nid](uint64_t id) {
			// IDs in the DOT language may not start with a digit (unless the whole thing is a numeral)
			return fmt::format("id_{}_{}", local_nid, id);
		};

		const auto print_vertex = [&](const command_printing_information& cmd) {
			static const char* const colors[] = {"black", "crimson", "dodgerblue4", "goldenrod", "maroon4", "springgreen2", "tan1", "chartreuse2"};

			const auto id = local_to_global_id(cmd.m_cid);
			const auto label = get_command_label(local_nid, cmd);
			const auto* const fontcolor = colors[local_nid % (sizeof(colors) / sizeof(char*))];
			const auto* const shape = cmd.m_task_id.has_value() ? "box" : "ellipse";
			return fmt::format("{}[label=<{}> fontcolor={} shape={}];", id, label, fontcolor, shape);
		};

		// we want to iterate over our command records in a sorted order, without moving everything around, and we aren't in C++20 (yet)
		std::vector<const command_printing_information*> sorted_cmd_pointers;
		for(const auto& cmd : recorder.get_commands()) {
			sorted_cmd_pointers.push_back(&cmd);
		}
		std::sort(sorted_cmd_pointers.begin(), sorted_cmd_pointers.end(), [](const auto* a, const auto* b) { return a->m_cid < b->m_cid; });

		for(const auto& cmd : sorted_cmd_pointers) {
			if(cmd->m_task_id.has_value()) {
				const auto tid = cmd->m_task_id.value();
				// Add to subgraph as well
				if(task_subgraph_dot.count(tid) == 0) {
					std::string task_label;
					fmt::format_to(std::back_inserter(task_label), "T{} ", tid);
					if(!cmd->m_task_name.empty()) { fmt::format_to(std::back_inserter(task_label), "\"{}\" ", cmd->m_task_name); }
					task_label += "(";
					task_label += task_type_string(cmd->m_task_type.value());
					if(cmd->m_task_type == task_type::collective) {
						fmt::format_to(std::back_inserter(task_label), " on CG{}", cmd->m_collective_group_id.value());
					}
					task_label += ")";

					task_subgraph_dot.emplace(tid,
					    fmt::format("subgraph cluster_{}{{label=<<font color=\"#606060\">{}</font>>;color=darkgray;", local_to_global_id(tid), task_label));
				}
				task_subgraph_dot[tid] += print_vertex(*cmd);
			} else {
				main_dot += print_vertex(*cmd);
			}

			for(const auto& d : cmd->m_dependencies) {
				fmt::format_to(std::back_inserter(main_dot), "{}->{}[{}];", local_to_global_id(d.node), local_to_global_id(cmd->m_cid), dependency_style(d));
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

} // namespace detail
} // namespace celerity
