#pragma once

#include "command.h"
#include "task.h"

namespace celerity::detail {

class buffer_manager;
class task_manager;

// General recording

struct access_record {
	const buffer_id bid;
	const std::string buffer_name;
	const access_mode mode;
	const region<3> req;
};
using access_list = std::vector<access_record>;

struct reduction_record {
	const reduction_id rid;
	const buffer_id bid;
	const std::string buffer_name;
	const bool init_from_buffer;
};
using reduction_list = std::vector<reduction_record>;

template <typename IdType>
struct dependency_record {
	const IdType node;
	const dependency_kind kind;
	const dependency_origin origin;
};

// Task recording

using task_dependency_list = std::vector<dependency_record<task_id>>;

struct task_record {
	task_record(const task& from, const buffer_manager* buff_mngr);

	const task_id tid;
	const std::string debug_name;
	const collective_group_id cgid;
	const task_type type;
	const task_geometry geometry;
	const reduction_list reductions;
	const access_list accesses;
	const side_effect_map side_effect_map;
	const task_dependency_list dependencies;
};

class task_recorder {
  public:
	using task_records = std::vector<task_record>;
	using task_callback = std::function<void(const task_record&)>;

	task_recorder(const buffer_manager* buff_mngr = nullptr) : m_buff_mngr(buff_mngr) {}

	void record_task(const task& tsk);

	void add_callback(task_callback callback);
	void invoke_callbacks(const task_record& tsk) const;

	const task_records& get_tasks() const { return m_recorded_tasks; }

  private:
	task_records m_recorded_tasks;
	std::vector<task_callback> m_callbacks{};
	const buffer_manager* m_buff_mngr;
};

// Command recording

using command_dependency_list = std::vector<dependency_record<command_id>>;

struct command_record {
	const command_id cid;
	const command_type type;

	const std::optional<epoch_action> epoch_action;
	const std::optional<subrange<3>> execution_range;
	const std::optional<reduction_id> reduction_id;
	const std::optional<buffer_id> buffer_id;
	const std::string buffer_name;
	const std::optional<node_id> target;
	const std::optional<region<3>> await_region;
	const std::optional<subrange<3>> push_range;
	const std::optional<transfer_id> transfer_id;
	const std::optional<task_id> task_id;
	const std::optional<task_geometry> task_geometry;
	const bool is_reduction_initializer;
	const std::optional<access_list> accesses;
	const std::optional<reduction_list> reductions;
	const std::optional<side_effect_map> side_effects;
	const command_dependency_list dependencies;
	const std::string task_name;
	const std::optional<task_type> task_type;
	const std::optional<collective_group_id> collective_group_id;

	command_record(const abstract_command& cmd, const task_manager* task_mngr, const buffer_manager* buff_mngr);
};

class command_recorder {
  public:
	using command_record = std::vector<command_record>;

	command_recorder(const task_manager* task_mngr, const buffer_manager* buff_mngr = nullptr) : m_task_mngr(task_mngr), m_buff_mngr(buff_mngr) {}

	void record_command(const abstract_command& com);

	const command_record& get_commands() const { return m_recorded_commands; }

  private:
	command_record m_recorded_commands;
	const task_manager* m_task_mngr;
	const buffer_manager* m_buff_mngr;
};

} // namespace celerity::detail

template <>
struct std::hash<celerity::detail::reduction_record> {
	std::size_t operator()(const celerity::detail::reduction_record& r) const noexcept {
		std::size_t seed = 0;
		celerity::detail::utils::hash_combine(seed, std::hash<celerity::detail::reduction_id>{}(r.rid), std::hash<celerity::detail::buffer_id>{}(r.bid),
		    std::hash<std::string>{}(r.buffer_name), std::hash<bool>{}(r.init_from_buffer));
		return seed;
	};
};

template <>
struct std::hash<celerity::detail::access_record> {
	std::size_t operator()(const celerity::detail::access_record& r) {
		std::size_t seed = 0;
		celerity::detail::utils::hash_combine(seed, std::hash<celerity::detail::buffer_id>{}(r.bid), std::hash<std::string>{}(r.buffer_name),
		    std::hash<celerity::access_mode>{}(r.mode), std::hash<celerity::detail::region<3>>{}(r.req));
		return seed;
	};
};

template <typename IdType>
struct std::hash<celerity::detail::dependency_record<IdType>> {
	std::size_t operator()(const celerity::detail::dependency_record<IdType>& r) const noexcept {
		std::size_t seed = 0;
		celerity::detail::utils::hash_combine(seed, std::hash<IdType>{}(r.node), std::hash<celerity::detail::dependency_kind>{}(r.kind),
		    std::hash<celerity::detail::dependency_origin>{}(r.origin));
		return seed;
	};
};

template <>
struct std::hash<celerity::detail::side_effect_map> {
	std::size_t operator()(const celerity::detail::side_effect_map& m) const noexcept {
		std::size_t seed = 0;
		for(auto& [hoid, order] : m) {
			celerity::detail::utils::hash_combine(
			    seed, std::hash<celerity::detail::host_object_id>{}(hoid), std::hash<celerity::experimental::side_effect_order>{}(order));
		}
		return seed;
	};
};

template <>
struct std::hash<celerity::detail::task_record> {
	std::size_t operator()(const celerity::detail::task_record& t) const noexcept {
		std::size_t seed = 0;
		celerity::detail::utils::hash_combine(seed, std::hash<celerity::detail::task_id>{}(t.tid), std::hash<std::string>{}(t.debug_name),
		    std::hash<celerity::detail::collective_group_id>{}(t.cgid), std::hash<celerity::detail::task_type>{}(t.type),
		    std::hash<celerity::detail::task_geometry>{}(t.geometry), celerity::detail::utils::vector_hash{}(t.reductions),
		    celerity::detail::utils::vector_hash{}(t.accesses), std::hash<celerity::detail::side_effect_map>{}(t.side_effect_map),
		    celerity::detail::utils::vector_hash{}(t.dependencies));

		return seed;
	};
};

template <>
struct fmt::formatter<celerity::detail::dependency_kind> : fmt::formatter<std::string> {
	static format_context::iterator format(const celerity::detail::dependency_kind& dk, format_context& ctx) {
		auto out = ctx.out();
		switch(dk) {
		case celerity::detail::dependency_kind::anti_dep: out = std::copy_n("anti-dep", 8, out); break;
		case celerity::detail::dependency_kind::true_dep: out = std::copy_n("true-dep", 8, out); break;
		}
		return out;
	}
};

template <>
struct fmt::formatter<celerity::detail::dependency_origin> : fmt::formatter<std::string> {
	static format_context::iterator format(const celerity::detail::dependency_origin& dk, format_context& ctx) {
		auto out = ctx.out();
		switch(dk) {
		case celerity::detail::dependency_origin::dataflow: out = std::copy_n("dataflow", 8, out); break;
		case celerity::detail::dependency_origin::collective_group_serialization: out = std::copy_n("collective-group-serialization", 31, out); break;
		case celerity::detail::dependency_origin::execution_front: out = std::copy_n("execution-front", 15, out); break;
		case celerity::detail::dependency_origin::last_epoch: out = std::copy_n("last-epoch", 10, out); break;
		}
		return out;
	}
};
