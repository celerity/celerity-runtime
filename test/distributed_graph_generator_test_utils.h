#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <type_traits>
#include <vector>

#include <catch2/catch_message.hpp>
#include <catch2/internal/catch_context.hpp>
#include <fmt/format.h>

#include "access_modes.h"
#include "command_graph.h"
#include "distributed_graph_generator.h"
#include "print_graph.h"
#include "recorders.h"
#include "task_manager.h"
#include "types.h"
#include "utils.h"

#include "test_utils.h"

using namespace celerity;
using namespace celerity::detail;

namespace celerity::test_utils {

class dist_cdag_test_context;
class idag_test_context;

template <typename TestContext>
class task_builder {
	friend class dist_cdag_test_context;
	friend class idag_test_context;

	using action = std::function<void(handler&)>;

	class step {
	  public:
		step(TestContext& dctx, action command, std::vector<action> requirements = {})
		    : m_tctx(dctx), m_command(std::move(command)), m_requirements(std::move(requirements)), m_uncaught_exceptions_before(std::uncaught_exceptions()) {}

		~step() noexcept(false) { // NOLINT(bugprone-exception-escape)
			if(std::uncaught_exceptions() == m_uncaught_exceptions_before && (m_command || !m_requirements.empty())) {
				throw std::runtime_error("Found incomplete task build. Did you forget to call submit()?");
			}
		}

		step(const step&) = delete;
		step(step&&) = delete;
		step& operator=(const step&) = delete;
		step& operator=(step&&) = delete;

		task_id submit() {
			assert(m_command);
			const auto tid = m_tctx.submit_command_group([this](handler& cgh) {
				for(auto& a : m_requirements) {
					a(cgh);
				}
				m_command(cgh);
			});
			m_tctx.build_task(tid);
			m_tctx.maybe_build_horizon();
			m_command = {};
			m_requirements = {};
			return tid;
		}

		step name(const std::string& name) {
			return chain<step>([&name](handler& cgh) { celerity::debug::set_task_name(cgh, name); });
		}

		template <typename BufferT, typename RangeMapper>
		step read(BufferT& buf, RangeMapper rmfn) {
			return chain<step>([&buf, rmfn](handler& cgh) { buf.template get_access<access_mode::read>(cgh, rmfn); });
		}

		template <typename BufferT, typename RangeMapper>
		step read_write(BufferT& buf, RangeMapper rmfn) {
			return chain<step>([&buf, rmfn](handler& cgh) { buf.template get_access<access_mode::read_write>(cgh, rmfn); });
		}

		template <typename BufferT, typename RangeMapper>
		step write(BufferT& buf, RangeMapper rmfn) {
			return chain<step>([&buf, rmfn](handler& cgh) { buf.template get_access<access_mode::write>(cgh, rmfn); });
		}

		template <typename BufferT, typename RangeMapper>
		step discard_write(BufferT& buf, RangeMapper rmfn) {
			return chain<step>([&buf, rmfn](handler& cgh) { buf.template get_access<access_mode::discard_write>(cgh, rmfn); });
		}

		template <typename BufferT>
		inline step reduce(BufferT& buf, const bool include_current_buffer_value) {
			return chain<step>([this, &buf, include_current_buffer_value](
			                       handler& cgh) { add_reduction(cgh, m_tctx.create_reduction(buf.get_id(), include_current_buffer_value)); });
		}

		template <typename HostObjT>
		step affect(HostObjT& host_obj, experimental::side_effect_order order = experimental::side_effect_order::sequential) {
			return chain<step>([&host_obj, order](handler& cgh) { host_obj.add_side_effect(cgh, order); });
		}

		template <int Dims>
		step constrain_split(const range<Dims>& constraint) {
			return chain<step>([constraint](handler& cgh) { experimental::constrain_split(cgh, constraint); });
		}

		template <typename Hint>
		step hint(Hint hint) {
			return chain<step>([&hint](handler& cgh) { experimental::hint(cgh, hint); });
		}

	  private:
		TestContext& m_tctx;
		action m_command;
		std::vector<action> m_requirements;
		int m_uncaught_exceptions_before;

		template <typename StepT>
		StepT chain(action a) {
			static_assert(std::is_base_of_v<step, StepT>);
			m_requirements.push_back(std::move(a));
			return StepT{m_tctx, std::move(m_command), std::move(m_requirements)};
		}
	};

  public:
	template <typename Name, int Dims>
	step device_compute(const range<Dims>& global_size, const id<Dims>& global_offset) {
		return step(m_dctx, [global_size, global_offset](handler& cgh) { cgh.parallel_for<Name>(global_size, global_offset, [](id<Dims>) {}); });
	}

	template <typename Name, int Dims>
	step device_compute(const nd_range<Dims>& execution_range) {
		return step(m_dctx, [execution_range](handler& cgh) { cgh.parallel_for<Name>(execution_range, [](nd_item<Dims>) {}); });
	}

	template <int Dims>
	step host_task(const range<Dims>& global_size) {
		return step(m_dctx, [global_size](handler& cgh) { cgh.host_task(global_size, [](partition<Dims>) {}); });
	}

	step master_node_host_task() {
		std::deque<action> actions;
		return step(m_dctx, [](handler& cgh) { cgh.host_task(on_master_node, [] {}); });
	}

	step collective_host_task(experimental::collective_group group) {
		return step(m_dctx, [group](handler& cgh) { cgh.host_task(experimental::collective(group), [](const experimental::collective_partition&) {}); });
	}

  private:
	TestContext& m_dctx;

	task_builder(TestContext& dctx) : m_dctx(dctx) {}
};

template <typename T>
constexpr static bool is_basic_query_filter_v = std::is_same_v<node_id, T> || std::is_same_v<task_id, T> || std::is_same_v<command_type, T>;

template <typename T>
constexpr static bool is_dependency_query_filter_v = std::is_same_v<dependency_kind, T> || std::is_same_v<dependency_origin, T>;

class command_query {
	friend struct command_query_testspy;
	friend class dist_cdag_test_context;

	class query_exception : public std::runtime_error {
		using std::runtime_error::runtime_error;
	};

  public:
	// -------------------------------------------------------------------------------------------------------------------------------------------------------
	// ------------------------------------------------------------------- Query functions -------------------------------------------------------------------
	// -------------------------------------------------------------------------------------------------------------------------------------------------------

	/**
	 * Finds all commands within the current set that match a given list of filters.

	 * Currently supported filters are node_id, task_id and command_type.
	 * Filters are applied conjunctively (AND), hence each type can be specified at most once.
	 */
	template <typename... Filters>
	command_query find_all(Filters... filters) const {
		assert_not_empty(__FUNCTION__);
		static_assert((is_basic_query_filter_v<Filters> && ...), "Unsupported filter");

		const auto node_filter = get_optional<node_id>(filters...);
		const auto task_filter = get_optional<task_id>(filters...);
		const auto type_filter = get_optional<command_type>(filters...);

		std::vector<std::unordered_set<const abstract_command*>> filtered(m_commands_by_node.size());
		for(node_id nid = 0; nid < m_commands_by_node.size(); ++nid) {
			if(node_filter.has_value() && *node_filter != nid) continue;
			for(const auto* cmd : m_commands_by_node[nid]) {
				if(task_filter.has_value()) {
					if(!utils::isa<task_command>(cmd)) continue;
					if(utils::as<task_command>(cmd)->get_tid() != *task_filter) continue;
				}
				if(type_filter.has_value()) {
					if(cmd->get_type() != *type_filter) continue;
				}
				filtered[nid].insert(cmd);
			}
		}

		return command_query{std::move(filtered)};
	}

	/**
	 * Returns a new command_query that contains all commands that precede the current set of commands.
	 */
	template <typename... Filters>
	command_query find_predecessors(Filters... filters) const {
		assert_not_empty(__FUNCTION__);
		return find_adjacent(true, filters...);
	}

	/**
	 * Returns a new command_query that contains all commands that succeed the current set of commands.
	 */
	template <typename... Filters>
	command_query find_successors(Filters... filters) const {
		assert_not_empty(__FUNCTION__);
		return find_adjacent(false, filters...);
	}

	/**
	 * Returns the total number of commands across all nodes.
	 */
	size_t count() const {
		return std::accumulate(
		    m_commands_by_node.begin(), m_commands_by_node.end(), size_t(0), [](size_t current, auto& cmds) { return current + cmds.size(); });
	}

	/**
	 * Returns the number of commands per node, if it is the same, throws otherwise.
	 */
	size_t count_per_node() const {
		if(m_commands_by_node.empty()) return 0;
		const size_t count = m_commands_by_node[0].size();
		for(size_t i = 1; i < m_commands_by_node.size(); ++i) {
			if(m_commands_by_node[i].size() != count) {
				throw query_exception(
				    fmt::format("Different number of commands across nodes (node 0: {}, node {}: {})", count, i, m_commands_by_node[i].size()));
			}
		}
		return count;
	}

	/**
	 * Chainable variant of count(), for use as part of larger query expressions.
	 */
	command_query assert_count(const size_t expected) const {
		if(count() != expected) { throw query_exception(fmt::format("Expected {} total command(s), found {}", expected, count())); }
		return *this;
	}

	/**
	 * Chainable variant of count_per_node(), for use as part of larger query expressions.
	 */
	command_query assert_count_per_node(const size_t expected) const {
		if(count_per_node() != expected) { throw query_exception(fmt::format("Expected {} command(s) per node, found {}", expected, count_per_node())); }
		return *this;
	}

	bool empty() const { return count() == 0; }

	// --------------------------------------------------------------------------------------------------------------------------------------------------------
	// -------------------------------------------------------------------- Set operations --------------------------------------------------------------------
	// --------------------------------------------------------------------------------------------------------------------------------------------------------

	friend command_query operator-(const command_query& lhs, const command_query& rhs) { return lhs.subtract(rhs); }
	friend command_query operator+(const command_query& lhs, const command_query& rhs) { return lhs.merge(rhs); }

	command_query subtract(const command_query& other) const {
		assert_not_empty(__FUNCTION__);
		assert(m_commands_by_node.size() == other.m_commands_by_node.size());
		std::vector<std::unordered_set<const abstract_command*>> result(m_commands_by_node.size());
		for(node_id nid = 0; nid < m_commands_by_node.size(); ++nid) {
			std::copy_if(m_commands_by_node[nid].cbegin(), m_commands_by_node[nid].cend(), std::inserter(result[nid], result[nid].begin()),
			    [&other, nid](const abstract_command* cmd) { return other.m_commands_by_node[nid].count(cmd) == 0; });
		}
		return command_query{std::move(result)};
	}

	command_query merge(const command_query& other) const {
		assert_not_empty(__FUNCTION__);
		assert(m_commands_by_node.size() == other.m_commands_by_node.size());
		std::vector<std::unordered_set<const abstract_command*>> result(m_commands_by_node.size());
		for(node_id nid = 0; nid < m_commands_by_node.size(); ++nid) {
			result[nid].insert(m_commands_by_node[nid].cbegin(), m_commands_by_node[nid].cend());
			result[nid].insert(other.m_commands_by_node[nid].cbegin(), other.m_commands_by_node[nid].cend());
		}
		return command_query{std::move(result)};
	}

	// --------------------------------------------------------------------------------------------------------------------------------------------------------
	// ---------------------------------------------------------------------- Predicates ----------------------------------------------------------------------
	// --------------------------------------------------------------------------------------------------------------------------------------------------------

	/**
	 * Returns whether all commands on all nodes are of the given type.
	 */
	bool have_type(const command_type expected) const {
		assert_not_empty(__FUNCTION__);
		return for_all_commands([expected](const node_id nid, const abstract_command* cmd) {
			const auto received = cmd->get_type();
			if(received != expected) {
				UNSCOPED_INFO(fmt::format("Expected command {} on node {} to have type '{}' but found type '{}'", cmd->get_cid(), nid, get_type_name(expected),
				    get_type_name(received)));
				return false;
			}
			return true;
		});
	}

	/**
	 * Returns whether the current set of commands is succeeded by ALL commands in successors on each node.
	 *
	 * Throws if successors is empty or contains commands for nodes not present in the current query.
	 *
	 * NOTE: Care has to be taken when using this function in negative assertions. For example the check
	 *       `CHECK_FALSE(q.find_all(task_a).have_successors(q.find_all(task_b))` can NOT be used to check
	 *       whether there are no true dependencies between tasks a and b: If there are multiple nodes and
	 *       for only one of them there is no dependency, the assertion will pass.
	 */
	bool have_successors(const command_query& successors, const std::optional<dependency_kind>& kind = std::nullopt,
	    const std::optional<dependency_origin>& origin = std::nullopt) const {
		assert_not_empty(__FUNCTION__);

		if(successors.count() == 0) { throw query_exception("Successor set is empty"); }

		assert(m_commands_by_node.size() == successors.m_commands_by_node.size());
		for(node_id nid = 0; nid < m_commands_by_node.size(); ++nid) {
			if(m_commands_by_node[nid].empty() && !successors.m_commands_by_node[nid].empty()) {
				throw query_exception(fmt::format("A.have_successors(B): B contains commands for node {}, whereas A does not", nid));
			}
		}

		return for_all_commands([&successors, &kind, &origin](const node_id nid, const abstract_command* cmd) {
			for(const auto* expected : successors.m_commands_by_node[nid]) {
				bool found = false;
				for(const auto received : cmd->get_dependents()) {
					if(received.node == expected) {
						found = true;
						if(kind.has_value() && received.kind != *kind) {
							UNSCOPED_INFO(fmt::format("Expected command {} on node {} to have successor {} with kind {}, but found kind {}", cmd->get_cid(),
							    nid, expected->get_cid(), *kind, received.kind));
							return false;
						}
						if(origin.has_value() && received.origin != *origin) {
							UNSCOPED_INFO(fmt::format("Expected command {} on node {} to have successor {} with origin {}, but found origin {}", cmd->get_cid(),
							    nid, expected->get_cid(), *origin, received.origin));
							return false;
						}
					}
				}
				if(!found) {
					UNSCOPED_INFO(fmt::format("Expected command {} on node {} to have successor {}", cmd->get_cid(), nid, expected->get_cid()));
					return false;
				}
			}
			return true;
		});
	}

	// --------------------------------------------------------------------------------------------------------------------------------------------------------
	// ------------------------------------------------------------------------ Other -------------------------------------------------------------------------
	// --------------------------------------------------------------------------------------------------------------------------------------------------------

	/**
	 * Call the provided function once for each node, with a subquery containing commands only for that node.
	 *
	 * Using this function is usually not necessary, as all predicates (have_successors, have_types, ...) apply
	 * simultaneously on all nodes.
	 */
	template <typename PerNodeCallback>
	void for_each_node(PerNodeCallback&& cb) const {
		assert_not_empty(__FUNCTION__);
		for(node_id nid = 0; nid < m_commands_by_node.size(); ++nid) {
			UNSCOPED_INFO(fmt::format("On node {}", nid));
			cb(find_all(nid));
		}
	}

	/**
	 * Returns the raw command pointers contained within the query, optionally limited to a given node.
	 */
	std::vector<const abstract_command*> get_raw(const std::optional<node_id>& nid = std::nullopt) const {
		std::vector<const abstract_command*> result;
		for_all_commands([&result, &nid](const node_id n, const abstract_command* const cmd) {
			if(nid.has_value() && n != nid) return;
			result.push_back(cmd);
		});
		return result;
	}

  private:
	std::vector<std::unordered_set<const abstract_command*>> m_commands_by_node;
	bool m_allow_empty_operations = false;

	// Constructor for initial top-level query (containing all commands)
	command_query(const std::vector<std::unique_ptr<command_graph>>& cdags) {
		for(const auto& cdag : cdags) {
			m_commands_by_node.push_back({cdag->all_commands().begin(), cdag->all_commands().end()});
		}
	}

	// Constructor for narrowed-down queries
	command_query(std::vector<std::unordered_set<const abstract_command*>> commands_by_node) : m_commands_by_node(std::move(commands_by_node)) {}

	void assert_not_empty(const std::string& op_name) const {
		if(m_allow_empty_operations) return;
		const bool all_empty = std::all_of(m_commands_by_node.cbegin(), m_commands_by_node.cend(), [](const auto& cmds) { return cmds.empty(); });
		if(all_empty) { throw query_exception(fmt::format("Operation '{}' not allowed on empty query", op_name)); }
	}

	template <typename Callback>
	bool for_all_commands(Callback&& cb) const {
		bool cont = true;
		for(node_id nid = 0; cont && nid < m_commands_by_node.size(); ++nid) {
			for(const auto* cmd : m_commands_by_node[nid]) {
				if constexpr(std::is_invocable_r_v<bool, Callback, node_id, decltype(cmd)>) {
					cont &= cb(nid, cmd);
				} else {
					cb(nid, cmd);
				}
				if(!cont) break;
			}
		}
		return cont;
	}

	template <typename... Filters>
	command_query find_adjacent(const bool find_predecessors, Filters... filters) const {
		static_assert(((is_basic_query_filter_v<Filters> || is_dependency_query_filter_v<Filters>)&&...), "Unsupported filter");
		const auto kind_filter = get_optional<dependency_kind>(filters...);
		const auto origin_filter = get_optional<dependency_origin>(filters...);

		std::vector<std::unordered_set<const abstract_command*>> adjacent(m_commands_by_node.size());
		for_all_commands([&adjacent, find_predecessors, kind_filter, origin_filter](const node_id nid, const abstract_command* cmd) {
			const auto iterable = find_predecessors ? cmd->get_dependencies() : cmd->get_dependents();
			for(auto it = iterable.begin(); it != iterable.end(); ++it) {
				if(kind_filter.has_value() && it->kind != *kind_filter) continue;
				if(origin_filter.has_value() && it->origin != *origin_filter) continue;
				adjacent[nid].insert(it->node);
			}
		});

		auto query = command_query{std::move(adjacent)};
		query.m_allow_empty_operations = true;
		// Filter resulting set of commands, but remove dependency_kind/origin filters (if present)
		return std::apply(
		    [&query](auto... fs) { return query.find_all(fs...); }, utils::tuple_without<dependency_kind, dependency_origin>(std::tuple{filters...}));
	}

	template <typename T, typename... Ts>
	static constexpr std::optional<T> get_optional(const std::tuple<Ts...>& tuple) {
		if constexpr((std::is_same_v<T, Ts> || ...)) { return std::get<T>(tuple); }
		return std::nullopt;
	}

	template <typename T, typename... Ts>
	static constexpr std::optional<T> get_optional(Ts... ts) {
		return get_optional<T>(std::tuple(ts...));
	}

	static std::string get_type_name(const command_type type) {
		switch(type) {
		case command_type::epoch: return "epoch";
		case command_type::horizon: return "horizon";
		case command_type::execution: return "execution";
		case command_type::push: return "push";
		case command_type::await_push: return "await_push";
		case command_type::reduction: return "reduction";
		case command_type::fence: return "fence";
		default: return "<unknown>";
		}
	}
};

inline std::string make_test_graph_title(const std::string& type) {
	const auto test_name = Catch::getResultCapture().getCurrentTestName();
	auto title = fmt::format("<br/>{}", type);
	if(!test_name.empty()) { fmt::format_to(std::back_inserter(title), "<br/><b>{}</b>", test_name); }
	return title;
}

inline std::string make_test_graph_title(
    const std::string& type, const size_t num_nodes, const node_id local_nid, const std::optional<size_t> num_devices_per_node = std::nullopt) {
	auto title = make_test_graph_title(type);
	fmt::format_to(std::back_inserter(title), "<br/>for N{} out of {} nodes", local_nid, num_nodes);
	if(num_devices_per_node.has_value()) { fmt::format_to(std::back_inserter(title), ", with {} devices / node", *num_devices_per_node); }
	return title;
}

class dist_cdag_test_context {
	friend class task_builder<dist_cdag_test_context>;

  public:
	struct policy_set {
		task_manager::policy_set tm;
		distributed_graph_generator::policy_set dggen;
	};

	dist_cdag_test_context(const size_t num_nodes, const policy_set& policy = {}) : m_num_nodes(num_nodes), m_tm(num_nodes, &m_task_recorder, policy.tm) {
		for(node_id nid = 0; nid < num_nodes; ++nid) {
			m_cdags.emplace_back(std::make_unique<command_graph>());
			m_cmd_recorders.emplace_back(std::make_unique<command_recorder>());
			m_dggens.emplace_back(std::make_unique<distributed_graph_generator>(num_nodes, nid, *m_cdags[nid], m_tm, m_cmd_recorders[nid].get(), policy.dggen));
		}
	}

	~dist_cdag_test_context() { maybe_log_graphs(); }

	dist_cdag_test_context(const dist_cdag_test_context&) = delete;
	dist_cdag_test_context(dist_cdag_test_context&&) = delete;
	dist_cdag_test_context& operator=(const dist_cdag_test_context&) = delete;
	dist_cdag_test_context& operator=(dist_cdag_test_context&&) = delete;

	template <int Dims>
	test_utils::mock_buffer<Dims> create_buffer(range<Dims> size, bool mark_as_host_initialized = false) {
		const buffer_id bid = m_next_buffer_id++;
		const auto buf = test_utils::mock_buffer<Dims>(bid, size);
		m_tm.notify_buffer_created(bid, range_cast<3>(size), mark_as_host_initialized);
		for(auto& dggen : m_dggens) {
			dggen->notify_buffer_created(bid, range_cast<3>(size), mark_as_host_initialized);
		}
		return buf;
	}

	test_utils::mock_host_object create_host_object(const bool owns_instance = true) {
		const host_object_id hoid = m_next_host_object_id++;
		m_tm.notify_host_object_created(hoid);
		for(auto& dggen : m_dggens) {
			dggen->notify_host_object_created(hoid);
		}
		return test_utils::mock_host_object(hoid);
	}

	// TODO: Do we want to duplicate all step functions here, or have some sort of .task() initial builder?

	template <typename Name = unnamed_kernel, int Dims>
	auto device_compute(const range<Dims>& global_size, const id<Dims>& global_offset = {}) {
		return task_builder(*this).template device_compute<Name>(global_size, global_offset);
	}

	template <typename Name = unnamed_kernel, int Dims>
	auto device_compute(const nd_range<Dims>& execution_range) {
		return task_builder(*this).template device_compute<Name>(execution_range);
	}

	template <int Dims>
	auto host_task(const range<Dims>& global_size) {
		return task_builder(*this).host_task(global_size);
	}

	auto master_node_host_task() { return task_builder(*this).master_node_host_task(); }

	auto collective_host_task(experimental::collective_group group = experimental::default_collective_group) {
		return task_builder(*this).collective_host_task(group);
	}

	task_id fence(test_utils::mock_host_object ho) {
		side_effect_map side_effects;
		side_effects.add_side_effect(ho.get_id(), experimental::side_effect_order::sequential);
		return fence({}, std::move(side_effects));
	}

	template <int Dims>
	task_id fence(test_utils::mock_buffer<Dims> buf, subrange<Dims> sr) {
		buffer_access_map access_map;
		access_map.add_access(buf.get_id(),
		    std::make_unique<range_mapper<Dims, celerity::access::fixed<Dims>>>(celerity::access::fixed<Dims>(sr), access_mode::read, buf.get_range()));
		return fence(std::move(access_map), {});
	}

	template <int Dims>
	task_id fence(test_utils::mock_buffer<Dims> buf) {
		return fence(buf, {{}, buf.get_range()});
	}

	task_id epoch(epoch_action action) {
		const auto tid = m_tm.generate_epoch_task(action);
		build_task(tid);
		return tid;
	}

	template <typename... Filters>
	command_query query(Filters... filters) {
		return command_query(m_cdags).find_all(filters...);
	}

	void set_horizon_step(const int step) { m_tm.set_horizon_step(step); }

	task_manager& get_task_manager() { return m_tm; }

	distributed_graph_generator& get_graph_generator(node_id nid) { return *m_dggens.at(nid); }

	[[nodiscard]] std::string print_task_graph() { return detail::print_task_graph(m_task_recorder, make_test_graph_title("Task Graph")); }
	[[nodiscard]] std::string print_command_graph(node_id nid) {
		return detail::print_command_graph(nid, *m_cmd_recorders[nid], make_test_graph_title("Command Graph"));
	}

  private:
	size_t m_num_nodes;
	buffer_id m_next_buffer_id = 0;
	host_object_id m_next_host_object_id = 0;
	reduction_id m_next_reduction_id = 1; // Start from 1 as rid 0 designates "no reduction" in push commands
	std::optional<task_id> m_most_recently_built_horizon;
	task_recorder m_task_recorder;
	task_manager m_tm;
	std::vector<std::unique_ptr<command_graph>> m_cdags;
	std::vector<std::unique_ptr<distributed_graph_generator>> m_dggens;
	std::vector<std::unique_ptr<command_recorder>> m_cmd_recorders;

	reduction_info create_reduction(const buffer_id bid, const bool include_current_buffer_value) {
		return reduction_info{m_next_reduction_id++, bid, include_current_buffer_value};
	}

	template <typename CGF, typename... Hints>
	task_id submit_command_group(CGF cgf, Hints... hints) {
		return m_tm.submit_command_group(cgf, hints...);
	}

	void build_task(const task_id tid) {
		for(auto& dggen : m_dggens) {
			dggen->build_task(*m_tm.get_task(tid));
		}
	}

	void maybe_build_horizon() {
		const auto current_horizon = task_manager_testspy::get_current_horizon(m_tm);
		if(m_most_recently_built_horizon != current_horizon) {
			assert(current_horizon.has_value());
			build_task(*current_horizon);
		}
		m_most_recently_built_horizon = current_horizon;
	}

	void maybe_log_graphs() {
		if(test_utils::g_print_graphs) {
			fmt::print("{}\n", std::string(79, '-'));
			if(const auto capture = Catch::getCurrentContext().getResultCapture()) { fmt::print("DAGs for [{}]\n", capture->getCurrentTestName()); }
			fmt::print("\n{}\n", print_task_graph());
			std::vector<std::string> graphs;
			for(node_id nid = 0; nid < m_num_nodes; ++nid) {
				graphs.push_back(print_command_graph(nid));
			}
			fmt::print("\n{}\n", combine_command_graphs(graphs, make_test_graph_title("Command Graph")));
			fmt::print("\n{}\n\n", std::string(79, '-'));
		}
	}

	task_id fence(buffer_access_map access_map, side_effect_map side_effects) {
		const auto tid = m_tm.generate_fence_task(std::move(access_map), std::move(side_effects), nullptr);
		build_task(tid);
		maybe_build_horizon();
		return tid;
	}
};

} // namespace celerity::test_utils
