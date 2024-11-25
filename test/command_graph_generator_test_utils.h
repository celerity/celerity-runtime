#pragma once

#include <memory>
#include <type_traits>
#include <vector>

#include <catch2/catch_message.hpp>
#include <catch2/internal/catch_context.hpp>
#include <fmt/format.h>
#include <fmt/ranges.h>

#include "command_graph.h"
#include "command_graph_generator.h"
#include "handler.h"
#include "print_graph.h"
#include "recorders.h"
#include "task_manager.h"
#include "types.h"

#include "graph_test_utils.h"
#include "test_utils.h"

using namespace celerity;
using namespace celerity::detail;

namespace celerity::test_utils {

template <int Dims>
std::vector<std::pair<node_id, region<3>>> push_regions(const std::vector<std::pair<node_id, region<Dims>>>& regions) {
	std::vector<std::pair<node_id, region<3>>> result;
	std::transform(regions.begin(), regions.end(), std::back_inserter(result), [](const auto& p) { return std::make_pair(p.first, region_cast<3>(p.second)); });
	return result;
}

template <typename Record>
struct command_matcher {
	static bool matches(const Record& cmd, const std::string& debug_name) {
		return matchbox::match(
		    cmd,                                                                                 //
		    [&](const execution_command_record& ecmd) { return ecmd.debug_name == debug_name; }, //
		    [](const auto& /* other */) { return false; });
	}

	static bool matches(const Record& cmd, const task_id tid) {
		if(const auto* tcmd = dynamic_cast<const task_command_record*>(&cmd); tcmd != nullptr) { return tcmd->tid == tid; }
		return false;
	}

	template <typename R = Record, std::enable_if_t<std::is_same_v<R, push_command_record> || std::is_same_v<R, await_push_command_record>, int> = 0>
	static bool matches(const R& cmd, const buffer_id bid) {
		return matchbox::match(
		    cmd,                                                             //
		    [&](const push_command_record& c) { return c.trid.bid == bid; }, //
		    [](const auto& /* other */) { return false; });
	}

	static std::string print_filter(const std::string& debug_name) { return fmt::format("\"{}\"", debug_name); }
	static std::string print_filter(const task_id tid) { return fmt::format("\"T{}\"", tid); }
	static std::string print_filter(const buffer_id bid) { return fmt::format("\"B{}\"", bid); }
};

using command_query = graph_query<command_record, command_record, command_recorder, command_matcher>;

/// Wrapper type around command_query that adds semantics for command graphs on multiple nodes.
template <typename Record = command_record>
class distributed_command_query {
  public:
	template <typename R>
	using command_query = graph_query<R, command_record, command_recorder, command_matcher>;

	explicit distributed_command_query(std::vector<command_query<Record>>&& queries) : m_queries(std::move(queries)) {}

	// allow upcast
	template <typename SpecificRecord, std::enable_if_t<std::is_base_of_v<Record, SpecificRecord> && !std::is_same_v<Record, SpecificRecord>, int> = 0>
	distributed_command_query(const distributed_command_query<SpecificRecord>& other)
	    : distributed_command_query(std::vector<command_query<Record>>(other.m_queries.begin(), other.m_queries.end())) {}

	// ------------------------- distributed graph interface -------------------------

	command_query<Record> on(const node_id nid) const {
		REQUIRE(nid < m_queries.size());
		return m_queries[nid];
	}

	size_t total_count() const {
		size_t sum = 0;
		for(const auto& q : m_queries) {
			sum += q.count();
		}
		return sum;
	}

	size_t count_per_node() const {
		const size_t expected = m_queries.at(0).count();
		for(node_id nid = 1; nid < m_queries.size(); ++nid) {
			REQUIRE(m_queries[nid].count() == expected);
		}
		return expected;
	}

	const distributed_command_query& assert_total_count(const size_t expected) const {
		REQUIRE(total_count() == expected);
		return *this;
	}

	const distributed_command_query& assert_count_per_node(const size_t expected) const {
		for(node_id nid = 0; nid < m_queries.size(); ++nid) {
			REQUIRE(m_queries[nid].count() == expected);
		}
		return *this;
	}

	const std::vector<command_query<Record>>& iterate_nodes() const& { return m_queries; }

	std::vector<command_query<Record>> iterate_nodes() && { return std::move(m_queries); }

	// ---------------------------- graph_query interface ----------------------------

	template <typename SpecificRecord = Record, typename... Filters>
	distributed_command_query<SpecificRecord> select_all(const Filters&... filters) const {
		return apply<SpecificRecord>([&filters...](auto& q) { return q.template select_all<SpecificRecord>(filters...); });
	}

	template <typename SpecificRecord = Record, typename... Filters>
	distributed_command_query<SpecificRecord> select_unique(const Filters&... filters) const {
		return apply<SpecificRecord>([&filters...](auto& q) { return q.template select_unique<SpecificRecord>(filters...); });
	}

	distributed_command_query<command_record> predecessors() const {
		return apply<command_record>([](auto& q) { return q.predecessors(); });
	}

	distributed_command_query<command_record> transitive_predecessors() const {
		return apply<command_record>([](auto& q) { return q.transitive_predecessors(); });
	}

	template <typename SpecificRecord = Record, typename... Filters>
	distributed_command_query<SpecificRecord> transitive_predecessors_across(const Filters&... filters) const {
		return apply<SpecificRecord>([&filters...](auto& q) { return q.template transitive_predecessors_across<SpecificRecord>(filters...); });
	}

	distributed_command_query<command_record> successors() const {
		return apply<command_record>([](auto& q) { return q.successors(); });
	}

	distributed_command_query<command_record> transitive_successors() const {
		return apply<command_record>([](auto& q) { return q.transitive_successors(); });
	}

	template <typename SpecificRecord = Record, typename... Filters>
	distributed_command_query<SpecificRecord> transitive_successors_across(const Filters&... filters) const {
		return apply<SpecificRecord>([&filters...](auto& q) { return q.template transitive_successors_across<SpecificRecord>(filters...); });
	}

	bool is_concurrent_with(const distributed_command_query& other) const {
		for(node_id nid = 0; nid < m_queries.size(); ++nid) {
			if(!m_queries[nid].is_concurrent_with(other.on(nid))) return false;
		}
		return true;
	}

	template <typename SpecificRecord = Record, typename... Filters>
	bool all_match(const Filters&... filters) const {
		return std::all_of(m_queries.begin(), m_queries.end(), [&filters...](const auto& q) { return q.template all_match<SpecificRecord>(filters...); });
	}

	template <typename SpecificRecord = Record, typename... Filters>
	distributed_command_query<SpecificRecord> assert_all(const Filters&... filters) const {
		return apply<SpecificRecord>([&filters...](auto& q) { return q.template assert_all<SpecificRecord>(filters...); });
	}

	bool contains(const distributed_command_query& subset) const {
		for(node_id nid = 0; nid < m_queries.size(); ++nid) {
			if(!m_queries[nid].contains(subset.on(nid))) return false;
		}
		return true;
	}

	bool operator==(const distributed_command_query& other) const {
		if(m_queries.size() != other.m_queries.size()) return false;
		for(node_id nid = 0; nid < m_queries.size(); ++nid) {
			if(m_queries[nid] != other.m_queries[nid]) return false;
		}
		return true;
	}
	bool operator!=(const distributed_command_query& other) const { return !(*this == other); }

	template <typename... DistributedCommandQueries>
	friend distributed_command_query union_of(const distributed_command_query& head, const DistributedCommandQueries&... tail) {
		REQUIRE(((head.m_queries.size() == tail.m_queries.size()) && ...));
		std::vector<command_query<Record>> result;
		for(node_id nid = 0; nid < head.m_queries.size(); ++nid) {
			result.push_back(union_of(head.m_queries[nid], tail.m_queries[nid]...));
		}
		return distributed_command_query{std::move(result)};
	}

	template <typename... DistributedCommandQueries>
	friend distributed_command_query intersection_of(const distributed_command_query& head, const DistributedCommandQueries&... tail) {
		REQUIRE(((head.m_queries.size() == tail.m_queries.size()) && ...));
		std::vector<command_query<Record>> result;
		for(node_id nid = 0; nid < head.m_queries.size(); ++nid) {
			result.push_back(intersection_of(head.m_queries[nid], tail.m_queries[nid]...));
		}
		return distributed_command_query{std::move(result)};
	}

	friend distributed_command_query difference_of(const distributed_command_query& lhs, const distributed_command_query& rhs) {
		REQUIRE(lhs.m_queries.size() == rhs.m_queries.size());
		std::vector<command_query<Record>> result;
		for(node_id nid = 0; nid < lhs.m_queries.size(); ++nid) {
			result.push_back(difference_of(lhs.m_queries[nid], rhs.m_queries[nid]));
		}
		return distributed_command_query{std::move(result)};
	}

  private:
	template <typename>
	friend class distributed_command_query;

	template <typename, typename, typename>
	friend struct fmt::formatter;

	std::vector<command_query<Record>> m_queries;

	template <typename SpecificRecord, typename Function>
	distributed_command_query<SpecificRecord> apply(const Function& fn) const {
		std::vector<command_query<SpecificRecord>> result;
		for(auto& q : m_queries) {
			result.push_back(fn(q));
		}
		return distributed_command_query<SpecificRecord>{std::move(result)};
	}
};

class cdag_test_context final : private task_manager::delegate {
	friend class task_builder<cdag_test_context>;

  public:
	struct policy_set {
		task_manager::policy_set tm;
		command_graph_generator::policy_set cggen;
	};

	cdag_test_context(const size_t num_nodes, const policy_set& policy = {})
	    : m_num_nodes(num_nodes), m_tm(num_nodes, m_tdag, &m_task_recorder, this, policy.tm) {
		for(node_id nid = 0; nid < num_nodes; ++nid) {
			m_cdags.emplace_back(std::make_unique<command_graph>());
			m_cmd_recorders.emplace_back(std::make_unique<command_recorder>());
			m_cggens.emplace_back(std::make_unique<command_graph_generator>(num_nodes, nid, *m_cdags[nid], m_cmd_recorders[nid].get(), policy.cggen));
		}
		m_initial_epoch_tid = m_tm.generate_epoch_task(epoch_action::init);
	}

	~cdag_test_context() { maybe_print_graphs(); }

	cdag_test_context(const cdag_test_context&) = delete;
	cdag_test_context(cdag_test_context&&) = delete;
	cdag_test_context& operator=(const cdag_test_context&) = delete;
	cdag_test_context& operator=(cdag_test_context&&) = delete;

	void task_created(const task* tsk) override {
		for(auto& cggen : m_cggens) {
			cggen->build_task(*tsk);
		}
	}

	template <int Dims>
	test_utils::mock_buffer<Dims> create_buffer(range<Dims> size, bool mark_as_host_initialized = false) {
		const buffer_id bid = m_next_buffer_id++;
		const auto buf = test_utils::mock_buffer<Dims>(bid, size);
		m_tm.notify_buffer_created(bid, range_cast<3>(size), mark_as_host_initialized);
		for(auto& cggen : m_cggens) {
			cggen->notify_buffer_created(bid, range_cast<3>(size), mark_as_host_initialized);
		}
		return buf;
	}

	test_utils::mock_host_object create_host_object(const bool owns_instance = true) {
		const host_object_id hoid = m_next_host_object_id++;
		m_tm.notify_host_object_created(hoid);
		for(auto& cggen : m_cggens) {
			cggen->notify_host_object_created(hoid);
		}
		return test_utils::mock_host_object(hoid);
	}

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

	auto collective_host_task(experimental::collective_group group = detail::default_collective_group) {
		return task_builder(*this).collective_host_task(group);
	}

	task_id fence(test_utils::mock_host_object ho) {
		host_object_effect effect{ho.get_id(), experimental::side_effect_order::sequential};
		return m_tm.generate_fence_task(effect, nullptr);
	}

	template <int Dims>
	task_id fence(test_utils::mock_buffer<Dims> buf, subrange<Dims> sr) {
		buffer_access access{buf.get_id(), access_mode::read,
		    std::make_unique<range_mapper<Dims, celerity::access::fixed<Dims>>>(celerity::access::fixed<Dims>(sr), buf.get_range())};
		return m_tm.generate_fence_task(std::move(access), nullptr);
	}

	template <int Dims>
	task_id fence(test_utils::mock_buffer<Dims> buf) {
		return fence(buf, {{}, buf.get_range()});
	}

	task_id epoch(epoch_action action) { return m_tm.generate_epoch_task(action); }

	template <typename SpecificRecord = command_record, typename... Filters>
	distributed_command_query<SpecificRecord> query(Filters... filters) {
		std::vector<typename distributed_command_query<>::command_query<command_record>> queries;
		for(auto& recorder : m_cmd_recorders) {
			queries.push_back(typename distributed_command_query<>::command_query<command_record>(*recorder));
		}
		return distributed_command_query(std::move(queries)).template select_all<SpecificRecord>(std::forward<Filters>(filters)...);
	}

	void set_horizon_step(const int step) { m_tm.set_horizon_step(step); }

	void set_test_chunk_multiplier(const size_t multiplier) {
		for(auto& cggen : m_cggens) {
			cggen->test_set_chunk_multiplier(multiplier);
		}
	}

	task_graph& get_task_graph() { return m_tdag; }

	task_manager& get_task_manager() { return m_tm; }

	command_graph& get_command_graph(node_id nid) { return *m_cdags.at(nid); }

	command_graph_generator& get_graph_generator(node_id nid) { return *m_cggens.at(nid); }

	task_id get_initial_epoch_task() const { return m_initial_epoch_tid; }

	[[nodiscard]] std::string print_task_graph() { return detail::print_task_graph(m_task_recorder, make_test_graph_title("Task Graph")); }

	[[nodiscard]] std::string print_command_graph(node_id nid) {
		// Don't include node id in title: All CDAG printouts must have identical preambles for combine_command_graphs to work
		return detail::print_command_graph(nid, *m_cmd_recorders[nid], make_test_graph_title("Command Graph"));
	}

  private:
	size_t m_num_nodes;
	buffer_id m_next_buffer_id = 0;
	host_object_id m_next_host_object_id = 0;
	reduction_id m_next_reduction_id = 1; // Start from 1 as rid 0 designates "no reduction" in push commands
	task_graph m_tdag;
	task_manager m_tm;
	task_recorder m_task_recorder;
	task_id m_initial_epoch_tid = 0;
	std::vector<std::unique_ptr<command_graph>> m_cdags;
	std::vector<std::unique_ptr<command_graph_generator>> m_cggens;
	std::vector<std::unique_ptr<command_recorder>> m_cmd_recorders;

	reduction_info create_reduction(const buffer_id bid, const bool include_current_buffer_value) {
		return reduction_info{m_next_reduction_id++, bid, include_current_buffer_value};
	}

	template <typename CGF>
	task_id submit_command_group(CGF cgf) {
		return m_tm.generate_command_group_task(invoke_command_group_function(cgf));
	}

	void maybe_print_graphs() {
		if(test_utils::g_print_graphs) {
			fmt::print("\n{}\n", print_task_graph());
			std::vector<std::string> graphs;
			for(node_id nid = 0; nid < m_num_nodes; ++nid) {
				graphs.push_back(print_command_graph(nid));
			}
			fmt::print("\n{}\n", combine_command_graphs(graphs, make_test_graph_title("Command Graph")));
		}
	}
};

} // namespace celerity::test_utils

template <typename Record>
struct fmt::formatter<celerity::test_utils::distributed_command_query<Record>> : fmt::formatter<size_t> {
	format_context::iterator format(const celerity::test_utils::distributed_command_query<Record>& dcq, format_context& ctx) const {
		auto out = ctx.out();
		fmt::format_to(out, "[{}]", fmt::join(dcq.m_queries, ", "));
		return out;
	}
};

template <typename Record>
struct Catch::StringMaker<celerity::test_utils::distributed_command_query<Record>> {
	static std::string convert(const celerity::test_utils::distributed_command_query<Record>& dcq) { return fmt::format("{}", dcq); }
};
