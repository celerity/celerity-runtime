#include <unordered_set>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>

#include "command_graph.h"

#include "distributed_graph_generator_test_utils.h"

using namespace celerity;
using namespace celerity::detail;
using namespace celerity::test_utils;

// Our testing utilities have become so sophisticated that they need tests of their own...

namespace celerity::test_utils {
struct command_query_testspy {
	static command_query create_for(const std::vector<std::unique_ptr<command_graph>>& cdags) { return command_query{cdags}; }
};
} // namespace celerity::test_utils

TEST_CASE("command_query::find_all supports various filters", "[command_query]") {
	std::vector<std::unique_ptr<command_graph>> cdags;
	cdags.emplace_back(std::make_unique<command_graph>());
	cdags.emplace_back(std::make_unique<command_graph>());

	auto* exe0_0 = cdags[0]->create<execution_command>(task_id(0), subrange<3>{});
	auto* exe0_1 = cdags[0]->create<execution_command>(task_id(1), subrange<3>{});
	auto* exe1_0 = cdags[1]->create<execution_command>(task_id(0), subrange<3>{});
	auto* exe1_1 = cdags[1]->create<execution_command>(task_id(1), subrange<3>{});

	auto* push0 = cdags[0]->create<push_command>(node_id(1), transfer_id(task_id(2), buffer_id(0), no_reduction_id), subrange<3>{});
	auto* push1 = cdags[1]->create<push_command>(node_id(0), transfer_id(task_id(2), buffer_id(0), no_reduction_id), subrange<3>{});

	const auto check_result = [](const command_query& query, const std::unordered_set<const abstract_command*>& expected) {
		CHECK(query.count() == expected.size());
		for(node_id nid = 0; nid < 2; ++nid) {
			for(const auto* cmd : query.get_raw(nid)) {
				REQUIRE_LOOP(expected.count(cmd) == 1);
			}
		}
	};

	const auto q = command_query_testspy::create_for(cdags);

	// Basic filtering
	check_result(q.find_all(), {exe0_0, exe0_1, exe1_0, exe1_1, push0, push1});
	check_result(q.find_all(node_id(0)), {exe0_0, exe0_1, push0});
	check_result(q.find_all(node_id(1)), {exe1_0, exe1_1, push1});
	check_result(q.find_all(task_id(0)), {exe0_0, exe1_0});
	check_result(q.find_all(task_id(1)), {exe0_1, exe1_1});
	check_result(q.find_all(command_type::execution), {exe0_0, exe0_1, exe1_0, exe1_1});
	check_result(q.find_all(command_type::push), {push0, push1});

	// Filter combinations (AND)
	check_result(q.find_all(node_id(1), command_type::execution), {exe1_0, exe1_1});
	check_result(q.find_all(task_id(0), command_type::push), {});
}

TEST_CASE("command_query::has_successor allows to match two set of commands against each other for each node", "[command_query]") {
	std::vector<std::unique_ptr<command_graph>> cdags;
	cdags.emplace_back(std::make_unique<command_graph>());
	cdags.emplace_back(std::make_unique<command_graph>());

	auto* exe0_0 = cdags[0]->create<execution_command>(task_id(0), subrange<3>{});
	auto* exe0_1 = cdags[0]->create<execution_command>(task_id(1), subrange<3>{});
	auto* exe0_2 = cdags[0]->create<execution_command>(task_id(2), subrange<3>{});
	auto* exe1_0 = cdags[1]->create<execution_command>(task_id(0), subrange<3>{});
	auto* exe1_1 = cdags[1]->create<execution_command>(task_id(1), subrange<3>{});
	auto* exe1_2 = cdags[1]->create<execution_command>(task_id(2), subrange<3>{});

	const auto q = command_query_testspy::create_for(cdags);

	// Shorthands for neater formatting
	const auto true_dep = dependency_kind::true_dep;
	const auto anti_dep = dependency_kind::anti_dep;
	const auto dataflow = dependency_origin::dataflow;
	const auto collective = dependency_origin::collective_group_serialization;

	SECTION("one-to-one relationship") {
		cdags[0]->add_dependency(exe0_1, exe0_0, true_dep, dataflow);
		CHECK /**/ (q.find_all(task_id(0), node_id(0)).have_successors(q.find_all(task_id(1), node_id(0))));
		CHECK_FALSE(q.find_all(task_id(0), node_id(1)).have_successors(q.find_all(task_id(1), node_id(1))));
		CHECK_FALSE(q.find_all(task_id(0) /*       */).have_successors(q.find_all(task_id(1))));
		CHECK_FALSE(q.find_all(task_id(1) /*       */).have_successors(q.find_all(task_id(2))));

		cdags[1]->add_dependency(exe1_1, exe1_0, anti_dep, collective);
		CHECK(q.find_all(task_id(0)).have_successors(q.find_all(task_id(1))));

		// Predicate can optionally also check for dependency_kind and dependency_origin
		CHECK /**/ (q.find_all(task_id(0), node_id(0)).have_successors(q.find_all(task_id(1), node_id(0)), true_dep, dataflow));
		CHECK_FALSE(q.find_all(task_id(0), node_id(0)).have_successors(q.find_all(task_id(1), node_id(0)), anti_dep, collective));
		CHECK_FALSE(q.find_all(task_id(0), node_id(1)).have_successors(q.find_all(task_id(1), node_id(1)), true_dep, dataflow));
		CHECK /**/ (q.find_all(task_id(0), node_id(1)).have_successors(q.find_all(task_id(1), node_id(1)), anti_dep, collective));

		CHECK_FALSE(q.find_all(task_id(0)).have_successors(q.find_all(task_id(1)), true_dep));
		CHECK_FALSE(q.find_all(task_id(0)).have_successors(q.find_all(task_id(1)), anti_dep));
		CHECK_FALSE(q.find_all(task_id(0)).have_successors(q.find_all(task_id(1)), std::nullopt, dataflow));
		CHECK_FALSE(q.find_all(task_id(0)).have_successors(q.find_all(task_id(1)), std::nullopt, collective));
	}

	SECTION("one-to-many relationship") {
		cdags[0]->add_dependency(exe0_1, exe0_0, true_dep, dataflow);
		cdags[1]->add_dependency(exe1_1, exe1_0, true_dep, dataflow);
		CHECK_FALSE(q.find_all(task_id(0)).have_successors(q.find_all(task_id(1)) + q.find_all(task_id(2))));
		cdags[0]->add_dependency(exe0_2, exe0_0, anti_dep, collective);
		CHECK_FALSE(q.find_all(task_id(0)).have_successors(q.find_all(task_id(1)) + q.find_all(task_id(2))));
		cdags[1]->add_dependency(exe1_2, exe1_0, anti_dep, collective);
		CHECK /**/ (q.find_all(task_id(0)).have_successors(q.find_all(task_id(1)) + q.find_all(task_id(2))));

		CHECK_FALSE(q.find_all(task_id(0)).have_successors(q.find_all(task_id(1)) + q.find_all(task_id(2)), true_dep));
		CHECK_FALSE(q.find_all(task_id(0)).have_successors(q.find_all(task_id(1)) + q.find_all(task_id(2)), std::nullopt, collective));
	}

	SECTION("many-to-one relationship") {
		cdags[0]->add_dependency(exe0_2, exe0_0, true_dep, dataflow);
		cdags[0]->add_dependency(exe0_2, exe0_1, true_dep, dataflow);
		CHECK /**/ (q.find_all(node_id(0)).subtract(q.find_all(task_id(2))).have_successors(q.find_all(node_id(0), task_id(2))));
		CHECK_FALSE(q.find_all(task_id(0)).merge(q.find_all(task_id(1))).have_successors(q.find_all(task_id(2))));
		cdags[1]->add_dependency(exe1_2, exe1_0, anti_dep, collective);
		cdags[1]->add_dependency(exe1_2, exe1_1, anti_dep, collective);
		CHECK /**/ (q.find_all(task_id(0)).merge(q.find_all(task_id(1))).have_successors(q.find_all(task_id(2))));

		CHECK_FALSE(q.find_all(task_id(0)).merge(q.find_all(task_id(1))).have_successors(q.find_all(task_id(2)), true_dep));
		CHECK_FALSE(q.find_all(task_id(0)).merge(q.find_all(task_id(1))).have_successors(q.find_all(task_id(2)), std::nullopt, collective));
	}

	SECTION("many-to-many relationship") {
		auto* exe0_3 = cdags[0]->create<execution_command>(task_id(3), subrange<3>{});
		auto* exe1_3 = cdags[1]->create<execution_command>(task_id(3), subrange<3>{});

		cdags[0]->add_dependency(exe0_2, exe0_0, true_dep, dataflow);
		cdags[0]->add_dependency(exe0_3, exe0_0, true_dep, dataflow);
		cdags[0]->add_dependency(exe0_2, exe0_1, true_dep, dataflow);
		cdags[0]->add_dependency(exe0_3, exe0_1, true_dep, dataflow);

		CHECK /**/ (q.find_all(node_id(0))
		                .subtract(q.find_all(task_id(2)).subtract(q.find_all(task_id(3))))
		                .have_successors(q.find_all(node_id(0), task_id(2)) + q.find_all(node_id(0), task_id(3))));
		CHECK_FALSE(q.find_all(task_id(0)).merge(q.find_all(task_id(1))).have_successors(q.find_all(task_id(2)) + q.find_all(task_id(3))));
		cdags[1]->add_dependency(exe1_2, exe1_0, anti_dep, collective);
		cdags[1]->add_dependency(exe1_3, exe1_0, anti_dep, collective);
		cdags[1]->add_dependency(exe1_2, exe1_1, anti_dep, collective);
		cdags[1]->add_dependency(exe1_3, exe1_1, anti_dep, collective);
		CHECK /**/ (q.find_all(task_id(0)).merge(q.find_all(task_id(1))).have_successors(q.find_all(task_id(2)) + q.find_all(task_id(3))));

		CHECK_FALSE(q.find_all(task_id(0)).merge(q.find_all(task_id(1))).have_successors(q.find_all(task_id(2)) + q.find_all(task_id(3)), true_dep));
		CHECK_FALSE(
		    q.find_all(task_id(0)).merge(q.find_all(task_id(1))).have_successors(q.find_all(task_id(2)) + q.find_all(task_id(3)), std::nullopt, collective));
	}

	SECTION("it throws if successor set is empty") {
		CHECK_THROWS_WITH(q.find_all(task_id(0)).have_successors(q.find_all(node_id(9999))), "Successor set is empty");
	}

	SECTION("it throws if successor set contains commands for nodes that have no commands in the query") {
		cdags[0]->add_dependency(exe0_1, exe0_0, true_dep, dataflow);
		cdags[1]->add_dependency(exe1_1, exe1_0, true_dep, dataflow);
		CHECK_THROWS_WITH(q.find_all(task_id(0), node_id(0)).have_successors(q.find_all(task_id(1))),
		    "A.have_successors(B): B contains commands for node 1, whereas A does not");
	}
}

TEST_CASE("command_query::count_per_node throws if nodes have different number of commands", "[command_query]") {
	std::vector<std::unique_ptr<command_graph>> cdags;
	cdags.emplace_back(std::make_unique<command_graph>());
	cdags.emplace_back(std::make_unique<command_graph>());

	cdags[0]->create<execution_command>(task_id(0), subrange<3>{});
	cdags[0]->create<execution_command>(task_id(1), subrange<3>{});
	cdags[1]->create<execution_command>(task_id(0), subrange<3>{});

	const auto q = command_query_testspy::create_for(cdags);
	CHECK(q.count() == 3);
	CHECK_THROWS_WITH(q.count_per_node(), "Different number of commands across nodes (node 0: 2, node 1: 1)");

	cdags[1]->create<execution_command>(task_id(1), subrange<3>{});
	const auto q2 = command_query_testspy::create_for(cdags);
	CHECK_NOTHROW(q2.count_per_node() == 2);
}

TEST_CASE("command_query::assert_count[_per_node] can be used to assert command counts while building larger expressions", "[command_query]") {
	std::vector<std::unique_ptr<command_graph>> cdags;
	cdags.emplace_back(std::make_unique<command_graph>());
	cdags.emplace_back(std::make_unique<command_graph>());

	cdags[0]->create<execution_command>(task_id(0), subrange<3>{});
	cdags[0]->create<execution_command>(task_id(1), subrange<3>{});
	cdags[1]->create<execution_command>(task_id(0), subrange<3>{});

	const auto q = command_query_testspy::create_for(cdags);
	CHECK(q.find_all(/*      */).assert_count(3).have_type(command_type::execution));
	CHECK(q.find_all(node_id(0)).assert_count(2).have_type(command_type::execution));
	CHECK_THROWS_WITH(q.assert_count(5).have_type(command_type::execution), "Expected 5 total command(s), found 3");

	CHECK(q.find_all(task_id(0)).assert_count_per_node(1).have_type(command_type::execution));
	CHECK_THROWS_WITH(q.find_all(command_type::await_push).assert_count_per_node(1), "Expected 1 command(s) per node, found 0");
}

TEST_CASE("most operations fail when called on an empty query", "[command_query]") {
	auto query = command_query_testspy::create_for({});

	CHECK_THROWS_WITH(query.find_all(), "Operation 'find_all' not allowed on empty query");
	CHECK_THROWS_WITH(query.find_predecessors(), "Operation 'find_predecessors' not allowed on empty query");
	CHECK_THROWS_WITH(query.find_successors(), "Operation 'find_successors' not allowed on empty query");
	CHECK_THROWS_WITH(query.for_each_node([](const auto&) {}), "Operation 'for_each_node' not allowed on empty query");
	CHECK_THROWS_WITH(query.have_successors(command_query_testspy::create_for({})), "Operation 'have_successors' not allowed on empty query");
	CHECK_THROWS_WITH(query.have_type(command_type::epoch), "Operation 'have_type' not allowed on empty query");
}

TEST_CASE("tests that log any message in excess of level::info fail by default", "[test_utils][log][!shouldfail]") { CELERITY_WARN("spooky message!"); }

// This is a non-default (i.e. manual) test, because it aborts when passing
TEST_CASE("tests that log messages in excess of level::info from a secondary thread abort", "[test_utils][log][!shouldfail][.]") {
	std::thread([] { CELERITY_WARN("abort() in 3... 2... 1..."); }).join();
}

TEST_CASE("test_utils::set_max_expected_log_level() allows tests with warning / error messages to pass", "[test_utils][log]") {
	test_utils::allow_max_log_level(spdlog::level::err);
	CELERITY_WARN("spooky message!");
	CELERITY_ERROR("spooky message!");
}
