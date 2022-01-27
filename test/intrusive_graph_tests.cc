#include <catch2/catch_test_macros.hpp>

#include "intrusive_graph.h"

namespace celerity {
namespace detail {

	struct my_graph_node : intrusive_graph_node<my_graph_node> {};

	TEST_CASE("intrusive_graph_node correctly handles adding and removing of", "[intrusive_graph_node]") {
		SECTION("true dependencies") {
			// Adding and removing true dependency
			{
				my_graph_node n0, n1;
				n0.add_dependency({&n1, dependency_kind::TRUE_DEP, dependency_origin::dataflow});
				REQUIRE(n0.has_dependency(&n1, dependency_kind::TRUE_DEP));
				REQUIRE(n1.has_dependent(&n0, dependency_kind::TRUE_DEP));
				n0.remove_dependency(&n1);
				REQUIRE_FALSE(n0.has_dependency(&n1));
				REQUIRE_FALSE(n1.has_dependent(&n0));
			}
			// Pseudo- or anti-dependency is upgraded to true dependency
			{
				my_graph_node n0, n1, n2;
				n0.add_dependency({&n1, dependency_kind::ANTI_DEP, dependency_origin::dataflow});
				n0.add_dependency({&n2, dependency_kind::TRUE_DEP, dependency_origin::dataflow});
				CHECK(n0.has_dependency(&n1, dependency_kind::ANTI_DEP));
				CHECK(n1.has_dependent(&n0, dependency_kind::ANTI_DEP));
				n0.add_dependency({&n1, dependency_kind::TRUE_DEP, dependency_origin::dataflow});
				n0.add_dependency({&n2, dependency_kind::ANTI_DEP, dependency_origin::dataflow});
				REQUIRE_FALSE(n0.has_dependency(&n1, dependency_kind::ANTI_DEP));
				REQUIRE_FALSE(n1.has_dependent(&n0, dependency_kind::ANTI_DEP));
				REQUIRE(n0.has_dependency(&n1, dependency_kind::TRUE_DEP));
				REQUIRE(n1.has_dependent(&n0, dependency_kind::TRUE_DEP));
				REQUIRE_FALSE(n0.has_dependency(&n2, dependency_kind::ANTI_DEP));
				REQUIRE_FALSE(n2.has_dependent(&n0, dependency_kind::ANTI_DEP));
			}
		}

		SECTION("anti-dependencies") {
			// True dependency cannot be downgraded to anti-dependency
			{
				my_graph_node n0, n1;
				n0.add_dependency({&n1, dependency_kind::TRUE_DEP, dependency_origin::dataflow});
				CHECK(n0.has_dependency(&n1, dependency_kind::TRUE_DEP));
				CHECK(n1.has_dependent(&n0, dependency_kind::TRUE_DEP));
				n0.add_dependency({&n1, dependency_kind::ANTI_DEP, dependency_origin::dataflow});
				REQUIRE_FALSE(n0.has_dependency(&n1, dependency_kind::ANTI_DEP));
				REQUIRE_FALSE(n1.has_dependent(&n0, dependency_kind::ANTI_DEP));
				REQUIRE(n0.has_dependency(&n1, dependency_kind::TRUE_DEP));
				REQUIRE(n1.has_dependent(&n0, dependency_kind::TRUE_DEP));
				n0.add_dependency({&n1, dependency_kind::TRUE_DEP, dependency_origin::dataflow});
				REQUIRE_FALSE(n0.has_dependency(&n1, dependency_kind::ANTI_DEP));
				REQUIRE_FALSE(n1.has_dependent(&n0, dependency_kind::ANTI_DEP));
				REQUIRE(n0.has_dependency(&n1, dependency_kind::TRUE_DEP));
				REQUIRE(n1.has_dependent(&n0, dependency_kind::TRUE_DEP));
			}
		}
	}

	TEST_CASE("intrusive_graph_node removes itself from all connected nodes upon destruction", "[intrusive_graph_node]") {
		my_graph_node n0, n2;
		auto n1 = std::make_unique<my_graph_node>();
		n0.add_dependency({n1.get(), dependency_kind::TRUE_DEP, dependency_origin::dataflow});
		n1->add_dependency({&n2, dependency_kind::TRUE_DEP, dependency_origin::dataflow});
		CHECK(n0.has_dependency(n1.get(), dependency_kind::TRUE_DEP));
		CHECK(n2.has_dependent(n1.get(), dependency_kind::TRUE_DEP));
		n1.reset();
		REQUIRE(std::distance(n0.get_dependencies().begin(), n0.get_dependencies().end()) == 0);
		REQUIRE(std::distance(n2.get_dependents().begin(), n2.get_dependents().end()) == 0);
	}

	TEST_CASE("intrusive_graph_node keeps track of the pseudo critical path length", "[intrusive_graph_node]") {
		my_graph_node n0, n1, n2, n3;
		REQUIRE(n3.get_pseudo_critical_path_length() == 0);
		n3.add_dependency({&n2, dependency_kind::TRUE_DEP, dependency_origin::dataflow});
		REQUIRE(n3.get_pseudo_critical_path_length() == 1);
		n3.add_dependency({&n0, dependency_kind::TRUE_DEP, dependency_origin::dataflow});
		REQUIRE(n3.get_pseudo_critical_path_length() == 1);
		n1.add_dependency({&n0, dependency_kind::TRUE_DEP, dependency_origin::dataflow});
		n3.add_dependency({&n1, dependency_kind::TRUE_DEP, dependency_origin::dataflow});
		REQUIRE(n3.get_pseudo_critical_path_length() == 2);
	}

} // namespace detail
} // namespace celerity
