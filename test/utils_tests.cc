#include "utils.h"

#include <catch2/catch_test_macros.hpp>

using namespace celerity::detail::utils;
using std::string;

TEST_CASE("name strings are correctly extracted from types", "[utils][simplify_task_name]") {
	const string simple = "name";
	CHECK(simplify_task_name(simple) == "name");

	const string namespaced = "ns::another::name2";
	CHECK(simplify_task_name(namespaced) == "name2");

	const string templated = "name3<class, int>";
	CHECK(simplify_task_name(templated) == "name3<...>");

	const string real = "set_identity<float>(celerity::distr_queue, celerity::buffer<float, 2>, "
	                    "bool)::{lambda(celerity::handler&)#1}::operator()(celerity::handler&) const::set_identity_kernel<const celerity::buffer&>";
	CHECK(simplify_task_name(real) == "set_identity_kernel<...>");
}

TEST_CASE("escaping of invalid characters for dot labels", "[utils][escape_for_dot_label]") {
	const string test = "hello<bla&>";
	CHECK(escape_for_dot_label(test) == "hello&lt;bla&amp;&gt;");
}
