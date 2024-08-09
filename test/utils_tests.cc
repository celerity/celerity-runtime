#include <celerity.h>

#include <catch2/catch_test_macros.hpp>

using namespace celerity;
using namespace celerity::detail;


// dummy definitions for get_simplified_type_name tests

namespace ns::another {
struct name2;
}

template <typename, typename>
class name3;

template <typename>
struct param {
	template <typename>
	struct nested_param;
};

template <typename, typename>
struct space {
	template <typename>
	struct nested_kernel;
};

template <typename T>
auto set_identity(distr_queue queue, buffer<T, 2> mat, bool reverse) {
	return [](handler& cgh) {
		class set_identity_kernel {};
		return set_identity_kernel{};
	};
}

using set_identity_name = decltype(set_identity(std::declval<distr_queue>(), std::declval<buffer<int, 2>>(), false)(std::declval<handler&>()));

TEST_CASE("name strings are correctly extracted from types", "[utils][get_simplified_type_name]") {
	CHECK(utils::get_simplified_type_name<class name>() == "name");
	CHECK(utils::get_simplified_type_name<ns::another::name2>() == "name2");
	CHECK(utils::get_simplified_type_name<name3<class foo, int>>() == "name3<...>");
	CHECK(utils::get_simplified_type_name<space<int, param<int>>::nested_kernel<param<float>::nested_param<int>>>() == "nested_kernel<...>");
	CHECK(utils::get_simplified_type_name<set_identity_name>() == "set_identity_kernel");
}


TEST_CASE("escaping of invalid characters for dot labels", "[utils][escape_for_dot_label]") {
	CHECK(utils::escape_for_dot_label("hello<bla&>") == "hello&lt;bla&amp;&gt;");
}

TEST_CASE("replace_all replaces all occurrences of a pattern in string", "[utils][replace_all]") {
	CHECK(utils::replace_all("hello world", " ", "_") == "hello_world");
	CHECK(utils::replace_all("hello world", "l", "") == "heo word");
	CHECK(utils::replace_all("hello world", " ", "<-->") == "hello<-->world");
	CHECK(utils::replace_all("hello world", "l", "_") == "he__o wor_d");
	CHECK(utils::replace_all("hello world", "o", "<->") == "hell<-> w<->rld");
	CHECK(utils::replace_all("hello world", "foo", "bar") == "hello world");
	CHECK(utils::replace_all("hello world", "", "bar") == "hello world");
	CHECK(utils::replace_all("", "", "bar") == "");
	CHECK(utils::replace_all("", "", "bar") == "");
}
