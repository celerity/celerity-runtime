#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>

#include <celerity.h>

#include "test_utils.h"

using namespace celerity;
using namespace celerity::detail;

class my_hint : public detail::hint_base {
  public:
	my_hint(int value) : m_value(value) {}
	int get_value() const { return m_value; }

  private:
	int m_value;
};

class my_other_hint : public detail::hint_base {
  private:
	void validate(const hint_base& other) const override {
		if(auto ptr = dynamic_cast<const my_hint*>(&other); ptr != nullptr) {
			if(ptr->get_value() != 1337) throw std::runtime_error("not leet enough");
		}
	}
};

TEST_CASE_METHOD(test_utils::runtime_fixture, "hints can be attached to and retrieved from tasks", "[task-hints]") {
	celerity::runtime::init(nullptr, nullptr);
	auto& tm = detail::runtime::get_instance().get_task_manager();
	const auto tid = test_utils::add_compute_task<class UKN(hint_task)>(tm, [&](handler& cgh) { experimental::hint(cgh, my_hint{1337}); });
	const auto tsk = tm.get_task(tid);
	const auto hint = tsk->get_hint<my_hint>();
	REQUIRE(hint != nullptr);
	CHECK(hint->get_value() == 1337);
}

TEST_CASE_METHOD(test_utils::runtime_fixture, "providing a hint of a particular type more than once throws", "[task-hints]") {
	celerity::runtime::init(nullptr, nullptr);
	auto& tm = detail::runtime::get_instance().get_task_manager();
	test_utils::add_compute_task<class UKN(hint_task)>(tm, [&](handler& cgh) {
		CHECK_NOTHROW(experimental::hint(cgh, my_hint{1337}));
		CHECK_NOTHROW(experimental::hint(cgh, my_other_hint{}));
		CHECK_THROWS_WITH(experimental::hint(cgh, my_hint{1337}), "Providing more than one hint of the same type is not allowed");
	});
}

TEST_CASE_METHOD(test_utils::runtime_fixture, "hints can ensure combinations with other hints are valid", "[task-hints]") {
	celerity::runtime::init(nullptr, nullptr);
	auto& tm = detail::runtime::get_instance().get_task_manager();
	test_utils::add_compute_task<class UKN(hint_task)>(tm, [&](handler& cgh) {
		CHECK_NOTHROW(experimental::hint(cgh, my_other_hint{}));
		CHECK_THROWS_WITH(experimental::hint(cgh, my_hint{1336}), "not leet enough");
	});
}

TEST_CASE_METHOD(test_utils::runtime_fixture, "split_1d and split_2d hints cannot be combined", "[task-hints]") {
	celerity::runtime::init(nullptr, nullptr);
	auto& tm = detail::runtime::get_instance().get_task_manager();
	SECTION("1d then 2d") {
		test_utils::add_compute_task<class UKN(hint_task)>(tm, [&](handler& cgh) {
			CHECK_NOTHROW(experimental::hint(cgh, experimental::hints::split_1d{}));
			CHECK_THROWS_WITH(experimental::hint(cgh, experimental::hints::split_2d{}), "Cannot combine split_1d and split_2d hints");
		});
	}
	SECTION("2d then 1d") {
		test_utils::add_compute_task<class UKN(hint_task)>(tm, [&](handler& cgh) {
			CHECK_NOTHROW(experimental::hint(cgh, experimental::hints::split_2d{}));
			CHECK_THROWS_WITH(experimental::hint(cgh, experimental::hints::split_1d{}), "Cannot combine split_1d and split_2d hints");
		});
	}
}
