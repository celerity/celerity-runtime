#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <celerity.h>

#include "divergence_check_test_utils.h"
#include "log_test_utils.h"
#include "test_utils.h"


using namespace celerity;
using namespace celerity::detail;
using namespace celerity::test_utils;
using celerity::access::fixed;

TEST_CASE("test diverged task execution on device tasks", "[divergence]") {
	using namespace cl::sycl::access;

	test_utils::task_test_context tt = test_utils::task_test_context{};
	test_utils::task_test_context tt_two = test_utils::task_test_context{};

	divergence_test_communicator_provider provider{2};
	std::vector<std::unique_ptr<divergence_block_chain>> div_tests;
	div_tests.emplace_back(std::make_unique<divergence_block_chain>(tt.trec, provider.create(0)));
	div_tests.emplace_back(std::make_unique<divergence_block_chain>(tt_two.trec, provider.create(1)));

	auto buf = tt.mbf.create_buffer(range<1>(128));
	auto buf_two = tt_two.mbf.create_buffer(range<1>(128));

	test_utils::add_compute_task<class UKN(task_a)>(tt.tm, [&](handler& cgh) { buf.get_access<mode::discard_write>(cgh, fixed<1>{{0, 64}}); });
	test_utils::add_compute_task<class UKN(task_b)>(tt.tm, [&](handler& cgh) { buf.get_access<mode::discard_write>(cgh, fixed<1>{{0, 128}}); });
	test_utils::add_compute_task<class UKN(task_b)>(tt_two.tm, [&](handler& cgh) { buf_two.get_access<mode::discard_write>(cgh, fixed<1>{{64, 128}}); });

	test_utils::log_capture log_capture;

	CHECK_THROWS(divergence_block_chain_testspy::call_check_for_divergence_with_pre_post(div_tests));

	CHECK_THAT(log_capture.get_log(), Catch::Matchers::ContainsSubstring("Divergence detected"));
}

TEST_CASE("test divergence free task execution on device", "[divergence]") {
	using namespace cl::sycl::access;

	auto tt = test_utils::task_test_context{};
	auto tt_two = test_utils::task_test_context{};

	{
		divergence_test_communicator_provider provider{2};
		std::vector<std::unique_ptr<divergence_block_chain>> div_tests;
		div_tests.emplace_back(std::make_unique<divergence_block_chain>(tt.trec, provider.create(0)));
		div_tests.emplace_back(std::make_unique<divergence_block_chain>(tt_two.trec, provider.create(1)));

		auto buf = tt.mbf.create_buffer(range<1>(128));
		auto buf_two = tt_two.mbf.create_buffer(range<1>(128));

		test_utils::add_compute_task<class UKN(task_a)>(tt.tm, [&](handler& cgh) {
			// manually set the name because SYCL needs the class tag to be unique making the default name different.
			celerity::debug::set_task_name(cgh, "task_a");
			buf.get_access<mode::discard_write>(cgh, fixed<1>{{0, 64}});
		});

		test_utils::add_compute_task<class UKN(task_a)>(tt_two.tm, [&](handler& cgh) {
			// manually set the name because SYCL needs the class tag to be unique making the default name different.
			celerity::debug::set_task_name(cgh, "task_a");
			buf_two.get_access<mode::discard_write>(cgh, fixed<1>{{0, 64}});
		});

		test_utils::log_capture log_capture;

		divergence_block_chain_testspy::call_check_for_divergence_with_pre_post(div_tests);

		CHECK_THAT(log_capture.get_log(), !Catch::Matchers::ContainsSubstring("Divergence detected"));
	}
}

TEST_CASE("test diverged task execution on host task", "[divergence]") {
	using namespace cl::sycl::access;

	auto tt = test_utils::task_test_context{};
	auto tt_two = test_utils::task_test_context{};

	divergence_test_communicator_provider provider{2};
	std::vector<std::unique_ptr<divergence_block_chain>> div_tests;
	div_tests.emplace_back(std::make_unique<divergence_block_chain>(tt.trec, provider.create(0)));
	div_tests.emplace_back(std::make_unique<divergence_block_chain>(tt_two.trec, provider.create(1)));

	auto buf = tt.mbf.create_buffer(range<1>(128));
	auto buf_two = tt_two.mbf.create_buffer(range<1>(128));

	test_utils::add_host_task(tt.tm, on_master_node, [&](handler& cgh) { buf.get_access<mode::discard_write>(cgh, fixed<1>({0, 128})); });
	test_utils::add_host_task(tt.tm, on_master_node, [&](handler& cgh) { buf.get_access<mode::discard_write>(cgh, fixed<1>({64, 128})); });
	test_utils::add_host_task(tt_two.tm, on_master_node, [&](handler& cgh) { buf_two.get_access<mode::discard_write>(cgh, fixed<1>({64, 128})); });

	test_utils::log_capture log_capture;

	CHECK_THROWS(divergence_block_chain_testspy::call_check_for_divergence_with_pre_post(div_tests));

	CHECK_THAT(log_capture.get_log(), Catch::Matchers::ContainsSubstring("Divergence detected"));
}

TEST_CASE("test divergence free task execution on host task", "[divergence]") {
	using namespace cl::sycl::access;

	auto tt = test_utils::task_test_context{};
	auto tt_two = test_utils::task_test_context{};

	{
		divergence_test_communicator_provider provider{2};
		std::vector<std::unique_ptr<divergence_block_chain>> div_tests;
		div_tests.emplace_back(std::make_unique<divergence_block_chain>(tt.trec, provider.create(0)));
		div_tests.emplace_back(std::make_unique<divergence_block_chain>(tt_two.trec, provider.create(1)));

		auto buf = tt.mbf.create_buffer(range<1>(128));
		auto buf_two = tt_two.mbf.create_buffer(range<1>(128));

		test_utils::add_host_task(tt.tm, on_master_node, [&](handler& cgh) { buf.get_access<mode::discard_write>(cgh, fixed<1>({0, 128})); });
		test_utils::add_host_task(tt.tm, on_master_node, [&](handler& cgh) { buf.get_access<mode::discard_write>(cgh, fixed<1>({64, 128})); });

		test_utils::add_host_task(tt_two.tm, on_master_node, [&](handler& cgh) { buf_two.get_access<mode::discard_write>(cgh, fixed<1>({0, 128})); });
		test_utils::add_host_task(tt_two.tm, on_master_node, [&](handler& cgh) { buf_two.get_access<mode::discard_write>(cgh, fixed<1>({64, 128})); });

		test_utils::log_capture log_capture;

		divergence_block_chain_testspy::call_check_for_divergence_with_pre_post(div_tests);

		CHECK_THAT(log_capture.get_log(), !Catch::Matchers::ContainsSubstring("Divergence detected"));
	}
}

TEST_CASE("test divergence warning for tasks that are stale longer than 10 seconds", "[divergence]") {
	using namespace cl::sycl::access;

	auto tt = test_utils::task_test_context{};
	auto tt_two = test_utils::task_test_context{};

	divergence_test_communicator_provider provider{2};
	std::vector<std::unique_ptr<divergence_block_chain>> div_tests;
	div_tests.emplace_back(std::make_unique<divergence_block_chain>(tt.trec, provider.create(0)));
	div_tests.emplace_back(std::make_unique<divergence_block_chain>(tt_two.trec, provider.create(1)));

	auto buf = tt.mbf.create_buffer(range<1>(128));

	test_utils::add_host_task(tt.tm, on_master_node, [&](handler& cgh) { buf.get_access<mode::discard_write>(cgh, fixed<1>({0, 128})); });

	test_utils::log_capture log_capture;

	// call two times because first time the start task has to be cleared
	divergence_block_chain_testspy::call_check_for_divergence_with_pre_post(div_tests);
	divergence_block_chain_testspy::set_last_cleared(*div_tests[0], (std::chrono::steady_clock::now() - std::chrono::seconds(10)));
	divergence_block_chain_testspy::call_check_for_divergence_with_pre_post(div_tests);

	CHECK_THAT(log_capture.get_log(),
	    Catch::Matchers::ContainsSubstring("After 10 seconds of waiting nodes 1, did not move to the next task. The runtime might be stuck."));
}

size_t get_hash(const std::vector<task_record>& tasks, size_t start, size_t end) {
	size_t seed = 0;
	for(size_t i = start; i <= end; i++) {
		utils::hash_combine(seed, std::hash<task_record>{}(tasks[i]));
	}
	return seed;
}

TEST_CASE("test correct output of 3 different divergent tasks", "[divergence]") {
	using namespace cl::sycl::access;

	auto tt = test_utils::task_test_context{};
	auto tt_two = test_utils::task_test_context{};
	auto tt_three = test_utils::task_test_context{};

	divergence_test_communicator_provider provider{3};
	std::vector<std::unique_ptr<divergence_block_chain>> div_tests;
	div_tests.emplace_back(std::make_unique<divergence_block_chain>(tt.trec, provider.create(0)));
	div_tests.emplace_back(std::make_unique<divergence_block_chain>(tt_two.trec, provider.create(1)));
	div_tests.emplace_back(std::make_unique<divergence_block_chain>(tt_three.trec, provider.create(2)));

	auto buf = tt.mbf.create_buffer(range<1>(128));
	auto buf_two = tt_two.mbf.create_buffer(range<1>(128));
	auto buf_three = tt_three.mbf.create_buffer(range<1>(128));

	test_utils::add_compute_task<class UKN(task_a)>(tt.tm, [&](handler& cgh) {
		celerity::debug::set_task_name(cgh, "task_a");
		buf.get_access<mode::discard_write>(cgh, fixed<1>{{0, 64}});
	});

	test_utils::add_compute_task<class UKN(task_a)>(tt_two.tm, [&](handler& cgh) {
		celerity::debug::set_task_name(cgh, "task_a");
		buf_two.get_access<mode::discard_write>(cgh, fixed<1>{{64, 128}});
	});

	test_utils::add_compute_task<class UKN(task_a)>(tt_three.tm, [&](handler& cgh) {
		celerity::debug::set_task_name(cgh, "task_a");
		buf_three.get_access<mode::discard_write>(cgh, fixed<1>{{0, 128}});
	});

	test_utils::log_capture log_capture;

	CHECK_THROWS(divergence_block_chain_testspy::call_check_for_divergence_with_pre_post(div_tests));

	CHECK_THAT(log_capture.get_log(), Catch::Matchers::ContainsSubstring("Divergence detected in task graph at index 0:\n\n"));
	CHECK_THAT(log_capture.get_log(), Catch::Matchers::ContainsSubstring("on nodes 2"));
	CHECK_THAT(log_capture.get_log(), Catch::Matchers::ContainsSubstring("on nodes 1"));
	CHECK_THAT(log_capture.get_log(), Catch::Matchers::ContainsSubstring("on nodes 0"));
}
