#include "double_buffered_queue.h"
#include "test_utils.h"

#include <catch2/catch_test_macros.hpp>

using namespace celerity;
using namespace celerity::detail;

TEST_CASE("double_buffered_queue works in a single-thread setup", "[double_buffered_queue]") {
	double_buffered_queue<int> dbq;
	CHECK(dbq.pop_all().empty());
	dbq.push(0);
	CHECK(dbq.pop_all() == std::vector{0});
	dbq.push(1);
	dbq.push(2);
	CHECK(dbq.pop_all() == std::vector{1, 2});
	dbq.push(3);
	dbq.push(4);
	dbq.push(5);
	dbq.wait_while_empty();
	CHECK(dbq.pop_all() == std::vector{3, 4, 5});
	CHECK(dbq.pop_all().empty());
}

TEST_CASE("double_buffered_queue provides ordered communication between threads", "[double_buffered_queue]") {
	double_buffered_queue<int> dbq;

	std::mutex m;
	int state = 0;
	std::condition_variable cv;

	const auto post_state = [&](int new_state) {
		CELERITY_DEBUG("post {}", new_state);
		std::lock_guard lock(m);
		state = new_state;
		cv.notify_all();
	};

	const auto await_state = [&](int target_state) {
		CELERITY_DEBUG("await {}", target_state);
		std::unique_lock lock(m);
		cv.wait(lock, [&] { return state == target_state; });
	};

	std::thread producer([&] {
		dbq.push(1);
		dbq.push(2);
		dbq.push(3);

		await_state(1);
		dbq.push(4);

		await_state(2);
		dbq.push(5);

		await_state(3);
		post_state(4);

		await_state(5);
		dbq.push(6);
		dbq.push(7);
		post_state(6);
	});

	std::thread consumer([&] {
		dbq.wait_while_empty();
		for(;;) {
			const auto& got = dbq.pop_all();
			CHECK(std::is_sorted(got.begin(), got.end()));
			if(got.back() == 3) break;
		}

		post_state(1);
		dbq.wait_while_empty();
		CHECK(dbq.pop_all() == std::vector{4});

		post_state(2);
		for(;;) {
			const auto& got = dbq.pop_all();
			if(!got.empty()) {
				CHECK(got == std::vector{5});
				break;
			}
		}

		post_state(3);
		await_state(4);
		CHECK(dbq.pop_all().empty());

		post_state(5);
		await_state(6);
		CHECK(dbq.pop_all() == std::vector{6, 7});
	});

	producer.join();
	consumer.join();
}
