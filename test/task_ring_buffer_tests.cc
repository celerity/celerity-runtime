#include "task_ring_buffer.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <celerity.h>

#include "test_utils.h"

namespace celerity::detail {

TEST_CASE_METHOD(test_utils::runtime_fixture, "freeing task ring buffer capacity via horizons continues execution in runtime", "[task_ring_buffer]") {
	using namespace std::chrono_literals;
	celerity::distr_queue q;

	std::atomic<bool> reached_ringbuffer_capacity = false;

	auto& tm = runtime::get_instance().get_task_manager(); // we promise using task_manager in a thread safe manner
	auto observer = std::thread([&] {
		while(tm.get_total_task_count() < task_ringbuffer_size) {}
		reached_ringbuffer_capacity = true;
	});

	int init = 42;
	celerity::buffer<int, 1> dependency{&init, 1};

	for(size_t i = 0; i < task_ringbuffer_size + 10; ++i) {
		q.submit([&](celerity::handler& cgh) {
			celerity::accessor acc{dependency, cgh, celerity::access::all{}, celerity::read_write_host_task};
			cgh.host_task(celerity::on_master_node, [=, &reached_ringbuffer_capacity] {
				(void)acc;
				while(!reached_ringbuffer_capacity.load())
					; // we wait in all tasks so that we can make sure to fill the ring buffer completely
					  // and therefore test that execution re-starts correctly once an epoch is reached
			});
		});
	}

	observer.join();

	q.slow_full_sync(); // `reach_ringbuffer_capacity` must not go out of scope before the host task has finished
}

TEST_CASE_METHOD(test_utils::runtime_fixture, "deadlock in task ring buffer due to slot exhaustion is reported", "[task_ring_buffer]") {
	celerity::distr_queue q;

	// set a high maximum so that we can actually run out of slots
	runtime::get_instance().get_task_manager().set_horizon_max_parallelism(task_ringbuffer_size * 16);

	CHECK_THROWS_WITH(
	    [&] {
		    for(size_t i = 0; i < task_ringbuffer_size + 1; ++i) {
			    q.submit([=](celerity::handler& cgh) { cgh.host_task(celerity::on_master_node, [=] {}); });
		    }
	    }(),
	    Catch::Matchers::ContainsSubstring("Exhausted task slots"));

	// we need to create a slot for the epoch task required for the runtime shutdown to succeed
	task_manager_testspy::create_task_slot(runtime::get_instance().get_task_manager());
}

} // namespace celerity::detail
