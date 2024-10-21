#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers.hpp>

#include "scheduler_test_utils.h"
#include "test_utils.h"


using namespace celerity;
using namespace celerity::detail;
namespace acc = celerity::access;


TEST_CASE("scheduler compiles non-allocating commands immediately", "[scheduler]") {
	test_utils::scheduler_test_context sctx(1 /* num nodes */, 0 /* my nid */, 1 /* num devices */);

	const auto lookahead = GENERATE(values({experimental::lookahead::none, experimental::lookahead::automatic, experimental::lookahead::infinite}));
	// invoke set_lookahead conditionally, because it has a side-effect on scheduler queueing
	if(lookahead != experimental::lookahead::automatic) { sctx.set_lookahead(lookahead); }

	sctx.inspect_instructions([](const test_utils::instruction_query& iq) {
		CHECK(iq.count() == 1);
		CHECK(iq.select_unique<epoch_instruction_record>()->epoch_action == epoch_action::init);
	});

	constexpr static size_t num_kernels_and_objects = 16;
	for(size_t i = 0; i < num_kernels_and_objects; ++i) {
		sctx.create_buffer(range(1));
		sctx.create_host_object();
		sctx.device_compute(range(1)).submit();
	}

	sctx.inspect_instructions([](const test_utils::instruction_query& iq) {
		CHECK(iq.count() == num_kernels_and_objects + 1);
		CHECK(iq.count<device_kernel_instruction_record>() == num_kernels_and_objects);
	});

	sctx.finish();

	sctx.inspect_instructions([](const test_utils::instruction_query& iq) {
		// buffers not accessed and thus never allocated
		CHECK(iq.count() == 2 * num_kernels_and_objects + 2);
		CHECK(iq.count<device_kernel_instruction_record>() == num_kernels_and_objects);
		CHECK(iq.count<destroy_host_object_instruction_record>() == num_kernels_and_objects);
		CHECK(iq.count<epoch_instruction_record>() == 2);
	});
}

TEST_CASE("scheduler(lookahead::automatic) flushes allocating commands after two horizons", "[scheduler][lookahead]") {
	test_utils::scheduler_test_context sctx(1 /* num nodes */, 0 /* my nid */, 1 /* num devices */);

	constexpr static size_t horizon_step = 4;
	sctx.set_horizon_step(horizon_step);

	auto buf = sctx.create_buffer(range(1));
	sctx.device_compute(range(1)).discard_write(buf, acc::all()).submit();

	for(size_t i = 1; i < 2 * horizon_step; ++i) {
		sctx.inspect_commands([i](const test_utils::command_query& cq) { CHECK(cq.count<execution_command_record>() == i); });
		sctx.inspect_instructions([](const test_utils::instruction_query& iq) { CHECK(iq.count<device_kernel_instruction_record>() == 0); });

		// read_write to generate a dependency chain and trigger horizon generation, scheduler knows subsequent accesses will not allocate
		sctx.device_compute(range(1)).read_write(buf, acc::all()).submit();
	}

	sctx.inspect_commands([](const test_utils::command_query& cq) {
		CHECK(cq.count<horizon_command_record>() == 2);
		CHECK(cq.count<execution_command_record>() == 2 * horizon_step);
	});
	sctx.inspect_instructions([](const test_utils::instruction_query& iq) {
		auto alloc = iq.select_unique<alloc_instruction_record>();
		CHECK(alloc->buffer_allocation->box == box<3>(zeros, ones));
		CHECK(iq.count<device_kernel_instruction_record>() == 2 * horizon_step);
	});
}

TEST_CASE("scheduler(lookahead::none) does not attempt to elide allocations", "[scheduler][lookahead]") {
	test_utils::allow_max_log_level(log_level::warn);

	test_utils::scheduler_test_context sctx(1 /* num nodes */, 0 /* my nid */, 1 /* num devices */);
	sctx.set_lookahead(experimental::lookahead::none);

	constexpr size_t num_timesteps = 20;

	auto buf = sctx.create_buffer(range(num_timesteps));
	for(size_t i = 0; i < num_timesteps; ++i) {
		sctx.inspect_instructions([=](const test_utils::instruction_query& iq) { CHECK(iq.count<device_kernel_instruction_record>() == i); });
		sctx.device_compute(range(1)).discard_write(buf, acc::fixed<1>({i, 1})).submit();
	}

	sctx.inspect_instructions([](const test_utils::instruction_query& iq) {
		CHECK(iq.count<alloc_instruction_record>() == num_timesteps);
		CHECK(iq.count<device_kernel_instruction_record>() == num_timesteps);
	});

	CHECK(test_utils::log_contains_exact(log_level::warn,
	    "Your program triggers frequent allocations or resizes for buffer B0, which may degrade performance. If possible, avoid "
	    "celerity::queue::fence(), celerity::queue::wait() and celerity::experimental::flush() between command groups of growing access "
	    "patterns, or try increasing scheduler lookahead via celerity::experimental::set_lookahead()."));
}

TEST_CASE("scheduler(lookahead::infinite) does not flush commands unless forced to", "[scheduler][lookahead]") {
	test_utils::scheduler_test_context sctx(1 /* num nodes */, 0 /* my nid */, 1 /* num devices */);
	sctx.set_lookahead(experimental::lookahead::infinite);

	constexpr static size_t horizon_step = 2;
	sctx.set_horizon_step(horizon_step);

	auto buf = sctx.create_buffer(range(1));
	sctx.device_compute(range(1)).discard_write(buf, acc::all()).submit();

	for(size_t i = 1; i < 4 * horizon_step; ++i) {
		sctx.inspect_commands([i](const test_utils::command_query& cq) { CHECK(cq.count<execution_command_record>() == i); });
		sctx.inspect_instructions([](const test_utils::instruction_query& iq) {
			CHECK(iq.count<alloc_instruction_record>() == 0);
			CHECK(iq.count<device_kernel_instruction_record>() == 0);
		});

		// read_write to generate a dependency chain and trigger horizon generation, scheduler knows subsequent accesses will not allocate
		sctx.device_compute(range(1)).read_write(buf, acc::all()).submit();
	}

	sctx.flush();

	sctx.inspect_instructions([](const test_utils::instruction_query& iq) {
		auto alloc = iq.select_unique<alloc_instruction_record>();
		CHECK(alloc->buffer_allocation->box == box<3>(zeros, ones));
		CHECK(iq.count<device_kernel_instruction_record>() == 4 * horizon_step);
	});
}

TEST_CASE("scheduler(lookahead::automatic) delays flushing while reallocations are pending", "[scheduler][lookahead]") {
	test_utils::scheduler_test_context sctx(1 /* num nodes */, 0 /* my nid */, 1 /* num devices */);

	sctx.set_horizon_step(2);
	constexpr size_t num_timesteps = 10;

	auto buf_a = sctx.create_buffer(range(num_timesteps));
	auto buf_b = sctx.create_buffer(range(num_timesteps));
	sctx.device_compute(range(1)).discard_write(buf_a, acc::one_to_one()).submit();

	for(size_t i = 1; i < num_timesteps; ++i) {
		sctx.device_compute(range(1)).read(buf_a, acc::fixed<1>({0, i})).discard_write(buf_b, acc::fixed<1>({0, i + 1})).submit();
		std::swap(buf_a, buf_b);
	}

	sctx.inspect_commands([](const test_utils::command_query& cq) {
		CHECK(cq.count<horizon_command_record>() > 2);
		CHECK(cq.count<execution_command_record>() == num_timesteps);
	});
	sctx.inspect_instructions([](const test_utils::instruction_query& iq) {
		CHECK(iq.count<horizon_command_record>() == 0);
		CHECK(iq.count<device_kernel_instruction_record>() == 0);
	});

	sctx.flush();

	sctx.inspect_instructions([&](const test_utils::instruction_query& iq) {
		auto all_allocs = iq.select_all<alloc_instruction_record>();
		CHECK(all_allocs.count() == 2);

		auto alloc_a = all_allocs.select_unique([&](const alloc_instruction_record& ainstr) { return ainstr.buffer_allocation->buffer_id == buf_a.get_id(); });
		CHECK(alloc_a->buffer_allocation->box == box_cast<3>(box<1>(0, num_timesteps)));
		auto alloc_b = all_allocs.select_unique([&](const alloc_instruction_record& ainstr) { return ainstr.buffer_allocation->buffer_id == buf_b.get_id(); });
		CHECK(alloc_b->buffer_allocation->box == box_cast<3>(box<1>(0, num_timesteps - 1)));

		CHECK(iq.count<free_instruction_record>() == 0); // deallocations happen after finish()
		CHECK(iq.count<device_kernel_instruction_record>() == num_timesteps);
	});

	sctx.finish();

	sctx.inspect_instructions([](const test_utils::instruction_query& iq) {
		CHECK(iq.count<free_instruction_record>() == 2); // buffers are destroyed on scheduler_test_context::finish
	});
}

TEST_CASE("epochs and fences implicitly flush the scheduler queue", "[scheduler][lookahead]") {
	const auto lookahead = GENERATE(values({experimental::lookahead::automatic, experimental::lookahead::infinite}));
	const auto task_type = GENERATE(values({task_type::epoch, task_type::fence}));
	CAPTURE(lookahead, task_type);

	test_utils::scheduler_test_context sctx(1 /* num nodes */, 0 /* my nid */, 1 /* num devices */);
	sctx.set_lookahead(lookahead);

	constexpr size_t num_timesteps = 20;

	auto buf = sctx.create_buffer(range(num_timesteps));
	for(size_t i = 0; i < num_timesteps; ++i) {
		sctx.device_compute(range(1)).discard_write(buf, acc::fixed<1>({i, 1})).submit();
	}

	sctx.inspect_instructions([](const test_utils::instruction_query& iq) {
		CHECK(iq.count() == 1); // init-epoch
	});

	if(task_type == task_type::epoch) {
		sctx.epoch(epoch_action::none);
	} else {
		sctx.fence(buf);
	}

	sctx.inspect_instructions([&](const test_utils::instruction_query& iq) {
		CHECK(iq.count<device_kernel_instruction_record>() == num_timesteps);
		CHECK(iq.count<epoch_instruction_record>() == 1 + (task_type == task_type::epoch));
		CHECK(iq.count<fence_instruction_record>() == (task_type == task_type::fence));
	});
}

TEST_CASE("scheduler(lookahead::automatic) avoids reallocations in the wave_sim pattern", "[scheduler][lookahead]") {
	constexpr size_t num_devices = 4;
	test_utils::scheduler_test_context sctx(1 /* num nodes */, 0 /* my nid */, num_devices);

	const int split_dims = GENERATE(values({1, 2}));
	const size_t oversub_factor = GENERATE(values({1, 4}));
	CAPTURE(split_dims, oversub_factor);

	sctx.set_horizon_step(2);
	constexpr size_t num_timesteps = 10;
	const auto size = range(16384, 16384);

	auto buf_a = sctx.create_buffer(size);
	auto buf_b = sctx.create_buffer(size);
	sctx.device_compute(size)
	    .discard_write(buf_a, acc::one_to_one())
	    .hint_if(split_dims == 2, experimental::hints::split_2d())
	    .hint(experimental::hints::oversubscribe(oversub_factor))
	    .submit();

	for(size_t i = 1; i < num_timesteps; ++i) {
		sctx.device_compute(size)
		    .read(buf_a, acc::neighborhood({1, 1}))
		    .discard_write(buf_b, acc::one_to_one())
		    .hint_if(split_dims == 2, experimental::hints::split_2d())
		    .hint(experimental::hints::oversubscribe(oversub_factor))
		    .submit();
		std::swap(buf_a, buf_b);
	}

	sctx.finish();

	sctx.inspect_instructions([](const test_utils::instruction_query& iq) {
		CHECK(iq.count<alloc_instruction_record>() == 2 * num_devices); // one for each buffer
		CHECK(iq.count<free_instruction_record>() == 2 * num_devices);
	});
}

TEST_CASE("scheduler(lookahead::automatic) avoids reallocations in the RSim pattern", "[scheduler][lookahead]") {
	constexpr size_t num_devices = 4;
	test_utils::scheduler_test_context sctx(1 /* num nodes */, 0 /* my nid */, num_devices);

	sctx.set_horizon_step(2);
	constexpr size_t num_timesteps = 10;
	constexpr size_t num_triangles = 4096;

	auto buf = sctx.create_buffer(range(num_triangles, num_triangles));
	for(size_t i = 0; i < num_timesteps; ++i) {
		const auto read_rm = [i](const chunk<1>& ck) { return subrange<2>({0, ck.offset[0]}, {i, ck.range[0]}); };
		const auto write_rm = [i](const chunk<1>& ck) { return subrange<2>({i, ck.offset[0]}, {1, ck.range[0]}); };
		sctx.device_compute(range(num_triangles)).read(buf, read_rm).discard_write(buf, write_rm).submit();
	}

	sctx.finish();

	sctx.inspect_instructions([](const test_utils::instruction_query& iq) {
		CHECK(iq.count<alloc_instruction_record>() == num_devices);
		CHECK(iq.count<free_instruction_record>() == num_devices);
	});
}
