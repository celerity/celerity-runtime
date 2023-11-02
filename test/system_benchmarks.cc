#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <celerity.h>

#include "test_utils.h"

using namespace celerity;

template <int Dims>
class bench_runtime_fixture : public test_utils::runtime_fixture {};

TEMPLATE_TEST_CASE_METHOD_SIG(
    bench_runtime_fixture, "benchmark independent task pattern with N tasks", "[benchmark][group:system][indep-tasks]", ((int N), N), 100, 1000, 5000) {
	constexpr size_t num_tasks = N;
	constexpr size_t num_repeats = 2;
	constexpr size_t items_per_task = 256;

#ifndef NDEBUG
	if(N > 100) { SKIP("Skipping larger-scale benchmark in debug build to save CI time"); }
#endif

	celerity::distr_queue queue;
	celerity::buffer<size_t, 2> buffer(celerity::range<2>(items_per_task, num_tasks));

	// initialize buffer
	queue.submit([&](celerity::handler& cgh) {
		celerity::accessor w{buffer, cgh, celerity::access::one_to_one{}, celerity::write_only, celerity::no_init};
		cgh.parallel_for(buffer.get_range(), [=](celerity::item<2> item) { w[item] = item.get_linear_id(); });
	});
	queue.slow_full_sync();

	size_t bench_repeats = 0;
	BENCHMARK("task generation") {
		for(size_t r = 0; r < num_repeats; ++r) {
			for(size_t i = 0; i < num_tasks; ++i) {
				queue.submit([&](celerity::handler& cgh) {
					celerity::accessor acc{buffer, cgh,
					    [=](celerity::chunk<1> c) { return celerity::subrange<2>(celerity::id<2>(c.offset.get(0), i), celerity::range<2>(c.range.get(0), 1)); },
					    celerity::read_write};
					cgh.parallel_for(celerity::range<1>(items_per_task), [=](celerity::item<1> item) { //
						acc[item[0]][i] += 1;
					});
				});
			}
		}
		queue.slow_full_sync();
		bench_repeats++;
	};

	// check result
	celerity::buffer<bool, 0> success_buffer = true;
	queue.submit([&](celerity::handler& cgh) {
		celerity::accessor r{buffer, cgh, celerity::access::all{}, celerity::read_only_host_task};
		celerity::accessor succ{success_buffer, cgh, celerity::access::all{}, celerity::write_only_host_task};
		cgh.host_task(celerity::on_master_node, [=]() {
			celerity::experimental::for_each_item(buffer.get_range(), [=](celerity::item<2> item) {
				size_t expected = item.get_linear_id() + (num_repeats * bench_repeats);
				if(r[item] != expected) {
					fmt::print("Mismatch at {}: {} != {}\n", item.get_linear_id(), r[item], expected);
					succ = false;
				}
			});
		});
	});
	CHECK(*queue.fence(success_buffer).get() == true);
}

TEMPLATE_TEST_CASE_METHOD_SIG(
    bench_runtime_fixture, "benchmark stencil pattern with N time steps", "[benchmark][group:system][stencil]", ((int N), N), 50, 1000) {
	constexpr size_t num_iterations = N;
	constexpr int side_length = 128; // sufficiently small to notice large-scale changes in runtime overhead

#ifndef NDEBUG
	if(N > 50) { SKIP("Skipping larger-scale benchmark in debug build to save CI time"); }
#endif

	celerity::distr_queue queue;

	const auto size = celerity::range<2>(side_length, side_length);
	celerity::buffer<float, 2> buffer_a(size);
	celerity::buffer<float, 2> buffer_b(size);

	// initialize buffer_a
	queue.submit([&](celerity::handler& cgh) {
		celerity::accessor w{buffer_a, cgh, celerity::access::one_to_one{}, celerity::write_only, celerity::no_init};
		cgh.parallel_for(size, [=](celerity::item<2> item) {
			// checkerboard
			w[item] = ((item.get_id(0) % 2) ^ (item.get_id(1) % 2)) == 0 ? 1.f : 0.f;
		});
	});
	queue.slow_full_sync();

	BENCHMARK("iterations") {
		for(size_t r = 0; r < num_iterations; ++r) {
			queue.submit([&](celerity::handler& cgh) {
				celerity::accessor read{buffer_a, cgh, celerity::access::neighborhood(1, 1), celerity::read_only};
				celerity::accessor write{buffer_b, cgh, celerity::access::one_to_one(), celerity::write_only, celerity::no_init};
				cgh.parallel_for(size, [=](celerity::item<2> item) {
					float sum = 0.f;
					float included_items = 0.f;
					for(int i = -1; i <= 1; ++i) {
						const int x = static_cast<int>(item.get_id(0)) + i;
						if(x < 0 || x >= side_length) continue;
						for(int j = -1; j <= 1; ++j) {
							const int y = static_cast<int>(item.get_id(1)) + j;
							if(y < 0 || y >= side_length) continue;
							sum += read[{static_cast<size_t>(x), static_cast<size_t>(y)}];
							included_items += 1.f;
						}
					}
					write[item] = 0.5f * read[item] + 0.5f * sum / included_items;
				});
			});
			std::swap(buffer_a, buffer_b);
		}
		queue.slow_full_sync();
	};

	// check result
	celerity::buffer<bool, 0> success_buffer = true;
	queue.submit([&](celerity::handler& cgh) {
		celerity::accessor r{buffer_a, cgh, celerity::access::all{}, celerity::read_only_host_task};
		celerity::accessor succ{success_buffer, cgh, celerity::access::all{}, celerity::write_only_host_task};
		cgh.host_task(celerity::on_master_node, [=]() {
			celerity::experimental::for_each_item(buffer_a.get_range(), [=](celerity::item<2> item) {
				constexpr float expected = 0.5f;
				constexpr float epsilon = 0.01f;
				if(std::fabs(r[item] - expected) > epsilon) {
					fmt::print("Mismatch at {}/{}: {} !~= {} +/- {}\n", item.get_id(0), item.get_id(1), r[item], expected, epsilon);
					succ = false;
				}
			});
		});
	});
	CHECK(*queue.fence(success_buffer).get() == true);
}
