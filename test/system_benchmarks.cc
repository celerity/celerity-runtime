#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <celerity.h>

#include "test_utils.h"

using namespace celerity;
using fixture = test_utils::runtime_fixture;

// This benchmark represents a set of parallel tasks working independently on a shared buffer

void run_indep_task_benchmark(size_t num_tasks) {
#ifndef NDEBUG
	if(num_tasks > 100) { SKIP("Skipping larger-scale benchmark in debug build to save CI time"); }
#endif

	constexpr size_t num_repeats = 2;
	constexpr size_t items_per_task = 256;

	queue queue;

	const auto size = range<2>(items_per_task, num_tasks);
	buffer<size_t, 2> buff_a(size);

	// initialize buffer
	queue.submit([&](handler& cgh) {
		accessor w{buff_a, cgh, access::one_to_one{}, write_only, no_init};
		cgh.parallel_for(size, [=](item<2> item) { w[item] = item.get_linear_id(); });
	});
	queue.wait();

	size_t bench_repeats = 0;
	BENCHMARK("task generation") {
		for(size_t r = 0; r < num_repeats; ++r) {
			for(size_t i = 0; i < num_tasks; ++i) {
				queue.submit([&](handler& cgh) {
					accessor acc{buff_a, cgh, [=](chunk<1> c) { return subrange<2>(id<2>(c.offset.get(0), i), range<2>(c.range.get(0), 1)); }, read_write};
					cgh.parallel_for(range<1>(items_per_task), [=](item<1> item) { //
						acc[item[0]][i] += 1;
					});
				});
			}
		}
		queue.wait();
		bench_repeats++;
	};

	// check result
	buffer<bool, 0> success_buffer = true;
	queue.submit([&](handler& cgh) {
		accessor r{buff_a, cgh, access::all{}, read_only_host_task};
		accessor succ{success_buffer, cgh, access::all{}, write_only_host_task};
		cgh.host_task(on_master_node, [=] {
			experimental::for_each_item(size, [=](item<2> item) {
				size_t expected = item.get_linear_id() + (num_repeats * bench_repeats);
				if(r[item] != expected) {
					fmt::print("Mismatch at {}: {} != {}\n", item.get_linear_id(), r[item], expected);
					succ = false;
				}
			});
		});
	});
	CHECK(*queue.fence(success_buffer).get() == true);
};

const auto indep_task_tags = "[benchmark][group:system][indep-tasks]";
TEST_CASE_METHOD(fixture, "benchmark independent task pattern with  100 tasks", indep_task_tags) { run_indep_task_benchmark(100); }
TEST_CASE_METHOD(fixture, "benchmark independent task pattern with  500 tasks", indep_task_tags) { run_indep_task_benchmark(500); }
TEST_CASE_METHOD(fixture, "benchmark independent task pattern with 2500 tasks", indep_task_tags) { run_indep_task_benchmark(2500); }


// This benchmark represents a basic 2D stencil, executed with 1D and 2D splits and varying levels of oversubscription

void run_stencil_benchmark(size_t num_iter, bool split2d, size_t oversub) {
	constexpr int side_length = 128; // sufficiently small to notice large-scale changes in runtime overhead
#ifndef NDEBUG
	if(num_iter > 50) { SKIP("Skipping larger-scale benchmark in debug build to save CI time"); }
#endif

	queue queue;

	const auto size = range<2>(side_length, side_length);
	buffer<float, 2> buffer_a(size);
	buffer<float, 2> buffer_b(size);

	// initialize buffer_a
	queue.submit([&](handler& cgh) {
		accessor w{buffer_a, cgh, access::one_to_one{}, write_only, no_init};
		cgh.parallel_for(size, [=](item<2> item) {
			// checkerboard
			w[item] = ((item.get_id(0) % 2) ^ (item.get_id(1) % 2)) == 0 ? 1.f : 0.f;
		});
	});
	queue.wait();

	BENCHMARK("iterations") {
		for(size_t r = 0; r < num_iter; ++r) {
			queue.submit([&](handler& cgh) {
				accessor read{buffer_a, cgh, access::neighborhood({1, 1}), read_only};
				accessor write{buffer_b, cgh, access::one_to_one(), write_only, no_init};
				if(split2d) { experimental::hint(cgh, experimental::hints::split_2d{}); }
				if(oversub != 1) { experimental::hint(cgh, experimental::hints::oversubscribe{oversub}); }
				cgh.parallel_for(size, [=](item<2> item) {
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
		queue.wait();
	};

	// check result
	buffer<bool, 0> success_buffer = true;
	queue.submit([&](handler& cgh) {
		accessor r{buffer_a, cgh, access::all{}, read_only_host_task};
		accessor succ{success_buffer, cgh, access::all{}, write_only_host_task};
		cgh.host_task(on_master_node, [=] {
			experimental::for_each_item(size, [=](item<2> item) {
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

constexpr auto stencil_tags = "[benchmark][group:system][stencil]";
TEST_CASE_METHOD(test_utils::runtime_fixture, "benchmark stencil: 1D  50 iters oversub 1", stencil_tags) { run_stencil_benchmark(50, false, 1); }
TEST_CASE_METHOD(test_utils::runtime_fixture, "benchmark stencil: 1D 500 iters oversub 1", stencil_tags) { run_stencil_benchmark(500, false, 1); }
TEST_CASE_METHOD(test_utils::runtime_fixture, "benchmark stencil: 1D  50 iters oversub 3", stencil_tags) { run_stencil_benchmark(50, false, 2); }
TEST_CASE_METHOD(test_utils::runtime_fixture, "benchmark stencil: 1D 500 iters oversub 3", stencil_tags) { run_stencil_benchmark(500, false, 2); }
TEST_CASE_METHOD(test_utils::runtime_fixture, "benchmark stencil: 2D  30 iters oversub 1", stencil_tags) { run_stencil_benchmark(30, true, 1); }
TEST_CASE_METHOD(test_utils::runtime_fixture, "benchmark stencil: 2D 300 iters oversub 1", stencil_tags) { run_stencil_benchmark(300, true, 1); }
TEST_CASE_METHOD(test_utils::runtime_fixture, "benchmark stencil: 2D  30 iters oversub 3", stencil_tags) { run_stencil_benchmark(30, true, 2); }
TEST_CASE_METHOD(test_utils::runtime_fixture, "benchmark stencil: 2D 300 iters oversub 3", stencil_tags) { run_stencil_benchmark(300, true, 2); }


// This benchmark represents the core "RSIM" compute step, notable for its growing buffer access pattern

void run_rsim_benchmark(size_t n_tris, size_t num_iter) {
#ifndef NDEBUG
	if(n_tris > 64 || num_iter > 50) { SKIP("Skipping larger-scale benchmark in debug build to save CI time"); }
#endif

	queue queue;

	// we simply set kij to all 1s
	const auto kij_size = range<2>(n_tris, n_tris);
	std::vector<float> kij_data(n_tris * n_tris, 1.f);
	buffer<float, 2> kij(kij_data.data(), kij_size);

	// rad starts as all 0s, but we need a new buffer every time the benchmark starts
	// otherwise we are not actually measuring the growing buffer access pattern
	const auto rad_size = range<2>(num_iter, n_tris);
	std::vector<float> rad_data(n_tris * num_iter, 0.f);

	// this is a separate result buffer that survives the benchmark section to allow checking the correctness
	buffer<float, 2> rad_result(rad_size);

	BENCHMARK("iterations") {
		buffer<float, 2> rad(rad_data.data(), rad_size);
		// set the first line of rad to 1
		queue.submit([&](handler& cgh) {
			auto write_rad_mapper = [](chunk<1> chnk) -> subrange<2> {
				return {
				    id(0, chnk.offset.get(0)),  // offset
				    range(1, chnk.range.get(0)) // range
				};
			};
			accessor write_rad{rad, cgh, write_rad_mapper, write_only, no_init};
			cgh.parallel_for(n_tris, [=](item<1> item) { write_rad[{0, item.get_id(0)}] = 1.f; });
		});

		for(size_t t = 1; t < num_iter; ++t) {
			queue.submit([&](handler& cgh) {
				// read everything written before the current timestep
				auto read_rad_mapper = [t](chunk<2> chnk) -> subrange<2> {
					return {
					    id(0, 0),                         // offset
					    range(t, chnk.global_size.get(0)) // range
					};
				};
				// only need to write to radiosities of own triangles in current timestep
				auto write_rad_mapper = [t](chunk<2> chnk) -> subrange<2> {
					return {
					    id(t, chnk.offset.get(0)),  // offset
					    range(1, chnk.range.get(0)) // range
					};
				};

				accessor read_kij{kij, cgh, access::one_to_one(), read_only};
				accessor write_rad{rad, cgh, write_rad_mapper, write_only, no_init};
				accessor read_rad{rad, cgh, read_rad_mapper, read_only};

				cgh.parallel_for(kij_size, [=](item<2> item) {
					float val = 0.f;
					float included_items = 0.f;
					for(size_t i = 0; i < t; ++i) {
						// printf("i: %lu, t: %lu, id: %lu/%lu, read_rad: %f, read_kij: %f\n", i, t, item.get_id(0), item.get_id(1), read_rad[{i,
						// item.get_id(0)}], read_kij[{item.get_id(0), item.get_id(1)}]);
						val += read_rad[{i, item.get_id(0)}] * read_kij[{item.get_id(0), item.get_id(1)}];
					}
					val /= (float)t;
					write_rad[{t, item.get_id(0)}] = val;
				});
			});
		}
		queue.submit([&](handler& cgh) {
			accessor read_rad{rad, cgh, access::one_to_one(), read_only};
			accessor write_rad_result{rad_result, cgh, access::one_to_one(), write_only};
			cgh.parallel_for(rad_size, [=](item<2> item) { write_rad_result[item] = read_rad[item]; });
		});
		queue.wait();
	};

	// check result
	buffer<bool, 0> success_buffer = true;
	queue.submit([&](handler& cgh) {
		accessor r{rad_result, cgh, access::all{}, read_only_host_task};
		accessor succ{success_buffer, cgh, access::all{}, write_only_host_task};
		cgh.host_task(on_master_node, [=] {
			experimental::for_each_item(rad_size, [=](item<2> item) {
				const float expected = 1.f;
				constexpr float epsilon = 0.01f;
				if(std::fabs(r[item] - expected) > epsilon) {
					fmt::print("Mismatch at {}/{}: {} !~= {} +/- {}\n", item.get_id(0), item.get_id(1), r[item], expected, epsilon);
					succ = false;
				}
			});
		});
	});
	CHECK(*queue.fence(success_buffer).get() == true);
};

constexpr auto rsim_tags = "[benchmark][group:system][rsim]";
TEST_CASE_METHOD(test_utils::runtime_fixture, "benchmark rsim:   64 tris  50 iters", rsim_tags) { run_rsim_benchmark(64, 50); }
TEST_CASE_METHOD(test_utils::runtime_fixture, "benchmark rsim: 1024 tris  50 iters", rsim_tags) { run_rsim_benchmark(1024, 50); }
TEST_CASE_METHOD(test_utils::runtime_fixture, "benchmark rsim:   64 tris 500 iters", rsim_tags) { run_rsim_benchmark(64, 500); }
TEST_CASE_METHOD(test_utils::runtime_fixture, "benchmark rsim: 1024 tris 500 iters", rsim_tags) { run_rsim_benchmark(1024, 500); }
