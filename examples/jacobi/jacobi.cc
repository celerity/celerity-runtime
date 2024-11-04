#include <celerity.h>
#include <fmt/ranges.h>

#include "../fvm/hash.h"

#include <span>

int main(int argc, char* argv[]) {
	const size_t buffer_size = argc > 1 ? std::atol(argv[1]) : 1024;
	const size_t num_iterations = argc > 2 ? std::atol(argv[2]) : 100;
	const bool use_pyramid = argc > 3 ? std::atoi(argv[3]) : 0;
	const size_t pyramid_height_limit = argc > 4 ? std::atoi(argv[4]) : 0;

	celerity::queue q;
	celerity::buffer<float, 1> buf_a{buffer_size};
	celerity::buffer<float, 1> buf_b{buffer_size};

	q.submit([&](celerity::handler& cgh) {
		celerity::accessor write_a{buf_a, cgh, celerity::access::one_to_one{}, celerity::write_only, celerity::no_init};
		celerity::accessor write_b{buf_b, cgh, celerity::access::one_to_one{}, celerity::write_only, celerity::no_init};
		celerity::debug::set_task_name(cgh, "Initialize");
		cgh.parallel_for(buffer_size, [=](celerity::id<1> idx) {
			write_a[idx] = idx[0];
			write_b[idx] = 0;
		});
	});

	// FIXME: Resize hack - otherwise we need a variable number of warmup iterations (depending on pyramid height, which is annoying for benchmarking)
	q.submit([&](celerity::handler& cgh) {
		celerity::accessor resize_a{buf_a, cgh, celerity::access::all{}, celerity::read_only};
		celerity::accessor resize_b{buf_b, cgh, celerity::access::all{}, celerity::read_only};
		celerity::debug::set_task_name(cgh, "resize hack");
		cgh.parallel_for(buffer_size, [=](celerity::id<1> idx) {
			(void)resize_a;
			(void)resize_b;
		});
	});

	const size_t num_chunks = 4;
	if(buffer_size % num_chunks != 0) throw std::runtime_error(fmt::format("buffer_size must be divisible by number of chunks ({})", num_chunks));
	const size_t pyramid_base = buffer_size / num_chunks;
	const size_t pyramid_height = pyramid_base / 2;
	const bool do_print = pyramid_height < 10;

	std::vector<celerity::subrange<1>> pyramid_chunks;
	pyramid_chunks.push_back(celerity::subrange<1>{1, pyramid_base - 2}); // First chunk starts at level 1 (output parallelism)
	for(size_t i = 1; i < pyramid_height - 1; ++i) {
		const auto prev = pyramid_chunks.back();
		// TODO API: For this we ideally want a "inset" functionality, and for inverse pyramids a "inverse" or something
		pyramid_chunks.push_back(celerity::subrange<1>{prev.offset[0] + 1, prev.range[0] - 2});
	}
	if(use_pyramid && do_print) CELERITY_CRITICAL("Pyramid chunks: {}", fmt::join(pyramid_chunks, ","));

	const size_t warmup_iterations = 10;
	fmt::print("Using {} warmup iterations\n", warmup_iterations);
	std::chrono::steady_clock::time_point start;
	for(size_t i = 0; i < num_iterations; ++i) {
		if(i + 1 == warmup_iterations) {
			q.wait(celerity::experimental::barrier);
			start = std::chrono::steady_clock::now();
		}

		if(use_pyramid) {
			// No height limit
			// const size_t pyramid_idx = i % (pyramid_height - 1);
			// Simple height limit: Still constructs tiny downward pyramids
			// const size_t pyramid_idx = pyramid_height_limit == 0 ? i % (pyramid_height - 1) : std::min(i % (pyramid_height - 1), pyramid_height_limit - 1);
			// Better height limit: Start inside of hierarchy
			const size_t actual_pyramid_height_limit = pyramid_height_limit == 0 ? pyramid_height - 1 : std::min(pyramid_height - 1, pyramid_height_limit);
			const size_t pyramid_start_offset = ([&]() -> size_t {
				if(pyramid_height_limit == 0 || pyramid_height_limit >= pyramid_height - 1) return 0;
				return (pyramid_height - 1) / 2 - pyramid_height_limit / 2;
			})(); // IIFE
			const size_t pyramid_idx = pyramid_start_offset + i % actual_pyramid_height_limit;
			if(do_print) CELERITY_CRITICAL("Iteration {}: Using pyramid level {}", i, pyramid_idx);

			// Upward pyramid
			{
				celerity::custom_task_geometry<1> geo;
				for(size_t j = 0; j < num_chunks; ++j) {
					auto sr = pyramid_chunks[pyramid_idx];
					sr.offset += j * pyramid_base;
					if(do_print) CELERITY_CRITICAL("it {}: Assigning upward chunk {} to did {}", i, sr, j);
					geo.assigned_chunks.push_back(celerity::geometry_chunk{.sr = celerity::detail::subrange_cast<3>(sr), .nid = 0, .did = j});
				}
				q.submit([&](celerity::handler& cgh) {
					celerity::accessor read_a{buf_a, cgh, celerity::access::neighborhood{1}, celerity::read_only};
					celerity::accessor write_b{buf_b, cgh, celerity::access::one_to_one{}, celerity::write_only, celerity::no_init};
					celerity::debug::set_task_name(cgh, "upward step");
					cgh.parallel_for(geo, [=](celerity::id<1> idx) {
						// actually jacobi is not ideal for tracing results
						// write_b[idx] = (read_a[idx[0] - 1] + read_a[idx[0] + 1]) / 2.f;
						write_b[idx] = read_a[idx[0] - 1] + read_a[idx[0] + 1];
					});
				});
			}
			// Downward pyramid
			{
				celerity::custom_task_geometry<1> geo;
				// Assign partial pyramid at beginning
				{
					auto sr = pyramid_chunks[pyramid_chunks.size() - pyramid_idx - 1];
					sr.offset[0] = 0;
					sr.range[0] /= 2;
					if(do_print) CELERITY_CRITICAL("it {}: Assigning partial downward chunk {} to did 0", i, sr);
					geo.assigned_chunks.push_back(celerity::geometry_chunk{.sr = celerity::detail::subrange_cast<3>(sr), .nid = 0, .did = 0});
				}

				// Assign full pyramids
				for(size_t j = 0; j < num_chunks - 1; ++j) {
					const auto upward_sr = pyramid_chunks[pyramid_idx];
					auto sr = pyramid_chunks[pyramid_chunks.size() - pyramid_idx - 1];
					sr.offset = upward_sr.offset + upward_sr.range;
					sr.offset += j * pyramid_base;
					// Shift assignment by 1, device 0 does the half pyramids on either side
					const celerity::detail::device_id did = j + 1;
					if(do_print) CELERITY_CRITICAL("it {}: Assigning downward chunk {} to did {}", i, sr, did);
					geo.assigned_chunks.push_back(celerity::geometry_chunk{.sr = celerity::detail::subrange_cast<3>(sr), .nid = 0, .did = did});
				}

				// Assign partial pyramid at end
				{
					const auto upward_sr = pyramid_chunks[pyramid_idx];
					auto sr = pyramid_chunks[pyramid_chunks.size() - pyramid_idx - 1];
					sr.offset = upward_sr.offset + upward_sr.range;
					sr.offset += (num_chunks - 1) * pyramid_base;
					sr.range[0] /= 2;
					// TODO: Assign this to device 0 or num_chunks - 1? Right now it looks like 0 is worse, even though it would mean more equal load
					// distribution ..?!
					const celerity::detail::device_id did = 0;
					if(do_print) CELERITY_CRITICAL("it {}: Assigning partial downward chunk {} to did {}", i, sr, did);
					geo.assigned_chunks.push_back(celerity::geometry_chunk{.sr = celerity::detail::subrange_cast<3>(sr), .nid = 0, .did = did});
				}

				// std::swap(buf_a, buf_b);
				q.submit([&](celerity::handler& cgh) {
					celerity::accessor read_a{buf_a, cgh, celerity::access::neighborhood{1}, celerity::read_only};
					celerity::accessor write_b{buf_b, cgh, celerity::access::one_to_one{}, celerity::write_only, celerity::no_init};
					celerity::debug::set_task_name(cgh, "downward step");
					cgh.parallel_for(geo, [=](celerity::id<1> idx) {
						const auto m = idx[0] > 0 ? idx[0] - 1 : 0;
						const auto p = idx[0] < buffer_size - 1 ? idx[0] + 1 : buffer_size - 1;
						// actually jacobi is not ideal for tracing results
						// write_b[idx] = (read_a[m] + read_a[p]) / 2.f;
						write_b[idx] = read_a[m] + read_a[p];
					});
				});
				// std::swap(buf_a, buf_b);
			}
		} else {
			q.submit([&](celerity::handler& cgh) {
				celerity::accessor read_a{buf_a, cgh, celerity::access::neighborhood{1}, celerity::read_only};
				celerity::accessor write_b{buf_b, cgh, celerity::access::one_to_one{}, celerity::write_only, celerity::no_init};
				celerity::debug::set_task_name(cgh, "step");
				cgh.parallel_for(buffer_size, [=](celerity::id<1> idx) {
					const auto m = idx[0] > 0 ? idx[0] - 1 : 0;
					const auto p = idx[0] < buffer_size - 1 ? idx[0] + 1 : buffer_size - 1;
					// actually jacobi is not ideal for tracing results
					// write_b[idx] = (read_a[m] + read_a[p]) / 2.f;
					write_b[idx] = read_a[m] + read_a[p];
				});
			});
		}
		std::swap(buf_a, buf_b);

		// q.submit([&](celerity::handler& cgh) {
		// 	celerity::accessor read_a{buf_a, cgh, celerity::access::all{}, celerity::read_only_host_task};
		// 	celerity::accessor read_b{buf_b, cgh, celerity::access::all{}, celerity::read_only_host_task};
		// 	celerity::debug::set_task_name(cgh, "print");
		// 	cgh.host_task(celerity::once, [=]() {
		// 		if(buffer_size < 128) {
		// 			CELERITY_CRITICAL("it {} a: {}", i, fmt::join(std::span(read_a.get_pointer(), read_a.get_pointer() + buffer_size), ","));
		// 			CELERITY_CRITICAL("it {} b: {}", i, fmt::join(std::span(read_b.get_pointer(), read_b.get_pointer() + buffer_size), ","));
		// 		}
		// 	});
		// });
		// q.wait(); // only for printing
	}
	q.wait(celerity::experimental::barrier);
	const auto after = std::chrono::steady_clock::now();

	q.submit([&](celerity::handler& cgh) {
		celerity::accessor read_a{buf_a, cgh, celerity::access::all{}, celerity::read_only_host_task};
		celerity::debug::set_task_name(cgh, "print");
		cgh.host_task(celerity::once, [=]() {
			CELERITY_CRITICAL("Time: {}ms", std::chrono::duration_cast<std::chrono::milliseconds>(after - start).count());
			if(buffer_size < 128) { CELERITY_CRITICAL("{}", fmt::join(std::span(read_a.get_pointer(), read_a.get_pointer() + buffer_size), ",")); }
			hash hsh;
			for(size_t i = 0; i < buffer_size; ++i) {
				hsh.add(read_a[i]);
			}
			fmt::print("Hash: {:x}\n", hsh.get());
		});
	});
}
