#include <vector>

#include <celerity.h>

int main(int argc, char* argv[]) {
	celerity::distr_queue queue;
	celerity::buffer<size_t, 2> buf(sycl::range<2>{512, 512});

	const auto access_up_to_ith_line_all = [](size_t i) { //
		return [i](celerity::chunk<2> chnk) { return celerity::subrange<2>({0, 0}, {i, size_t(-1)}); };
	};

	const auto access_ith_line_1to1 = [](size_t i) {
		return [i](celerity::chunk<2> chnk) { return celerity::subrange<2>({i, chnk.offset[0]}, {1, chnk.range[0]}); };
	};

	queue.slow_full_sync();
	const auto before = std::chrono::steady_clock::now();

	for(size_t t = 0; t < 100; ++t) {
		queue.submit([&](celerity::handler& cgh) {
			celerity::accessor a{buf, cgh, access_up_to_ith_line_all(t), celerity::read_only};
			celerity::accessor b{buf, cgh, access_ith_line_1to1(t), celerity::write_only, celerity::no_init};
			cgh.parallel_for<class rsim_pattern>(buf.get_range(), [=](celerity::item<2>) {
				(void)a;
				(void)b;
			});
		});
	}

	queue.slow_full_sync();
	const auto after = std::chrono::steady_clock::now();

	fmt::print("Time: {}ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(after - before).count());

	return 0;
}
