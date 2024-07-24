#include <celerity.h>

#define ASSERT(...)                                                                                                                                            \
	if(!(__VA_ARGS__)) {                                                                                                                                       \
		CELERITY_ERROR("ASSERT({}) on line {} failed\n", #__VA_ARGS__, __LINE__);                                                                              \
		std::abort();                                                                                                                                          \
	}

template <int Dims>
celerity::range<Dims> truncate_range(const celerity::range<3>& r3) {
	celerity::range<Dims> r = celerity::detail::zeros;
	for(int d = 0; d < Dims; ++d) {
		r[d] = r3[d];
	}
	return r;
}

template <int Dims>
celerity::id<Dims> truncate_id(const celerity::id<3>& i3) {
	celerity::id<Dims> i;
	for(int d = 0; d < Dims; ++d) {
		i[d] = i3[d];
	}
	return i;
}

template <int Dims>
celerity::subrange<Dims> truncate_subrange(const celerity::subrange<3>& sr3) {
	celerity::subrange<Dims> sr;
	for(int d = 0; d < Dims; ++d) {
		sr.offset[d] = sr3.offset[d];
		sr.range[d] = sr3.range[d];
	}
	return sr;
}


template <typename T, int Dims>
struct kernel_name {};

template <int Dims>
void test_copy(celerity::distr_queue& q) {
	celerity::buffer<size_t, Dims> buf(truncate_range<Dims>({5, 7, 9}));

	// Initialize on device
	q.submit([&](celerity::handler& cgh) {
		celerity::accessor acc{buf, cgh, celerity::access::one_to_one{}, celerity::write_only, celerity::no_init};
		cgh.parallel_for<kernel_name<class init, Dims>>(buf.get_range(), [=](celerity::item<Dims> itm) { acc[itm] = itm.get_linear_id(); });
	});

	// Check and modify partially on host
	const auto sr = truncate_subrange<Dims>({{1, 2, 3}, {3, 4, 5}});
	const auto sr3 = celerity::detail::subrange_cast<3>(sr);
	q.submit([&](celerity::handler& cgh) {
		celerity::accessor acc{buf, cgh, celerity::access::fixed<Dims>{sr}, celerity::read_write_host_task};
		cgh.host_task(celerity::on_master_node, [acc, sr3, buf_range = buf.get_range()] {
			for(size_t k = 0; k < sr3.range[0]; ++k) {
				for(size_t j = 0; j < sr3.range[1]; ++j) {
					for(size_t i = 0; i < sr3.range[2]; ++i) {
						const auto idx = truncate_id<Dims>({sr3.offset[0] + k, sr3.offset[1] + j, sr3.offset[2] + i});
						const auto linear_id = celerity::detail::get_linear_index(buf_range, idx);
						ASSERT(acc[idx] == linear_id);
						acc[idx] *= 2;
					}
				}
			}
		});
	});

	// Modify everything on device
	q.submit([&](celerity::handler& cgh) {
		celerity::accessor acc{buf, cgh, celerity::access::one_to_one{}, celerity::read_write};
		cgh.parallel_for<kernel_name<class modify, Dims>>(buf.get_range(), [=](celerity::item<Dims> itm) { acc[itm] += 1; });
	});

	// Check everything on host
	q.submit([&](celerity::handler& cgh) {
		celerity::accessor acc{buf, cgh, celerity::access::all{}, celerity::read_only_host_task};
		cgh.host_task(celerity::on_master_node, [acc, sr, buf_range = buf.get_range()] {
			const auto r3 = celerity::detail::range_cast<3>(buf_range);
			for(size_t k = 0; k < r3[0]; ++k) {
				for(size_t j = 0; j < r3[1]; ++j) {
					for(size_t i = 0; i < r3[2]; ++i) {
						const auto idx = truncate_id<Dims>({k, j, i});
						const auto is_in_sr = (celerity::detail::all_true(idx >= sr.offset) && celerity::detail::all_true(idx < sr.offset + sr.range));
						const auto linear_id = celerity::detail::get_linear_index(buf_range, idx);
						if(is_in_sr) {
							ASSERT(acc[idx] == 2 * linear_id + 1);
						} else {
							ASSERT(acc[idx] == linear_id + 1);
						}
					}
				}
			}
		});
	});
}

int main(int argc, char* argv[]) {
	if(argc > 2) {
		fmt::print(stderr, "Usage: %s [device index]\n", argv[0]);
		return EXIT_FAILURE;
	}

	const auto all_devices = sycl::device::get_devices();
	if(argc == 1) {
		for(const auto& d : all_devices) {
			fmt::print("{} {}\n", d.get_platform().get_info<sycl::info::platform::name>(), d.get_info<sycl::info::device::name>());
		}
		return 0;
	}

	const auto device_idx = std::atoi(argv[1]);
	celerity::distr_queue q{{all_devices[device_idx]}};

	test_copy<1>(q);
	test_copy<2>(q);
	test_copy<3>(q);

	return EXIT_SUCCESS;
}
