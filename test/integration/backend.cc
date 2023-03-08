#include <celerity.h>

#define ASSERT(...)                                                                                                                                            \
	if(!(__VA_ARGS__)) {                                                                                                                                       \
		CELERITY_ERROR("ASSERT({}) on line {} failed\n", #__VA_ARGS__, __LINE__);                                                                              \
		std::abort();                                                                                                                                          \
	}

template <typename T, int Dims>
struct kernel_name {};

template <int Dims>
void test_copy(celerity::distr_queue& q) {
	celerity::buffer<size_t, Dims> buf(celerity::detail::range_cast<Dims>(celerity::range<3>{5, 7, 9}));

	// Initialize on device
	q.submit([=](celerity::handler& cgh) {
		celerity::accessor acc{buf, cgh, celerity::access::one_to_one<>{}, celerity::write_only, celerity::no_init};
		cgh.parallel_for<kernel_name<class init, Dims>>(buf.get_range(), [=](celerity::item<Dims> itm) { acc[itm] = itm.get_linear_id(); });
	});

	// Check and modify partially on host
	const auto sr = celerity::detail::subrange_cast<Dims>(celerity::subrange<3>{{1, 2, 3}, {3, 4, 5}});
	const auto sr3 = celerity::detail::subrange_cast<3>(sr);
	q.submit([=](celerity::handler& cgh) {
		celerity::accessor acc{buf, cgh, celerity::access::fixed<Dims>{sr}, celerity::read_write_host_task};
		cgh.host_task(celerity::on_master_node, [=]() {
			for(size_t k = 0; k < sr3.range[0]; ++k) {
				for(size_t j = 0; j < sr3.range[1]; ++j) {
					for(size_t i = 0; i < sr3.range[2]; ++i) {
						const celerity::id<3> idx{sr3.offset[0] + k, sr3.offset[1] + j, sr3.offset[2] + i};
						const auto linear_id = celerity::detail::get_linear_index(buf.get_range(), celerity::detail::id_cast<Dims>(idx));
						ASSERT(acc[celerity::detail::id_cast<Dims>(idx)] == linear_id);
						acc[celerity::detail::id_cast<Dims>(idx)] *= 2;
					}
				}
			}
		});
	});

	// Modify everything on device
	q.submit([=](celerity::handler& cgh) {
		celerity::accessor acc{buf, cgh, celerity::access::one_to_one<>{}, celerity::read_write};
		cgh.parallel_for<kernel_name<class modify, Dims>>(buf.get_range(), [=](celerity::item<Dims> itm) { acc[itm] += 1; });
	});

	// Check everything on host
	q.submit([=](celerity::handler& cgh) {
		celerity::accessor acc{buf, cgh, celerity::access::all{}, celerity::read_only_host_task};
		cgh.host_task(celerity::on_master_node, [=]() {
			const auto r3 = celerity::detail::range_cast<3>(buf.get_range());
			for(size_t k = 0; k < r3[0]; ++k) {
				for(size_t j = 0; j < r3[1]; ++j) {
					for(size_t i = 0; i < r3[2]; ++i) {
						const celerity::id<3> idx{k, j, i};
						const auto is_in_sr =
						    ((idx >= sr3.offset == celerity::id<3>(true, true, true)) && (idx < sr3.offset + sr3.range == celerity::id<3>(true, true, true)));
						const auto linear_id = celerity::detail::get_linear_index(buf.get_range(), celerity::detail::id_cast<Dims>(idx));
						if(is_in_sr) {
							ASSERT(acc[celerity::detail::id_cast<Dims>(idx)] == 2 * linear_id + 1);
						} else {
							ASSERT(acc[celerity::detail::id_cast<Dims>(idx)] == linear_id + 1);
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
	celerity::distr_queue q{all_devices[device_idx]};

	test_copy<1>(q);
	test_copy<2>(q);
	test_copy<3>(q);

	return EXIT_SUCCESS;
}