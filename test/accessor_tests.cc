#include "sycl_wrappers.h"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <celerity.h>

#include "ranges.h"

#include "buffer_manager_test_utils.h"

namespace celerity {
namespace detail {

	using celerity::access::all;
	using celerity::access::fixed;
	using celerity::access::one_to_one;

	TEST_CASE_METHOD(test_utils::runtime_fixture, "accessors behave correctly for 0-dimensional master node kernels", "[accessor]") {
		distr_queue q;
		std::vector mem_a{42};
		buffer<int, 1> buf_a(mem_a.data(), cl::sycl::range<1>{1});
		q.submit([=](handler& cgh) {
			auto a = buf_a.get_access<cl::sycl::access::mode::read_write, target::host_task>(cgh, fixed<1>({0, 1}));
			cgh.host_task(on_master_node, [=] { ++a[{0}]; });
		});
		int out = 0;
		q.submit(celerity::allow_by_ref, [=, &out](handler& cgh) {
			auto a = buf_a.get_access<cl::sycl::access::mode::read, target::host_task>(cgh, fixed<1>({0, 1}));
			cgh.host_task(on_master_node, [=, &out] { out = a[0]; });
		});
		q.slow_full_sync();
		CHECK(out == 43);
	}

	TEST_CASE_METHOD(test_utils::runtime_fixture, "accessors mode and target deduced correctly from SYCL 2020 tag types and no_init property", "[accessor]") {
		buffer<int, 1> buf_a(cl::sycl::range<1>(32));
		auto& tm = runtime::get_instance().get_task_manager();
		detail::task_id tid;

		SECTION("Device Accessors") {
			tid = test_utils::add_compute_task<class get_access_with_tag>(
			    tm,
			    [&](handler& cgh) {
				    accessor acc1{buf_a, cgh, one_to_one{}, celerity::write_only};
				    static_assert(std::is_same_v<accessor<int, 1, access_mode::write, target::device>, decltype(acc1)>);

				    accessor acc2{buf_a, cgh, one_to_one{}, celerity::read_only};
				    static_assert(std::is_same_v<accessor<int, 1, access_mode::read, target::device>, decltype(acc2)>);

				    accessor acc3{buf_a, cgh, one_to_one{}, celerity::read_write};
				    static_assert(std::is_same_v<accessor<int, 1, access_mode::read_write, target::device>, decltype(acc3)>);

				    accessor acc4{buf_a, cgh, one_to_one{}, celerity::write_only, celerity::no_init};
				    static_assert(std::is_same_v<accessor<int, 1, access_mode::discard_write, target::device>, decltype(acc4)>);

				    accessor acc5{buf_a, cgh, one_to_one{}, celerity::read_write, celerity::no_init};
				    static_assert(std::is_same_v<accessor<int, 1, access_mode::discard_read_write, target::device>, decltype(acc5)>);
			    },
			    buf_a.get_range());
		}


		SECTION("Host Accessors") {
			tid = test_utils::add_host_task(tm, on_master_node, [&](handler& cgh) {
				//   The following line is commented because it produces a compile error but it is still a case we wanted to test.
				//   Since we can not check the content of a property list at compile time, for now it is only accepted to pass either the property
				//   celerity::no_init or nothing.
				// accessor acc0{buf_a, cgh, one_to_one{}, cl::sycl::write_only_host_task, celerity::property_list{celerity::no_init}};

				accessor acc1{buf_a, cgh, one_to_one{}, celerity::write_only_host_task};
				static_assert(std::is_same_v<accessor<int, 1, access_mode::write, target::host_task>, decltype(acc1)>);

				accessor acc2{buf_a, cgh, one_to_one{}, celerity::read_only_host_task};
				static_assert(std::is_same_v<accessor<int, 1, access_mode::read, target::host_task>, decltype(acc2)>);

				accessor acc3{buf_a, cgh, fixed<1>{{0, 1}}, celerity::read_write_host_task};
				static_assert(std::is_same_v<accessor<int, 1, access_mode::read_write, target::host_task>, decltype(acc3)>);

				accessor acc4{buf_a, cgh, one_to_one{}, celerity::write_only_host_task, celerity::no_init};
				static_assert(std::is_same_v<accessor<int, 1, access_mode::discard_write, target::host_task>, decltype(acc4)>);

				accessor acc5{buf_a, cgh, one_to_one{}, celerity::read_write_host_task, celerity::no_init};
				static_assert(std::is_same_v<accessor<int, 1, access_mode::discard_read_write, target::host_task>, decltype(acc5)>);
			});
		}

		const auto tsk = tm.get_task(tid);
		const auto buff_id = detail::get_buffer_id(buf_a);

		REQUIRE(tsk->get_buffer_access_map().get_access_modes(buff_id).count(access_mode::write) == 1);
		REQUIRE(tsk->get_buffer_access_map().get_access_modes(buff_id).count(access_mode::read) == 1);
		REQUIRE(tsk->get_buffer_access_map().get_access_modes(buff_id).count(access_mode::read_write) == 1);
		REQUIRE(tsk->get_buffer_access_map().get_access_modes(buff_id).count(access_mode::discard_write) == 1);
		REQUIRE(tsk->get_buffer_access_map().get_access_modes(buff_id).count(access_mode::discard_read_write) == 1);
	}

	template <int>
	class accessor_fixture : public test_utils::buffer_manager_fixture {};

	template <int>
	class kernel_multi_dim_accessor_write_;

	template <int>
	class kernel_multi_dim_accessor_read_;

	template <int>
	class check_multi_dim_accessor;

	TEMPLATE_TEST_CASE_METHOD_SIG(accessor_fixture, "accessor supports multi-dimensional subscript operator", "[accessor]", ((int Dims), Dims), 2, 3) {
		// This test *used* to fill a buffer<sycl::id<Dims>> and check that the correct indices have been written. However, this caused the ComputeCpp 2.6.0
		// compiler to segfault on a device-code recursion detection step while traversing the following call path:
		//
		// 0. buffer_manager_fixture::get_device_accessor()
		// 1. buffer_manager::get_device_buffer()
		// 2. new device_buffer_storage<DataT, Dims>::device_buffer_storage()
		//
		// Stripping the device_buffer_storage constructor call in device code (where it is never actually called, this is all pure host code) through
		// #if __SYCL_DEVICE_ONLY__ did get rid of the segfault, but caused the test to fail with a heap corruption at runtime. Instead, replacing sycl::id with
		// size_t seems to resolve the problem.

		const auto range = cl::sycl::range<3>(2, 3, 4);
		auto& bm = accessor_fixture<Dims>::get_buffer_manager();
		auto bid = bm.template register_buffer<size_t, Dims>(range);

		auto& q = accessor_fixture<Dims>::get_device_queue();
		auto sr = subrange<3>({}, range);
		live_pass_device_handler cgh(nullptr, sr, true, q);

		// this kernel initializes the buffer what will be read after.
		auto acc_write =
		    accessor_fixture<Dims>::template get_device_accessor<size_t, Dims, cl::sycl::access::mode::discard_write>(cgh, bid, range_cast<Dims>(range), {});
		cgh.parallel_for<class kernel_multi_dim_accessor_write_<Dims>>(
		    range_cast<Dims>(range), [=](celerity::item<Dims> item) { acc_write[item] = item.get_linear_id(); });
		cgh.get_submission_event().wait();

		SECTION("for device buffers") {
			auto acc_read =
			    accessor_fixture<Dims>::template get_device_accessor<size_t, Dims, cl::sycl::access::mode::read>(cgh, bid, range_cast<Dims>(range), {});
			auto acc = accessor_fixture<Dims>::template get_device_accessor<size_t, Dims, cl::sycl::access::mode::discard_write>(
			    cgh, bid, range_cast<Dims>(range), {});
			cgh.parallel_for<class kernel_multi_dim_accessor_read_<Dims>>(range_cast<Dims>(range), [=](celerity::item<Dims> item) {
				size_t i = item[0];
				size_t j = item[1];
				if constexpr(Dims == 2) {
					acc[i][j] = acc_read[i][j];
				} else {
					size_t k = item[2];
					acc[i][j][k] = acc_read[i][j][k];
				}
			});
			cgh.get_submission_event().wait();
		}

		SECTION("for host buffers") {
			auto acc_read = accessor_fixture<Dims>::template get_host_accessor<size_t, Dims, cl::sycl::access::mode::read>(bid, range_cast<Dims>(range), {});
			auto acc =
			    accessor_fixture<Dims>::template get_host_accessor<size_t, Dims, cl::sycl::access::mode::discard_write>(bid, range_cast<Dims>(range), {});
			for(size_t i = 0; i < range[0]; i++) {
				for(size_t j = 0; j < range[1]; j++) {
					for(size_t k = 0; k < (Dims == 2 ? 1 : range[2]); k++) {
						if constexpr(Dims == 2) {
							acc[i][j] = acc_read[i][j];
						} else {
							acc[i][j][k] = acc_read[i][j][k];
						}
					}
				}
			}
		}

		typename accessor_fixture<Dims>::access_target tgt = accessor_fixture<Dims>::access_target::host;
		bool acc_check = accessor_fixture<Dims>::template buffer_reduce<size_t, Dims, class check_multi_dim_accessor<Dims>>(bid, tgt, range_cast<Dims>(range),
		    {}, true,
		    [range = range_cast<Dims>(range)](cl::sycl::id<Dims> idx, bool current, size_t value) { return current && value == get_linear_index(range, idx); });

		REQUIRE(acc_check);
	}

	TEST_CASE_METHOD(test_utils::runtime_fixture, "conflicts between producer-accessors and reductions are reported", "[task-manager]") {
		runtime::init(nullptr, nullptr);
		auto& tm = runtime::get_instance().get_task_manager();
		test_utils::mock_buffer_factory mbf{tm};
		test_utils::mock_reduction_factory mrf;

		auto buf_0 = mbf.create_buffer(cl::sycl::range<1>{1});

		CHECK_THROWS(test_utils::add_compute_task<class UKN(task_reduction_conflict)>(tm, [&](handler& cgh) {
			test_utils::add_reduction(cgh, mrf, buf_0, false);
			test_utils::add_reduction(cgh, mrf, buf_0, false);
		}));

		CHECK_THROWS(test_utils::add_compute_task<class UKN(task_reduction_access_conflict)>(tm, [&](handler& cgh) {
			test_utils::add_reduction(cgh, mrf, buf_0, false);
			buf_0.get_access<access_mode::read>(cgh, fixed<1>({0, 1}));
		}));

		CHECK_THROWS(test_utils::add_compute_task<class UKN(task_reduction_access_conflict)>(tm, [&](handler& cgh) {
			test_utils::add_reduction(cgh, mrf, buf_0, false);
			buf_0.get_access<access_mode::write>(cgh, fixed<1>({0, 1}));
		}));

		CHECK_THROWS(test_utils::add_compute_task<class UKN(task_reduction_access_conflict)>(tm, [&](handler& cgh) {
			test_utils::add_reduction(cgh, mrf, buf_0, false);
			buf_0.get_access<access_mode::read_write>(cgh, fixed<1>({0, 1}));
		}));

		CHECK_THROWS(test_utils::add_compute_task<class UKN(task_reduction_access_conflict)>(tm, [&](handler& cgh) {
			test_utils::add_reduction(cgh, mrf, buf_0, false);
			buf_0.get_access<access_mode::discard_write>(cgh, fixed<1>({0, 1}));
		}));
	}

	template <access_mode>
	class empty_access_kernel;

	template <access_mode Mode>
	static void test_empty_access(distr_queue& q, buffer<int, 1>& test_buf) {
		CAPTURE(Mode);
		bool verified = false;
		buffer<bool> verify_buf{&verified, 1};
		q.submit([=](handler& cgh) {
			// access with offset == buffer range just to mess with things
			const auto offset = id_cast<1>(test_buf.get_range());
			const auto test_acc = test_buf.get_access<Mode>(cgh, [=](chunk<1>) { return subrange<1>{offset, 0}; });
			const auto verify_acc = verify_buf.get_access<access_mode::write>(cgh, one_to_one{});
			cgh.parallel_for<empty_access_kernel<Mode>>(range<1>{1}, [=](item<1>) {
				(void)test_acc;
				verify_acc[0] = true;
			});
		});
		q.submit(allow_by_ref, [&](handler& cgh) {
			const accessor verify_acc{verify_buf, cgh, all{}, read_only_host_task};
			cgh.host_task(on_master_node, [=, &verified] { verified = verify_acc[0]; });
		});
		q.slow_full_sync();
		CHECK(verified);
	};

	TEST_CASE_METHOD(test_utils::runtime_fixture, "kernels gracefully handle access to empty buffers", "[accessor]") {
		distr_queue q;
		buffer<int> buf{0};

		test_empty_access<access_mode::discard_write>(q, buf);
		test_empty_access<access_mode::read_write>(q, buf);
		test_empty_access<access_mode::read>(q, buf);
	}

	TEST_CASE_METHOD(test_utils::runtime_fixture, "kernels gracefully handle empty access ranges", "[accessor]") {
		distr_queue q;
		std::optional<buffer<int>> buf;

		int init = 0;
		SECTION("when the buffer is uninitialized") { buf = buffer<int>{1}; };
		SECTION("when the buffer is host-initialized") { buf = buffer<int>{&init, 1}; };

		test_empty_access<access_mode::discard_write>(q, *buf);
		test_empty_access<access_mode::read_write>(q, *buf);
		test_empty_access<access_mode::read>(q, *buf);
	}

	TEST_CASE_METHOD(test_utils::runtime_fixture, "host accessor get_allocation_window produces the correct memory layout", "[task][accessor]") {
		distr_queue q;

		std::vector<char> memory1d(10);
		buffer<char, 1> buf1d(memory1d.data(), cl::sycl::range<1>(10));

		q.submit([=](handler& cgh) {
			accessor b{buf1d, cgh, all{}, celerity::write_only_host_task, celerity::no_init};
			cgh.host_task(on_master_node, [=](partition<0> part) {
				auto aw = b.get_allocation_window(part);
				CHECK(aw.get_window_offset_in_buffer()[0] == 0);
				CHECK(aw.get_window_offset_in_allocation()[0] == 0);
				CHECK(aw.get_buffer_range()[0] == 10);
				CHECK(aw.get_allocation_range()[0] >= 10);
				CHECK(aw.get_window_range()[0] == 10);
			});
		});

		q.submit([=](handler& cgh) {
			accessor b{buf1d, cgh, one_to_one{}, celerity::write_only_host_task, celerity::no_init};
			cgh.host_task(cl::sycl::range<1>(6), cl::sycl::id<1>(2), [=](partition<1> part) {
				auto aw = b.get_allocation_window(part);
				CHECK(aw.get_window_offset_in_buffer()[0] == 2);
				CHECK(aw.get_window_offset_in_allocation()[0] <= 2);
				CHECK(aw.get_buffer_range()[0] == 10);
				CHECK(aw.get_allocation_range()[0] >= 6);
				CHECK(aw.get_allocation_range()[0] <= 10);
				CHECK(aw.get_window_range()[0] == 6);
			});
		});

		std::vector<char> memory2d(10 * 10);
		buffer<char, 2> buf2d(memory2d.data(), cl::sycl::range<2>(10, 10));

		q.submit([=](handler& cgh) {
			accessor b{buf2d, cgh, one_to_one{}, celerity::write_only_host_task, celerity::no_init};
			cgh.host_task(cl::sycl::range<2>(5, 6), cl::sycl::id<2>(1, 2), [=](partition<2> part) {
				auto aw = b.get_allocation_window(part);
				CHECK(aw.get_window_offset_in_buffer()[0] == 1);
				CHECK(aw.get_buffer_range()[0] == 10);
				CHECK(aw.get_window_offset_in_allocation()[0] <= 1);
				CHECK(aw.get_allocation_range()[0] >= 6);
				CHECK(aw.get_allocation_range()[0] <= 10);
				CHECK(aw.get_window_range()[0] == 5);
				CHECK(aw.get_window_offset_in_buffer()[1] == 2);
				CHECK(aw.get_buffer_range()[1] == 10);
				CHECK(aw.get_window_range()[1] == 6);
			});
		});

		std::vector<char> memory3d(10 * 10 * 10);
		buffer<char, 3> buf3d(memory3d.data(), cl::sycl::range<3>(10, 10, 10));

		q.submit([=](handler& cgh) {
			accessor b{buf3d, cgh, one_to_one{}, celerity::write_only_host_task, celerity::no_init};
			cgh.host_task(cl::sycl::range<3>(5, 6, 7), cl::sycl::id<3>(1, 2, 3), [=](partition<3> part) {
				auto aw = b.get_allocation_window(part);
				CHECK(aw.get_window_offset_in_buffer()[0] == 1);
				CHECK(aw.get_window_offset_in_allocation()[0] <= 1);
				CHECK(aw.get_buffer_range()[0] == 10);
				CHECK(aw.get_allocation_range()[0] >= 5);
				CHECK(aw.get_allocation_range()[0] <= 10);
				CHECK(aw.get_window_range()[0] == 5);
				CHECK(aw.get_window_offset_in_buffer()[1] == 2);
				CHECK(aw.get_window_offset_in_allocation()[1] <= 2);
				CHECK(aw.get_buffer_range()[1] == 10);
				CHECK(aw.get_allocation_range()[1] >= 6);
				CHECK(aw.get_allocation_range()[1] <= 10);
				CHECK(aw.get_window_range()[1] == 6);
				CHECK(aw.get_window_offset_in_buffer()[2] == 3);
				CHECK(aw.get_buffer_range()[2] == 10);
				CHECK(aw.get_window_range()[2] == 7);
			});
		});
	}

} // namespace detail
} // namespace celerity
