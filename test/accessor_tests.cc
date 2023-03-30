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
		buffer<int, 1> buf_a(mem_a.data(), celerity::range<1>{1});
		q.submit([=](handler& cgh) {
			auto a = buf_a.get_access<cl::sycl::access::mode::read_write, target::host_task>(cgh, fixed<1>({0, 1}));
			cgh.host_task(on_master_node, [=] { ++a[0]; });
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
		using buf1d_t = buffer<int, 1>&;
		using buf0d_t = buffer<int, 0>&;

		SECTION("device accessors") {
			// This currently throws an error at runtime, because we cannot infer whether the access is a discard_* from the property list parameter.
			using acc0 = decltype(accessor{std::declval<buf1d_t>(), std::declval<handler&>(), one_to_one{}, celerity::write_only, celerity::property_list{}});
			STATIC_REQUIRE(std::is_same_v<accessor<int, 1, access_mode::write, target::device>, acc0>);

			using acc1 = decltype(accessor{std::declval<buf1d_t>(), std::declval<handler&>(), one_to_one{}, celerity::write_only});
			STATIC_REQUIRE(std::is_same_v<accessor<int, 1, access_mode::write, target::device>, acc1>);

			using acc2 = decltype(accessor{std::declval<buf1d_t>(), std::declval<handler&>(), one_to_one{}, celerity::read_only});
			STATIC_REQUIRE(std::is_same_v<accessor<int, 1, access_mode::read, target::device>, acc2>);

			using acc3 = decltype(accessor{std::declval<buf1d_t>(), std::declval<handler&>(), one_to_one{}, celerity::read_write});
			STATIC_REQUIRE(std::is_same_v<accessor<int, 1, access_mode::read_write, target::device>, acc3>);

			using acc4 = decltype(accessor{std::declval<buf1d_t>(), std::declval<handler&>(), one_to_one{}, celerity::write_only, celerity::no_init});
			STATIC_REQUIRE(std::is_same_v<accessor<int, 1, access_mode::discard_write, target::device>, acc4>);

			using acc5 = decltype(accessor{std::declval<buf1d_t>(), std::declval<handler&>(), one_to_one{}, celerity::read_write, celerity::no_init});
			STATIC_REQUIRE(std::is_same_v<accessor<int, 1, access_mode::discard_read_write, target::device>, acc5>);

			// This currently throws an error at runtime, because we cannot infer whether the access is a discard_* from the property list parameter.
			using acc6 = decltype(accessor{std::declval<buf0d_t>(), std::declval<handler&>(), celerity::write_only, celerity::property_list{}});
			STATIC_REQUIRE(std::is_same_v<accessor<int, 0, access_mode::write, target::device>, acc6>);

			using acc7 = decltype(accessor{std::declval<buf0d_t>(), std::declval<handler&>(), celerity::read_only});
			STATIC_REQUIRE(std::is_same_v<accessor<int, 0, access_mode::read, target::device>, acc7>);

			using acc8 = decltype(accessor{std::declval<buf0d_t>(), std::declval<handler&>(), celerity::write_only, celerity::no_init});
			STATIC_REQUIRE(std::is_same_v<accessor<int, 0, access_mode::discard_write, target::device>, acc8>);
		}

		SECTION("host accessors") {
			// This currently throws an error at runtime, because we cannot infer whether the access is a discard_* from the property list parameter.
			using acc0 =
			    decltype(accessor{std::declval<buf1d_t>(), std::declval<handler&>(), one_to_one{}, celerity::write_only_host_task, celerity::property_list{}});
			STATIC_REQUIRE(std::is_same_v<accessor<int, 1, access_mode::write, target::host_task>, acc0>);

			using acc1 = decltype(accessor{std::declval<buf1d_t>(), std::declval<handler&>(), one_to_one{}, celerity::write_only_host_task});
			STATIC_REQUIRE(std::is_same_v<accessor<int, 1, access_mode::write, target::host_task>, acc1>);

			using acc2 = decltype(accessor{std::declval<buf1d_t>(), std::declval<handler&>(), one_to_one{}, celerity::read_only_host_task});
			STATIC_REQUIRE(std::is_same_v<accessor<int, 1, access_mode::read, target::host_task>, acc2>);

			using acc3 = decltype(accessor{std::declval<buf1d_t>(), std::declval<handler&>(), one_to_one{}, celerity::read_write_host_task});
			STATIC_REQUIRE(std::is_same_v<accessor<int, 1, access_mode::read_write, target::host_task>, acc3>);

			using acc4 = decltype(accessor{std::declval<buf1d_t>(), std::declval<handler&>(), one_to_one{}, celerity::write_only_host_task, celerity::no_init});
			STATIC_REQUIRE(std::is_same_v<accessor<int, 1, access_mode::discard_write, target::host_task>, acc4>);

			using acc5 = decltype(accessor{std::declval<buf1d_t>(), std::declval<handler&>(), one_to_one{}, celerity::read_write_host_task, celerity::no_init});
			STATIC_REQUIRE(std::is_same_v<accessor<int, 1, access_mode::discard_read_write, target::host_task>, acc5>);

			// This currently throws an error at runtime, because we cannot infer whether the access is a discard_* from the property list parameter.
			using acc6 = decltype(accessor{std::declval<buf0d_t>(), std::declval<handler&>(), celerity::write_only_host_task, celerity::property_list{}});
			STATIC_REQUIRE(std::is_same_v<accessor<int, 0, access_mode::write, target::host_task>, acc6>);

			using acc7 = decltype(accessor{std::declval<buf0d_t>(), std::declval<handler&>(), celerity::read_only_host_task});
			STATIC_REQUIRE(std::is_same_v<accessor<int, 0, access_mode::read, target::host_task>, acc7>);

			using acc8 = decltype(accessor{std::declval<buf0d_t>(), std::declval<handler&>(), celerity::write_only_host_task, celerity::no_init});
			STATIC_REQUIRE(std::is_same_v<accessor<int, 0, access_mode::discard_write, target::host_task>, acc8>);
		}
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
		// #if __SYCL_DEVICE_ONLY__ did get rid of the segfault, but caused the test to fail with a heap corruption at runtime. Instead, replacing celerity::id
		// with size_t seems to resolve the problem.

		const auto range = celerity::range<3>(2, 3, 4);
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
		    [range = range_cast<Dims>(range)](celerity::id<Dims> idx, bool current, size_t value) { return current && value == get_linear_index(range, idx); });

		REQUIRE(acc_check);
	}

	TEST_CASE_METHOD(test_utils::runtime_fixture, "conflicts between producer-accessors and reductions are reported", "[task-manager]") {
		runtime::init(nullptr, nullptr);
		auto& tm = runtime::get_instance().get_task_manager();
		test_utils::mock_buffer_factory mbf{tm};
		test_utils::mock_reduction_factory mrf;

		auto buf_0 = mbf.create_buffer(celerity::range<1>{1});

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
		buffer<char, 1> buf1d(memory1d.data(), celerity::range<1>(10));

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
			cgh.host_task(celerity::range<1>(6), celerity::id<1>(2), [=](partition<1> part) {
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
		buffer<char, 2> buf2d(memory2d.data(), celerity::range<2>(10, 10));

		q.submit([=](handler& cgh) {
			accessor b{buf2d, cgh, one_to_one{}, celerity::write_only_host_task, celerity::no_init};
			cgh.host_task(celerity::range<2>(5, 6), celerity::id<2>(1, 2), [=](partition<2> part) {
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
		buffer<char, 3> buf3d(memory3d.data(), celerity::range<3>(10, 10, 10));

		q.submit([=](handler& cgh) {
			accessor b{buf3d, cgh, one_to_one{}, celerity::write_only_host_task, celerity::no_init};
			cgh.host_task(celerity::range<3>(5, 6, 7), celerity::id<3>(1, 2, 3), [=](partition<3> part) {
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

	TEST_CASE_METHOD(test_utils::runtime_fixture, "kernels can access 0-dimensional buffers", "[buffer]") {
		constexpr float value_a = 13.37f;
		constexpr float value_b = 42.0f;

		buffer<float, 0> buf_0 = value_a;
		buffer<float, 1> buf_1(100);

		distr_queue q;
		q.submit([=](handler& cgh) {
			accessor acc_0(buf_0, cgh, read_write_host_task);
			cgh.host_task(on_master_node, [=] {
				CHECK(*acc_0 == value_a);
				*acc_0 = value_b;
			});
		});
		q.submit([=](handler& cgh) {
			accessor acc_0(buf_0, cgh, read_only);
			accessor acc_1(buf_1, cgh, one_to_one(), write_only, no_init);
			cgh.parallel_for<class UKN(device)>(buf_1.get_range(), [=](item<1> it) { acc_1[it] = *acc_0; });
		});
		q.submit([=](handler& cgh) {
			accessor acc_1(buf_1, cgh, all(), read_only_host_task);
			cgh.host_task(on_master_node, [=] {
				for(size_t i = 0; i < buf_1.get_range().size(); ++i) {
					REQUIRE_LOOP(acc_1[i] == value_b);
				}
			});
		});
	}

	TEST_CASE_METHOD(test_utils::runtime_fixture, "0-dimensional kernels can access arbitrary-dimensional buffers", "[buffer]") {
		buffer<float, 0> buf_0d;
		buffer<float, 1> buf_1d({100});
		buffer<float, 2> buf_2d({10, 10});
		buffer<float, 3> buf_3d({5, 5, 5});

		distr_queue q;
		q.submit([=](handler& cgh) {
			accessor acc_0d(buf_0d, cgh, write_only, no_init);
			accessor acc_1d(buf_1d, cgh, all(), write_only, no_init);
			accessor acc_2d(buf_2d, cgh, all(), write_only, no_init);
			accessor acc_3d(buf_3d, cgh, all(), write_only, no_init);
			cgh.parallel_for<class UKN(device)>(range<0>(), [=](item<0>) {
				*acc_0d = 1;
				acc_1d[99] = 2;
				acc_2d[9][9] = 3;
				acc_3d[4][4][4] = 4;
			});
		});
		q.submit([=](handler& cgh) {
			accessor acc_0d(buf_0d, cgh, read_write_host_task);
			accessor acc_1d(buf_1d, cgh, all(), read_write_host_task);
			accessor acc_2d(buf_2d, cgh, all(), read_write_host_task);
			accessor acc_3d(buf_3d, cgh, all(), read_write_host_task);
			cgh.host_task(range<0>(), [=](partition<0>) {
				*acc_0d += 9;
				acc_1d[99] += 9;
				acc_2d[9][9] += 9;
				acc_3d[4][4][4] += 9;
			});
		});
		q.submit([=](handler& cgh) {
			accessor acc_0d(buf_0d, cgh, read_only_host_task);
			accessor acc_1d(buf_1d, cgh, all(), read_only_host_task);
			accessor acc_2d(buf_2d, cgh, all(), read_only_host_task);
			accessor acc_3d(buf_3d, cgh, all(), read_only_host_task);
			cgh.host_task(on_master_node, [=] {
				CHECK(*acc_0d == 10);
				CHECK(acc_1d[99] == 11);
				CHECK(acc_2d[9][9] == 12);
				CHECK(acc_3d[4][4][4] == 13);
			});
		});
	}

	TEST_CASE_METHOD(test_utils::runtime_fixture, "0-dimensional local accessors behave as expected", "[buffer]") {
		constexpr float value_a = 13.37f;
		constexpr float value_b = 42.0f;

		buffer<float, 1> buf_1(32);

		distr_queue q;
		q.submit([=](handler& cgh) {
			accessor acc_1(buf_1, cgh, one_to_one(), write_only, no_init);
			cgh.parallel_for<class UKN(device)>(buf_1.get_range(), [=](item<1> it) { acc_1[it] = value_a; });
		});
		q.submit([=](handler& cgh) {
			accessor acc_1(buf_1, cgh, one_to_one(), write_only);
			local_accessor<float, 0> local_0(cgh);
			cgh.parallel_for<class UKN(device)>(nd_range(buf_1.get_range(), buf_1.get_range()), [=](nd_item<1> it) {
				if(it.get_local_id() == 0) { *local_0 = value_b; }
				group_barrier(it.get_group());
				acc_1[it.get_global_id()] = *local_0;
			});
		});
		q.submit([=](handler& cgh) {
			accessor acc_1(buf_1, cgh, all(), read_only_host_task);
			cgh.host_task(on_master_node, [=] {
				for(size_t i = 0; i < buf_1.get_range().size(); ++i) {
					REQUIRE_LOOP(acc_1[i] == value_b);
				}
			});
		});
	}

} // namespace detail
} // namespace celerity
