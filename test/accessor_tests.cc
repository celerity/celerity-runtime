#include "sycl_wrappers.h"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <celerity.h>

#include "ranges.h"

#include "buffer_manager_test_utils.h"
#include "log_test_utils.h"

// NOTE: There are some additional accessor tests in buffer_manager_tests.cc

namespace celerity {
namespace detail {

	using celerity::access::all;
	using celerity::access::fixed;
	using celerity::access::one_to_one;

	TEST_CASE_METHOD(test_utils::runtime_fixture, "accessors behave correctly for 0-dimensional master node kernels", "[accessor]") {
		distr_queue q;
		std::vector mem_a{42};
		buffer<int, 1> buf_a(mem_a.data(), range<1>{1});
		q.submit([&](handler& cgh) {
			auto a = buf_a.get_access<cl::sycl::access::mode::read_write, target::host_task>(cgh, fixed<1>({0, 1}));
			cgh.host_task(on_master_node, [=] { ++a[0]; });
		});
		int out = 0;
		q.submit([&](handler& cgh) {
			auto a = buf_a.get_access<cl::sycl::access::mode::read, target::host_task>(cgh, fixed<1>({0, 1}));
			cgh.host_task(on_master_node, [=, &out] { out = a[0]; });
		});
		q.slow_full_sync();
		CHECK(out == 43);
	}

	TEST_CASE("accessors mode and target deduced correctly from SYCL 2020 tag types and no_init property", "[accessor]") {
		using buf1d_t = buffer<int, 1>&;
		using buf0d_t = buffer<int, 0>&;

		SECTION("device accessors") {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
			// This currently throws an error at runtime, because we cannot infer whether the access is a discard_* from the property list parameter.
			using acc0 = decltype(accessor{std::declval<buf1d_t>(), std::declval<handler&>(), one_to_one{}, celerity::write_only, celerity::property_list{}});
			STATIC_REQUIRE(std::is_same_v<accessor<int, 1, access_mode::write, target::device>, acc0>);
#pragma GCC diagnostic pop

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

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
			// This currently throws an error at runtime, because we cannot infer whether the access is a discard_* from the property list parameter.
			using acc6 = decltype(accessor{std::declval<buf0d_t>(), std::declval<handler&>(), all(), celerity::write_only, celerity::property_list{}});
			STATIC_REQUIRE(std::is_same_v<accessor<int, 0, access_mode::write, target::device>, acc6>);
#pragma GCC diagnostic pop

			using acc7 = decltype(accessor{std::declval<buf0d_t>(), std::declval<handler&>(), all(), celerity::read_only});
			STATIC_REQUIRE(std::is_same_v<accessor<int, 0, access_mode::read, target::device>, acc7>);

			using acc8 = decltype(accessor{std::declval<buf0d_t>(), std::declval<handler&>(), all(), celerity::write_only, celerity::no_init});
			STATIC_REQUIRE(std::is_same_v<accessor<int, 0, access_mode::discard_write, target::device>, acc8>);

			// This currently throws an error at runtime, because we cannot infer whether the access is a discard_* from the property list parameter.
			using acc9 = decltype(accessor{std::declval<buf0d_t>(), std::declval<handler&>(), celerity::write_only, celerity::property_list{}});
			STATIC_REQUIRE(std::is_same_v<accessor<int, 0, access_mode::write, target::device>, acc9>);

			using acc10 = decltype(accessor{std::declval<buf0d_t>(), std::declval<handler&>(), celerity::read_only});
			STATIC_REQUIRE(std::is_same_v<accessor<int, 0, access_mode::read, target::device>, acc10>);

			using acc11 = decltype(accessor{std::declval<buf0d_t>(), std::declval<handler&>(), celerity::write_only, celerity::no_init});
			STATIC_REQUIRE(std::is_same_v<accessor<int, 0, access_mode::discard_write, target::device>, acc11>);
		}

		SECTION("host accessors") {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
			// This currently throws an error at runtime, because we cannot infer whether the access is a discard_* from the property list parameter.
			using acc0 =
			    decltype(accessor{std::declval<buf1d_t>(), std::declval<handler&>(), one_to_one{}, celerity::write_only_host_task, celerity::property_list{}});
			STATIC_REQUIRE(std::is_same_v<accessor<int, 1, access_mode::write, target::host_task>, acc0>);
#pragma GCC diagnostic pop

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

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
			// This currently throws an error at runtime, because we cannot infer whether the access is a discard_* from the property list parameter.
			using acc6 =
			    decltype(accessor{std::declval<buf0d_t>(), std::declval<handler&>(), all(), celerity::write_only_host_task, celerity::property_list{}});
			STATIC_REQUIRE(std::is_same_v<accessor<int, 0, access_mode::write, target::host_task>, acc6>);
#pragma GCC diagnostic pop

			using acc7 = decltype(accessor{std::declval<buf0d_t>(), std::declval<handler&>(), all(), celerity::read_only_host_task});
			STATIC_REQUIRE(std::is_same_v<accessor<int, 0, access_mode::read, target::host_task>, acc7>);

			using acc8 = decltype(accessor{std::declval<buf0d_t>(), std::declval<handler&>(), all(), celerity::write_only_host_task, celerity::no_init});
			STATIC_REQUIRE(std::is_same_v<accessor<int, 0, access_mode::discard_write, target::host_task>, acc8>);

			// This currently throws an error at runtime, because we cannot infer whether the access is a discard_* from the property list parameter.
			using acc9 = decltype(accessor{std::declval<buf0d_t>(), std::declval<handler&>(), celerity::write_only_host_task, celerity::property_list{}});
			STATIC_REQUIRE(std::is_same_v<accessor<int, 0, access_mode::write, target::host_task>, acc9>);

			using acc10 = decltype(accessor{std::declval<buf0d_t>(), std::declval<handler&>(), celerity::read_only_host_task});
			STATIC_REQUIRE(std::is_same_v<accessor<int, 0, access_mode::read, target::host_task>, acc10>);

			using acc11 = decltype(accessor{std::declval<buf0d_t>(), std::declval<handler&>(), celerity::write_only_host_task, celerity::no_init});
			STATIC_REQUIRE(std::is_same_v<accessor<int, 0, access_mode::discard_write, target::host_task>, acc11>);
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
		// #if __SYCL_DEVICE_ONLY__ did get rid of the segfault, but caused the test to fail with a heap corruption at runtime. Instead, replacing id
		// with size_t seems to resolve the problem.

		const auto range = test_utils::truncate_range<Dims>({2, 3, 4});
		auto& bm = accessor_fixture<Dims>::get_buffer_manager();
		auto bid = bm.template register_buffer<size_t, Dims>(range_cast<3>(range));

		auto& q = accessor_fixture<Dims>::get_device_queue();
		auto sr = subrange<3>({}, range_cast<3>(range));

		// this kernel initializes the buffer what will be read after.
		auto acc_write = accessor_fixture<Dims>::template get_device_accessor<size_t, Dims, access_mode::discard_write>(bid, range, {});
		test_utils::run_parallel_for<class kernel_multi_dim_accessor_write_<Dims>>(accessor_fixture<Dims>::get_device_queue().get_sycl_queue(),
		    range, {}, [=](celerity::item<Dims> item) { acc_write[item] = item.get_linear_id(); });

		SECTION("for device buffers") {
			auto acc_read = accessor_fixture<Dims>::template get_device_accessor<size_t, Dims, access_mode::read>(bid, range, {});
			auto acc = accessor_fixture<Dims>::template get_device_accessor<size_t, Dims, access_mode::discard_write>(bid, range, {});
			test_utils::run_parallel_for<class kernel_multi_dim_accessor_read_<Dims>>(
			    accessor_fixture<Dims>::get_device_queue().get_sycl_queue(), range, {}, [=](celerity::item<Dims> item) {
				    size_t i = item[0];
				    size_t j = item[1];
				    if constexpr(Dims == 2) {
					    acc[i][j] = acc_read[i][j];
				    } else {
					    size_t k = item[2];
					    acc[i][j][k] = acc_read[i][j][k];
				    }
			    });
		}

		SECTION("for host buffers") {
			auto acc_read = accessor_fixture<Dims>::template get_host_accessor<size_t, Dims, access_mode::read>(bid, range, {});
			auto acc = accessor_fixture<Dims>::template get_host_accessor<size_t, Dims, access_mode::discard_write>(bid, range, {});
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
		bool acc_check = accessor_fixture<Dims>::template buffer_reduce<size_t, Dims, class check_multi_dim_accessor<Dims>>(bid, tgt, range,
		    {}, true, [range = range](id<Dims> idx, bool current, size_t value) { return current && value == get_linear_index(range, idx); });

		REQUIRE(acc_check);
	}

	TEST_CASE_METHOD(test_utils::runtime_fixture, "conflicts between producer-accessors and reductions are reported", "[task-manager]") {
		runtime::init(nullptr, nullptr);
		auto& tm = runtime::get_instance().get_task_manager();
		test_utils::mock_buffer_factory mbf{tm};
		test_utils::mock_reduction_factory mrf;

		auto buf_0 = mbf.create_buffer(range<1>{1});

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
		q.submit([&](handler& cgh) {
			// access with offset == buffer range just to mess with things
			const auto offset = id(test_buf.get_range());
			const auto test_acc = test_buf.get_access<Mode>(cgh, [=](chunk<1>) { return subrange<1>{offset, 0}; });
			const auto verify_acc = verify_buf.get_access<access_mode::write>(cgh, one_to_one{});
			cgh.parallel_for<empty_access_kernel<Mode>>(range<1>{1}, [=](item<1>) {
				(void)test_acc;
				verify_acc[0] = true;
			});
		});
		q.submit([&](handler& cgh) {
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
		buffer<char, 1> buf1d(memory1d.data(), range<1>(10));

		q.submit([&](handler& cgh) {
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

		q.submit([&](handler& cgh) {
			accessor b{buf1d, cgh, one_to_one{}, celerity::write_only_host_task, celerity::no_init};
			cgh.host_task(range<1>(6), id<1>(2), [=](partition<1> part) {
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
		buffer<char, 2> buf2d(memory2d.data(), range<2>(10, 10));

		q.submit([&](handler& cgh) {
			accessor b{buf2d, cgh, one_to_one{}, celerity::write_only_host_task, celerity::no_init};
			cgh.host_task(range<2>(5, 6), id<2>(1, 2), [=](partition<2> part) {
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
		buffer<char, 3> buf3d(memory3d.data(), range<3>(10, 10, 10));

		q.submit([&](handler& cgh) {
			accessor b{buf3d, cgh, one_to_one{}, celerity::write_only_host_task, celerity::no_init};
			cgh.host_task(range<3>(5, 6, 7), id<3>(1, 2, 3), [=](partition<3> part) {
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
		q.submit([&](handler& cgh) {
			accessor acc_0(buf_0, cgh, read_write_host_task);
			cgh.host_task(on_master_node, [=] {
				CHECK(acc_0 == value_a);
				CHECK(*acc_0 == value_a);
				CHECK(*acc_0.operator->() == value_a);
				CHECK(acc_0[id<0>()] == value_a);
				acc_0[id<0>()] = value_b;
				*acc_0 = value_b;
			});
		});
		q.submit([&](handler& cgh) {
			accessor acc_0(buf_0, cgh, read_only);
			accessor acc_1(buf_1, cgh, one_to_one(), write_only, no_init);
			cgh.parallel_for<class UKN(device)>(buf_1.get_range(), [=](item<1> it) {
				acc_1[it] = acc_0;
				acc_1[it] = *acc_0;
				acc_1[it] = *acc_0.operator->();
				acc_1[it] = acc_0[id<0>()];
			});
		});
		q.submit([&](handler& cgh) {
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
		q.submit([&](handler& cgh) {
			accessor acc_0d(buf_0d, cgh, all(), write_only, no_init);
			accessor acc_1d(buf_1d, cgh, all(), write_only, no_init);
			accessor acc_2d(buf_2d, cgh, all(), write_only, no_init);
			accessor acc_3d(buf_3d, cgh, all(), write_only, no_init);
			cgh.parallel_for<class UKN(device)>(range<0>(), [=](item<0>) {
				acc_0d = 1;
				*acc_0d = 1;
				*acc_0d.operator->() = 1;
				acc_0d[id<0>()] = 1;
				acc_1d[99] = 2;
				acc_2d[9][9] = 3;
				acc_3d[4][4][4] = 4;
			});
		});
		q.submit([&](handler& cgh) {
			accessor acc_0d(buf_0d, cgh, all(), read_write_host_task);
			accessor acc_1d(buf_1d, cgh, all(), read_write_host_task);
			accessor acc_2d(buf_2d, cgh, all(), read_write_host_task);
			accessor acc_3d(buf_3d, cgh, all(), read_write_host_task);
			cgh.host_task(range<0>(), [=](partition<0>) {
				float& ref_0d = acc_0d;
				ref_0d += 1;
				*acc_0d += 2;
				*acc_0d.operator->() += 3;
				acc_0d[id<0>()] += 3;
				acc_1d[99] += 9;
				acc_2d[9][9] += 9;
				acc_3d[4][4][4] += 9;
			});
		});
		q.submit([&](handler& cgh) {
			accessor acc_0d(buf_0d, cgh, all(), read_only_host_task);
			accessor acc_1d(buf_1d, cgh, all(), read_only_host_task);
			accessor acc_2d(buf_2d, cgh, all(), read_only_host_task);
			accessor acc_3d(buf_3d, cgh, all(), read_only_host_task);
			cgh.host_task(on_master_node, [=] {
				CHECK(acc_0d == 10);
				CHECK(*acc_0d == 10);
				CHECK(*acc_0d.operator->() == 10);
				CHECK(acc_0d[id<0>()] == 10);
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
		q.submit([&](handler& cgh) {
			accessor acc_1(buf_1, cgh, one_to_one(), write_only, no_init);
			cgh.parallel_for<class UKN(device)>(buf_1.get_range(), [=](item<1> it) { acc_1[it] = value_a; });
		});
		q.submit([&](handler& cgh) {
			accessor acc_1(buf_1, cgh, one_to_one(), write_only);
			local_accessor<float, 0> local_0(cgh);
			cgh.parallel_for<class UKN(device)>(nd_range(buf_1.get_range(), buf_1.get_range()), [=](nd_item<1> it) {
				if(it.get_local_id() == 0) {
					local_0 = value_b;
					*local_0 = value_b;
					*local_0.operator->() = value_b;
					local_0[id<0>()] = value_b;
				}
				group_barrier(it.get_group());
				acc_1[it.get_global_id()] = local_0;
				acc_1[it.get_global_id()] = *local_0;
				acc_1[it.get_global_id()] = *local_0.operator->();
				acc_1[it.get_global_id()] = local_0[id<0>()];
			});
		});
		q.submit([&](handler& cgh) {
			accessor acc_1(buf_1, cgh, all(), read_only_host_task);
			cgh.host_task(on_master_node, [=] {
				for(size_t i = 0; i < buf_1.get_range().size(); ++i) {
					REQUIRE_LOOP(acc_1[i] == value_b);
				}
			});
		});
	}

	TEST_CASE_METHOD(test_utils::runtime_fixture, "accessors are default-constructible", "[accessor]") {
		buffer<float, 0> buf_0;
		buffer<float, 1> buf_1(1);

		distr_queue q;

		q.submit([&](handler& cgh) {
			accessor<float, 0, access_mode::discard_write, target::device> device_acc_0;
			accessor<float, 1, access_mode::discard_write, target::device> device_acc_1;
			local_accessor<float, 0> local_acc_0;
			local_accessor<float, 1> local_acc_1;

			device_acc_0 = decltype(device_acc_0)(buf_0, cgh, all());
			device_acc_1 = decltype(device_acc_1)(buf_1, cgh, all());
			local_acc_0 = decltype(local_acc_0)(cgh);
			local_acc_1 = decltype(local_acc_1)(1, cgh);
			cgh.parallel_for<class UKN(device_kernel_1)>(
			    nd_range(1, 1), [=](nd_item<1> /* it */) { (void)device_acc_0, (void)local_acc_0, (void)device_acc_1, (void)local_acc_1; });
		});

		q.submit([&](handler& cgh) {
			accessor<float, 0, access_mode::discard_write, target::host_task> host_acc_0;
			accessor<float, 1, access_mode::discard_write, target::host_task> host_acc_1;
			host_acc_0 = decltype(host_acc_0)(buf_0, cgh, all());
			host_acc_1 = decltype(host_acc_1)(buf_1, cgh, all());
			cgh.host_task(on_master_node, [=] { (void)host_acc_0, (void)host_acc_1; });
		});
	}

	TEST_CASE("0-dimensional accessors are pointer-sized", "[accessor]") {
		if(!CELERITY_DETAIL_HAS_NO_UNIQUE_ADDRESS) SKIP("[[no_unique_address]] not available on this compiler");
#if CELERITY_ACCESSOR_BOUNDARY_CHECK
		SKIP("no accessor size guarantees when CELERITY_ACCESSOR_BOUNDARY_CHECK=1.");
#endif

		// these checks are not static_asserts because they depend on an (optional) compiler layout optimization
		CHECK(sizeof(accessor<int, 0, access_mode::read, target::device>) == sizeof(int*));
		CHECK(sizeof(accessor<int, 0, access_mode::read, target::host_task>) == sizeof(int*));
	}

	TEST_CASE("0-dimensional local accessor has no overhead over SYCL", "[accessor]") {
		if(!CELERITY_DETAIL_HAS_NO_UNIQUE_ADDRESS) SKIP("[[no_unique_address]] not available on this compiler");

		// this check is not a static_assert because it depends on an (optional) compiler layout optimization
		CHECK(sizeof(local_accessor<int, 0>) == sizeof(accessor_testspy::declval_sycl_accessor<local_accessor<int, 0>>()));
	}

	TEST_CASE_METHOD(accessor_fixture<0>, "closure_hydrator provides correct pointers to host and device accessors", "[closure_hydrator][accessor]") {
		auto& bm = get_buffer_manager();
		auto bid = bm.register_buffer<size_t, 1>({100, 1, 1});
		auto& q = get_device_queue();

		SECTION("host accessor") {
			auto access_info = bm.access_host_buffer<size_t, 1>(bid, access_mode::discard_write, {{}, {100}});
			std::vector<closure_hydrator::accessor_info> infos;
			infos.push_back({access_info.ptr, access_info.backing_buffer_range, access_info.backing_buffer_offset, subrange<3>{{}, {100, 1, 1}}});
			auto acc = accessor_testspy::make_host_accessor<size_t, 1, access_mode::discard_write>(subrange<1>({}, {100}), hydration_id(1),
			    detail::id_cast<1>(access_info.backing_buffer_offset), detail::range_cast<1>(access_info.backing_buffer_range),
			    detail::range_cast<1>(bm.get_buffer_info(bid).range));
			CHECK(accessor_testspy::get_pointer(acc) != access_info.ptr);
			closure_hydrator::get_instance().arm(target::host_task, std::move(infos));
			const auto run_check = closure_hydrator::get_instance().hydrate<target::host_task>(
			    [&, hydrated_acc = acc] { CHECK(accessor_testspy::get_pointer(hydrated_acc) == access_info.ptr); });
			run_check();
		}

		SECTION("device accessor") {
			auto access_info = bm.access_device_buffer<size_t, 1>(bid, access_mode::discard_write, {{}, {100}});
			std::vector<closure_hydrator::accessor_info> infos;
			infos.push_back({access_info.ptr, access_info.backing_buffer_range, access_info.backing_buffer_offset, subrange<3>{{}, {100, 1, 1}}});
			auto acc = accessor_testspy::make_device_accessor<size_t, 1, access_mode::discard_write>(
			    hydration_id(1), id_cast<1>(access_info.backing_buffer_offset), detail::range_cast<1>(access_info.backing_buffer_range));
			CHECK(accessor_testspy::get_pointer(acc) != access_info.ptr);
			accessor<size_t, 1, access_mode::discard_write, target::device> hydrated_acc;
			closure_hydrator::get_instance().arm(target::device, std::move(infos));
			q.get_sycl_queue().submit([&](sycl::handler& cgh) {
				closure_hydrator::get_instance().hydrate<target::device>(cgh, [&hydrated_acc, acc]() { hydrated_acc = acc; })(/* call to hydrate */);
				cgh.single_task<class UKN(nop)>([] {});
			});
			CHECK(accessor_testspy::get_pointer(hydrated_acc) == access_info.ptr);
		}
	}

	TEST_CASE_METHOD(accessor_fixture<0>, "closure_hydrator correctly handles unused and duplicate accessors", "[closure_hydrator][accessor]") {
		auto& bm = get_buffer_manager();
		auto& q = get_device_queue();

		std::vector<closure_hydrator::accessor_info> infos;
		hydration_id next_hid = 1;
		auto create_accessor = [&](const buffer_id bid) {
			auto access_info = bm.access_host_buffer<size_t, 1>(bid, access_mode::discard_write, {{}, {10}});
			infos.push_back({access_info.ptr, access_info.backing_buffer_range, access_info.backing_buffer_offset, subrange<3>{{}, {10, 1, 1}}});
			auto acc = accessor_testspy::make_host_accessor<size_t, 1, access_mode::discard_write>(subrange<1>({}, {10}), next_hid++,
			    id_cast<1>(access_info.backing_buffer_offset), detail::range_cast<1>(access_info.backing_buffer_range),
			    detail::range_cast<1>(bm.get_buffer_info(bid).range));
			return std::pair{acc, access_info.ptr};
		};

		const auto bid1 = bm.register_buffer<size_t, 1>({10, 1, 1});
		[[maybe_unused]] const auto [acc1, ptr1] = create_accessor(bid1);
		const auto bid2 = bm.register_buffer<size_t, 1>({20, 1, 1});
		const auto [acc2, ptr2] = create_accessor(bid2);
		const auto bid3 = bm.register_buffer<size_t, 1>({30, 1, 1});
		[[maybe_unused]] const auto [acc3, ptr3] = create_accessor(bid3);
		const auto bid4 = bm.register_buffer<size_t, 1>({40, 1, 1});
		const auto [acc4, ptr4] = create_accessor(bid4);
		auto acc5 = acc4;

		auto closure = [acc2 = acc2, acc4 = acc4, acc5 = acc5] {
			return std::tuple{accessor_testspy::get_pointer(acc2), accessor_testspy::get_pointer(acc4), accessor_testspy::get_pointer(acc5)};
		};
		closure_hydrator::get_instance().arm(target::host_task, std::move(infos));
		auto hydrated_closure = closure_hydrator::get_instance().hydrate<target::host_task>(closure);
		CHECK(ptr2 == std::get<0>(hydrated_closure()));
		CHECK(ptr4 == std::get<1>(hydrated_closure()));
		CHECK(ptr4 == std::get<2>(hydrated_closure()));
	}

	TEST_CASE_METHOD(accessor_fixture<0>, "closure_hydrator correctly hydrates local_accessor", "[closure_hydrator][accessor][smoke-test]") {
		auto& q = get_device_queue();
		auto local_acc = accessor_testspy::make_local_accessor<size_t, 1>(range<1>(2));
		size_t* const result = sycl::malloc_device<size_t>(2, q.get_sycl_queue());
		auto closure = [=](sycl::nd_item<1> itm) {
			// We can't really check pointers or anything, so this is a smoke test
			local_acc[itm.get_local_id()] = 100 + itm.get_local_id(0) * 10;
			sycl::group_barrier(itm.get_group());
			// Write other item's value
			result[itm.get_local_id(0)] = local_acc[1 - itm.get_local_id()];
		};
		closure_hydrator::get_instance().arm(target::device, {});
		q.submit([&](sycl::handler& cgh) {
			 auto hydrated_closure = closure_hydrator::get_instance().hydrate<target::device>(cgh, closure);
			 cgh.parallel_for(sycl::nd_range<1>{{2}, {2}}, [=](sycl::nd_item<1> itm) { hydrated_closure(itm); });
		 }).wait_and_throw();
		size_t result_host[2];
		q.get_sycl_queue().memcpy(&result_host[0], result, 2 * sizeof(size_t)).wait_and_throw();
		CHECK(result_host[0] == 110);
		CHECK(result_host[1] == 100);
		sycl::free(result, q.get_sycl_queue());
	}

	template <int>
	class oob_fixture : public test_utils::runtime_fixture {};

	template <int>
	class acc_out_of_bounds_kernel {};

	TEMPLATE_TEST_CASE_METHOD_SIG(oob_fixture, "accessor reports out-of-bounds accesses", "[accessor][oob]", ((int Dims), Dims), 1, 2, 3) {
#if !CELERITY_ACCESSOR_BOUNDARY_CHECK
		SKIP("CELERITY_ACCESSOR_BOUNDARY_CHECK=0");
#endif
		buffer<int, Dims> buff(test_utils::truncate_range<Dims>({10, 20, 30}));
		const auto accessible_sr = test_utils::truncate_subrange<Dims>({{5, 10, 15}, {1, 2, 3}});
		const auto oob_idx_lo = test_utils::truncate_id<Dims>({1, 2, 3});
		const auto oob_idx_hi = test_utils::truncate_id<Dims>({7, 13, 25});

		// we need to be careful about the orderign of the construction and destruction
		// of the Celerity queue and the log capturing utility here
		std::unique_ptr<celerity::test_utils::log_capture> lc;
		{
			distr_queue q;

			lc = std::make_unique<celerity::test_utils::log_capture>();

			q.submit([&](handler& cgh) {
				accessor acc(buff, cgh, celerity::access::fixed(accessible_sr), celerity::write_only, celerity::no_init);
				cgh.parallel_for<acc_out_of_bounds_kernel<Dims>>(range<Dims>(ones), [=](item<Dims>) {
					acc[oob_idx_lo] = 0;
					acc[oob_idx_hi] = 0;
				});
			});
			q.slow_full_sync();
		}

		const auto attempted_sr = subrange<3>{id_cast<3>(oob_idx_lo), range_cast<3>(oob_idx_hi - oob_idx_lo + id_cast<Dims>(range<Dims>(ones)))};
		const auto error_message = fmt::format("Out-of-bounds access in kernel 'celerity::detail::acc_out_of_bounds_kernel<{}>' detected: Accessor 0 for "
		                                       "buffer 0 attempted to access indices between {} which are outside of mapped subrange {}",
		    Dims, attempted_sr, subrange_cast<3>(accessible_sr));
		CHECK_THAT(lc->get_log(), Catch::Matchers::ContainsSubstring(error_message));
	}

} // namespace detail
} // namespace celerity
