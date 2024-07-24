#include "sycl_wrappers.h"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <celerity.h>

#include "test_utils.h"

namespace celerity {
namespace detail {

	struct accessor_testspy {
		template <typename DataT, int Dims, access_mode Mode, typename... Args>
		static accessor<DataT, Dims, Mode, target::device> make_device_accessor(Args&&... args) {
			return {std::forward<Args>(args)...};
		}

		template <typename DataT, int Dims, access_mode Mode, typename... Args>
		static accessor<DataT, Dims, Mode, target::host_task> make_host_accessor(Args&&... args) {
			return {std::forward<Args>(args)...};
		}

		// It appears to be impossible to make a private member type visible through a typedef here, so we opt for a declval-like function declaration instead
		template <typename LocalAccessor>
		static typename LocalAccessor::sycl_accessor declval_sycl_accessor() {
			static_assert(constexpr_false<LocalAccessor>, "declval_sycl_accessor cannot be used in an evaluated context");
		}

		template <typename DataT, int Dims, typename... Args>
		static local_accessor<DataT, Dims> make_local_accessor(Args&&... args) {
			return local_accessor<DataT, Dims>{std::forward<Args>(args)...};
		}

		template <typename DataT, int Dims, access_mode Mode, target Tgt>
		static DataT* get_pointer(const accessor<DataT, Dims, Mode, Tgt>& acc) {
			if constexpr(Tgt == target::device) {
				return acc.m_device_ptr;
			} else {
				return acc.m_host_ptr;
			}
		}
	};

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
	class kernel_multi_dim_accessor_write_;

	template <int>
	class kernel_multi_dim_accessor_read_;

	template <int>
	class check_multi_dim_accessor;

	TEMPLATE_TEST_CASE_METHOD_SIG(
	    test_utils::runtime_fixture_dims, "accessor supports multi-dimensional subscript operator", "[accessor]", ((int Dims), Dims), 2, 3) {
		distr_queue q;

		const auto range = test_utils::truncate_range<Dims>({2, 3, 4});
		buffer<size_t, Dims> buf_in(range);
		buffer<size_t, Dims> buf_out(range);

		q.submit([&](handler& cgh) {
			accessor acc_write(buf_in, cgh, one_to_one(), write_only, no_init);
			cgh.parallel_for(range, [=](celerity::item<Dims> item) { acc_write[item] = item.get_linear_id(); });
		});

		SECTION("for device accessors") {
			q.submit([&](handler& cgh) {
				accessor acc_read(buf_in, cgh, one_to_one(), read_only);
				accessor acc_write(buf_out, cgh, one_to_one(), write_only, no_init);
				cgh.parallel_for(range, [=](celerity::item<Dims> item) {
					size_t i = item[0];
					size_t j = item[1];
					if constexpr(Dims == 2) {
						acc_write[i][j] = acc_read[i][j];
					} else {
						size_t k = item[2];
						acc_write[i][j][k] = acc_read[i][j][k];
					}
				});
			});
		}

		SECTION("for host accessors") {
			q.submit([&](handler& cgh) {
				accessor acc_read(buf_in, cgh, one_to_one(), read_only_host_task);
				accessor acc_write(buf_out, cgh, one_to_one(), write_only_host_task, no_init);
				cgh.host_task(range, [=](celerity::partition<Dims> part) {
					experimental::for_each_item(range, [&](celerity::item<Dims> item) {
						size_t i = item[0];
						size_t j = item[1];
						if constexpr(Dims == 2) {
							acc_write[i][j] = acc_read[i][j];
						} else {
							size_t k = item[2];
							acc_write[i][j][k] = acc_read[i][j][k];
						}
					});
				});
			});
		}

		const auto result = q.fence(buf_out).get();
		for(size_t i = 0; i < range.size(); ++i) {
			REQUIRE_LOOP(result.get_data()[i] == i);
		}
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
				CHECK(aw.get_allocation_range()[0] >= 5);
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
				// accessor<T, 0, read> is implicitly convertible to const T&, but due to a bug in GCC, that conversion is not considered in overload resolution
				// for operator==, so we implicitly convert by assigning to a variable (https://gcc.gnu.org/bugzilla/show_bug.cgi?id=113052)
				const float val = acc_0;
				CHECK(val == value_a);

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
			cgh.host_task(on_master_node, [=, range = buf_1.get_range()] {
				for(size_t i = 0; i < range.size(); ++i) {
					REQUIRE_LOOP(acc_1[i] == value_b);
				}
			});
		});
	}

	TEST_CASE_METHOD(test_utils::runtime_fixture, "0-dimensional kernels can access arbitrary-dimensional buffers", "[buffer]") {
		buffer<float, 0> buf_0d;
		buffer<float, 1> buf_1d(100);
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
				// accessor<T, 0, read> is implicitly convertible to const T&, but due to a bug in GCC, that conversion is not considered in overload resolution
				// for operator==, so we implicitly convert by assigning to a variable (https://gcc.gnu.org/bugzilla/show_bug.cgi?id=113052)
				const float val = acc_0d;
				CHECK(val == 10);
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
			cgh.host_task(on_master_node, [=, range = buf_1.get_range()] {
				for(size_t i = 0; i < range.size(); ++i) {
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

	TEST_CASE("closure_hydrator provides correct pointers to host and device accessors", "[closure_hydrator][accessor]") {
		const auto allocation = reinterpret_cast<void*>(0x12345abcdef);
		const range<3> buffer_range(30, 29, 28);
		const id<3> allocation_offset_in_buffer(4, 8, 12);
		const range<3> allocation_range(5, 7, 9);
		const box<3> box = subrange(allocation_offset_in_buffer, allocation_range);
		const std::vector<closure_hydrator::accessor_info> infos{{allocation, box, box}};

		SECTION("host accessor") {
			auto acc = accessor_testspy::make_host_accessor<size_t, 3, access_mode::discard_write>(
			    box.get_subrange(), hydration_id(1), allocation_offset_in_buffer, allocation_range, buffer_range);
			CHECK(accessor_testspy::get_pointer(acc) != allocation);
			closure_hydrator::get_instance().arm(target::host_task, infos);
			const auto hydrated_closure =
			    closure_hydrator::get_instance().hydrate<target::host_task>([&, hydrated_acc = acc] { return accessor_testspy::get_pointer(hydrated_acc); });
			CHECK(hydrated_closure() == allocation);
		}

		SECTION("device accessor") {
			auto acc =
			    accessor_testspy::make_device_accessor<size_t, 3, access_mode::discard_write>(hydration_id(1), allocation_offset_in_buffer, allocation_range);
			CHECK(accessor_testspy::get_pointer(acc) != allocation);
			accessor<size_t, 3, access_mode::discard_write, target::device> hydrated_acc;
			closure_hydrator::get_instance().arm(target::device, infos);
			test_utils::sycl_queue_fixture()
			    .get_sycl_queue()
			    .submit([&](sycl::handler& cgh) {
				    closure_hydrator::get_instance().hydrate<target::device>(cgh, [&hydrated_acc, acc]() { hydrated_acc = acc; })(/* call to hydrate */);
				    cgh.single_task<class UKN(nop)>([] {});
			    })
			    .wait();
			CHECK(accessor_testspy::get_pointer(hydrated_acc) == allocation);
		}
	}

	TEST_CASE("closure_hydrator correctly handles unused and duplicate accessors", "[closure_hydrator][accessor]") {
		hydration_id next_hid = 1;
		std::vector<closure_hydrator::accessor_info> infos;
		auto create_accessor = [&](const test_utils::mock_buffer<1>& buf) {
			const auto allocation = reinterpret_cast<void*>(0x12345abc000 + buf.get_id());
			const subrange<1> sr(0, 10);
			infos.push_back({allocation, subrange_cast<3>(sr), subrange_cast<3>(sr)});
			auto acc = accessor_testspy::make_host_accessor<size_t, 1, access_mode::discard_write>(sr, next_hid++, sr.offset, sr.range, buf.get_range());
			return std::pair{acc, allocation};
		};

		test_utils::mock_buffer_factory mbf;
		const auto buf1 = mbf.create_buffer<1>(10);
		[[maybe_unused]] const auto [acc1, ptr1] = create_accessor(buf1);
		const auto buf2 = mbf.create_buffer<1>(20);
		const auto [acc2, ptr2] = create_accessor(buf2);
		const auto buf3 = mbf.create_buffer<1>(30);
		[[maybe_unused]] const auto [acc3, ptr3] = create_accessor(buf3);
		const auto buf4 = mbf.create_buffer<1>(40);
		const auto [acc4, ptr4] = create_accessor(buf4);
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

	TEST_CASE_METHOD(test_utils::sycl_queue_fixture, "closure_hydrator correctly hydrates local_accessor", "[closure_hydrator][accessor][smoke-test]") {
		auto local_acc = accessor_testspy::make_local_accessor<size_t, 1>(range<1>(2));
		size_t* const result = sycl::malloc_device<size_t>(2, get_sycl_queue());
		auto closure = [=](sycl::nd_item<1> itm) {
			// We can't really check pointers or anything, so this is a smoke test
			local_acc[itm.get_local_id()] = 100 + itm.get_local_id(0) * 10;
			sycl::group_barrier(itm.get_group());
			// Write other item's value
			result[itm.get_local_id(0)] = local_acc[1 - itm.get_local_id(0)];
		};
		closure_hydrator::get_instance().arm(target::device, {});
		get_sycl_queue()
		    .submit([&](sycl::handler& cgh) {
			    auto hydrated_closure = closure_hydrator::get_instance().hydrate<target::device>(cgh, closure);
			    cgh.parallel_for(sycl::nd_range<1>{{2}, {2}}, [=](sycl::nd_item<1> itm) { hydrated_closure(itm); });
		    })
		    .wait_and_throw();
		size_t result_host[2];
		get_sycl_queue().memcpy(&result_host[0], result, 2 * sizeof(size_t)).wait_and_throw();
		CHECK(result_host[0] == 110);
		CHECK(result_host[1] == 100);
		sycl::free(result, get_sycl_queue());
	}

	TEMPLATE_TEST_CASE_METHOD_SIG(
	    test_utils::runtime_fixture_dims, "device accessor reports out-of-bounds accesses", "[accessor][oob]", ((int Dims), Dims), 1, 2, 3) {
#if !CELERITY_ACCESSOR_BOUNDARY_CHECK
		SKIP("CELERITY_ACCESSOR_BOUNDARY_CHECK=0");
#endif
		test_utils::allow_max_log_level(spdlog::level::err);

		distr_queue q;

		buffer<int, Dims> unnamed_buff(test_utils::truncate_range<Dims>({10, 20, 30}));
		buffer<int, Dims> named_buff(test_utils::truncate_range<Dims>({10, 20, 30}));
		const auto accessible_sr = test_utils::truncate_subrange<Dims>({{5, 10, 15}, {1, 2, 3}});
		const auto oob_idx_lo = test_utils::truncate_id<Dims>({1, 2, 3});
		const auto oob_idx_hi = test_utils::truncate_id<Dims>({7, 13, 25});
		const auto buffer_name = "oob_buffer";
		const auto task_name = "oob_task";

		celerity::debug::set_buffer_name(named_buff, buffer_name);

		q.submit([&](handler& cgh) {
			debug::set_task_name(cgh, task_name);
			accessor unnamed_acc(unnamed_buff, cgh, celerity::access::fixed(accessible_sr), celerity::write_only, celerity::no_init);
			accessor named_acc(named_buff, cgh, celerity::access::fixed(accessible_sr), celerity::write_only, celerity::no_init);

			cgh.parallel_for(range<Dims>(ones), [=](item<Dims>) {
				unnamed_acc[oob_idx_lo] = 0;
				unnamed_acc[oob_idx_hi] = 0;

				named_acc[oob_idx_lo] = 0;
				named_acc[oob_idx_hi] = 0;
			});
		});
		q.slow_full_sync();

		const auto accessible_box = box(subrange_cast<3>(accessible_sr));
		const auto attempted_box = box_cast<3>(box(oob_idx_lo, oob_idx_hi + id<Dims>(ones)));
		const auto unnamed_error_message = fmt::format("Out-of-bounds access detected in device kernel T1 \"{}\": accessor 0 attempted to access buffer B0 "
		                                               "indicies between {} and outside the declared range {}.",
		    task_name, attempted_box, accessible_box);
		CHECK(test_utils::log_contains_substring(log_level::err, unnamed_error_message));

		const auto named_error_message = fmt::format("Out-of-bounds access detected in device kernel T1 \"{}\": accessor 1 attempted to access buffer B1 "
		                                             "\"{}\" indicies between {} and outside the declared range {}.",
		    task_name, buffer_name, attempted_box, accessible_box);
		CHECK(test_utils::log_contains_substring(log_level::err, named_error_message));
	}

	TEMPLATE_TEST_CASE_METHOD_SIG(
	    test_utils::runtime_fixture_dims, "host accessor reports out-of-bounds accesses", "[accessor][oob]", ((int Dims), Dims), 1, 2, 3) {
#if !CELERITY_ACCESSOR_BOUNDARY_CHECK
		SKIP("CELERITY_ACCESSOR_BOUNDARY_CHECK=0");
#endif
		test_utils::allow_max_log_level(spdlog::level::err);

		distr_queue q;

		buffer<int, Dims> unnamed_buff(test_utils::truncate_range<Dims>({10, 20, 30}));
		buffer<int, Dims> named_buff(test_utils::truncate_range<Dims>({10, 20, 30}));
		const auto accessible_sr = test_utils::truncate_subrange<Dims>({{5, 10, 15}, {1, 2, 3}});
		const auto oob_idx_lo = test_utils::truncate_id<Dims>({1, 2, 3});
		const auto oob_idx_hi = test_utils::truncate_id<Dims>({7, 13, 25});
		const auto buffer_name = "oob_buffer";
		const auto task_name = "oob_task";

		celerity::debug::set_buffer_name(named_buff, buffer_name);

		q.submit([&](handler& cgh) {
			debug::set_task_name(cgh, task_name);
			accessor unnamed_acc(unnamed_buff, cgh, celerity::access::fixed(accessible_sr), celerity::write_only_host_task, celerity::no_init);
			accessor nambed_acc(named_buff, cgh, celerity::access::fixed(accessible_sr), celerity::write_only_host_task, celerity::no_init);

			cgh.host_task(range<Dims>(ones), [=](partition<Dims>) {
				unnamed_acc[oob_idx_lo] = 0;
				unnamed_acc[oob_idx_hi] = 0;

				nambed_acc[oob_idx_lo] = 0;
				nambed_acc[oob_idx_hi] = 0;
			});
		});

		q.slow_full_sync();

		const auto accessible_box = box(subrange_cast<3>(accessible_sr));
		const auto attempted_box = box_cast<3>(box(oob_idx_lo, oob_idx_hi + id<Dims>(ones)));
		const auto unnamed_error_message = fmt::format("Out-of-bounds access detected in host-compute task T1 \"{}\": accessor 0 attempted to access buffer B0 "
		                                               "indicies between {} and outside the declared range {}.",
		    task_name, attempted_box, accessible_box);
		CHECK(test_utils::log_contains_substring(log_level::err, unnamed_error_message));

		const auto named_error_message = fmt::format("Out-of-bounds access detected in host-compute task T1 \"{}\": accessor 1 attempted to access buffer B1 "
		                                             "\"{}\" indicies between {} and outside the declared range {}.",
		    task_name, buffer_name, attempted_box, accessible_box);
		CHECK(test_utils::log_contains_substring(log_level::err, named_error_message));
	}

	TEST_CASE_METHOD(test_utils::sycl_queue_fixture, "accessor correctly handles backing buffer offsets", "[accessor]") {
		const box<2> allocation({16, 0}, {64, 32});
		const auto kernel_range = range<2>(32, 32);
		const auto kernel_offset = id<2>(32, 0);

		const auto ptr = sycl::malloc_host<size_t>(allocation.get_area(), get_sycl_queue());

		SECTION("when using device accessors") {
			const auto acc =
			    detail::accessor_testspy::make_device_accessor<size_t, 2, access_mode::discard_write>(ptr, allocation.get_offset(), allocation.get_range());

			parallel_for(kernel_range, kernel_offset, [=](id<2> id) { acc[id] = id[0] + id[1]; });

			for(size_t i = 32; i < 64; ++i) {
				for(size_t j = 0; j < 32; ++j) {
					REQUIRE_LOOP(ptr[(i - allocation.get_offset()[0]) * 32 + j - allocation.get_offset()[1]] == i + j);
				}
			}
		}

		SECTION("when using host accessors") {
			const auto acc = detail::accessor_testspy::make_host_accessor<size_t, 2, access_mode::discard_write>(
			    subrange<2>(kernel_offset, kernel_range), ptr, allocation.get_offset(), allocation.get_range(), kernel_range);
			for(size_t i = 32; i < 64; ++i) {
				for(size_t j = 0; j < 32; ++j) {
					acc[{i, j}] = i + j;
				}
			}
			for(size_t i = 32; i < 64; ++i) {
				for(size_t j = 0; j < 32; ++j) {
					REQUIRE_LOOP(ptr[(i - allocation.get_offset()[0]) * 32 + j - allocation.get_offset()[1]] == i + j);
				}
			}
		}

		sycl::free(ptr, get_sycl_queue());
	}

	TEST_CASE_METHOD(test_utils::sycl_queue_fixture, "accessor supports SYCL special member and hidden friend functions", "[accessor]") {
		constexpr static range<1> range(32);
		constexpr static id<1> offset(0);
		constexpr static subrange sr(offset, range);

		constexpr auto make_all_device_accessor = [](size_t* ptr) {
			return detail::accessor_testspy::make_device_accessor<size_t, 1, access_mode::discard_write>(ptr, offset, range);
		};
		constexpr auto make_all_host_accessor = [](size_t* ptr) {
			return detail::accessor_testspy::make_host_accessor<size_t, 1, access_mode::discard_write>(sr, ptr, offset, range, range);
		};

		test_utils::mock_buffer_factory mbf;
		auto [buf_a, ptr_a] = std::pair{mbf.create_buffer<1>(range), sycl::malloc_host<size_t>(32, get_sycl_queue())};
		auto [buf_b, ptr_b] = std::pair{mbf.create_buffer<1>(range), sycl::malloc_host<size_t>(32, get_sycl_queue())};
		auto [buf_c, ptr_c] = std::pair{mbf.create_buffer<1>(range), sycl::malloc_host<size_t>(32, get_sycl_queue())};
		auto [buf_d, ptr_d] = std::pair{mbf.create_buffer<1>(range), sycl::malloc_host<size_t>(32, get_sycl_queue())};

		SECTION("when using device accessors") {
			// For device accessors we test this both on host and device

			// Copy ctor
			auto device_acc_a = make_all_device_accessor(ptr_a);
			decltype(device_acc_a) device_acc_a1(device_acc_a);

			// Move ctor
			auto device_acc_b = make_all_device_accessor(ptr_b);
			decltype(device_acc_b) device_acc_b1(std::move(device_acc_b));

			// Copy assignment
			auto device_acc_c = make_all_device_accessor(ptr_c);
			auto device_acc_c1 = make_all_device_accessor(ptr_a);
			device_acc_c1 = device_acc_c;

			// Move assignment
			auto device_acc_d = make_all_device_accessor(ptr_d);
			auto device_acc_d1 = make_all_device_accessor(ptr_a);
			device_acc_d1 = std::move(device_acc_d);

			// Hidden friends (equality operators)
			REQUIRE(device_acc_a == device_acc_a1);
			REQUIRE(device_acc_a1 != device_acc_b1);

			parallel_for(range, offset, [=](id<1> id) {
				// Copy ctor
				decltype(device_acc_a1) device_acc_a2(device_acc_a1);
				device_acc_a2[id] = 1 * id[0];

				// Move ctor
				decltype(device_acc_b1) device_acc_b2(std::move(device_acc_b1));
				device_acc_b2[id] = 2 * id[0];

				// Copy assignment
				auto device_acc_c2 = device_acc_a1;
				device_acc_c2 = device_acc_c1;
				device_acc_c2[id] = 3 * id[0];

				// Move assignment
				auto device_acc_d2 = device_acc_a1;
				device_acc_d2 = std::move(device_acc_d1);
				device_acc_d2[id] = 4 * id[0];

				// Hidden friends (equality operators) are only required to be defined on the host
			});
		}

		SECTION("when using host buffers") {
			{
				// Copy ctor
				auto acc_a = make_all_host_accessor(ptr_a);
				decltype(acc_a) acc_a1(acc_a);

				// Move ctor
				auto acc_b = make_all_host_accessor(ptr_b);
				decltype(acc_b) acc_b1(std::move(acc_b));

				// Copy assignment
				auto acc_c = make_all_host_accessor(ptr_c);
				auto acc_c1 = make_all_host_accessor(ptr_a);
				acc_c1 = acc_c;

				// Move assignment
				auto acc_d = make_all_host_accessor(ptr_d);
				auto acc_d1 = make_all_host_accessor(ptr_a);
				acc_d1 = std::move(acc_d);

				// Hidden friends (equality operators)
				REQUIRE(acc_a == acc_a1);
				REQUIRE(acc_a1 != acc_b1);

				for(size_t i = 0; i < 32; ++i) {
					acc_a1[i] = 1 * i;
					acc_b1[i] = 2 * i;
					acc_c1[i] = 3 * i;
					acc_d1[i] = 4 * i;
				}
			}
		}

		for(size_t i = 0; i < 32; ++i) {
			REQUIRE_LOOP(ptr_a[i] == 1 * i);
			REQUIRE_LOOP(ptr_b[i] == 2 * i);
			REQUIRE_LOOP(ptr_c[i] == 3 * i);
			REQUIRE_LOOP(ptr_d[i] == 4 * i);
		}

		sycl::free(ptr_a, get_sycl_queue());
		sycl::free(ptr_b, get_sycl_queue());
		sycl::free(ptr_c, get_sycl_queue());
		sycl::free(ptr_d, get_sycl_queue());
	}

	TEMPLATE_TEST_CASE_SIG("host accessor supports get_pointer", "[accessor]", ((int Dims), Dims), 0, 1, 2, 3) {
		const auto ptr = reinterpret_cast<size_t*>(0x1234567890);
		const auto buffer_range = test_utils::truncate_range<Dims>({8, 8, 8});
		const auto alloc_box = box(subrange<Dims>(zeros, buffer_range));
		const auto accessed_box = alloc_box;

		const auto acc = detail::accessor_testspy::make_host_accessor<size_t, Dims, access_mode::discard_write>(
		    accessed_box.get_subrange(), ptr, alloc_box.get_offset(), alloc_box.get_range(), buffer_range);
		CHECK(acc.get_pointer() == ptr);
	}

	TEST_CASE("host accessor throws when calling get_pointer for a backing buffer with different stride or nonzero offset", "[accessor]") {
		constexpr auto get_pointer = [](const auto& buffer_range, const auto& alloc_box, const auto& accessed_box) {
			constexpr auto dims = std::remove_reference_t<decltype(buffer_range)>::dimensions;
			const auto ptr = reinterpret_cast<size_t*>(0x1234567890);
			const auto acc = detail::accessor_testspy::make_host_accessor<size_t, dims, access_mode::discard_write>(
			    accessed_box.get_subrange(), ptr, alloc_box.get_offset(), alloc_box.get_range(), buffer_range);
			return acc.get_pointer();
		};

		const std::string error_msg = "Buffer cannot be accessed with expected stride";

		// This is not allowed, as the backing buffer hasn't been allocated from offset 0, which means the pointer would point to offset 32.
		CHECK_THROWS_WITH(get_pointer(range(128), box<1>(32, 64), box<1>(32, 64)), error_msg);

		// This is fine, as the backing buffer has been resized to start from 0 now.
		CHECK_NOTHROW(get_pointer(range(128), box<1>(0, 64), box<1>(0, 64)));

		// This is now also okay, as the backing buffer starts at 0, and the pointer points to offset 0.
		// (Same semantics as SYCL accessor with offset, i.e., UB outside of requested range).
		CHECK_NOTHROW(get_pointer(range(128), box<1>(0, 64), box<1>(32, 64)));

		// In 2D (and 3D) it's trickier, as the stride of the backing buffer must also match what the user expects.
		// This is not allowed, even though the offset is 0.
		CHECK_THROWS_WITH(get_pointer(range(128, 128), box<2>({0, 0}, {64, 64}), box<2>({0, 0}, {64, 64})), error_msg);

		// This is allowed, as we request the full buffer.
		CHECK_NOTHROW(get_pointer(range(128, 128), box<2>({0, 0}, {128, 128}), box<2>({0, 0}, {128, 128})));

		// This is now allowed, as the backing buffer has the expected stride.
		CHECK_NOTHROW(get_pointer(range(128, 128), box<2>({0, 0}, {128, 128}), box<2>({0, 0}, {64, 64})));

		// Passing an offset is now also possible.
		CHECK_NOTHROW(get_pointer(range(128, 128), box<2>({0, 0}, {128, 128}), box<2>({32, 32}, {96, 96})));
	}

} // namespace detail
} // namespace celerity
