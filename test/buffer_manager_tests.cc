#include "sycl_wrappers.h"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <celerity.h>

#include "ranges.h"

#include "buffer_manager_test_utils.h"
#include "log_test_utils.h"

namespace celerity {
namespace detail {

	TEST_CASE_METHOD(test_utils::runtime_fixture, "buffer_manager allows buffer deallocation", "[buffer_manager][dealloc]") {
		distr_queue q;
		buffer_id b_id;
		auto& bm = runtime::get_instance().get_buffer_manager();
		auto& tm = runtime::get_instance().get_task_manager();
		constexpr int new_horizon_step = 2;
		tm.set_horizon_step(new_horizon_step);
		{
			celerity::buffer<int, 1> b(range<1>(128));
			b_id = celerity::detail::get_buffer_id(b);
			q.submit([&](celerity::handler& cgh) {
				celerity::accessor a{b, cgh, celerity::access::all(), celerity::write_only};
				cgh.parallel_for<class UKN(i)>(b.get_range(), [=](celerity::item<1> it) { (void)a; });
			});
			REQUIRE(bm.has_buffer(b_id));
		}
		celerity::buffer<int, 1> c(range<1>(128));
		// we need horizon_step_size * 3 + 1 tasks to generate the third horizon,
		// and one extra task to trigger the clean_up process
		for(int i = 0; i < (new_horizon_step * 3 + 2); i++) {
			q.submit([&](celerity::handler& cgh) {
				celerity::accessor a{c, cgh, celerity::access::all(), celerity::write_only};
				cgh.parallel_for<class UKN(i)>(c.get_range(), [=](celerity::item<1>) { (void)a; });
			});
			// this sync is inside the loop because otherwise there is a race between this thread and the executor informing the TDAG
			// of the executed horizons, meaning that task deletion is not guaranteed.
			q.slow_full_sync();
		}
		// require buffer b was indeed unregistered.
		REQUIRE_FALSE(bm.has_buffer(b_id));

		// TODO: check whether error was printed or not
		test_utils::maybe_print_graph(celerity::detail::runtime::get_instance().get_task_manager());
	}

	TEST_CASE_METHOD(test_utils::buffer_manager_fixture, "buffer_manager keeps track of buffers", "[buffer_manager]") {
		std::vector<std::pair<buffer_manager::buffer_lifecycle_event, buffer_id>> cb_calls;
		initialize([&](buffer_manager::buffer_lifecycle_event e, buffer_id bid) { cb_calls.push_back({e, bid}); });
		auto& bm = get_buffer_manager();

		REQUIRE_FALSE(bm.has_active_buffers());

		REQUIRE_FALSE(bm.has_buffer(0));
		bm.register_buffer<float, 1>({1024, 1, 1});
		REQUIRE(bm.has_buffer(0));
		REQUIRE(bm.has_active_buffers());
		REQUIRE(bm.get_buffer_info(0).range == range<3>{1024, 1, 1});
		REQUIRE(bm.get_buffer_info(0).is_host_initialized == false);
		REQUIRE(cb_calls.size() == 1);
		REQUIRE(cb_calls[0] == std::make_pair(buffer_manager::buffer_lifecycle_event::registered, buffer_id(0)));

		std::vector<float> host_buf(5 * 6 * 7);
		bm.register_buffer<float, 3>({5, 6, 7}, host_buf.data());
		REQUIRE(bm.has_buffer(1));
		REQUIRE(bm.get_buffer_info(1).range == range<3>{5, 6, 7});
		REQUIRE(bm.get_buffer_info(1).is_host_initialized == true);
		REQUIRE(cb_calls.size() == 2);
		REQUIRE(cb_calls[1] == std::make_pair(buffer_manager::buffer_lifecycle_event::registered, buffer_id(1)));

		bm.unregister_buffer(0);
		REQUIRE(cb_calls.size() == 3);
		REQUIRE(cb_calls[2] == std::make_pair(buffer_manager::buffer_lifecycle_event::unregistered, buffer_id(0)));
		REQUIRE(bm.has_active_buffers());

		bm.unregister_buffer(1);
		REQUIRE_FALSE(bm.has_active_buffers());
	}

	TEST_CASE_METHOD(test_utils::buffer_manager_fixture, "buffer_manager creates appropriately sized buffers as needed", "[buffer_manager]") {
		auto& bm = get_buffer_manager();
		auto bid = bm.register_buffer<float, 1>(range<3>(3072, 1, 1));

		auto run_test = [&](auto access_buffer) {
			auto buf_info = access_buffer(1024, 0);

			// Even though we registered the buffer with a size of 3072, the actual backing buffer is only 1024
			REQUIRE(buf_info.backing_buffer_range == range<3>(1024, 1, 1));

			// Requesting smaller portions of the buffer will re-use the existing backing buffer
			for(auto s = 512; s > 2; s >>= 2) {
				auto smallbuf_info = access_buffer(s, 0);
				REQUIRE_LOOP(smallbuf_info.ptr == buf_info.ptr);
			}

			// As long as we're not exceeding the original 1024 items, changing the offset will also re-use the backing buffer
			for(auto o = 512; o > 2; o >>= 2) {
				auto smallbuf_info = access_buffer(512, o);
				REQUIRE_LOOP(smallbuf_info.ptr == buf_info.ptr);
			}

			// If we however exceed the original 1024 by passing an offset, the backing buffer will be resized
			{
				auto buf_info = access_buffer(1024, 512);
				// Since the BM cannot discard the previous contents at offset 0, the new buffer includes them as well
				REQUIRE(buf_info.backing_buffer_range == range<3>(1024 + 512, 1, 1));
			}

			// Likewise, requesting a larger range will cause the backing buffer to be resized
			{
				auto buf_info = access_buffer(2048, 0);
				REQUIRE(buf_info.backing_buffer_range == range<3>(2048, 1, 1));
			}

			// Lastly, requesting a totally different (non-overlapping) sub-range will require the buffer to be resized
			// such that it contains both the previous and the new ranges.
			{
				auto buf_info = access_buffer(512, 2560);
				REQUIRE(buf_info.backing_buffer_range == range<3>(3072, 1, 1));
			}
		};

		SECTION("when using device buffers") {
			run_test([&bm, bid](size_t range, size_t offset) { return bm.access_device_buffer<float, 1>(bid, access_mode::read, {offset, range}); });
		}

		SECTION("when using host buffers") {
			run_test([&bm, bid](size_t range, size_t offset) { return bm.access_host_buffer<float, 1>(bid, access_mode::read, {offset, range}); });
		}
	}

	TEST_CASE_METHOD(test_utils::buffer_manager_fixture, "buffer_manager returns correct access offset for backing buffers larger than the requested range",
	    "[buffer_manager]") {
		auto& bm = get_buffer_manager();
		auto bid = bm.register_buffer<float, 1>(range<3>(2048, 1, 1));

		auto run_test = [&](auto access_buffer) {
			// The returned offset indicates where the backing buffer starts, relative to the virtual buffer.
			REQUIRE(access_buffer(1024, 1024).backing_buffer_offset == id<3>(1024, 0, 0));
			REQUIRE(access_buffer(1024, 512).backing_buffer_offset == id<3>(512, 0, 0));
			REQUIRE(access_buffer(1024, 1024).backing_buffer_offset == id<3>(512, 0, 0));
			REQUIRE(access_buffer(256, 1024).backing_buffer_offset == id<3>(512, 0, 0));
			REQUIRE(access_buffer(1024, 0).backing_buffer_offset == id<3>(0, 0, 0));
		};

		SECTION("when using device buffers") {
			run_test([&bm, bid](size_t range, size_t offset) { return bm.access_device_buffer<float, 1>(bid, access_mode::read, {offset, range}); });
		}

		SECTION("when using host buffers") {
			run_test([&bm, bid](size_t range, size_t offset) { return bm.access_host_buffer<float, 1>(bid, access_mode::read, {offset, range}); });
		}
	}

	TEST_CASE_METHOD(test_utils::buffer_manager_fixture, "buffer_manager retains existing data when resizing buffers", "[buffer_manager]") {
		auto& bm = get_buffer_manager();

		auto run_1_d_test = [&](access_target tgt) {
			auto bid = bm.register_buffer<size_t, 1>(range<3>(160, 1, 1));

			// Request a 64 element buffer at offset 32 and initialize it with known values.
			buffer_for_each<size_t, 1, access_mode::discard_write, class UKN(partial_init)>(
			    bid, tgt, {64}, {32}, [](id<1> idx, size_t& value) { value = idx[0]; });

			// Now request a 128 element buffer at offset 32, requiring the backing device buffer to be resized.
			{
				bool valid = buffer_reduce<size_t, 1, class UKN(check)>(bid, tgt, {128}, {32}, true, [](id<1> idx, bool current, size_t value) {
					if(idx[0] < 96) return current && value == idx[0];
					return current;
				});
				REQUIRE(valid);
			}

			// Finally, request 128 elements at offset 0, again requiring the backing device buffer to be resized.
			{
				bool valid = buffer_reduce<size_t, 1, class UKN(check)>(bid, tgt, {128}, {0}, true, [](id<1> idx, bool current, size_t value) {
					if(idx[0] >= 32 && idx[0] < 96) return current && value == idx[0];
					return current;
				});
				REQUIRE(valid);
			}
		};

		SECTION("when using 1D device buffers") { run_1_d_test(access_target::device); }
		SECTION("when using 1D host buffers") { run_1_d_test(access_target::host); }

		auto run_2_d_test = [&](access_target tgt) {
			auto bid = bm.register_buffer<size_t, 2>(range<3>(128, 128, 1));

			// Request a set of columns and initialize it with known values.
			buffer_for_each<size_t, 2, access_mode::discard_write, class UKN(partial_init)>(
			    bid, tgt, {128, 64}, {0, 64}, [](id<2> idx, size_t& value) { value = idx[0] * 100 + idx[1]; });

			// Now request a set of rows that partially intersect the columns from before, requiring the backing device buffer to be resized.
			{
				bool valid = buffer_reduce<size_t, 2, class UKN(check)>(bid, tgt, {64, 128}, {64, 0}, true, [](id<2> idx, bool current, size_t value) {
					if(idx[1] >= 64) return current && value == idx[0] * 100 + idx[1];
					return current;
				});
				REQUIRE(valid);
			}
		};

		SECTION("when using 2D device buffers") { run_2_d_test(access_target::device); }
		SECTION("when using 2D host buffers") { run_2_d_test(access_target::host); }

		// While the fix for bug that warranted adding a 2D test *should* also cover 3D buffers, it would be good to have a 3D test here as well.
		// TODO: Can we also come up with a good 3D case?
	}

	template <typename T>
	bool is_valid_buffer_test_mode_pattern(T value) {
		static_assert(sizeof(T) % sizeof(buffer_manager::test_mode_pattern) == 0);
		for(size_t i = 0; i < sizeof(T) / sizeof(buffer_manager::test_mode_pattern); ++i) {
			if(reinterpret_cast<decltype(buffer_manager::test_mode_pattern)*>(&value)[i] != buffer_manager::test_mode_pattern) { return false; }
		}
		return true;
	}

	TEST_CASE_METHOD(test_utils::buffer_manager_fixture, "buffer_manager does not retain existing data when resizing buffer using a pure producer access mode",
	    "[buffer_manager][performance]") {
		auto& bm = get_buffer_manager();
		auto bid = bm.register_buffer<size_t, 1>(range<3>(128, 1, 1));

		auto run_test = [&](access_target tgt, bool partial_overwrite) {
			// Initialize 64 element buffer at offset 0
			buffer_for_each<size_t, 1, access_mode::discard_write, class UKN(partial_init)>(
			    bid, tgt, {64}, {0}, [](id<1> idx, size_t& value) { value = 1337 + idx[0]; });

			// Resize it to 128 elements using a pure producer mode
			buffer_for_each<size_t, 1, access_mode::discard_write, class UKN(faux_overwrite)>(bid, tgt, {partial_overwrite == false ? size_t(128) : size_t(96)},
			    {partial_overwrite == false ? size_t(0) : size_t(32)}, [](id<1> idx, size_t& value) { /* NOP */ });

			// Verify that the original 64 elements have not been retained during the resizing (unless we did a partial overwrite)
			{
				bool valid = buffer_reduce<size_t, 1, class UKN(check)>(bid, tgt, {128}, {0}, true, [=](id<1> idx, bool current, size_t value) {
					if(partial_overwrite) {
						// If we did a partial overwrite, the first 32 elements should have been retained
						if(idx[0] < 32) return current && value == 1337 + idx[0];
					}
					if(idx[0] < 64) return current && is_valid_buffer_test_mode_pattern(value);
					return current;
				});
				REQUIRE(valid);
			}
		};

		SECTION("when using device buffers") { run_test(access_target::device, false); }
		SECTION("when using host buffers") { run_test(access_target::host, false); }

		SECTION("unless accessed range does not fully cover previous buffer size (using device buffers)") { run_test(access_target::device, true); }
		SECTION("unless accessed range does not fully cover previous buffer size (using host buffers)") { run_test(access_target::host, true); }
	}

	TEST_CASE_METHOD(test_utils::buffer_manager_fixture, "buffer_manager ensures coherence between device and host buffers", "[buffer_manager]") {
		auto& bm = get_buffer_manager();
		auto bid = bm.register_buffer<size_t, 1>(range<3>(512, 1, 1));

		auto run_test1 = [&](access_target tgt) {
			// Initialize first half of buffer on this side
			buffer_for_each<size_t, 1, access_mode::discard_write, class UKN(init_first_half)>(
			    bid, tgt, {256}, {0}, [](id<1> idx, size_t& value) { value = idx[0]; });

			// Initialize second half of buffer on other side
			buffer_for_each<size_t, 1, access_mode::discard_write, class UKN(init_second_half)>(
			    bid, get_other_target(tgt), {256}, {256}, [](id<1> idx, size_t& value) { value = (512 - idx[0]) * 2; });

			// Verify coherent full buffer is available on this side
			bool valid = buffer_reduce<size_t, 1, class UKN(check)>(bid, tgt, {512}, {0}, true,
			    [](id<1> idx, bool current, size_t value) { return current && value == (idx[0] < 256 ? idx[0] : (512 - idx[0]) * 2); });
			REQUIRE(valid);
		};

		SECTION("when writing separate parts on host and device, verifying on device") { run_test1(access_target::device); }
		SECTION("when writing separate parts on host and device, verifying on host") { run_test1(access_target::host); }

		// This test can be run in two slightly different variations, as overwriting a larger range incurs
		// a resize operation internally, which then leads to a somewhat different code path during the coherency update.
		auto run_test2 = [&](access_target tgt, size_t overwrite_range) {
			// Initialize on this side
			buffer_for_each<size_t, 1, access_mode::discard_write, class UKN(init)>(bid, tgt, {256}, {0}, [](id<1> idx, size_t& value) { value = idx[0]; });

			// Update (potentially larger portion, depending on `overwrite_range`) on other side
			buffer_for_each<size_t, 1, access_mode::read_write, class UKN(update)>(
			    bid, get_other_target(tgt), {overwrite_range}, {0}, [](id<1> idx, size_t& value) { value = (idx[0] < 256 ? value * 2 : 33); });

			// Verify result on this side
			bool valid = buffer_reduce<size_t, 1, class UKN(check)>(bid, tgt, {overwrite_range}, {0}, true, [](id<1> idx, bool current, size_t value) {
				if(idx[0] < 256) return current && value == idx[0] * 2;
				return current && value == 33;
			});
			REQUIRE(valid);
		};

		SECTION("when initializing on device, updating on host, verifying on device") { run_test2(access_target::device, 256); }
		SECTION("when initializing on host, updating on device, verifying on host") { run_test2(access_target::host, 256); }

		SECTION("when initializing on device, partially updating larger portion on host, verifying on device") { run_test2(access_target::device, 512); }
		SECTION("when initializing on host, partially updating larger portion on device, verifying on host") { run_test2(access_target::host, 512); }
	}

	TEST_CASE_METHOD(
	    test_utils::buffer_manager_fixture, "buffer_manager does not ensure coherence when access mode is pure producer", "[buffer_manager][performance]") {
		auto& bm = get_buffer_manager();
		auto bid = bm.register_buffer<size_t, 1>(range<3>(128, 1, 1));

		auto run_test = [&](access_target tgt) {
			// Initialize on other side
			buffer_for_each<size_t, 1, access_mode::discard_write, class UKN(init)>(
			    bid, get_other_target(tgt), {128}, {0}, [](id<1> idx, size_t& value) { value = 1337 + idx[0]; });

			// Overwrite on this side (but not really) using a pure producer mode
			buffer_for_each<size_t, 1, access_mode::discard_write, class UKN(faux_overwrite)>(bid, tgt, {128}, {0}, [](id<1> idx, size_t& value) { /* NOP */ });

			// Verify that buffer does not have initialized contents
			bool valid = buffer_reduce<size_t, 1, class UKN(check)>(
			    bid, tgt, {128}, {0}, true, [](id<1> idx, bool current, size_t value) { return current && is_valid_buffer_test_mode_pattern(value); });
			REQUIRE(valid);
		};

		SECTION("when initializing on host, verifying on device") { run_test(access_target::device); }
		SECTION("when initializing on device, verifying on host") { run_test(access_target::host); }
	}

	TEST_CASE_METHOD(test_utils::buffer_manager_fixture,
	    "buffer_manager correctly updates buffer versioning for pure producer accesses that do not require a resize", "[buffer_manager]") {
		auto& bm = get_buffer_manager();
		auto bid = bm.register_buffer<size_t, 1>(range<3>(128, 1, 1));

		auto run_test = [&](access_target tgt) {
			// Initialize on other side
			buffer_for_each<size_t, 1, access_mode::discard_write, class UKN(init)>(
			    bid, get_other_target(tgt), {128}, {0}, [](id<1> idx, size_t& value) { value = idx[0]; });

			// Read buffer on this side at both ends (but not in between), forcing a resize without full replication
			buffer_for_each<size_t, 1, access_mode::read, class UKN(force_copy1)>(bid, tgt, {1}, {0}, [](id<1> idx, size_t value) { /* NOP */ });
			buffer_for_each<size_t, 1, access_mode::read, class UKN(force_copy2)>(bid, tgt, {1}, {127}, [](id<1> idx, size_t value) { /* NOP */ });

			// Overwrite on this side using a pure producer mode, without requiring a resize
			buffer_for_each<size_t, 1, access_mode::discard_write, class UKN(overwrite)>(bid, tgt, {128}, {0}, [](id<1> idx, size_t& value) { value = 33; });

			// Verify that buffer contains new values
			{
				bool valid = buffer_reduce<size_t, 1, class UKN(check)>(
				    bid, tgt, {128}, {0}, true, [](id<1> idx, bool current, size_t value) { return current && value == 33; });
				REQUIRE(valid);
			}
		};

		SECTION("when using device buffers") { run_test(access_target::device); }
		SECTION("when using host buffers") { run_test(access_target::host); }
	}

	/**
	 * This test ensures that the BM doesn't generate superfluous H <-> D data transfers after coherence has already been established
	 * by a previous access. For this to work, we have to cheat a bit and update a buffer after a second access call, which is technically not allowed.
	 */
	TEST_CASE_METHOD(
	    test_utils::buffer_manager_fixture, "buffer_manager remembers coherency replications between consecutive accesses", "[buffer_manager][performance]") {
		auto& bm = get_buffer_manager();
		auto& dq = get_device_queue();

		auto bid = bm.register_buffer<size_t, 1>(range<3>(32, 1, 1));

		SECTION("when using device buffers") {
			size_t* host_ptr = nullptr;

			// Remember host buffer for later.
			{
				auto info = bm.access_host_buffer<size_t, 1>(bid, access_mode::discard_write, {0, 32});
				host_ptr = static_cast<size_t*>(info.ptr);
			}

			// Initialize buffer on host.
			buffer_for_each<size_t, 1, access_mode::discard_write, class UKN(init)>(
			    bid, access_target::host, {32}, {}, [](id<1> idx, size_t& value) { value = idx[0]; });

			// Read buffer on device. This makes the device buffer coherent with the host buffer.
			bm.access_device_buffer<size_t, 1>(bid, access_mode::read, {0, 32});

			// Here we cheat: We override the host data using the pointer we kept from before, without telling the BM (which is not allowed).
			for(size_t i = 0; i < 32; ++i) {
				host_ptr[i] = 33;
			}

			// Now access the buffer on device again for reading and writing. The buffer manager should realize that the newest version is already on the
			// device. After this, the device holds the newest version of the buffer.
			bm.access_device_buffer<size_t, 1>(bid, access_mode::read_write, {0, 32});

			// Verify that the data is still what we expect.
			{
				bool valid = buffer_reduce<size_t, 1, class UKN(check)>(
				    bid, access_target::host, {32}, {0}, true, [](id<1> idx, bool current, size_t value) { return current && value == idx[0]; });
				REQUIRE(valid);
			}

			// Finally, also check the other way round: Accessing the device buffer now doesn't generate a superfluous H -> D transfer.
			// First, we cheat again.
			for(size_t i = 0; i < 32; ++i) {
				host_ptr[i] = 34;
			}

			// Access device buffer. This should still contain the original data.
			{
				bool valid = buffer_reduce<size_t, 1, class UKN(check)>(
				    bid, access_target::device, {32}, {0}, true, [](id<1> idx, bool current, size_t value) { return current && value == idx[0]; });
				REQUIRE(valid);
			}
		}

		SECTION("when using host buffers") {
			size_t* device_ptr = nullptr;

			// Remember device buffer for later.
			{
				auto info = bm.access_device_buffer<size_t, 1>(bid, access_mode::discard_write, {0, 32});
				device_ptr = static_cast<size_t*>(info.ptr);
			}

			// Initialize buffer on device.
			buffer_for_each<size_t, 1, access_mode::discard_write, class UKN(init)>(
			    bid, access_target::device, {32}, {}, [](id<1> idx, size_t& value) { value = idx[0]; });

			// Read buffer on host. This makes the host buffer coherent with the device buffer.
			bm.access_host_buffer<size_t, 1>(bid, access_mode::read, {0, 32});

			// Here we cheat: We override the device data using the pointer we kept from before, without telling the BM (which is not allowed).
			dq.get_sycl_queue()
			    .submit([&](cl::sycl::handler& cgh) {
				    cgh.parallel_for<class UKN(overwrite_buf)>(sycl::range<1>(32), [=](sycl::item<1> item) { device_ptr[item[0]] = 33; });
			    })
			    .wait();

			// Now access the buffer on host again for reading and writing. The buffer manager should realize that the newest version is already on the
			// host. After this, the host holds the newest version of the buffer.
			bm.access_host_buffer<size_t, 1>(bid, access_mode::read_write, {0, 32});

			// Verify that the data is still what we expect.
			{
				auto info = bm.access_device_buffer<size_t, 1>(bid, access_mode::read, {0, 32});
				std::vector<size_t> tmp_host(info.backing_buffer_range.size());
				dq.get_sycl_queue().memcpy(tmp_host.data(), info.ptr, sizeof(size_t) * info.backing_buffer_range.size()).wait();
				for(size_t i = 0; i < 32; ++i) {
					REQUIRE_LOOP(tmp_host[i] == i);
				}
			}

			// Finally, also check the other way round: Accessing the host buffer now doesn't generate a superfluous D -> H transfer.
			// First, we cheat again.
			dq.get_sycl_queue()
			    .submit([&](cl::sycl::handler& cgh) {
				    cgh.parallel_for<class UKN(overwrite_buf)>(sycl::range<1>(32), [=](cl::sycl::item<1> item) { device_ptr[item[0]] = 34; });
			    })
			    .wait();

			// Access host buffer. This should still contain the original data.
			{
				auto info = bm.access_host_buffer<size_t, 1>(bid, access_mode::read, {0, 32});
				for(size_t i = 0; i < 32; ++i) {
					REQUIRE_LOOP(static_cast<size_t*>(info.ptr)[i] == i);
				}
			}
		}
	}

	TEST_CASE_METHOD(test_utils::buffer_manager_fixture, "buffer_manager retains data that exists on both host and device when resizing buffers",
	    "[buffer_manager][performance]") {
		auto& bm = get_buffer_manager();
		auto bid = bm.register_buffer<size_t, 1>(range<3>(128, 1, 1));

		auto run_test = [&](access_target tgt) {
			// Initialize full buffer on other side.
			buffer_for_each<size_t, 1, access_mode::discard_write, class UKN(init)>(
			    bid, get_other_target(tgt), {128}, {0}, [](id<1> idx, size_t& value) { value = idx[0]; });

			// Request the first half on this side for reading, so that after this, the first half will exist on both sides.
			{
				bool valid = buffer_reduce<size_t, 1, class UKN(check)>(
				    bid, tgt, {64}, {0}, true, [](id<1> idx, bool current, size_t value) { return current && value == idx[0]; });
				CHECK(valid);
				CHECK(get_backing_buffer_range<size_t, 1>(bid, tgt, {64}, {0}) == range<1>{64});
			}

			// Now request the second half on this side for reading.
			// Since the first half exists on both sides, technically there is no need to retain the previous buffer's contents.
			// While this causes the buffer to be larger than necessary, it saves us an extra transfer (and re-allocation) in the future,
			// in case we ever need the first half again on this side.
			// TODO: This is a time-memory tradeoff and something we might want to change at some point.
			//		 => In particular, if we won't need the first half ever again, this wastes both time and memory!
			{
				bool valid = buffer_reduce<size_t, 1, class UKN(check)>(
				    bid, tgt, {64}, {64}, true, [](id<1> idx, bool current, size_t value) { return current && value == idx[0]; });
				CHECK(valid);
				// Check that the buffer has been resized to accomodate both halves.
				REQUIRE(get_backing_buffer_range<size_t, 1>(bid, tgt, {64}, {64}) == range<1>{128});
			}
		};

		SECTION("when using device buffers") { run_test(access_target::device); }
		SECTION("when using host buffers") { run_test(access_target::host); }
	}

	// This test is in response to a bug that was caused by computing the region to be retained upon buffer resizing as the bounding box of the coherence
	// subrange as well as the old buffer range. While that works fine in 1D, in 2D (and 3D) it can introduce unnecessary H<->D coherence updates.
	// TODO: Ideally we'd also test this for 3D buffers
	TEST_CASE_METHOD(test_utils::buffer_manager_fixture, "buffer_manager does not introduce superfluous coherence updates when retaining 2D buffers",
	    "[buffer_manager][performance]") {
		auto& bm = get_buffer_manager();
		auto bid = bm.register_buffer<size_t, 2>(range<3>(128, 128, 1));

		auto run_test = [&](access_target tgt) {
			// Initialize whole buffer to known value on other side.
			buffer_for_each<size_t, 2, access_mode::discard_write, class UKN(init)>(
			    bid, get_other_target(tgt), {128, 128}, {0, 0}, [](id<2> idx, size_t& value) { value = 1337 * idx[0] + 42 + idx[1]; });

			// Request a set of columns on this side, causing a coherence update.
			{
				bool valid = buffer_reduce<size_t, 2, class UKN(check)>(
				    bid, tgt, {128, 64}, {0, 64}, true, [](id<2> idx, bool current, size_t value) { return current && value == 1337 * idx[0] + 42 + idx[1]; });
				CHECK(valid);
			}

			// Now request a set of rows that partially intersect the columns from before, requiring the backing buffer on this side to be resized.
			// This resizing should retain the columns, but not introduce an additional coherence update in the empty area not covered by
			// either the columns or rows (i.e., [0,0]-[64,64]).
			{
				bool valid = buffer_reduce<size_t, 2, class UKN(check)>(bid, tgt, {64, 128}, {64, 0}, true, [](id<2> idx, bool current, size_t value) {
					if(idx[1] >= 64) return current && value == 1337 * idx[0] + 42 + idx[1];
					return current;
				});
				CHECK(valid);
			}

			// Do a faux overwrite on this side to ensure that no coherence update will be done for the next call to buffer_reduce.
			buffer_for_each<size_t, 2, access_mode::discard_write, class UKN(faux_overwrite)>(
			    bid, tgt, {128, 128}, {0, 0}, [](id<2> idx, size_t& value) { /* NOP */ });

			// While the backing buffer also includes the [0,0]-[64,64] region, this part should still be uninitialized.
			{
				bool valid = buffer_reduce<size_t, 2, class UKN(check)>(bid, tgt, {64, 64}, {0, 0}, true,
				    [=](id<2> idx, bool current, size_t value) { return current && is_valid_buffer_test_mode_pattern(value); });
				REQUIRE(valid);
			}
		};

		SECTION("when using device buffers") { run_test(access_target::device); }
		SECTION("when using host buffers") { run_test(access_target::host); }
	}

	TEST_CASE_METHOD(test_utils::buffer_manager_fixture, "buffer_manager correctly updates buffer versioning for queued transfers", "[buffer_manager]") {
		auto& bm = get_buffer_manager();
		auto bid = bm.register_buffer<size_t, 1>(range<3>(64, 1, 1));

		auto run_test = [&](access_target tgt) {
			// Initialize buffer on the other side
			buffer_for_each<size_t, 1, access_mode::discard_write, class UKN(initialize)>(
			    bid, get_other_target(tgt), {64}, {0}, [](id<1>, size_t& value) { value = 33; });

			// Add transfer for second half on this side
			auto data = make_uninitialized_payload<size_t>(32);
			std::uninitialized_fill_n(static_cast<size_t*>(data.get_pointer()), 32, size_t{77});
			bm.set_buffer_data(bid, {{32, 0, 0}, {32, 1, 1}}, std::move(data));

			// Check that transfer has been correctly ingested
			{
				bool valid = buffer_reduce<size_t, 1, class UKN(check_second_half)>(
				    bid, tgt, {64}, {0}, true, [](id<1> idx, bool current, size_t value) { return current && (value == (idx[0] < 32 ? 33 : 77)); });
				REQUIRE(valid);
			}

			// Finally, check that accessing the other side now copies the transfer data as well
			{
				bool valid = buffer_reduce<size_t, 1, class UKN(check_second_half)>(bid, get_other_target(tgt), {64}, {0}, true,
				    [](id<1> idx, bool current, size_t value) { return current && (value == (idx[0] < 32 ? 33 : 77)); });
				REQUIRE(valid);
			}
		};

		SECTION("when using device buffers") { run_test(access_target::device); }
		SECTION("when using host buffers") { run_test(access_target::host); }
	}

	TEST_CASE_METHOD(test_utils::buffer_manager_fixture, "buffer_manager prioritizes queued transfers over resize/coherency copies for the same ranges",
	    "[buffer_manager]") {
		auto& bm = get_buffer_manager();
		auto bid = bm.register_buffer<size_t, 1>(range<3>(128, 1, 1));

		auto run_test = [&](access_target tgt, bool resize, bool coherency) {
			// Write first half of buffer.
			buffer_for_each<size_t, 1, access_mode::discard_write, class UKN(write_buf)>(
			    bid, coherency ? get_other_target(tgt) : tgt, {resize ? size_t(64) : size_t(128)}, {0}, [](id<1>, size_t& value) { value = 33; });

			// Set full range to new value.
			{
				auto data = make_uninitialized_payload<size_t>(128);
				std::uninitialized_fill_n(static_cast<size_t*>(data.get_pointer()), 128, size_t{77});
				bm.set_buffer_data(bid, {{0, 0, 0}, {128, 1, 1}}, std::move(data));
			}

			// Now read full range.
			{
				bool valid = buffer_reduce<size_t, 1, class UKN(check)>(
				    bid, tgt, {96}, {0}, true, [](id<1>, bool current, size_t value) { return current && value == 77; });
				REQUIRE(valid);
			}
		};

		SECTION("when initializing, resizing and verifying on device") { run_test(access_target::device, true, false); }
		SECTION("when initializing, resizing and verifying on host") { run_test(access_target::host, true, false); }

		SECTION("when initializing on host, verifying on device") { run_test(access_target::device, false, true); }
		SECTION("when initializing on device, verifying on host") { run_test(access_target::host, false, true); }
	}

	TEST_CASE_METHOD(test_utils::buffer_manager_fixture, "buffer_manager correctly handles transfers that partially overlap with requested buffer range",
	    "[buffer_manager]") {
		auto& bm = get_buffer_manager();

		auto bid = bm.register_buffer<size_t, 1>(range<3>(128, 1, 1));

		auto run_test = [&](access_target tgt) {
			// Initialize first half of buffer with linear index.
			buffer_for_each<size_t, 1, access_mode::discard_write, class UKN(init)>(bid, tgt, {64}, {0}, [](id<1> idx, size_t& value) { value = idx[0]; });

			// Set data that only partially overlaps with currently allocated range.
			{
				auto data = make_uninitialized_payload<size_t>(64);
				std::uninitialized_fill_n(static_cast<size_t*>(data.get_pointer()), 64, size_t{99});
				bm.set_buffer_data(bid, {{32, 0, 0}, {64, 1, 1}}, std::move(data));
			}

			// Check that second half of buffer has been updated...
			{
				bool valid = buffer_reduce<size_t, 1, class UKN(check)>(
				    bid, tgt, {64}, {0}, true, [](id<1> idx, bool current, size_t value) { return current && (value == (idx[0] < 32 ? idx[0] : 99)); });
				REQUIRE(valid);
			}

			// ...without changing its original size.
			REQUIRE(get_backing_buffer_range<size_t, 1>(bid, tgt, {64}, {0})[0] == 64);

			// Check that remainder of buffer has been updated as well.
			{
				bool valid = buffer_reduce<size_t, 1, class UKN(check)>(
				    bid, tgt, {96}, {0}, true, [](id<1> idx, bool current, size_t value) { return current && (value == (idx[0] < 32 ? idx[0] : 99)); });
				REQUIRE(valid);
			}
		};

		SECTION("when using device buffers") { run_test(access_target::device); }
		SECTION("when using host buffers") { run_test(access_target::host); }
	}

	TEST_CASE_METHOD(test_utils::buffer_manager_fixture, "buffer_manager returns the newest raw buffer data when requested", "[buffer_manager]") {
		auto& bm = get_buffer_manager();
		auto bid = bm.register_buffer<size_t, 1>(range<3>(32, 1, 1));

		auto run_test = [&](access_target tgt) {
			buffer_for_each<size_t, 1, access_mode::discard_write, class UKN(write_buffer)>(
			    bid, get_other_target(tgt), {32}, {0}, [](id<1> idx, size_t& value) { value = idx[0]; });
			buffer_for_each<size_t, 1, access_mode::read_write, class UKN(update_buffer)>(bid, tgt, {32}, {0}, [](id<1> idx, size_t& value) { value += 1; });
			std::vector<size_t> data(32);
			bm.get_buffer_data(bid, {{0, 0, 0}, {32, 1, 1}}, data.data());
			for(size_t i = 0; i < 32; ++i) {
				REQUIRE_LOOP(data[i] == i + 1);
			}
		};

		SECTION("when newest data is on device") { run_test(access_target::device); }
		SECTION("when newest data is on host") { run_test(access_target::host); }

		SECTION("when newest data is split across host and device") {
			buffer_for_each<size_t, 1, access_mode::discard_write, class UKN(write_first_half)>(
			    bid, access_target::device, {16}, {0}, [](id<1> idx, size_t& value) { value = idx[0]; });
			buffer_for_each<size_t, 1, access_mode::discard_write, class UKN(write_second_half)>(
			    bid, access_target::host, {16}, {16}, [](id<1> idx, size_t& value) { value = idx[0] * 2; });
			std::vector<size_t> data(32);
			bm.get_buffer_data(bid, {{0, 0, 0}, {32, 1, 1}}, data.data());
			for(size_t i = 0; i < 32; ++i) {
				REQUIRE_LOOP(data[i] == (i < 16 ? i : 2 * i));
			}
		}
	}

	TEST_CASE_METHOD(test_utils::buffer_manager_fixture, "buffer_manager correctly handles host-initialized buffers", "[buffer_manager]") {
		auto& bm = get_buffer_manager();

		constexpr size_t size = 64;
		std::vector<size_t> host_buf(size * size);
		for(size_t i = 0; i < 7; ++i) {
			for(size_t j = 0; j < 5; ++j) {
				host_buf[i * size + j] = i * 5 + j;
			}
		}

		auto bid = bm.register_buffer<size_t, 2>(range<3>(size, size, 1), host_buf.data());

		SECTION("when accessed on host") {
			// Host buffers need to accomodate the full host-initialized data range.
			REQUIRE(get_backing_buffer_range<size_t, 2>(bid, access_target::host, {7, 5}, {0, 0}) == range<2>{size, size});

			bool valid = buffer_reduce<size_t, 2, class UKN(check)>(bid, access_target::host, {7, 5}, {0, 0}, true,
			    [](id<2> idx, bool current, size_t value) { return current && (value == idx[0] * 5 + idx[1]); });
			REQUIRE(valid);
		}

		SECTION("when accessed on device") {
			// Device buffers still are only as large as required.
			REQUIRE(get_backing_buffer_range<size_t, 2>(bid, access_target::device, {7, 5}, {0, 0}) == range<2>{7, 5});

			bool valid = buffer_reduce<size_t, 2, class UKN(check)>(bid, access_target::device, {7, 5}, {0, 0}, true,
			    [](id<2> idx, bool current, size_t value) { return current && (value == idx[0] * 5 + idx[1]); });
			REQUIRE(valid);
		}
	}

	TEST_CASE_METHOD(test_utils::buffer_manager_fixture, "buffer_manager correctly handles locking", "[buffer_manager]") {
		auto& bm = get_buffer_manager();

		// Check that basic functionality works.
		REQUIRE(bm.try_lock(0, {}));
		bm.unlock(0);

		// Lock buffers 1 - 3.
		CHECK(bm.try_lock(1, {1, 2, 3}));

		// No combination of these buffers can be locked.
		REQUIRE(!bm.try_lock(2, {1}));
		REQUIRE(!bm.try_lock(2, {1, 2}));
		REQUIRE(!bm.try_lock(2, {1, 3}));
		REQUIRE(!bm.try_lock(2, {2}));
		REQUIRE(!bm.try_lock(2, {2, 3}));
		REQUIRE(!bm.try_lock(2, {3}));

		// However another buffer can be locked.
		REQUIRE(bm.try_lock(2, {4}));

		// A single locked buffer prevents an entire set of otherwise unlocked buffers from being locked.
		REQUIRE(!bm.try_lock(3, {4, 5, 6}));

		// After unlocking buffer 4 can be locked again.
		bm.unlock(2);
		REQUIRE(bm.try_lock(3, {4, 5, 6}));

		// But 1 - 3 are still locked.
		REQUIRE(!bm.try_lock(4, {1}));
		REQUIRE(!bm.try_lock(4, {2}));
		REQUIRE(!bm.try_lock(4, {3}));

		// Now they can be locked again as well.
		bm.unlock(1);
		REQUIRE(bm.try_lock(4, {3}));
	}

	TEST_CASE_METHOD(test_utils::buffer_manager_fixture, "buffer_manager throws if accessing locked buffers in unsupported order", "[buffer_manager]") {
		auto& bm = get_buffer_manager();
		auto bid = bm.register_buffer<size_t, 1>(range<3>(128, 1, 1));

		task_id tid = 0;
		auto run_test = [&](auto test_fn) {
			CHECK(bm.try_lock(tid, {bid}));
			test_fn();
			bm.unlock(tid);
			tid++;
		};

		const std::string resize_error_msg = "You are requesting multiple accessors for the same buffer, with later ones requiring a larger part of the "
		                                     "buffer, causing a backing buffer reallocation. "
		                                     "This is currently unsupported. Try changing the order of your calls to buffer::get_access.";
		const std::string discard_error_msg =
		    "You are requesting multiple accessors for the same buffer, using a discarding access mode first, followed by a non-discarding mode. "
		    "This is currently unsupported. Try changing the order of your calls to buffer::get_access.";

		SECTION("when running on device, requiring resize on second access") {
			run_test([&]() {
				bm.access_device_buffer<size_t, 1>(bid, access_mode::read, {0, 64});
				REQUIRE_THROWS_WITH((bm.access_device_buffer<size_t, 1>(bid, access_mode::read, {0, 128})), resize_error_msg);
			});
		}

		SECTION("when running on host, requiring resize on second access") {
			run_test([&]() {
				bm.access_host_buffer<size_t, 1>(bid, access_mode::read, {0, 64});
				REQUIRE_THROWS_WITH((bm.access_host_buffer<size_t, 1>(bid, access_mode::read, {0, 128})), resize_error_msg);
			});
		}

		SECTION("when running on device, using consumer after discard access") {
			run_test([&]() {
				bm.access_device_buffer<size_t, 1>(bid, access_mode::discard_write, {0, 64});
				REQUIRE_THROWS_WITH((bm.access_device_buffer<size_t, 1>(bid, access_mode::read, {0, 64})), discard_error_msg);
			});
		}

		SECTION("when running on host, using consumer after discard access") {
			run_test([&]() {
				bm.access_host_buffer<size_t, 1>(bid, access_mode::discard_write, {0, 64});
				REQUIRE_THROWS_WITH((bm.access_host_buffer<size_t, 1>(bid, access_mode::read, {0, 64})), discard_error_msg);
			});
		}
	}

	TEST_CASE_METHOD(test_utils::buffer_manager_fixture, "accessor correctly handles backing buffer offsets", "[accessor][buffer_manager]") {
		auto& bm = get_buffer_manager();
		auto& dq = get_device_queue();
		auto bid = bm.register_buffer<size_t, 2>(range<3>(64, 32, 1));

		SECTION("when using device buffers") {
			const auto range = celerity::range<2>(32, 32);
			const auto offset = id<2>(32, 0);
			auto sr = subrange<3>(id_cast<3>(offset), range_cast<3>(range));

			get_device_accessor<size_t, 2, access_mode::discard_write>(bid, {48, 32}, {16, 0});
			auto acc = get_device_accessor<size_t, 2, access_mode::discard_write>(bid, {32, 32}, {32, 0});

			test_utils::run_parallel_for<class UKN(write_buf)>(dq.get_sycl_queue(), range, offset, [=](id<2> id) { acc[id] = id[0] + id[1]; });

			auto buf_info = bm.access_host_buffer<size_t, 2>(bid, access_mode::read, {{32, 0}, {32, 32}});
			for(size_t i = 32; i < 64; ++i) {
				for(size_t j = 0; j < 32; ++j) {
					REQUIRE_LOOP(
					    static_cast<size_t*>(buf_info.ptr)[(i - buf_info.backing_buffer_offset[0]) * 32 + j - buf_info.backing_buffer_offset[1]] == i + j);
				}
			}
		}

		SECTION("when using host buffers") {
			get_host_accessor<size_t, 2, access_mode::discard_write>(bid, {48, 32}, {16, 0});
			auto acc = get_host_accessor<size_t, 2, access_mode::discard_write>(bid, {32, 32}, {32, 0});
			for(size_t i = 32; i < 64; ++i) {
				for(size_t j = 0; j < 32; ++j) {
					acc[{i, j}] = i + j;
				}
			}
			auto buf_info = bm.access_host_buffer<size_t, 2>(bid, access_mode::read, {{32, 0}, {32, 32}});
			for(size_t i = 32; i < 64; ++i) {
				for(size_t j = 0; j < 32; ++j) {
					REQUIRE_LOOP(
					    static_cast<size_t*>(buf_info.ptr)[(i - buf_info.backing_buffer_offset[0]) * 32 + j - buf_info.backing_buffer_offset[1]] == i + j);
				}
			}
		}
	}

	TEST_CASE_METHOD(test_utils::buffer_manager_fixture, "accessor supports SYCL special member and hidden friend functions", "[accessor]") {
		auto& bm = get_buffer_manager();
		auto& dq = get_device_queue();

		auto bid_a = bm.register_buffer<size_t, 1>(range<3>(32, 1, 1));
		auto bid_b = bm.register_buffer<size_t, 1>(range<3>(32, 1, 1));
		auto bid_c = bm.register_buffer<size_t, 1>(range<3>(32, 1, 1));
		auto bid_d = bm.register_buffer<size_t, 1>(range<3>(32, 1, 1));

		SECTION("when using device buffers") {
			auto range = celerity::range<1>(32);
			auto sr = subrange<3>({}, range_cast<3>(range));

			// For device accessors we test this both on host and device

			// Copy ctor
			auto device_acc_a = get_device_accessor<size_t, 1, access_mode::discard_write>(bid_a, {32}, {0});
			decltype(device_acc_a) device_acc_a1(device_acc_a);

			// Move ctor
			auto device_acc_b = get_device_accessor<size_t, 1, access_mode::discard_write>(bid_b, {32}, {0});
			decltype(device_acc_b) device_acc_b1(std::move(device_acc_b));

			// Copy assignment
			auto device_acc_c = get_device_accessor<size_t, 1, access_mode::discard_write>(bid_c, {32}, {0});
			auto device_acc_c1 = get_device_accessor<size_t, 1, access_mode::discard_write>(bid_a, {32}, {0});
			device_acc_c1 = device_acc_c;

			// Move assignment
			auto device_acc_d = get_device_accessor<size_t, 1, access_mode::discard_write>(bid_d, {32}, {0});
			auto device_acc_d1 = get_device_accessor<size_t, 1, access_mode::discard_write>(bid_a, {32}, {0});
			device_acc_d1 = std::move(device_acc_d);

			// Hidden friends (equality operators)
			REQUIRE(device_acc_a == device_acc_a1);
			REQUIRE(device_acc_a1 != device_acc_b1);

			test_utils::run_parallel_for<class UKN(member_fn_test)>(dq.get_sycl_queue(), range, {}, [=](id<1> id) {
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

			auto host_acc_a = get_host_accessor<size_t, 1, access_mode::read>(bid_a, {32}, {0});
			auto host_acc_b = get_host_accessor<size_t, 1, access_mode::read>(bid_b, {32}, {0});
			auto host_acc_c = get_host_accessor<size_t, 1, access_mode::read>(bid_c, {32}, {0});
			auto host_acc_d = get_host_accessor<size_t, 1, access_mode::read>(bid_d, {32}, {0});
			for(size_t i = 0; i < 32; ++i) {
				REQUIRE_LOOP(host_acc_a[i] == 1 * i);
				REQUIRE_LOOP(host_acc_b[i] == 2 * i);
				REQUIRE_LOOP(host_acc_c[i] == 3 * i);
				REQUIRE_LOOP(host_acc_d[i] == 4 * i);
			}
		}

		SECTION("when using host buffers") {
			{
				// Copy ctor
				auto acc_a = get_host_accessor<size_t, 1, access_mode::discard_write>(bid_a, {32}, {0});
				decltype(acc_a) acc_a1(acc_a);

				// Move ctor
				auto acc_b = get_host_accessor<size_t, 1, access_mode::discard_write>(bid_b, {32}, {0});
				decltype(acc_b) acc_b1(std::move(acc_b));

				// Copy assignment
				auto acc_c = get_host_accessor<size_t, 1, access_mode::discard_write>(bid_c, {32}, {0});
				auto acc_c1 = get_host_accessor<size_t, 1, access_mode::discard_write>(bid_a, {32}, {0});
				acc_c1 = acc_c;

				// Move assignment
				auto acc_d = get_host_accessor<size_t, 1, access_mode::discard_write>(bid_d, {32}, {0});
				auto acc_d1 = get_host_accessor<size_t, 1, access_mode::discard_write>(bid_a, {32}, {0});
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

			auto acc_a = get_host_accessor<size_t, 1, access_mode::read>(bid_a, {32}, {0});
			auto acc_b = get_host_accessor<size_t, 1, access_mode::read>(bid_b, {32}, {0});
			auto acc_c = get_host_accessor<size_t, 1, access_mode::read>(bid_c, {32}, {0});
			auto acc_d = get_host_accessor<size_t, 1, access_mode::read>(bid_d, {32}, {0});
			for(size_t i = 0; i < 32; ++i) {
				REQUIRE_LOOP(acc_a[i] == 1 * i);
				REQUIRE_LOOP(acc_b[i] == 2 * i);
				REQUIRE_LOOP(acc_c[i] == 3 * i);
				REQUIRE_LOOP(acc_d[i] == 4 * i);
			}
		}
	}

	TEST_CASE_METHOD(test_utils::buffer_manager_fixture, "host accessor supports get_pointer", "[accessor]") {
		auto& bm = get_buffer_manager();

		auto check_values = [&](const id<3>* ptr, range<3> range) {
			for(size_t i = 0; i < range[0]; ++i) {
				for(size_t j = 0; j < range[1]; ++j) {
					for(size_t k = 0; k < range[2]; ++k) {
						const auto offset = i * range[1] * range[2] + j * range[2] + k;
						REQUIRE_LOOP(ptr[offset] == id<3>(i, j, k));
					}
				}
			}
		};

		SECTION("for 1D buffers") {
			auto bid = bm.register_buffer<id<3>, 1>(range<3>(8, 1, 1));
			buffer_for_each<id<3>, 1, access_mode::discard_write, class UKN(init)>(
			    bid, access_target::device, {8}, {0}, [](id<1> idx, id<3>& value) { value = id_cast<3>(idx); });
			auto acc = get_host_accessor<id<3>, 1, access_mode::read>(bid, {8}, {0});
			check_values(acc.get_pointer(), {8, 1, 1});
		}

		SECTION("for 2D buffers") {
			auto bid = bm.register_buffer<id<3>, 2>(range<3>(8, 8, 1));
			buffer_for_each<id<3>, 2, access_mode::discard_write, class UKN(init)>(
			    bid, access_target::device, {8, 8}, {0, 0}, [](id<2> idx, id<3>& value) { value = id_cast<3>(idx); });
			auto acc = get_host_accessor<id<3>, 2, access_mode::read>(bid, {8, 8}, {0, 0});
			check_values(acc.get_pointer(), {8, 8, 1});
		}

		SECTION("for 3D buffers") {
			auto bid = bm.register_buffer<id<3>, 3>(range<3>(8, 8, 8));
			buffer_for_each<id<3>, 3, access_mode::discard_write, class UKN(init)>(
			    bid, access_target::device, {8, 8, 8}, {0, 0, 0}, [](id<3> idx, id<3>& value) { value = id_cast<3>(idx); });
			auto acc = get_host_accessor<id<3>, 3, access_mode::read>(bid, {8, 8, 8}, {0, 0, 0});
			check_values(acc.get_pointer(), {8, 8, 8});
		}
	}

	TEST_CASE_METHOD(test_utils::buffer_manager_fixture,
	    "host accessor throws when calling get_pointer for a backing buffer with different stride or nonzero offset", "[accessor]") {
		auto& bm = get_buffer_manager();
		auto bid_a = bm.register_buffer<size_t, 1>(range<3>(128, 1, 1));
		auto bid_b = bm.register_buffer<size_t, 2>(range<3>(128, 128, 1));

		const std::string error_msg = "Buffer cannot be accessed with expected stride";

		// TODO: Use single lambda once https://github.com/KhronosGroup/SYCL-Docs/pull/351 is merged and implemented
		auto get_pointer_1d = [this](const buffer_id bid, const range<1>& range, const id<1>& offset) {
			get_host_accessor<size_t, 1, access_mode::discard_write>(bid, range, offset).get_pointer();
		};

		auto get_pointer_2d = [this](const buffer_id bid, const range<2>& range, const id<2>& offset) {
			get_host_accessor<size_t, 2, access_mode::discard_write>(bid, range, offset).get_pointer();
		};

		// This is not allowed, as the backing buffer hasn't been allocated from offset 0, which means the pointer would point to offset 32.
		REQUIRE_THROWS_WITH(get_pointer_1d(bid_a, {32}, {32}), error_msg);

		// This is fine, as the backing buffer has been resized to start from 0 now.
		REQUIRE_NOTHROW(get_pointer_1d(bid_a, {64}, {0}));

		// This is now also okay, as the backing buffer starts at 0, and the pointer points to offset 0.
		// (Same semantics as SYCL accessor with offset, i.e., UB outside of requested range).
		REQUIRE_NOTHROW(get_pointer_1d(bid_a, {32}, {32}));

		// In 2D (and 3D) it's trickier, as the stride of the backing buffer must also match what the user expects.
		// This is not allowed, even though the offset is 0.
		REQUIRE_THROWS_WITH(get_pointer_2d(bid_b, {64, 64}, {0, 0}), error_msg);

		// This is allowed, as we request the full buffer.
		REQUIRE_NOTHROW(get_pointer_2d(bid_b, {128, 128}, {0, 0}));

		// This is now allowed, as the backing buffer has the expected stride.
		REQUIRE_NOTHROW(get_pointer_2d(bid_b, {64, 64}, {0, 0}));

		// Passing an offset is now also possible.
		REQUIRE_NOTHROW(get_pointer_2d(bid_b, {64, 64}, {32, 32}));
	}

	TEST_CASE_METHOD(test_utils::buffer_manager_fixture, "empty access ranges do not inflate backing buffer allocations", "[buffer_manager]") {
		auto& bm = get_buffer_manager();
		const auto bid = bm.register_buffer<int, 2>(range<3>(32, 32, 1));

		const auto access_1_sr = GENERATE(values({subrange<2>{{1, 1}, {0, 0}}, subrange<2>{{0, 0}, {8, 8}}}));
		const auto access_2_sr = GENERATE(values({subrange<2>{{16, 16}, {0, 0}}, subrange<2>{{16, 16}, {8, 8}}}));

		const auto access_1_empty = access_1_sr.range.size() == 0;
		CAPTURE(access_1_empty);
		const auto backing_buffer_1 = bm.access_device_buffer<int, 2>(bid, access_mode::discard_write, access_1_sr);
		auto* const backing_buffer_1_ptr = backing_buffer_1.ptr;
		if(access_1_empty) {
			CHECK(backing_buffer_1.backing_buffer_range.size() == 0);
		} else {
			CHECK(range_cast<2>(backing_buffer_1.backing_buffer_range) == access_1_sr.range);
			CHECK(id_cast<2>(backing_buffer_1.backing_buffer_offset) == access_1_sr.offset);
		}

		const auto access_2_empty = access_2_sr.range.size() == 0;
		CAPTURE(access_2_empty);
		const auto backing_buffer_2 = bm.access_device_buffer<int, 2>(bid, access_mode::write, access_2_sr);
		auto* const backing_buffer_2_ptr = backing_buffer_2.ptr;
		if(access_2_empty) {
			CHECK(backing_buffer_2_ptr == backing_buffer_1_ptr); // no re-allocation was made
		}
		if(access_2_empty && !access_1_empty) {
			CHECK(range_cast<2>(backing_buffer_1.backing_buffer_range) == access_1_sr.range);
			CHECK(id_cast<2>(backing_buffer_1.backing_buffer_offset) == access_1_sr.offset);
		}
		if(access_1_empty && access_2_empty) { CHECK(backing_buffer_2.backing_buffer_range.size() == 0); }
		if(access_1_empty && !access_2_empty) {
			CHECK(range_cast<2>(backing_buffer_2.backing_buffer_range) == access_2_sr.range);
			CHECK(id_cast<2>(backing_buffer_2.backing_buffer_offset) == access_2_sr.offset);
		}
	}

	TEST_CASE_METHOD(test_utils::runtime_fixture, "buffer_manager allows to set buffer debug names on  buffers", "[buffer_manager]") {
		celerity::buffer<int, 1> buff_a(16);
		std::string buff_name{"my_buffer"};
		detail::runtime::get_instance().get_buffer_manager().set_debug_name(detail::get_buffer_id(buff_a), buff_name);
		CHECK(detail::runtime::get_instance().get_buffer_manager().get_debug_name(detail::get_buffer_id(buff_a)) == buff_name);
	}

	TEST_CASE_METHOD(test_utils::device_queue_fixture, "device_queue allows to allocate device memory and query usage", "[device_queue]") {
		auto& dq = get_device_queue();
		const size_t ten_mib = 1024ul * 1024ul * 10ul;
		auto alloc1 = dq.malloc<char>(ten_mib);
		CHECK(alloc1.size_bytes == ten_mib);
		CHECK(dq.get_global_memory_allocated_bytes() == ten_mib);
		auto alloc2 = dq.malloc<char>(2 * ten_mib);
		CHECK(alloc2.size_bytes == 2 * ten_mib);
		CHECK(dq.get_global_memory_allocated_bytes() == 3 * ten_mib);
		dq.free(alloc1);
		CHECK(dq.get_global_memory_allocated_bytes() == 2 * ten_mib);
		dq.free(alloc2);
	}

	TEST_CASE_METHOD(test_utils::device_queue_fixture, "device_queue throws on allocation if device is out of memory", "[device_queue]") {
#if CELERITY_DPCPP
		SKIP("DPC++ swaps to system memory instead of failing");
#else
		auto& dq = get_device_queue();
		const auto one_quarter = dq.get_global_memory_total_size_bytes() / 4;
		// The device queue expects the user to keep track of memory usage (and asserts that an allocation will fit),
		// so we have to allocate manually through SYCL to trigger an OOM error.
		auto my_queue = sycl::queue{dq.get_sycl_queue().get_device(), [](const sycl::exception_list&) {}};
		auto* const ptr = sycl::malloc_device(2 * one_quarter, my_queue);
		CHECK(ptr != nullptr);
		auto alloc1 = dq.malloc<char>(one_quarter);
		CHECK_THROWS_MATCHES(dq.malloc<char>(2 * one_quarter), allocation_error,
		    Catch::Matchers::Message(fmt::format("Allocation of {} bytes failed; likely out of memory. Currently allocated: {} out of {} bytes.",
		        2 * one_quarter, one_quarter, dq.get_global_memory_total_size_bytes())));
		dq.free(alloc1);
		sycl::free(ptr, my_queue);
		auto alloc2 = dq.malloc<char>(2 * one_quarter);
		dq.free(alloc2);
#endif
	}

	TEST_CASE_METHOD(test_utils::buffer_manager_fixture, "buffer_manager can resize large buffers by going through the host", "[buffer_manager]") {
		auto& bm = get_buffer_manager();

		// Set memory usage limit to something low so the test doesn't run forever
		const size_t buf_size_bytes = 100 * sizeof(size_t);
		REQUIRE(buf_size_bytes < get_device_queue().get_global_memory_total_size_bytes());
		bm.set_max_device_global_memory_usage(
		    static_cast<double>(buf_size_bytes) / static_cast<double>(get_device_queue().get_global_memory_total_size_bytes()));

		const auto one_quarter_elements = buf_size_bytes / (4 * sizeof(size_t));
		const auto bid = bm.register_buffer<size_t, 1>(range<3>(4 * one_quarter_elements, 1, 1));

		// Initialize buffer to use 75% of available device memory
		buffer_for_each<size_t, 1, access_mode::discard_write, class UKN(write_linear_id)>(
		    bid, access_target::device, {3 * one_quarter_elements}, {0}, [](id<1> idx, size_t& value) { value = idx[0]; });
		const auto dinfo1 = bm.access_device_buffer<size_t, 1>(bid, access_mode::read, {{}, {3 * one_quarter_elements}});

		test_utils::log_capture lc(spdlog::level::warn);

		// Now access one additional element, which requires a resize
		buffer_for_each<size_t, 1, access_mode::read_write, class UKN(update)>(
		    bid, access_target::device, {3 * one_quarter_elements + 1}, {0}, [=](id<1> idx, size_t& value) {
			    value *= 2;
			    if(idx[0] == 3 * one_quarter_elements) { value = 1337; }
		    });
		const auto dinfo2 = bm.access_device_buffer<size_t, 1>(bid, access_mode::read, {{}, {3 * one_quarter_elements + 1}});

		// Sanity check: Did we actually reallocate the buffer?
		CHECK_FALSE((dinfo1.ptr == dinfo2.ptr && dinfo1.backing_buffer_offset == dinfo2.backing_buffer_offset
		             && dinfo1.backing_buffer_range == dinfo2.backing_buffer_range));

		CHECK_THAT(lc.get_log(), Catch::Matchers::ContainsSubstring(
		                             fmt::format("Resize of buffer {} requires temporarily copying to host memory. Performance may be degraded.", bid)));

		// Verify that data was correctly copied back from host
		bool valid = buffer_reduce<size_t, 1, class UKN(check)>(
		    bid, access_target::device, {3 * one_quarter_elements}, {0}, true, [=](id<1> idx, bool current, size_t value) {
			    if(idx[0] == 3 * one_quarter_elements) { return current && value == 1337; }
			    return current && (value == idx[0] * 2);
		    });
		REQUIRE(valid);
	}

	TEST_CASE_METHOD(
	    test_utils::buffer_manager_fixture, "buffer_manager does not retain regions that will be overwritten when resizing through host", "[buffer_manager]") {
		auto& bm = get_buffer_manager();

		// Set memory usage limit to something low so the test doesn't run forever
		const size_t buf_size_bytes = 100 * sizeof(size_t);
		REQUIRE(buf_size_bytes < get_device_queue().get_global_memory_total_size_bytes());
		bm.set_max_device_global_memory_usage(
		    static_cast<double>(buf_size_bytes) / static_cast<double>(get_device_queue().get_global_memory_total_size_bytes()));

		const auto one_quarter_elements = buf_size_bytes / (4 * sizeof(size_t));
		const auto bid = bm.register_buffer<size_t, 1>(range<3>(4 * one_quarter_elements, 1, 1));

		// Initialize full buffer on host with known values
		buffer_for_each<size_t, 1, access_mode::discard_write, class UKN(write_linear_id)>(
		    bid, access_target::host, {4 * one_quarter_elements}, {0}, [](id<1> idx, size_t& value) { value = idx[0]; });
		const auto hinfo1 = bm.access_host_buffer<size_t, 1>(bid, access_mode::read, {{}, {4 * one_quarter_elements}});

		// Allocate 0-75 of buffer on device
		buffer_for_each<size_t, 1, access_mode::read_write, class UKN(update)>(
		    bid, access_target::device, {3 * one_quarter_elements}, {0}, [](id<1> idx, size_t& value) { value *= 2; });
		const auto dinfo1 = bm.access_device_buffer<size_t, 1>(bid, access_mode::read, {{}, {3 * one_quarter_elements}});

		// Now access 25-75 + 1, forcing reallocation. Important: Use discarding access mode so 25-75 does not need to be retained on host
		buffer_for_each<size_t, 1, access_mode::discard_write, class UKN(overwrite)>(
		    bid, access_target::device, {2 * one_quarter_elements + 1}, {one_quarter_elements}, [](id<1> idx, size_t& value) { value = 3 * idx[0]; });
		const auto dinfo2 = bm.access_device_buffer<size_t, 1>(bid, access_mode::read, {{one_quarter_elements}, {2 * one_quarter_elements + 1}});

		// Sanity check: Did we actually reallocate the buffer?
		CHECK_FALSE((dinfo1.ptr == dinfo2.ptr && dinfo1.backing_buffer_offset == dinfo2.backing_buffer_offset
		             && dinfo1.backing_buffer_range == dinfo2.backing_buffer_range));

		// Access full buffer on host
		const auto hinfo2 = bm.access_host_buffer<size_t, 1>(bid, access_mode::read, {{}, {100}});
		// Host buffer should not have been reallocated
		CHECK((hinfo1.ptr == hinfo2.ptr && hinfo1.backing_buffer_offset == hinfo2.backing_buffer_offset
		       && hinfo1.backing_buffer_range == hinfo2.backing_buffer_range));
		const size_t* const hptr = static_cast<size_t*>(hinfo1.ptr);

		// Retained from device
		for(size_t i = 0; i < 25; ++i) {
			REQUIRE_LOOP(hptr[i] == i * 2);
		}
		// Updated on device
		for(size_t i = 25; i < 76; ++i) {
			REQUIRE_LOOP(hptr[i] == i * 3);
		}
		// Initial host values
		for(size_t i = 76; i < 100; ++i) {
			REQUIRE_LOOP(hptr[i] == i);
		}
	}

	TEST_CASE_METHOD(test_utils::buffer_manager_fixture, "buffer_manager spills unused parts to host for large accesses", "[buffer_manager]") {
		auto& bm = get_buffer_manager();

		// Set memory usage limit to something low so the test doesn't run forever
		const size_t buf_size_bytes = 100 * sizeof(size_t);
		REQUIRE(buf_size_bytes < get_device_queue().get_global_memory_total_size_bytes());
		bm.set_max_device_global_memory_usage(
		    static_cast<double>(buf_size_bytes) / static_cast<double>(get_device_queue().get_global_memory_total_size_bytes()));

		const auto bid = bm.register_buffer<size_t, 2>(range<3>(100, 100, 1));

		// Access "top left" of buffer first
		buffer_for_each<size_t, 2, access_mode::discard_write, class UKN(write_linear_id)>(
		    bid, access_target::device, {8, 8}, {0, 0}, [](id<2> idx, size_t& value) { value = idx[0] * 100 + idx[1]; });
		const auto dinfo1 = bm.access_device_buffer<size_t, 2>(bid, access_mode::read, {{}, {8, 8}});

		test_utils::log_capture lc(spdlog::level::warn);

		// Now access "bottom right", which normally would result in the full buffer to be allocated
		buffer_for_each<size_t, 2, access_mode::discard_write, class UKN(write_linear_id)>(
		    bid, access_target::device, {8, 8}, {91, 91}, [](id<2> idx, size_t& value) { value = idx[0] * 100 + idx[1]; });
		const auto dinfo2 = bm.access_device_buffer<size_t, 2>(bid, access_mode::read, {{91, 91}, {8, 8}});

		// Sanity check: Did we actually reallocate the buffer?
		CHECK_FALSE((dinfo1.ptr == dinfo2.ptr && dinfo1.backing_buffer_offset == dinfo2.backing_buffer_offset
		             && dinfo1.backing_buffer_range == dinfo2.backing_buffer_range));

		CHECK_THAT(
		    lc.get_log(), Catch::Matchers::ContainsSubstring(fmt::format("Buffer {} cannot be resized to fit fully into device memory, spilling partially to "
		                                                                 "host and only storing requested range on device. Performance may be degraded.",
		                      bid)));

		// Verify that data is fully available on the host
		const auto acc = get_host_accessor<size_t, 2, access_mode::read>(bid, {100, 100}, {0, 0});
		for(size_t i = 0; i < 8; ++i) {
			for(size_t j = 0; j < 8; ++j) {
				REQUIRE_LOOP(acc[i][j] == i * 100 + j);
				REQUIRE_LOOP(acc[91 + i][91 + j] == (91 + i) * 100 + 91 + j);
			}
		}
	}

	TEST_CASE_METHOD(test_utils::buffer_manager_fixture, "buffer_manager throws if buffer access exceeds available memory", "[buffer_manager]") {
#if CELERITY_DPCPP
		SKIP("DPC++ swaps to system memory instead of failing");
#else  // CELERITY_DPCPP
		using Catch::Matchers::ContainsSubstring;
		using Catch::Matchers::MessageMatches;
		auto& bm = get_buffer_manager();

		const size_t global_mem_size_bytes = get_device_queue().get_global_memory_total_size_bytes();
		const auto one_quarter_elements = global_mem_size_bytes / (4 * sizeof(size_t));
		const auto bid0 = bm.register_buffer<size_t, 1>(range<3>(10 * one_quarter_elements, 1, 1));
		const auto bid1 = bm.register_buffer<size_t, 1>(range<3>(10 * one_quarter_elements, 1, 1));

		bm.access_device_buffer(bid0, access_mode::discard_write, subrange<3>{{}, {one_quarter_elements, 1, 1}});

		// Order of listed buffers is implementation defined (unordered_map), so we match them separately
		const auto error_msg = fmt::format("Unable to allocate buffer {} of size {}.\n\nCurrent allocations:", bid1, 4 * one_quarter_elements * sizeof(size_t));
		const auto buf0_size = fmt::format("Buffer {}: {} bytes", bid0, one_quarter_elements * sizeof(size_t));

		SECTION("upon first creation") {
			CHECK_THROWS_MATCHES(bm.access_device_buffer(bid1, access_mode::discard_write, subrange<3>{{}, {4 * one_quarter_elements, 1, 1}}), allocation_error,
			    MessageMatches(ContainsSubstring(error_msg) && ContainsSubstring(buf0_size)
			                   && ContainsSubstring(fmt::format("Total usage: {} / {} bytes ({:.1f}%)", one_quarter_elements * sizeof(size_t),
			                       global_mem_size_bytes, 100 * static_cast<double>(one_quarter_elements * sizeof(size_t)) / global_mem_size_bytes))));
		}
		SECTION("upon resizing") {
			const auto buf1_size = fmt::format("Buffer {}: {} bytes", bid1, 2 * one_quarter_elements * sizeof(size_t));
			CHECK_NOTHROW(bm.access_device_buffer(bid1, access_mode::discard_write, subrange<3>{{}, {2 * one_quarter_elements, 1, 1}}));
			CHECK_THROWS_MATCHES(bm.access_device_buffer(bid1, access_mode::discard_write, subrange<3>{{}, {4 * one_quarter_elements, 1, 1}}), allocation_error,
			    MessageMatches(ContainsSubstring(error_msg) && ContainsSubstring(buf0_size) && ContainsSubstring(buf1_size)
			                   && ContainsSubstring(fmt::format("Total usage: {} / {} bytes ({:.1f}%)", 3 * one_quarter_elements * sizeof(size_t),
			                       global_mem_size_bytes, 100 * static_cast<double>(3 * one_quarter_elements * sizeof(size_t)) / global_mem_size_bytes))));
		}
#endif // CELERITY_DPCPP
	}

} // namespace detail
} // namespace celerity
