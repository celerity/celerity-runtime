#include <memory>
#include <random>

#include <SYCL/sycl.hpp>

#define CATCH_CONFIG_MAIN
#include <catch.hpp>

#define CELERITY_TEST
#include <celerity.h>

GridBox<3> make_grid_box(cl::sycl::range<3> range, cl::sycl::id<3> offset = {}) {
	const auto end = cl::sycl::range<3>(offset) + range;
	return GridBox<3>(celerity::detail::sycl_range_to_grid_point(cl::sycl::range<3>(offset)), celerity::detail::sycl_range_to_grid_point(end));
}

GridRegion<3> make_grid_region(cl::sycl::range<3> range, cl::sycl::id<3> offset = {}) {
	return GridRegion<3>(make_grid_box(range, offset));
}

struct test_context {
	std::unique_ptr<celerity::distr_queue> queue = nullptr;
	test_context() {
		celerity::runtime::init_for_testing();
		queue = std::make_unique<celerity::distr_queue>();
	}
};

namespace celerity {

TEST_CASE("Basic", "[buffer_state]") {
	detail::buffer_state bs(cl::sycl::range<3>(256, 1, 1), 2);

	auto sn = bs.get_source_nodes(make_grid_region({256, 1, 1}));
	REQUIRE(sn.size() == 1);
	REQUIRE(sn[0].first == make_grid_box({256, 1, 1}));
	REQUIRE(sn[0].second.size() == 2);
	REQUIRE(sn[0].second.count(0) == 1);
	REQUIRE(sn[0].second.count(1) == 1);
}

TEST_CASE("UpdateRegion", "[buffer_state]") {
	detail::buffer_state bs(cl::sycl::range<3>(256, 1, 1), 2);
	bs.update_region(make_grid_region({128, 1, 1}), {1});

	auto sn = bs.get_source_nodes(make_grid_region({32, 1, 1}, {32, 0, 0}));
	REQUIRE(sn.size() == 1);
	REQUIRE(sn[0].first == make_grid_box({32, 1, 1}, {32, 0, 0}));
	REQUIRE(sn[0].second.size() == 1);
	REQUIRE(sn[0].second.count(1) == 1);

	sn = bs.get_source_nodes(make_grid_region({256, 1, 1}));
	REQUIRE(sn.size() == 2);
	REQUIRE(sn[0].first == make_grid_box({128, 1, 1}, {128, 0, 0}));
	REQUIRE(sn[0].second.size() == 2);
	REQUIRE(sn[0].second.count(0) == 1);
	REQUIRE(sn[0].second.count(1) == 1);
	REQUIRE(sn[1].first == make_grid_box({128, 1, 1}));
	REQUIRE(sn[1].second.size() == 1);
	REQUIRE(sn[1].second.count(1) == 1);
}

TEST_CASE("CollapseRegions", "[buffer_state]") {
	// We test buffer_state<>::collapse_regions by observing the order of the
	// returned boxes. This somewhat relies on implementation details of
	// buffer_state<>::get_source_nodes.
	// TODO: We may want to test this directly instead
	detail::buffer_state bs(cl::sycl::range<3>(256, 1, 1), 2);
	bs.update_region(make_grid_region({64, 1, 1}, {64, 0, 0}), {1});
	bs.update_region(make_grid_region({64, 1, 1}, {192, 0, 0}), {1});

	auto sn = bs.get_source_nodes(make_grid_region({192, 1, 1}, {64, 0, 0}));
	REQUIRE(sn.size() == 3);
	REQUIRE(sn[0].first == make_grid_box({64, 1, 1}, {64, 0, 0}));
	REQUIRE(sn[0].second.size() == 1);
	REQUIRE(sn[0].second.count(1) == 1);

	// Since this one is returned before the [128,192) box,
	// the {[64,128), [192,256)} region must exist internally.
	// REQUIRE(sn[1].first == GridBox<1>(192, 256));
	REQUIRE(sn[1].first == make_grid_box({64, 1, 1}, {192, 0, 0}));
	REQUIRE(sn[1].second.size() == 1);
	REQUIRE(sn[1].second.count(1) == 1);

	REQUIRE(sn[2].first == make_grid_box({64, 1, 1}, {128, 0, 0}));
	REQUIRE(sn[2].second.size() == 2);
	REQUIRE(sn[2].second.count(0) == 1);
	REQUIRE(sn[2].second.count(1) == 1);
}

TEST_CASE("host_accessor 1D indexing behaves the same way as a SYCL host-accessor", "[host_accessor]") {
	test_context ctx;
	auto cel_host_buffer = std::make_shared<detail::buffer_storage<float, 1>>(cl::sycl::range<1>(15));
	auto cel_device_buffer = std::make_shared<detail::buffer_storage<float, 1>>(cl::sycl::range<1>(15));
	auto sycl_buffer = std::make_shared<detail::buffer_storage<float, 1>>(cl::sycl::range<1>(15));
	cel_host_buffer->set_type(detail::buffer_type::HOST_BUFFER);
	cel_device_buffer->set_type(detail::buffer_type::DEVICE_BUFFER);
	sycl_buffer->set_type(detail::buffer_type::DEVICE_BUFFER);

	float test_values[15];
	std::mt19937 gen(1337);
	std::uniform_real_distribution<float> dis(0.f, 10.f);
	for(int i = 0; i < 15; ++i) {
		test_values[i] = dis(gen);
	}
	{
		host_accessor<float, 1, cl::sycl::access::mode::write> cel_host_acc(cel_host_buffer, cl::sycl::range<1>(6), cl::sycl::id<1>(8));
		host_accessor<float, 1, cl::sycl::access::mode::write> cel_device_acc(cel_device_buffer, cl::sycl::range<1>(6), cl::sycl::id<1>(8));
		auto sycl_acc = sycl_buffer->get_sycl_buffer().get_access<cl::sycl::access::mode::write>(cl::sycl::range<1>(6), cl::sycl::id<1>(8));
		for(auto i = 8u; i < 14; ++i) {
			cel_host_acc[i] = test_values[i];
			cel_device_acc[i] = test_values[i];
			sycl_acc[i] = test_values[i];
		}
	}
	host_accessor<float, 1, cl::sycl::access::mode::read> cel_host_acc(cel_host_buffer, cl::sycl::range<1>(15));
	host_accessor<float, 1, cl::sycl::access::mode::read> cel_device_acc(cel_device_buffer, cl::sycl::range<1>(15));
	auto sycl_acc = sycl_buffer->get_sycl_buffer().get_access<cl::sycl::access::mode::read>(cl::sycl::range<1>(15));
	for(auto i = 0u; i < 15; ++i) {
		REQUIRE(cel_host_acc[i] == ((i > 7 && i < 14) ? test_values[i] : 0.f));
		REQUIRE(cel_device_acc[i] == ((i > 7 && i < 14) ? test_values[i] : 0.f));
		REQUIRE(sycl_acc[i] == ((i > 7 && i < 14) ? test_values[i] : 0.f));
	}
}

TEST_CASE("host_accessor 2D indexing behaves the same way as a SYCL host-accessor", "[host_accessor]") {
	test_context ctx;
	auto cel_host_buffer = std::make_shared<detail::buffer_storage<float, 2>>(cl::sycl::range<2>(37, 22));
	auto cel_device_buffer = std::make_shared<detail::buffer_storage<float, 2>>(cl::sycl::range<2>(37, 22));
	auto sycl_buffer = std::make_shared<detail::buffer_storage<float, 2>>(cl::sycl::range<2>(37, 22));
	cel_host_buffer->set_type(detail::buffer_type::HOST_BUFFER);
	cel_device_buffer->set_type(detail::buffer_type::DEVICE_BUFFER);
	sycl_buffer->set_type(detail::buffer_type::DEVICE_BUFFER);

	float test_values[7 * 6];
	std::mt19937 gen(1337);
	std::uniform_real_distribution<float> dis(0.f, 10.f);
	for(int i = 0; i < 7 * 6; ++i) {
		test_values[i] = dis(gen);
	}
	{
		host_accessor<float, 2, cl::sycl::access::mode::write> cel_host_acc(cel_host_buffer, cl::sycl::range<2>(7, 6), cl::sycl::id<2>(30, 16));
		host_accessor<float, 2, cl::sycl::access::mode::write> cel_device_acc(cel_device_buffer, cl::sycl::range<2>(7, 6), cl::sycl::id<2>(30, 16));
		auto sycl_acc = sycl_buffer->get_sycl_buffer().get_access<cl::sycl::access::mode::write>(cl::sycl::range<2>(7, 6), cl::sycl::id<2>(30, 16));
		for(auto i = 30u; i < 37; ++i) {
			for(auto j = 16u; j < 22; ++j) {
				cel_host_acc[{i, j}] = test_values[(i - 30) * 6 + j - 16];
				cel_device_acc[{i, j}] = test_values[(i - 30) * 6 + j - 16];
				sycl_acc[{i, j}] = test_values[(i - 30) * 6 + j - 16];
			}
		}
	}
	host_accessor<float, 2, cl::sycl::access::mode::read> cel_host_acc(cel_host_buffer, cl::sycl::range<2>(7, 6), cl::sycl::id<2>(30, 16));
	host_accessor<float, 2, cl::sycl::access::mode::read> cel_device_acc(cel_device_buffer, cl::sycl::range<2>(7, 6), cl::sycl::id<2>(30, 16));
	auto sycl_acc = sycl_buffer->get_sycl_buffer().get_access<cl::sycl::access::mode::read>(cl::sycl::range<2>(7, 6), cl::sycl::id<2>(30, 16));
	for(auto i = 30u; i < 37; ++i) {
		for(auto j = 16u; j < 22; ++j) {
			REQUIRE(cel_host_acc[{i, j}] == test_values[(i - 30) * 6 + j - 16]);
			REQUIRE(cel_device_acc[{i, j}] == test_values[(i - 30) * 6 + j - 16]);
			REQUIRE(sycl_acc[{i, j}] == test_values[(i - 30) * 6 + j - 16]);
		}
	}
}

TEST_CASE("host_accessor 3D indexing behaves the same way as a SYCL host-accessor", "[host_accessor]") {
	// TODO
}

} // namespace celerity
