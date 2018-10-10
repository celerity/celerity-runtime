#include <memory>
#include <random>

#include <SYCL/sycl.hpp>

#define CATCH_CONFIG_MAIN
#include <catch.hpp>

#define CELERITY_TEST
#include <celerity.h>

#include "buffer_state.h"

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
	REQUIRE(sn[1].first == make_grid_box({64, 1, 1}, {192, 0, 0}));
	REQUIRE(sn[1].second.size() == 1);
	REQUIRE(sn[1].second.count(1) == 1);

	REQUIRE(sn[2].first == make_grid_box({64, 1, 1}, {128, 0, 0}));
	REQUIRE(sn[2].second.size() == 2);
	REQUIRE(sn[2].second.count(0) == 1);
	REQUIRE(sn[2].second.count(1) == 1);
}

TEST_CASE("Merging states", "[buffer_state]") {
	detail::buffer_state bs1(cl::sycl::range<3>(128, 64, 32), 3);
	detail::buffer_state bs2(cl::sycl::range<3>(128, 64, 32), 3);

	bs1.update_region(make_grid_region({128, 64, 32}, {0, 0, 0}), {0});
	bs2.update_region(make_grid_region({128, 8, 1}, {0, 24, 0}), {1});
	bs2.update_region(make_grid_region({128, 24, 1}, {0, 0, 0}), {2});
	bs1.merge(bs2);

	const auto sn = bs1.get_source_nodes(make_grid_region({128, 64, 32}, {0, 0, 0}));
	REQUIRE(sn.size() == 4);
	REQUIRE(sn[0].first == make_grid_box({128, 32, 31}, {0, 0, 1}));
	REQUIRE(sn[0].second.size() == 1);
	REQUIRE(sn[0].second.count(0) == 1);

	REQUIRE(sn[1].first == make_grid_box({128, 32, 32}, {0, 32, 0}));
	REQUIRE(sn[1].second.size() == 1);
	REQUIRE(sn[1].second.count(0) == 1);

	REQUIRE(sn[2].first == make_grid_box({128, 24, 1}, {0, 0, 0}));
	REQUIRE(sn[2].second.size() == 1);
	REQUIRE(sn[2].second.count(2) == 1);

	REQUIRE(sn[3].first == make_grid_box({128, 8, 1}, {0, 24, 0}));
	REQUIRE(sn[3].second.size() == 1);
	REQUIRE(sn[3].second.count(1) == 1);

	// Attempting to merge buffer states with incompatible dimensions or numbers of nodes should throw
	const detail::buffer_state bs_incompat1(cl::sycl::range<3>(128, 64, 32), 2);
	REQUIRE_THROWS_WITH(bs1.merge(bs_incompat1), Catch::Equals("Incompatible buffer state"));
	const detail::buffer_state bs_incompat2(cl::sycl::range<3>(128, 64, 30), 3);
	REQUIRE_THROWS_WITH(bs1.merge(bs_incompat2), Catch::Equals("Incompatible buffer state"));
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

TEST_CASE("task_manager invokes callback upon task creation", "[task_manager]") {
	detail::task_manager task_mngr;
	size_t call_counter = 0;
	task_mngr.register_task_callback([&call_counter]() { call_counter++; });
	task_mngr.create_compute_task([](auto& cgh) {});
	REQUIRE(call_counter == 1);
	task_mngr.create_master_access_task([](auto& mah) {});
	REQUIRE(call_counter == 2);
}

TEST_CASE("task_manager correctly records compute task information", "[task_manager]") {
	detail::task_manager task_mngr;
	task_mngr.create_compute_task([](auto& cgh) { cgh.template parallel_for<class my_kernel>(cl::sycl::range<2>{32, 128}, [](cl::sycl::item<2>) {}); });
	const auto tsk = task_mngr.get_task(0);
	REQUIRE(tsk->get_type() == task_type::COMPUTE);
	const auto ctsk = dynamic_cast<const compute_task*>(tsk.get());
	REQUIRE(ctsk->get_dimensions() == 2);
	REQUIRE(ctsk->get_global_size() == cl::sycl::range<3>{32, 128, 1});

	task_mngr.add_requirement(0, 32, cl::sycl::access::mode::read,
	    std::make_unique<detail::range_mapper<2, 2>>(
	        [](celerity::chunk<2> chnk) {
		        chnk.range += 3;
		        return chnk;
	        },
	        cl::sycl::access::mode::read));
	const auto& rms = ctsk->get_range_mappers();
	REQUIRE(rms.size() == 1);
	REQUIRE(rms.at(32).size() == 1);
	auto result = rms.at(32)[0]->map_2(celerity::chunk<2>({0, 0}, {5, 7}, {99, 99}));
	REQUIRE(result.range == cl::sycl::range<2>(8, 10));
}

TEST_CASE("task_manager correctly records master access task information", "[task_manager]") {
	detail::task_manager task_mngr;
	size_t call_counter = 0;
	const auto ma_functor = [&call_counter]() { call_counter++; };
	task_mngr.create_master_access_task([=](auto& mah) { mah.run(ma_functor); });
	const auto tsk = task_mngr.get_task(0);
	REQUIRE(tsk->get_type() == task_type::MASTER_ACCESS);
	const auto matsk = dynamic_cast<const master_access_task*>(tsk.get());
	master_access_livepass_handler mah;
	matsk->get_functor()(mah);
	REQUIRE(call_counter == 1);

	task_mngr.add_requirement(0, 99, cl::sycl::access::mode::read, cl::sycl::range<3>{32, 48, 96}, cl::sycl::id<3>{4, 8, 16});
	task_mngr.add_requirement(0, 99, cl::sycl::access::mode::write, cl::sycl::range<3>{21, 84, 75}, cl::sycl::id<3>{9, 2, 44});
	task_mngr.add_requirement(0, 21, cl::sycl::access::mode::write, cl::sycl::range<3>{1, 7, 19}, cl::sycl::id<3>{0, 3, 8});
	const auto& acc = matsk->get_accesses();
	REQUIRE(acc.size() == 2);
	REQUIRE(acc.at(99).at(0).mode == cl::sycl::access::mode::read);
	REQUIRE(acc.at(99).at(0).range == cl::sycl::range<3>(32, 48, 96));
	REQUIRE(acc.at(99).at(0).offset == cl::sycl::id<3>(4, 8, 16));
	REQUIRE(acc.at(99).at(1).mode == cl::sycl::access::mode::write);
	REQUIRE(acc.at(99).at(1).range == cl::sycl::range<3>(21, 84, 75));
	REQUIRE(acc.at(99).at(1).offset == cl::sycl::id<3>(9, 2, 44));
	REQUIRE(acc.at(21).at(0).mode == cl::sycl::access::mode::write);
	REQUIRE(acc.at(21).at(0).range == cl::sycl::range<3>(1, 7, 19));
	REQUIRE(acc.at(21).at(0).offset == cl::sycl::id<3>(0, 3, 8));
}

TEST_CASE("task_manager records task dependencies based on buffer accesses", "[task_manager]") {
	detail::task_manager task_mngr;
	const buffer_id buf_a = 0;
	const buffer_id buf_b = 1;
	const buffer_id buf_c = 2;
	task_mngr.create_compute_task([&](auto& cgh) { task_mngr.add_requirement(0, buf_a, cl::sycl::access::mode::write, nullptr); });
	task_mngr.create_compute_task([&](auto& cgh) { task_mngr.add_requirement(1, buf_b, cl::sycl::access::mode::write, nullptr); });
	REQUIRE(task_mngr.has_dependency(1, 0) == false);
	REQUIRE(task_mngr.has_dependency(0, 1) == false);
	task_mngr.create_compute_task([&](auto& cgh) {
		task_mngr.add_requirement(2, buf_a, cl::sycl::access::mode::read, nullptr);
		task_mngr.add_requirement(2, buf_b, cl::sycl::access::mode::read, nullptr);
		task_mngr.add_requirement(2, buf_c, cl::sycl::access::mode::write, nullptr);
	});
	REQUIRE(task_mngr.has_dependency(2, 0) == true);
	REQUIRE(task_mngr.has_dependency(2, 1) == true);
	REQUIRE(task_mngr.has_dependency(0, 2) == false);
	REQUIRE(task_mngr.has_dependency(1, 2) == false);
	task_mngr.create_master_access_task([&](auto& cgh) {
		task_mngr.add_requirement(3, buf_a, cl::sycl::access::mode::read, {}, {});
		task_mngr.add_requirement(3, buf_c, cl::sycl::access::mode::read, {}, {});
	});
	REQUIRE(task_mngr.has_dependency(3, 0) == true);
	REQUIRE(task_mngr.has_dependency(3, 1) == true); // transitively!
	REQUIRE(task_mngr.has_dependency(3, 2) == true);
	task_mngr.create_compute_task([&](auto& cgh) {
		task_mngr.add_requirement(4, buf_a, cl::sycl::access::mode::write, nullptr);
		task_mngr.add_requirement(4, buf_b, cl::sycl::access::mode::write, nullptr);
		task_mngr.add_requirement(4, buf_c, cl::sycl::access::mode::write, nullptr);
	});
	// No dependencies since task 4 only writes
	REQUIRE(task_mngr.has_dependency(4, 0) == false);
	REQUIRE(task_mngr.has_dependency(4, 1) == false);
	REQUIRE(task_mngr.has_dependency(4, 2) == false);
	REQUIRE(task_mngr.has_dependency(4, 3) == false);
};

TEST_CASE("task_manager manages the number of unsatisfied dependencies for each task", "[task_manager]") {
	detail::task_manager task_mngr;
	const buffer_id buf_a = 0;
	task_mngr.create_compute_task([&](auto& cgh) { task_mngr.add_requirement(0, buf_a, cl::sycl::access::mode::write, nullptr); });
	REQUIRE((*task_mngr.get_task_graph())[0].num_unsatisfied == 0);
	REQUIRE((*task_mngr.get_task_graph())[0].processed == false);
	task_mngr.create_compute_task([&](auto& cgh) { task_mngr.add_requirement(1, buf_a, cl::sycl::access::mode::read, nullptr); });
	REQUIRE((*task_mngr.get_task_graph())[1].num_unsatisfied == 1);
	task_mngr.mark_task_as_processed(0);
	REQUIRE((*task_mngr.get_task_graph())[0].processed == true);
	REQUIRE((*task_mngr.get_task_graph())[1].num_unsatisfied == 0);
	task_mngr.create_compute_task([&](auto& cgh) { task_mngr.add_requirement(2, buf_a, cl::sycl::access::mode::read, nullptr); });
	REQUIRE((*task_mngr.get_task_graph())[2].num_unsatisfied == 0);
};

} // namespace celerity
