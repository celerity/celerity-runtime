#include "../test_utils.h"

#include <algorithm>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <mpi.h>

#include <celerity.h>

#include "../log_test_utils.h"

namespace celerity {
namespace detail {

#if CELERITY_FEATURE_SIMPLE_SCALAR_REDUCTIONS

	template <typename T>
	struct unknown_identity_maximum {
		T operator()(T a, T b) const { return a < b ? b : a; }
	};

	TEST_CASE_METHOD(test_utils::runtime_fixture, "simple reductions produce the expected results", "[reductions]") {
		size_t N = 1000;
		buffer<size_t, 1> sum_buf{{1}};
		buffer<size_t, 1> max_buf{{1}};

		distr_queue q;
		const auto initialize_to_identity = cl::sycl::property::reduction::initialize_to_identity{};

#if !CELERITY_FEATURE_SCALAR_REDUCTIONS // DPC++ can handle at most 1 reduction variable per kernel
		q.submit([=](handler& cgh) {
			auto sum_r = reduction(sum_buf, cgh, cl::sycl::plus<size_t>{}, initialize_to_identity);
			cgh.parallel_for<class UKN(kernel)>(cl::sycl::range{N}, cl::sycl::id{1}, sum_r, [=](celerity::item<1> item, auto& sum) { sum += item.get_id(0); });
		});

		q.submit([=](handler& cgh) {
			auto max_r = reduction(max_buf, cgh, size_t{0}, unknown_identity_maximum<size_t>{}, initialize_to_identity);
			cgh.parallel_for<class UKN(kernel)>(
			    cl::sycl::range{N}, cl::sycl::id{1}, max_r, [=](celerity::item<1> item, auto& max) { max.combine(item.get_id(0)); });
		});
#else
		q.submit([=](handler& cgh) {
			auto sum_r = reduction(sum_buf, cgh, cl::sycl::plus<size_t>{}, initialize_to_identity);
			auto max_r = reduction(max_buf, cgh, size_t{0}, unknown_identity_maximum<size_t>{}, initialize_to_identity);
			cgh.parallel_for<class UKN(kernel)>(cl::sycl::range{N}, cl::sycl::id{1}, sum_r, max_r, [=](celerity::item<1> item, auto& sum, auto& max) {
				sum += item.get_id(0);
				max.combine(item.get_id(0));
			});
		});
#endif

		q.submit([=](handler& cgh) {
			accessor sum_acc{sum_buf, cgh, celerity::access::all{}, celerity::read_only_host_task};
			accessor max_acc{max_buf, cgh, celerity::access::all{}, celerity::read_only_host_task};
			cgh.host_task(on_master_node, [=] {
				CHECK(sum_acc[0] == (N + 1) * (N / 2));
				CHECK(max_acc[0] == N);
			});
		});
	}

	// Regression test: The host -> device transfer previously caused an illegal nested sycl::queue::submit call which deadlocks
	// Distributed test, since the single-node case optimizes the reduction command away
	TEST_CASE_METHOD(test_utils::runtime_fixture, "reduction commands perform host -> device transfers if necessary", "[reductions]") {
		distr_queue q;

		REQUIRE(runtime::get_instance().get_num_nodes() > 1);

		const int N = 1000;
		const int init = 42;
		buffer<int, 1> sum(&init, cl::sycl::range{1});
		q.submit([=](handler& cgh) {
			cgh.parallel_for<class UKN(kernel)>(cl::sycl::range{N}, reduction(sum, cgh, cl::sycl::plus<int>{} /* don't initialize to identity */),
			    [=](celerity::item<1> item, auto& sum) { sum += 1; });
		});

		q.submit([=](handler& cgh) {
			accessor acc{sum, cgh, celerity::access::all{}, celerity::read_only_host_task};
			cgh.host_task(on_master_node, [=] { CHECK(acc[0] == N + init); });
		});
	}

	TEST_CASE_METHOD(test_utils::runtime_fixture, "multiple chained reductions produce correct results", "[reductions]") {
		distr_queue q;

		const int N = 1000;

		buffer<int, 1> sum(cl::sycl::range{1});
		q.submit([=](handler& cgh) {
			cgh.parallel_for<class UKN(kernel)>(cl::sycl::range{N},
			    reduction(sum, cgh, cl::sycl::plus<int>{}, cl::sycl::property::reduction::initialize_to_identity{}),
			    [=](celerity::item<1> item, auto& sum) { sum += 1; });
		});

		q.submit([=](handler& cgh) {
			cgh.parallel_for<class UKN(kernel)>(cl::sycl::range{N}, reduction(sum, cgh, cl::sycl::plus<int>{} /* include previous reduction result */),
			    [=](celerity::item<1> item, auto& sum) { sum += 2; });
		});

		q.submit([=](handler& cgh) {
			accessor acc{sum, cgh, celerity::access::all{}, celerity::read_only_host_task};
			cgh.host_task(on_master_node, [=] { CHECK(acc[0] == 3 * N); });
		});
	}

	TEST_CASE_METHOD(
	    test_utils::runtime_fixture, "subsequently requiring reduction results on different subsets of nodes produces correct data flow", "[reductions]") {
		distr_queue q;

		const int N = 1000;

		buffer<int, 1> sum(cl::sycl::range{1});
		q.submit([=](handler& cgh) {
			cgh.parallel_for<class UKN(produce)>(cl::sycl::range{N},
			    reduction(sum, cgh, cl::sycl::plus<int>{}, cl::sycl::property::reduction::initialize_to_identity{}),
			    [=](celerity::item<1> item, auto& sum) { sum += static_cast<int>(item.get_linear_id()); });
		});

		const int expected = (N - 1) * N / 2;

		q.submit([=](handler& cgh) {
			accessor acc{sum, cgh, celerity::access::all{}, celerity::read_only_host_task};
			cgh.host_task(on_master_node, [=] { CHECK(acc[0] == expected); });
		});

		q.submit([=](handler& cgh) {
			accessor acc{sum, cgh, celerity::access::all{}, celerity::read_only_host_task};
			cgh.host_task(experimental::collective, [=](experimental::collective_partition p) {
				INFO("Node " << p.get_node_index());
				CHECK(acc[0] == expected);
			});
		});
	}

	TEST_CASE_METHOD(
	    test_utils::runtime_fixture, "runtime-shutdown graph printing works in the presence of a finished reduction", "[reductions][print_graph][smoke-test]") {
		// init runtime early so the distr_queue ctor doesn't override the log level set by log_capture
		runtime::init(nullptr, nullptr);
		const bool is_master_node = runtime::get_instance().is_master_node();

		test_utils::log_capture log_capture;

		{
			distr_queue q;
			buffer<int, 1> sum(cl::sycl::range{1});
			q.submit([=](handler& cgh) {
				cgh.parallel_for<class UKN(produce)>(cl::sycl::range{100},
				    reduction(sum, cgh, cl::sycl::plus<int>{}, cl::sycl::property::reduction::initialize_to_identity{}),
				    [](celerity::item<1> item, auto& sum) {});
			});
			q.submit([=](handler& cgh) {
				accessor acc{sum, cgh, celerity::access::all{}, celerity::read_only_host_task};
				cgh.host_task(on_master_node, [] {});
			});
		} // shutdown runtime and print graph

		if(is_master_node) {
			using Catch::Matchers::ContainsSubstring;
			const auto log = log_capture.get_log();
			CHECK_THAT(log, ContainsSubstring("digraph G{label=\"Command Graph\""));
			CHECK_THAT(log, ContainsSubstring("(R1) <b>await push</b> from N1"));
			CHECK_THAT(log, ContainsSubstring("<b>reduction</b> R1<br/> B0 {[[0,0,0] - [1,1,1]]}"));
		}
	}

#endif // CELERITY_FEATURE_SIMPLE_SCALAR_REDUCTIONS

	template <int Dims>
	class kernel_name_nd_geometry;

	// This should be a template, but the ComputeCpp compiler segfaults if DataT of a buffer is a template type
	struct geometry {
		struct {
			size_t group_linear_id = 0;
			cl::sycl::range<3> group_range = zero_range;
			cl::sycl::id<3> local_id;
			size_t local_linear_id = 0;
			cl::sycl::range<3> local_range = zero_range;
			cl::sycl::id<3> global_id;
			size_t global_linear_id = 0;
			cl::sycl::range<3> global_range = zero_range;
		} item;
		struct {
			cl::sycl::id<3> group_id;
			size_t group_linear_id = 0;
			cl::sycl::range<3> group_range = zero_range;
			cl::sycl::id<3> local_id;
			size_t local_linear_id = 0;
			cl::sycl::range<3> local_range = zero_range;
		} group;
	};

	template <int Dims>
	class dimension_runtime_fixture : public test_utils::runtime_fixture {};

	TEMPLATE_TEST_CASE_METHOD_SIG(
	    dimension_runtime_fixture, "nd_item and group return correct execution space geometry", "[item]", ((int Dims), Dims), 1, 2, 3) {
		distr_queue q;
		auto n = runtime::get_instance().get_num_nodes();

		// Note: We assume a local range size of 165 here, this may not be supported by all devices.

		auto global_range = range_cast<Dims>(cl::sycl::range<3>{n * 4 * 3, 3 * 5, 2 * 11});
		auto local_range = range_cast<Dims>(cl::sycl::range<3>{3, 5, 11});
		auto group_range = global_range / local_range;
		auto global_offset = id_cast<Dims>(cl::sycl::id<3>{47, 53, 59});

		buffer<geometry, Dims> geo(global_range);

		q.submit([=](handler& cgh) {
			accessor g{geo, cgh, celerity::access::one_to_one{}, write_only, no_init};
			cgh.parallel_for<kernel_name_nd_geometry<Dims>>(celerity::nd_range{global_range, local_range}, /* global_offset,*/ [=](nd_item<Dims> item) {
				auto group = item.get_group();
				g[item.get_global_id()] = geometry{//
				    {item.get_group_linear_id(), range_cast<3>(item.get_group_range()), range_cast<3>(item.get_local_id()), item.get_local_linear_id(),
				        range_cast<3>(item.get_local_range()), id_cast<3>(item.get_global_id()), item.get_global_linear_id(),
				        range_cast<3>(item.get_global_range())},
				    {id_cast<3>(group.get_group_id()), group.get_group_linear_id(), range_cast<3>(group.get_group_range()), id_cast<3>(group.get_local_id()),
				        group.get_local_linear_id(), range_cast<3>(group.get_local_range())}};
			});
		});

		q.submit([=](handler& cgh) {
			accessor g{geo, cgh, celerity::access::all{}, read_only_host_task};
			cgh.host_task(on_master_node, [=] {
				for(size_t global_linear_id = 0; global_linear_id < global_range.size(); ++global_linear_id) {
					cl::sycl::id<Dims> global_id;
					{
						size_t relative = global_linear_id;
						for(int nd = 0; nd < Dims; ++nd) {
							int d = Dims - 1 - nd;
							global_id[d] = relative % global_range[d];
							relative /= global_range[d];
						}
					}
					auto group_id = global_id / local_range;
					auto local_id = global_id % local_range;
					auto local_linear_id = get_linear_index(local_range, local_id);
					auto group_linear_id = get_linear_index(group_range, group_id);

					REQUIRE_LOOP(g[global_id].item.group_linear_id == group_linear_id);
					REQUIRE_LOOP(range_cast<Dims>(g[global_id].item.group_range) == group_range);
					REQUIRE_LOOP(id_cast<Dims>(g[global_id].item.local_id) == local_id);
					REQUIRE_LOOP(g[global_id].item.local_linear_id == local_linear_id);
					REQUIRE_LOOP(range_cast<Dims>(g[global_id].item.local_range) == local_range);
					REQUIRE_LOOP(id_cast<Dims>(g[global_id].item.global_id) == global_id);
					REQUIRE_LOOP(g[global_id].item.global_linear_id == global_linear_id);
					REQUIRE_LOOP(range_cast<Dims>(g[global_id].item.global_range) == global_range);
					REQUIRE_LOOP(id_cast<Dims>(g[global_id].group.group_id) == group_id);
					REQUIRE_LOOP(g[global_id].group.group_linear_id == group_linear_id);
					REQUIRE_LOOP(range_cast<Dims>(g[global_id].group.group_range) == group_range);
					REQUIRE_LOOP(id_cast<Dims>(g[global_id].group.local_id) == local_id);
					REQUIRE_LOOP(g[global_id].group.local_linear_id == local_linear_id);
					REQUIRE_LOOP(range_cast<Dims>(g[global_id].group.local_range) == local_range);
				}
			});
		});
	}

	TEST_CASE_METHOD(test_utils::runtime_fixture, "generating same task graph on different nodes", "[task-graph]") {
		distr_queue q;
		REQUIRE(runtime::get_instance().get_num_nodes() > 1);

		constexpr int N = 1000;

		buffer<int, 1> buff_a(cl::sycl::range<1>{1});
		q.submit([=](handler& cgh) {
			accessor write_a{buff_a, cgh, celerity::access::all{}, celerity::write_only, celerity::no_init};
			cgh.parallel_for<class UKN(write_a)>(cl::sycl::range<1>{N}, [=](celerity::item<1> item) {});
		});

		buffer<int, 1> buff_b(cl::sycl::range<1>{1});
		q.submit([=](handler& cgh) {
			accessor write_b{buff_b, cgh, celerity::access::all{}, celerity::write_only, celerity::no_init};
			cgh.parallel_for<class UKN(write_b)>(cl::sycl::range<1>{N}, [=](celerity::item<1> item) {});
		});

		q.submit([=](handler& cgh) {
			accessor read_write_a{buff_a, cgh, celerity::access::all{}, celerity::read_write};
			cgh.parallel_for<class UKN(read_write_a)>(cl::sycl::range<1>{N}, [=](celerity::item<1> item) {});
		});

		q.submit([=](handler& cgh) {
			accessor read_write_a{buff_a, cgh, celerity::access::all{}, celerity::read_write};
			accessor read_write_b{buff_b, cgh, celerity::access::all{}, celerity::read_write};
			cgh.parallel_for<class UKN(read_write_a_b)>(cl::sycl::range<1>{N}, [=](celerity::item<1> item) {});
		});

		q.submit([=](handler& cgh) {
			accessor write_a{buff_a, cgh, celerity::access::all{}, celerity::write_only, celerity::no_init};
			cgh.parallel_for<class UKN(write_a_again)>(cl::sycl::range<1>{N}, [=](celerity::item<1> item) {});
		});

		q.slow_full_sync();

		int global_rank;
		MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);

		MPI_Group world_group;
		MPI_Comm_group(MPI_COMM_WORLD, &world_group);

		MPI_Comm test_communicator;
		MPI_Comm_create(MPI_COMM_WORLD, world_group, &test_communicator);

		const auto graph_str = runtime::get_instance().get_task_manager().print_graph(100);
		REQUIRE(graph_str.has_value());
		const int graph_str_length = graph_str->length();
		REQUIRE(graph_str_length > 0);

		if(global_rank == 1) {
			MPI_Send(&graph_str_length, 1, MPI_INT, 0, 0, test_communicator);
			MPI_Send(graph_str->c_str(), graph_str_length, MPI_BYTE, 0, 0, test_communicator);
		} else if(global_rank == 0) {
			int rec_graph_str_length = 0;
			MPI_Recv(&rec_graph_str_length, 1, MPI_INT, 1, 0, test_communicator, MPI_STATUS_IGNORE);
			CHECK(rec_graph_str_length == graph_str_length);
			std::string received_graph(rec_graph_str_length, 'X');
			MPI_Recv(received_graph.data(), rec_graph_str_length, MPI_BYTE, 1, 0, test_communicator, MPI_STATUS_IGNORE);
			CHECK(received_graph == graph_str);
		}
	}

	TEST_CASE_METHOD(test_utils::runtime_fixture, "nodes do not receive commands for empty chunks", "[command-graph]") {
		distr_queue q;
		auto n = runtime::get_instance().get_num_nodes();
		REQUIRE(n > 1);

		buffer<float, 2> buf{{1, 100}};

		const auto chunk_check_rm = [buf_range = buf.get_range()](const chunk<2>& chnk) {
			CHECK(chnk.range == buf_range);
			return celerity::access::one_to_one{}(chnk);
		};

		q.submit([=](handler& cgh) {
			accessor acc{buf, cgh, chunk_check_rm, write_only, no_init};
			// The kernel has a size of 1 in dimension 0, so it will not be split into
			// more than one chunk (assuming current naive split behavior).
			cgh.parallel_for<class UKN(kernel)>(buf.get_range(), [=](item<2> it) { acc[it] = 0; });
		});
	}

	TEST_CASE_METHOD(test_utils::runtime_fixture, "fences transfer data correctly between nodes", "[fence]") {
		buffer<int, 3> buf{{5, 4, 7}}; // Use an oddly-sized buffer to test the buffer subrange extraction logic
		experimental::host_object<int> obj;

		distr_queue q;
		q.submit([=](handler& cgh) {
			experimental::side_effect eff{obj, cgh};
			cgh.host_task(experimental::collective, [=](experimental::collective_partition p) { *eff = static_cast<int>(p.get_node_index()); });
		});
		q.submit([=](handler& cgh) {
			accessor acc{buf, cgh, celerity::access::all{}, write_only_host_task, no_init};
			cgh.host_task(on_master_node, [=] { acc[{1, 2, 3}] = 42; });
		});

		const auto gathered_from_master = experimental::fence(q, buf, subrange<3>({1, 2, 3}, {1, 1, 1})).get();
		const auto host_rank = experimental::fence(q, obj).get();

		REQUIRE(gathered_from_master.get_range() == range<3>{1, 1, 1});
		CHECK(gathered_from_master[0][0][0] == 42);

		int global_rank;
		MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
		CHECK(host_rank == global_rank);
	}

} // namespace detail
} // namespace celerity