#include "../test_utils.h"

#include <algorithm>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <libenvpp/env.hpp>

#include <mpi.h>

#include <celerity.h>


namespace celerity {
namespace detail {

	template <typename T>
	struct unknown_identity_maximum {
		T operator()(T a, T b) const { return a < b ? b : a; }
	};

	TEST_CASE_METHOD(test_utils::runtime_fixture, "simple reductions produce the expected results", "[reductions]") {
		size_t N = 1000;
		buffer<size_t, 1> sum_buf{{1}};
		buffer<size_t, 1> max_buf{{1}};

		distr_queue q;
		const auto initialize_to_identity = sycl::property::reduction::initialize_to_identity{};

		q.submit([&](handler& cgh) {
			auto sum_r = reduction(sum_buf, cgh, sycl::plus<size_t>{}, initialize_to_identity);
			auto max_r = reduction(max_buf, cgh, size_t{0}, unknown_identity_maximum<size_t>{}, initialize_to_identity);
			cgh.parallel_for<class UKN(kernel)>(range{N}, id{1}, sum_r, max_r, [=](celerity::item<1> item, auto& sum, auto& max) {
				sum += item.get_id(0);
				max.combine(item.get_id(0));
			});
		});

		q.submit([&](handler& cgh) {
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
		const auto num_nodes = runtime_testspy::get_num_nodes(runtime::get_instance());
		if(num_nodes < 2) { SKIP("Test needs at least 2 participating nodes"); }

		const int N = 1000;
		const int init = 42;
		buffer<int, 1> sum(&init, range{1});
		q.submit([&](handler& cgh) {
			cgh.parallel_for<class UKN(kernel)>(
			    range{N}, reduction(sum, cgh, sycl::plus<int>{} /* don't initialize to identity */), [=](celerity::item<1> item, auto& sum) { sum += 1; });
		});

		q.submit([&](handler& cgh) {
			accessor acc{sum, cgh, celerity::access::all{}, celerity::read_only_host_task};
			cgh.host_task(on_master_node, [=] { CHECK(acc[0] == N + init); });
		});
	}

	TEST_CASE_METHOD(test_utils::runtime_fixture, "multiple chained reductions produce correct results", "[reductions]") {
		distr_queue q;

		const int N = 1000;

		buffer<int, 1> sum(range(1));
		q.submit([&](handler& cgh) {
			cgh.parallel_for<class UKN(kernel)>(range{N}, reduction(sum, cgh, sycl::plus<int>{}, sycl::property::reduction::initialize_to_identity{}),
			    [=](celerity::item<1> item, auto& sum) { sum += 1; });
		});

		q.submit([&](handler& cgh) {
			cgh.parallel_for<class UKN(kernel)>(
			    range{N}, reduction(sum, cgh, sycl::plus<int>{} /* include previous reduction result */), [=](celerity::item<1> item, auto& sum) { sum += 2; });
		});

		q.submit([&](handler& cgh) {
			accessor acc{sum, cgh, celerity::access::all{}, celerity::read_only_host_task};
			cgh.host_task(on_master_node, [=] { CHECK(acc[0] == 3 * N); });
		});
	}

	TEST_CASE_METHOD(
	    test_utils::runtime_fixture, "subsequently requiring reduction results on different subsets of nodes produces correct data flow", "[reductions]") {
		distr_queue q;

		const int N = 1000;

		buffer<int, 1> sum(range(1));
		q.submit([&](handler& cgh) {
			cgh.parallel_for<class UKN(produce)>(range{N}, reduction(sum, cgh, sycl::plus<int>{}, sycl::property::reduction::initialize_to_identity{}),
			    [=](celerity::item<1> item, auto& sum) { sum += static_cast<int>(item.get_linear_id()); });
		});

		const int expected = (N - 1) * N / 2;

		q.submit([&](handler& cgh) {
			accessor acc{sum, cgh, celerity::access::all{}, celerity::read_only_host_task};
			cgh.host_task(on_master_node, [=] { CHECK(acc[0] == expected); });
		});

		q.submit([&](handler& cgh) {
			accessor acc{sum, cgh, celerity::access::all{}, celerity::read_only_host_task};
			cgh.host_task(experimental::collective, [=](experimental::collective_partition p) {
				INFO("Node " << p.get_node_index());
				CHECK(acc[0] == expected);
			});
		});
	}

	TEST_CASE_METHOD(test_utils::runtime_fixture, "runtime-shutdown graph printing works in the presence of a finished reduction",
	    "[reductions][print_graph][smoke-test]") //
	{
		env::scoped_test_environment test_env(print_graphs_env_setting);
		// init runtime early so the distr_queue ctor doesn't override the log level set by log_capture
		runtime::init(nullptr, nullptr);

		const auto num_nodes = runtime_testspy::get_num_nodes(runtime::get_instance());
		if(num_nodes < 2) { SKIP("Test needs at least 2 participating nodes"); }

		const bool is_node_0 = runtime_testspy::get_local_nid(runtime::get_instance()) == 0; // runtime instance is gone after queue destruction

		{
			distr_queue q;
			buffer<int, 1> sum(range(1));
			q.submit([&](handler& cgh) {
				cgh.parallel_for<class UKN(produce)>(range{100}, reduction(sum, cgh, sycl::plus<int>{}, sycl::property::reduction::initialize_to_identity{}),
				    [](celerity::item<1> item, auto& sum) {});
			});
			q.submit([&](handler& cgh) {
				accessor acc{sum, cgh, celerity::access::all{}, celerity::read_only_host_task};
				cgh.host_task(on_master_node, [=] { (void)acc; });
			});
		} // shutdown runtime and print graph

		if(is_node_0) { // We log graphs only on node 0
			CHECK(test_utils::log_contains_substring(log_level::info, "digraph G{label=<Command Graph>"));
			CHECK(test_utils::log_contains_substring(log_level::info, "(R1) <b>await push</b>"));
			CHECK(test_utils::log_contains_substring(log_level::info, "<b>reduction</b> R1<br/> B0 {[0,0,0] - [1,1,1]}"));
		}
	}

	template <int Dims>
	class kernel_name_nd_geometry;

	// This should be a template, but the ComputeCpp compiler segfaults if DataT of a buffer is a template type
	struct geometry {
		struct {
			size_t group_linear_id = 0;
			range<3> group_range = zeros;
			id<3> local_id;
			size_t local_linear_id = 0;
			range<3> local_range = zeros;
			id<3> global_id;
			size_t global_linear_id = 0;
			range<3> global_range = zeros;
		} item;
		struct {
			id<3> group_id;
			size_t group_linear_id = 0;
			range<3> group_range = zeros;
			id<3> local_id;
			size_t local_linear_id = 0;
			range<3> local_range = zeros;
		} group;
	};

	template <int Dims>
	class dimension_runtime_fixture : public test_utils::runtime_fixture {};

	TEMPLATE_TEST_CASE_METHOD_SIG(
	    dimension_runtime_fixture, "nd_item and group return correct execution space geometry", "[item]", ((int Dims), Dims), 1, 2, 3) {
		distr_queue q;
		const auto num_nodes = runtime_testspy::get_num_nodes(runtime::get_instance());
		if(num_nodes < 2) { SKIP("Test needs at least 2 participating nodes"); }

		// Note: We assume a local range size of 165 here, this may not be supported by all devices.

		const auto global_range = test_utils::truncate_range<Dims>({num_nodes * 4 * 3, 3 * 5, 2 * 11});
		const auto local_range = test_utils::truncate_range<Dims>({3, 5, 11});
		const auto group_range = global_range / local_range;

		buffer<geometry, Dims> geo(global_range);

		q.submit([&](handler& cgh) {
			accessor g{geo, cgh, celerity::access::one_to_one{}, write_only, no_init};
			cgh.parallel_for<kernel_name_nd_geometry<Dims>>(celerity::nd_range{global_range, local_range}, [=](nd_item<Dims> item) {
				auto group = item.get_group();
				g[item.get_global_id()] = geometry{//
				    {item.get_group_linear_id(), range_cast<3>(item.get_group_range()), id_cast<3>(item.get_local_id()), item.get_local_linear_id(),
				        range_cast<3>(item.get_local_range()), id_cast<3>(item.get_global_id()), item.get_global_linear_id(),
				        range_cast<3>(item.get_global_range())},
				    {id_cast<3>(group.get_group_id()), group.get_group_linear_id(), range_cast<3>(group.get_group_range()), id_cast<3>(group.get_local_id()),
				        group.get_local_linear_id(), range_cast<3>(group.get_local_range())}};
			});
		});

		q.submit([&](handler& cgh) {
			accessor g{geo, cgh, celerity::access::all{}, read_only_host_task};
			cgh.host_task(on_master_node, [=] {
				for(size_t global_linear_id = 0; global_linear_id < global_range.size(); ++global_linear_id) {
					id<Dims> global_id;
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
		env::scoped_test_environment tenv(print_graphs_env_setting);
		distr_queue q;
		const auto num_nodes = runtime_testspy::get_num_nodes(runtime::get_instance());
		if(num_nodes < 2) { SKIP("Test needs at least 2 participating nodes"); }

		constexpr int N = 1000;

		buffer<int, 1> buff_a(N);
		q.submit([&](handler& cgh) {
			accessor write_a{buff_a, cgh, celerity::access::one_to_one{}, celerity::write_only, celerity::no_init};
			cgh.parallel_for<class UKN(write_a)>(range<1>{N}, [=](celerity::item<1> item) { (void)write_a; });
		});

		buffer<int, 1> buff_b(N);
		q.submit([&](handler& cgh) {
			accessor write_b{buff_b, cgh, celerity::access::one_to_one{}, celerity::write_only, celerity::no_init};
			cgh.parallel_for<class UKN(write_b)>(range<1>{N}, [=](celerity::item<1> item) { (void)write_b; });
		});

		q.submit([&](handler& cgh) {
			accessor read_write_a{buff_a, cgh, celerity::access::one_to_one{}, celerity::read_write};
			cgh.parallel_for<class UKN(read_write_a)>(range<1>{N}, [=](celerity::item<1> item) { (void)read_write_a; });
		});

		q.submit([&](handler& cgh) {
			accessor read_write_a{buff_a, cgh, celerity::access::one_to_one{}, celerity::read_write};
			accessor read_write_b{buff_b, cgh, celerity::access::one_to_one{}, celerity::read_write};
			cgh.parallel_for<class UKN(read_write_a_b)>(range<1>{N}, [=](celerity::item<1> item) {
				(void)read_write_a;
				(void)read_write_b;
			});
		});

		q.submit([&](handler& cgh) {
			accessor write_a{buff_a, cgh, celerity::access::one_to_one{}, celerity::write_only, celerity::no_init};
			cgh.parallel_for<class UKN(write_a_again)>(range<1>{N}, [=](celerity::item<1> item) { (void)write_a; });
		});

		q.slow_full_sync();

		int global_rank;
		MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);

		MPI_Group world_group;
		MPI_Comm_group(MPI_COMM_WORLD, &world_group);

		MPI_Comm test_communicator;
		MPI_Comm_create(MPI_COMM_WORLD, world_group, &test_communicator);

		const auto graph_str = runtime_testspy::print_task_graph(runtime::get_instance());
		const int graph_str_length = static_cast<int>(graph_str.length());
		REQUIRE(graph_str_length > 0);

		if(global_rank == 1) {
			MPI_Send(&graph_str_length, 1, MPI_INT, 0, 0, test_communicator);
			MPI_Send(graph_str.c_str(), graph_str_length, MPI_BYTE, 0, 0, test_communicator);
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
		const auto num_nodes = runtime_testspy::get_num_nodes(runtime::get_instance());
		if(num_nodes < 2) { SKIP("Test needs at least 2 participating nodes"); }

		buffer<float, 2> buf{{1, 100}};

		const auto chunk_check_rm = [buf_range = buf.get_range()](const chunk<2>& chnk) {
			CHECK(chnk.range == buf_range);
			return celerity::access::one_to_one{}(chnk);
		};

		q.submit([&](handler& cgh) {
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
		q.submit([&](handler& cgh) {
			experimental::side_effect eff{obj, cgh};
			cgh.host_task(experimental::collective, [=](experimental::collective_partition p) { *eff = static_cast<int>(p.get_node_index()); });
		});
		q.submit([&](handler& cgh) {
			accessor acc{buf, cgh, celerity::access::all{}, write_only_host_task, no_init};
			cgh.host_task(on_master_node, [=] { acc[{1, 2, 3}] = 42; });
		});

		const auto gathered_from_master = q.fence(buf, subrange<3>({1, 2, 3}, {1, 1, 1})).get();
		const auto host_rank = q.fence(obj).get();

		REQUIRE(gathered_from_master.get_range() == range<3>{1, 1, 1});
		CHECK(gathered_from_master[0][0][0] == 42);

		int global_rank = -1;
		MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
		CHECK(host_rank == global_rank);
	}

	TEST_CASE_METHOD(test_utils::mpi_fixture, "command graph can be collected across distributed nodes", "[print_graph]") {
		int num_nodes = -1;
		int local_nid = -1;
		MPI_Comm_size(MPI_COMM_WORLD, &num_nodes);
		MPI_Comm_rank(MPI_COMM_WORLD, &local_nid);
		if(num_nodes != 2) { SKIP("can only perform this test when invoked for exactly 2 participating nodes"); }

		const std::string my_cdag = local_nid == 0 //
		                                ? "digraph G{label=<Command Graph>;pad=0.2;subgraph cluster_id_0_0{label=<l>;id_0_0[label=<C0 on N0>;shape=box];}}"
		                                : "digraph G{label=<Command Graph>;pad=0.2;subgraph cluster_id_1_0{label=<l>;id_1_0[label=<C0 on N1>;shape=box];}}";
		const auto combined_cdag = gather_command_graph(my_cdag, static_cast<size_t>(num_nodes), static_cast<node_id>(local_nid));

		if(local_nid == 0) {
			const auto expected = "digraph G{label=<Command Graph>;pad=0.2;subgraph cluster_id_0_0{label=<l>;id_0_0[label=<C0 on N0>;shape=box];}subgraph "
			                      "cluster_id_1_0{label=<l>;id_1_0[label=<C0 on N1>;shape=box];}}";
			CHECK(combined_cdag == expected);
		}
	}

	TEST_CASE_METHOD(test_utils::runtime_fixture, "runtime logs errors on overlapping writes between commands iff access pattern diagnostics are enabled",
	    "[runtime][diagnostics]") //
	{
		test_utils::allow_max_log_level(detail::log_level::err);

		distr_queue q;
		const auto num_nodes = runtime_testspy::get_num_nodes(runtime::get_instance());
		if(num_nodes < 2) { SKIP("Test needs at least 2 participating nodes"); }

		buffer<int, 1> buf(1);

		SECTION("in distributed device kernels") {
			q.submit([&](handler& cgh) {
				accessor acc(buf, cgh, celerity::access::all(), write_only, no_init);
				cgh.parallel_for(range(num_nodes), [=](item<1>) { (void)acc; });
			});
		}

		SECTION("in collective host tasks") {
			q.submit([&](handler& cgh) {
				accessor acc(buf, cgh, celerity::access::all(), write_only_host_task, no_init);
				cgh.host_task(celerity::experimental::collective, [=](experimental::collective_partition) { (void)acc; });
			});
		}

		q.slow_full_sync();

		const auto error_message = "has overlapping writes between multiple nodes in B0 {[0,0,0] - [1,1,1]}. Choose a non-overlapping range mapper for this "
		                           "write access or constrain the split via experimental::constrain_split to make the access non-overlapping.";
		CHECK(test_utils::log_contains_substring(log_level::err, error_message) == CELERITY_ACCESS_PATTERN_DIAGNOSTICS);
	}

	TEST_CASE_METHOD(test_utils::runtime_fixture, "buffer transfers are performed correctly on transposed access patterns", "[runtime]") {
		const bool oversubscribed_producer = GENERATE(values<int>({false, true}));
		const bool oversubscribed_consumer = GENERATE(values<int>({false, true}));
		CAPTURE(oversubscribed_producer, oversubscribed_consumer);

		// This test covers split/await-receive operations. Assumes a 1d split along dim0.
		distr_queue q;

		// 1. Write with row/col transposed access, such that each device owns a column of data.
		buffer<id<2>, 2> transposed_buf(range(1024, 1024));
		q.submit([&](handler& cgh) {
			const auto transposed = [](const celerity::chunk<2> in) -> subrange<2> { return {{in.offset[1], in.offset[0]}, {in.range[1], in.range[0]}}; };
			accessor acc(transposed_buf, cgh, transposed, write_only, no_init);
			if(oversubscribed_producer) { experimental::hint(cgh, experimental::hints::oversubscribe(2)); }
			cgh.parallel_for(transposed_buf.get_range(), [=](item<2> item) {
				const id transposed_id(item[1], item[0]);
				acc[transposed_id] = transposed_id;
			});
		});

		// 2. Read with 1:1 access such that each device must collect one row of data. On a multi-device system this will generate a split_receive_instruction,
		// since it is not known on the consumer side how the producer will subdivide its pushes.
		buffer<id<2>, 2> copy_buf(transposed_buf.get_range());
		q.submit([&](handler& cgh) {
			accessor acc_in(transposed_buf, cgh, celerity::access::one_to_one(), read_only);
			accessor acc_out(copy_buf, cgh, celerity::access::one_to_one(), write_only, no_init);
			if(oversubscribed_consumer) { experimental::hint(cgh, experimental::hints::oversubscribe(2)); }
			cgh.parallel_for(copy_buf.get_range(), [=](item<2> item) { acc_out[item] = acc_in[item]; });
		});

		// 3. Verify the result
		auto result = q.fence(copy_buf).get();
		experimental::for_each_item(copy_buf.get_range(), [&](const item<2>& item) {
			REQUIRE_LOOP(result[item] == item.get_id()); //
		});
	}

} // namespace detail
} // namespace celerity
