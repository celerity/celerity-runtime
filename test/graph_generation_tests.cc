#include <algorithm>
#include <iostream>
#include <map>
#include <set>

// Use custom main(), see below
#define CATCH_CONFIG_RUNNER
#include <catch.hpp>

#include <boost/optional.hpp>

#define CELERITY_TEST
#include <celerity.h>

#include "access_modes.h"
#include "graph_builder.h"
#include "graph_generator.h"
#include "graph_utils.h"
#include "task_manager.h"

#include "test_utils.h"

// To avoid having to come up with tons of unique kernel names, we simply use the CPP counter.
// This is non-standard but widely supported.
#define _UKN_CONCAT2(x, y) x##_##y
#define _UKN_CONCAT(x, y) _UKN_CONCAT2(x, y)
#define UKN(name) _UKN_CONCAT(name, __COUNTER__)

// Printing of graphs can be enabled using the "--print-graphs" command line flag
bool print_graphs = false;
celerity::detail::logger graph_logger{"graph"};

namespace celerity {
namespace detail {

	void compare_cmd_subrange(const command_subrange& sr, const cl::sycl::id<3>& offset, const cl::sycl::range<3>& range) {
		REQUIRE(sr.offset[0] == offset[0]);
		REQUIRE(sr.offset[1] == offset[1]);
		REQUIRE(sr.offset[2] == offset[2]);
		REQUIRE(sr.range[0] == range[0]);
		REQUIRE(sr.range[1] == range[1]);
		REQUIRE(sr.range[2] == range[2]);
	}

	template <typename GraphOwner>
	void maybe_print_graph(GraphOwner& go) {
		if(print_graphs) { go.print_graph(graph_logger); }
	}

	bool has_dependency(const task_manager& tm, task_id dependant, task_id dependency, bool anti = false) {
		const auto tdag = tm.get_task_graph();
		const auto ed = boost::edge(dependency, dependant, *tdag);
		if(!ed.second) return false;
		return (*tdag)[ed.first].anti_dependency == anti;
	}

	bool has_dependency(const command_dag& cdag, command_id dependant, command_id dependency, bool anti = false) {
		const auto dependant_v = GRAPH_PROP(cdag, command_vertices.at(dependant));
		const auto dependency_v = GRAPH_PROP(cdag, command_vertices.at(dependency));
		const auto ed = boost::edge(dependency_v, dependant_v, cdag);
		if(!ed.second) return false;
		return cdag[ed.first].anti_dependency == anti;
	}

	TEST_CASE("task_manager does not create multiple dependencies between the same tasks", "[task_manager][task-graph]") {
		using namespace cl::sycl::access;
		task_manager tm{true};
		test_utils::mock_buffer_factory mbf(&tm);
		auto buf_a = mbf.create_buffer(cl::sycl::range<1>(128));
		auto buf_b = mbf.create_buffer(cl::sycl::range<1>(128));

		SECTION("true dependencies") {
			const auto tid_a = test_utils::add_master_access_task(tm, [&](handler& cgh) {
				buf_a.get_access<mode::discard_write>(cgh, 128);
				buf_b.get_access<mode::discard_write>(cgh, 128);
			});
			const auto tid_b = test_utils::add_master_access_task(tm, [&](handler& cgh) {
				buf_a.get_access<mode::read>(cgh, 128);
				buf_b.get_access<mode::read>(cgh, 128);
			});
			CHECK(has_dependency(tm, tid_b, tid_a));

			const auto its = boost::out_edges(tid_a, *tm.get_task_graph());
			REQUIRE(std::distance(its.first, its.second) == 1);

			maybe_print_graph(tm);
		}

		SECTION("anti-dependencies") {
			const auto tid_a = test_utils::add_master_access_task(tm, [&](handler& cgh) {
				buf_a.get_access<mode::discard_write>(cgh, 128);
				buf_b.get_access<mode::discard_write>(cgh, 128);
			});
			const auto tid_b = test_utils::add_master_access_task(tm, [&](handler& cgh) {
				buf_a.get_access<mode::discard_write>(cgh, 128);
				buf_b.get_access<mode::discard_write>(cgh, 128);
			});
			CHECK(has_dependency(tm, tid_b, tid_a, true));

			const auto its = boost::out_edges(tid_a, *tm.get_task_graph());
			REQUIRE(std::distance(its.first, its.second) == 1);

			maybe_print_graph(tm);
		}

		// Here we also check that true dependencies always take precedence
		SECTION("true and anti-dependencies combined") {
			SECTION("if true is declared first") {
				const auto tid_a = test_utils::add_master_access_task(tm, [&](handler& cgh) {
					buf_a.get_access<mode::discard_write>(cgh, 128);
					buf_b.get_access<mode::discard_write>(cgh, 128);
				});
				const auto tid_b = test_utils::add_master_access_task(tm, [&](handler& cgh) {
					buf_a.get_access<mode::read>(cgh, 128);
					buf_b.get_access<mode::discard_write>(cgh, 128);
				});
				CHECK(has_dependency(tm, tid_b, tid_a));
				CHECK_FALSE(has_dependency(tm, tid_b, tid_a, true));

				const auto its = boost::out_edges(tid_a, *tm.get_task_graph());
				REQUIRE(std::distance(its.first, its.second) == 1);

				maybe_print_graph(tm);
			}

			SECTION("if anti is declared first") {
				const auto tid_a = test_utils::add_master_access_task(tm, [&](handler& cgh) {
					buf_a.get_access<mode::discard_write>(cgh, 128);
					buf_b.get_access<mode::discard_write>(cgh, 128);
				});
				const auto tid_b = test_utils::add_master_access_task(tm, [&](handler& cgh) {
					buf_a.get_access<mode::discard_write>(cgh, 128);
					buf_b.get_access<mode::read>(cgh, 128);
				});
				CHECK(has_dependency(tm, tid_b, tid_a));
				CHECK_FALSE(has_dependency(tm, tid_b, tid_a, true));

				const auto its = boost::out_edges(tid_a, *tm.get_task_graph());
				REQUIRE(std::distance(its.first, its.second) == 1);

				maybe_print_graph(tm);
			}
		}
	}

	TEST_CASE("task_manager respects range mapper results for finding dependencies", "[task_manager][task-graph]") {
		using namespace cl::sycl::access;
		task_manager tm{true};
		test_utils::mock_buffer_factory mbf(&tm);
		auto buf = mbf.create_buffer(cl::sycl::range<1>(128));

		const auto tid_a = test_utils::add_compute_task<class UKN(task_a)>(tm, [&](handler& cgh) {
			buf.get_access<mode::discard_write>(cgh, access::fixed<2, 1>({0, 64}));
		});
		const auto tid_b = test_utils::add_compute_task<class UKN(task_b)>(tm, [&](handler& cgh) {
			buf.get_access<mode::read>(cgh, access::fixed<2, 1>({0, 128}));
		});
		REQUIRE(has_dependency(tm, tid_b, tid_a));

		const auto tid_c = test_utils::add_compute_task<class UKN(task_c)>(tm, [&](handler& cgh) {
			buf.get_access<mode::read>(cgh, access::fixed<2, 1>({64, 128}));
		});
		REQUIRE_FALSE(has_dependency(tm, tid_c, tid_a));

		maybe_print_graph(tm);
	}

	TEST_CASE("task_manager correctly generates anti-dependencies", "[task_manager][task-graph]") {
		using namespace cl::sycl::access;
		task_manager tm{true};
		test_utils::mock_buffer_factory mbf(&tm);
		auto buf = mbf.create_buffer(cl::sycl::range<1>(128));

		// Write to the full buffer
		const auto tid_a = test_utils::add_compute_task<class UKN(task_a)>(tm, [&](handler& cgh) {
			buf.get_access<mode::discard_write>(cgh, access::fixed<2, 1>({0, 128}));
		});
		// Read the first half of the buffer
		const auto tid_b = test_utils::add_compute_task<class UKN(task_b)>(tm, [&](handler& cgh) {
			buf.get_access<mode::read>(cgh, access::fixed<2, 1>({0, 64}));
		});
		CHECK(has_dependency(tm, tid_b, tid_a));
		// Overwrite the second half - no anti-dependency onto task_b should exist (but onto task_a)
		const auto tid_c = test_utils::add_compute_task<class UKN(task_c)>(tm, [&](handler& cgh) {
			buf.get_access<mode::discard_write>(cgh, access::fixed<2, 1>({64, 64}));
		});
		REQUIRE(has_dependency(tm, tid_c, tid_a, true));
		REQUIRE_FALSE(has_dependency(tm, tid_c, tid_b, true));
		// Overwrite the first half - now only an anti-dependency onto task_b should exist
		const auto tid_d = test_utils::add_compute_task<class UKN(task_d)>(tm, [&](handler& cgh) {
			buf.get_access<mode::discard_write>(cgh, access::fixed<2, 1>({0, 64}));
		});
		REQUIRE_FALSE(has_dependency(tm, tid_d, tid_a, true));
		REQUIRE(has_dependency(tm, tid_d, tid_b, true));

		maybe_print_graph(tm);
	}

	TEST_CASE("task_manager correctly handles host-initialized buffers", "[task_manager][task-graph]") {
		using namespace cl::sycl::access;
		task_manager tm{true};
		test_utils::mock_buffer_factory mbf(&tm);
		auto host_init_buf = mbf.create_buffer(cl::sycl::range<1>(128), true);
		auto non_host_init_buf = mbf.create_buffer(cl::sycl::range<1>(128), false);

		const auto tid_a = test_utils::add_compute_task<class UKN(task_a)>(tm, [&](handler& cgh) {
			host_init_buf.get_access<mode::read>(cgh, access::fixed<2, 1>({0, 128}));
		});
		REQUIRE(has_dependency(tm, tid_a, 0)); // This task has a dependency on the init task (tid 0)
		const auto tid_b = test_utils::add_compute_task<class UKN(task_b)>(tm, [&](handler& cgh) {
			non_host_init_buf.get_access<mode::read>(cgh, access::fixed<2, 1>({0, 128}));
		});
		REQUIRE_FALSE(has_dependency(tm, tid_b, 0));

		const auto tid_c = test_utils::add_compute_task<class UKN(task_c)>(tm, [&](handler& cgh) {
			host_init_buf.get_access<mode::discard_write>(cgh, access::fixed<2, 1>({0, 128}));
		});
		REQUIRE(has_dependency(tm, tid_c, tid_a, true));
		const auto tid_d = test_utils::add_compute_task<class UKN(task_d)>(tm, [&](handler& cgh) {
			non_host_init_buf.get_access<mode::discard_write>(cgh, access::fixed<2, 1>({0, 128}));
		});
		// Since task b is essentially reading uninitialized garbage, it doesn't make a difference if we write into it concurrently
		REQUIRE_FALSE(has_dependency(tm, tid_d, tid_b, true));

		maybe_print_graph(tm);
	}

	template <int Dims, typename Handler, typename Functor>
	void dispatch_get_access(test_utils::mock_buffer<Dims>& mb, Handler& handler, cl::sycl::access::mode mode, Functor rmfn) {
		using namespace cl::sycl::access;
		switch(mode) {
		case mode::read: mb.template get_access<mode::read>(handler, rmfn); break;
		case mode::write: mb.template get_access<mode::write>(handler, rmfn); break;
		case mode::read_write: mb.template get_access<mode::read_write>(handler, rmfn); break;
		case mode::discard_write: mb.template get_access<mode::discard_write>(handler, rmfn); break;
		case mode::discard_read_write: mb.template get_access<mode::discard_read_write>(handler, rmfn); break;
		case mode::atomic: mb.template get_access<mode::atomic>(handler, rmfn); break;
		default: assert(false);
		}
	}

	TEST_CASE("task_manager correctly handles dependencies for R/W modes", "[task_manager][task-graph]") {
		using namespace cl::sycl::access;
		// A read-write access can also be implicitly created using a separate write and read, which is why we operate on "mode sets" here.
		const std::vector<std::vector<mode>> rw_mode_sets = {{mode::discard_read_write}, {mode::read_write}, {mode::atomic}, {mode::discard_write, mode::read}};

		for(const auto& mode_set : rw_mode_sets) {
			task_manager tm{true};
			test_utils::mock_buffer_factory mbf(&tm);
			auto buf = mbf.create_buffer(cl::sycl::range<1>(128), true);

			const auto tid_a = test_utils::add_compute_task<class UKN(task_a)>(tm, [&](handler& cgh) {
				for(const auto& m : mode_set) {
					dispatch_get_access(buf, cgh, m, access::fixed<2, 1>({0, 128}));
				}
			});
			const auto tid_b = test_utils::add_compute_task<class UKN(task_b)>(tm, [&](handler& cgh) {
				buf.get_access<mode::discard_write>(cgh, access::fixed<2, 1>({0, 128}));
			});
			REQUIRE(has_dependency(tm, tid_b, tid_a, true));
		}
	}

	TEST_CASE("task_manager handles all producer/consumer combinations correctly", "[task_manager][task-graph]") {
		using namespace cl::sycl::access;
		for(const auto& consumer_mode : access::detail::consumer_modes) {
			for(const auto& producer_mode : access::detail::producer_modes) {
				CAPTURE(consumer_mode);
				CAPTURE(producer_mode);
				task_manager tm{true};
				test_utils::mock_buffer_factory mbf(&tm);
				auto buf = mbf.create_buffer(cl::sycl::range<1>(128), false);

				const task_id tid_a = test_utils::add_compute_task<class UKN(task_a)>(tm, [&](handler& cgh) {
					dispatch_get_access(buf, cgh, producer_mode, celerity::access::fixed<2, 1>({0, 128}));
				});

				const task_id tid_b = test_utils::add_compute_task<class UKN(task_b)>(tm, [&](handler& cgh) {
					dispatch_get_access(buf, cgh, consumer_mode, celerity::access::fixed<2, 1>({0, 128}));
				});
				REQUIRE(has_dependency(tm, tid_b, tid_a));

				const task_id tid_c = test_utils::add_compute_task<class UKN(task_c)>(tm, [&](handler& cgh) {
					dispatch_get_access(buf, cgh, producer_mode, celerity::access::fixed<2, 1>({0, 128}));
				});
				const bool pure_consumer = consumer_mode == mode::read;
				const bool pure_producer = producer_mode == mode::discard_read_write || producer_mode == mode::discard_write;
				REQUIRE(has_dependency(tm, tid_c, tid_b, pure_consumer || pure_producer));
			}
		}
	}

	TEST_CASE("task_manager manages the number of unsatisfied dependencies for each task", "[task_manager]") {
		task_manager tm{true};
		test_utils::mock_buffer_factory mbf(&tm);
		auto buf = mbf.create_buffer(cl::sycl::range<2>(30, 40));

		const task_id tid_a = test_utils::add_compute_task<class UKN(task_a)>(tm, [&](handler& cgh) {
			buf.get_access<cl::sycl::access::mode::discard_write>(cgh, access::fixed<2, 2>({{}, {32, 23}}));
		});
		REQUIRE((*tm.get_task_graph())[tid_a].num_unsatisfied == 0);
		REQUIRE((*tm.get_task_graph())[tid_a].processed == false);
		const task_id tid_b = test_utils::add_compute_task<class UKN(task_b)>(tm, [&](handler& cgh) {
			buf.get_access<cl::sycl::access::mode::read>(cgh, access::fixed<2, 2>({{}, {32, 23}}));
		});
		REQUIRE((*tm.get_task_graph())[tid_b].num_unsatisfied == 1);
		tm.mark_task_as_processed(tid_a);
		REQUIRE((*tm.get_task_graph())[tid_a].processed == true);
		REQUIRE((*tm.get_task_graph())[tid_b].num_unsatisfied == 0);
		const task_id tid_c = test_utils::add_compute_task<class UKN(task_c)>(tm, [&](handler& cgh) {
			buf.get_access<cl::sycl::access::mode::read>(cgh, access::fixed<2, 2>({{}, {32, 23}}));
		});
		REQUIRE((*tm.get_task_graph())[tid_c].num_unsatisfied == 0);
	}

	TEST_CASE("graph_builder correctly handles command ids", "[graph_builder]") {
		command_dag cdag;
		REQUIRE(GRAPH_PROP(cdag, next_cmd_id) == 0);
		REQUIRE(GRAPH_PROP(cdag, command_vertices).empty());
		graph_builder gb(cdag);
		const auto cid_0 = gb.add_command(cdag_vertex_none, cdag_vertex_none, 0, 0, command::NOP, command_data{}, "Foo");
		const auto cid_1 = gb.add_command(cdag_vertex_none, cdag_vertex_none, 0, 0, command::NOP, command_data{}, "Foo");
		REQUIRE(GRAPH_PROP(cdag, next_cmd_id) == 2);
		gb.commit();
		REQUIRE(GRAPH_PROP(cdag, command_vertices).count(cid_0) == 1);
		REQUIRE(GRAPH_PROP(cdag, command_vertices).count(cid_1) == 1);
	}

	TEST_CASE("graph_builder correctly creates dependencies", "[graph_builder]") {
		command_dag cdag;
		graph_builder gb(cdag);
		const auto cid_0 = gb.add_command(cdag_vertex_none, cdag_vertex_none, 0, 0, command::NOP, command_data{}, "Foo");
		const auto cid_1 = gb.add_command(cdag_vertex_none, cdag_vertex_none, 0, 0, command::NOP, command_data{}, "Foo");
		gb.add_dependency(cid_1, cid_0, true);
		gb.add_dependency(cid_1, cid_0, true);
		gb.commit();
		REQUIRE(has_dependency(cdag, cid_1, cid_0, true));
		{
			// Don't create multiple dependencies between the same commands
			const auto its = boost::out_edges(GRAPH_PROP(cdag, command_vertices)[cid_0], cdag);
			REQUIRE(std::distance(its.first, its.second) == 1);
		}
		// Adding a true dependency overwrites anti-dependencies
		gb.add_dependency(cid_1, cid_0, false);
		gb.commit();
		REQUIRE(has_dependency(cdag, cid_1, cid_0, false));
		REQUIRE_FALSE(has_dependency(cdag, cid_1, cid_0, true));
		{
			const auto its = boost::out_edges(GRAPH_PROP(cdag, command_vertices)[cid_0], cdag);
			REQUIRE(std::distance(its.first, its.second) == 1);
		}
	}

	TEST_CASE("graph_builder correctly splits commands", "[graph_builder]") {
		task_dag tdag;
		boost::add_vertex(tdag);
		tdag[0].label = "Foo Task";

		command_dag cdag;
		graph_builder gb(cdag);

		cdag_vertex begin_task_cmd_v, end_task_cmd_v;
		std::tie(begin_task_cmd_v, end_task_cmd_v) = create_task_commands(tdag, cdag, gb, 0);
		gb.commit();
		REQUIRE(cdag.m_vertices.size() == 2);

		command_data compute_data{};
		compute_data.compute.subrange = subrange<2>{cl::sycl::id<2>{64, 0}, cl::sycl::range<2>{192, 512}};
		const auto compute_cid = gb.add_command(begin_task_cmd_v, end_task_cmd_v, 0, 0, command::COMPUTE, compute_data);
		gb.commit();
		const auto first_chunk = chunk<3>{cl::sycl::id<3>{64, 0, 0}, cl::sycl::range<3>{64, 256, 1}, cl::sycl::range<3>{192, 512, 1}};
		const auto second_chunk = chunk<3>{cl::sycl::id<3>{128, 256, 0}, cl::sycl::range<3>{128, 256, 1}, cl::sycl::range<3>{192, 512, 1}};
		const std::vector<chunk<3>> split_chunks = {first_chunk, second_chunk};
		const std::vector<node_id> nodes = {3, 5};
		gb.split_command(compute_cid, split_chunks, nodes);
		gb.commit();

		// Verify that original command has been deleted
		REQUIRE(cdag.m_vertices.size() == 4);
		REQUIRE(GRAPH_PROP(cdag, command_vertices).count(compute_cid) == 0);

		// Check that new commands have been properly created
		const auto first_v = GRAPH_PROP(cdag, command_vertices).at(3);
		const auto& first_data = cdag[first_v];
		REQUIRE(first_data.cmd == command::COMPUTE);
		REQUIRE(first_data.tid == 0);
		REQUIRE(first_data.cid == 3);
		REQUIRE(first_data.nid == 3);
		compare_cmd_subrange(first_data.data.compute.subrange, {64, 0, 0}, {64, 256, 1});

		const auto second_v = GRAPH_PROP(cdag, command_vertices).at(4);
		const auto& second_data = cdag[second_v];
		REQUIRE(second_data.cmd == command::COMPUTE);
		REQUIRE(second_data.tid == 0);
		REQUIRE(second_data.cid == 4);
		REQUIRE(second_data.nid == 5);
		compare_cmd_subrange(second_data.data.compute.subrange, {128, 256, 0}, {128, 256, 1});
	}

	TEST_CASE("graph_builder throws if split chunks don't add up to original chunk") {
#ifdef NDEBUG
		std::cerr << "NOTE: Some tests only run in debug builds" << std::endl;
#else
		task_dag tdag;
		boost::add_vertex(tdag);
		tdag[0].label = "Foo Task";

		command_dag cdag;
		graph_builder gb(cdag);

		cdag_vertex begin_task_cmd_v, end_task_cmd_v;
		std::tie(begin_task_cmd_v, end_task_cmd_v) = detail::create_task_commands(tdag, cdag, gb, 0);
		gb.commit();
		REQUIRE(cdag.m_vertices.size() == 2);

		command_data compute_data{};
		compute_data.compute.subrange = subrange<2>{cl::sycl::id<2>{64, 0}, cl::sycl::range<2>{192, 512}};
		const auto compute_cid = gb.add_command(begin_task_cmd_v, end_task_cmd_v, 0, 0, command::COMPUTE, compute_data);
		gb.commit();
		const auto first_chunk = chunk<3>{cl::sycl::id<3>{32, 0, 0}, cl::sycl::range<3>{64, 256, 1}, cl::sycl::range<3>{192, 512, 1}};
		const auto second_chunk = chunk<3>{cl::sycl::id<3>{128, 256, 0}, cl::sycl::range<3>{64, 128, 1}, cl::sycl::range<3>{192, 512, 1}};
		const std::vector<chunk<3>> split_chunks = {first_chunk, second_chunk};
		const std::vector<node_id> nodes = {3, 5};
		REQUIRE_THROWS_WITH(gb.split_command(compute_cid, split_chunks, nodes), Catch::Equals("Invalid split"));
#endif
	}

	// TODO: Move this elsewhere
	class cdag_inspector {
	  public:
		auto get_cb() {
			return [this](node_id nid, command_pkg pkg, const std::vector<command_id>& dependencies) {
				const command_id cid = pkg.cid;
				commands[cid] = {nid, pkg, dependencies};
				by_task[pkg.tid].insert(cid);
				by_node[nid].insert(cid);
			};
		}

		std::set<command_id> get_commands(boost::optional<task_id> tid, boost::optional<node_id> nid, boost::optional<command> cmd) const {
			std::set<command_id> result;
			std::transform(commands.cbegin(), commands.cend(), std::inserter(result, result.begin()), [](auto p) { return p.first; });

			if(tid != boost::none) {
				auto& task_set = by_task.at(*tid);
				std::set<command_id> new_result;
				std::set_intersection(result.cbegin(), result.cend(), task_set.cbegin(), task_set.cend(), std::inserter(new_result, new_result.begin()));
				result = std::move(new_result);
			}
			if(nid != boost::none) {
				auto& node_set = by_node.at(*nid);
				std::set<command_id> new_result;
				std::set_intersection(result.cbegin(), result.cend(), node_set.cbegin(), node_set.cend(), std::inserter(new_result, new_result.begin()));
				result = std::move(new_result);
			}
			if(cmd != boost::none) {
				std::set<command_id> new_result;
				std::copy_if(result.cbegin(), result.cend(), std::inserter(new_result, new_result.begin()),
				    [this, cmd](command_id cid) { return commands.at(cid).pkg.cmd == cmd; });
				result = std::move(new_result);
			}

			return result;
		}

		bool has_dependency(command_id dependant, command_id dependency) {
			const auto& deps = commands.at(dependant).dependencies;
			return std::find(deps.cbegin(), deps.cend(), dependency) != deps.cend();
		}

	  private:
		struct cmd_info {
			node_id nid;
			command_pkg pkg;
			std::vector<command_id> dependencies;
		};

		std::map<command_id, cmd_info> commands;
		std::map<task_id, std::set<command_id>> by_task;
		std::map<node_id, std::set<command_id>> by_node;
	};

	task_id build_and_flush(detail::graph_generator& ggen, task_id tid) {
		ggen.build_task(tid);
		ggen.flush(tid);
		return tid;
	}

	TEST_CASE("graph_generator generates required data transfer commands", "[graph_generator][command-graph]") {
		using namespace cl::sycl::access;

		task_manager tm{true};
		cdag_inspector inspector;
		graph_generator ggen(4, tm, inspector.get_cb());
		test_utils::mock_buffer_factory mbf(&tm, &ggen);
		auto buf = mbf.create_buffer(cl::sycl::range<1>(300));

		const auto tid_a = build_and_flush(ggen, test_utils::add_compute_task<class UKN(task_a)>(tm,
		                                             [&](handler& cgh) {
			                                             buf.get_access<mode::discard_write>(cgh, [](chunk<1> chnk) {
				                                             if(chnk.offset[0] == 0) return subrange<1>(100, 100);
				                                             if(chnk.offset[0] == 100) return subrange<1>(0, 100);
				                                             return subrange<1>(chnk);
			                                             });
		                                             },
		                                             cl::sycl::range<1>{300}));

		CHECK(inspector.get_commands(tid_a, boost::none, command::COMPUTE).size() == 3);
		CHECK(inspector.get_commands(tid_a, node_id(1), command::COMPUTE).size() == 1);
		CHECK(inspector.get_commands(tid_a, node_id(2), command::COMPUTE).size() == 1);
		CHECK(inspector.get_commands(tid_a, node_id(3), command::COMPUTE).size() == 1);

		const auto tid_b = build_and_flush(ggen, test_utils::add_compute_task<class UKN(task_b)>(tm,
		                                             [&](handler& cgh) { buf.get_access<mode::read>(cgh, access::one_to_one<1>()); }, cl::sycl::range<1>{300}));

		REQUIRE(inspector.get_commands(tid_b, boost::none, command::PUSH).size() == 2);
		REQUIRE(inspector.get_commands(tid_b, node_id(1), command::PUSH).size() == 1);
		REQUIRE(inspector.get_commands(tid_b, node_id(2), command::PUSH).size() == 1);
		REQUIRE(inspector.get_commands(tid_b, boost::none, command::AWAIT_PUSH).size() == 2);
		REQUIRE(inspector.get_commands(tid_b, node_id(1), command::AWAIT_PUSH).size() == 1);
		REQUIRE(inspector.get_commands(tid_b, node_id(2), command::AWAIT_PUSH).size() == 1);

		maybe_print_graph(tm);
		maybe_print_graph(ggen);
	}

	TEST_CASE("graph_generator doesn't generate data transfer commands for the same buffer and range more than once", "[graph_generator][command-graph]") {
		using namespace cl::sycl::access;

		task_manager tm{true};
		cdag_inspector inspector;
		graph_generator ggen(2, tm, inspector.get_cb());
		test_utils::mock_buffer_factory mbf(&tm, &ggen);
		auto buf_a = mbf.create_buffer(cl::sycl::range<1>(100));

		SECTION("when used in the same task") {
			build_and_flush(ggen, test_utils::add_compute_task<class UKN(task_a)>(
			                          tm, [&](handler& cgh) { buf_a.get_access<mode::discard_write>(cgh, access::one_to_one<1>()); }, cl::sycl::range<1>{100}));

			const auto tid_b = build_and_flush(ggen, test_utils::add_master_access_task(tm, [&](handler& cgh) {
				// Both of theses are consumer modes, meaning that both have a requirement on the buffer range produced in task_a
				buf_a.get_access<mode::read>(cgh, {100});
				buf_a.get_access<mode::write>(cgh, {100});
			}));

			REQUIRE(inspector.get_commands(tid_b, boost::none, command::PUSH).size() == 1);
			REQUIRE(inspector.get_commands(tid_b, node_id(1), command::PUSH).size() == 1);
			REQUIRE(inspector.get_commands(tid_b, boost::none, command::AWAIT_PUSH).size() == 1);
			REQUIRE(inspector.get_commands(tid_b, node_id(0), command::AWAIT_PUSH).size() == 1);

			maybe_print_graph(tm);
			maybe_print_graph(ggen);
		}

		SECTION("when used in consecutive tasks") {
			auto buf_b = mbf.create_buffer(cl::sycl::range<1>(100));

			build_and_flush(ggen, test_utils::add_compute_task<class UKN(task_a)>(
			                          tm, [&](handler& cgh) { buf_a.get_access<mode::discard_write>(cgh, access::one_to_one<1>()); }, cl::sycl::range<1>{100}));

			const auto tid_b = build_and_flush(ggen, test_utils::add_master_access_task(tm, [&](handler& cgh) {
				buf_a.get_access<mode::read>(cgh, {100});
				buf_b.get_access<mode::discard_write>(cgh, {100});
			}));

			REQUIRE(inspector.get_commands(tid_b, boost::none, command::PUSH).size() == 1);
			REQUIRE(inspector.get_commands(tid_b, node_id(1), command::PUSH).size() == 1);
			REQUIRE(inspector.get_commands(tid_b, boost::none, command::AWAIT_PUSH).size() == 1);
			REQUIRE(inspector.get_commands(tid_b, node_id(0), command::AWAIT_PUSH).size() == 1);

			const auto tid_c = build_and_flush(ggen, test_utils::add_master_access_task(tm, [&](handler& cgh) {
				buf_a.get_access<mode::read>(cgh, {100});
				buf_b.get_access<mode::read>(cgh, {100});
			}));

			REQUIRE(inspector.get_commands(tid_c, boost::none, command::PUSH).empty());
			REQUIRE(inspector.get_commands(tid_c, boost::none, command::AWAIT_PUSH).empty());

			maybe_print_graph(tm);
			maybe_print_graph(ggen);
		}

		SECTION("when used in parallel tasks") {
			build_and_flush(ggen, test_utils::add_compute_task<class UKN(task_a)>(
			                          tm, [&](handler& cgh) { buf_a.get_access<mode::discard_write>(cgh, access::one_to_one<1>()); }, cl::sycl::range<1>{100}));

			const auto tid_b = build_and_flush(ggen, test_utils::add_master_access_task(tm, [&](handler& cgh) { buf_a.get_access<mode::read>(cgh, {100}); }));

			REQUIRE(inspector.get_commands(tid_b, boost::none, command::PUSH).size() == 1);
			REQUIRE(inspector.get_commands(tid_b, node_id(1), command::PUSH).size() == 1);
			REQUIRE(inspector.get_commands(tid_b, boost::none, command::AWAIT_PUSH).size() == 1);
			REQUIRE(inspector.get_commands(tid_b, node_id(0), command::AWAIT_PUSH).size() == 1);

			const auto tid_c = build_and_flush(ggen, test_utils::add_master_access_task(tm, [&](handler& cgh) { buf_a.get_access<mode::read>(cgh, {100}); }));

			REQUIRE(inspector.get_commands(tid_c, boost::none, command::PUSH).empty());
			REQUIRE(inspector.get_commands(tid_c, boost::none, command::AWAIT_PUSH).empty());

			maybe_print_graph(tm);
			maybe_print_graph(ggen);
		}
	}

	// Currently fails as this optimization is NYI and the test just exists for documentation purposes.
	TEST_CASE("graph_generator consolidates PUSH commands for adjacent subranges", "[graph_generator][command-graph][!shouldfail]") {
		using namespace cl::sycl::access;

		task_manager tm{true};
		cdag_inspector inspector;

		graph_generator ggen(2, tm, inspector.get_cb());
		test_utils::mock_buffer_factory mbf(&tm, &ggen);
		auto buf = mbf.create_buffer(cl::sycl::range<1>(128));

		const auto tid_a = build_and_flush(
		    ggen, test_utils::add_compute_task<class UKN(task_a)>(tm, [&](handler& cgh) { buf.get_access<mode::discard_write>(cgh, access::one_to_one<1>()); },
		              cl::sycl::range<1>{64}, cl::sycl::id<1>{0}));
		CHECK(inspector.get_commands(tid_a, boost::none, command::COMPUTE).size() == 1);

		const auto tid_b = build_and_flush(
		    ggen, test_utils::add_compute_task<class UKN(task_b)>(tm, [&](handler& cgh) { buf.get_access<mode::discard_write>(cgh, access::one_to_one<1>()); },
		              cl::sycl::range<1>{64}, cl::sycl::id<1>{64}));
		CHECK(inspector.get_commands(tid_b, boost::none, command::COMPUTE).size() == 1);

		const auto tid_c = build_and_flush(ggen, test_utils::add_master_access_task(tm, [&](handler& cgh) { buf.get_access<mode::read>(cgh, 128); }));
		REQUIRE(inspector.get_commands(tid_c, node_id(1), command::PUSH).size() == 1);

		maybe_print_graph(tm);
		maybe_print_graph(ggen);
	}

	TEST_CASE("graph_generator generates dependencies for PUSH commands", "[graph_generator][command-graph]") {
		using namespace cl::sycl::access;

		task_manager tm{true};
		cdag_inspector inspector;

		SECTION("if data is produced by an execution command") {
			graph_generator ggen(2, tm, inspector.get_cb());
			test_utils::mock_buffer_factory mbf(&tm, &ggen);
			auto buf = mbf.create_buffer(cl::sycl::range<1>(100));

			const auto tid_a =
			    build_and_flush(ggen, test_utils::add_compute_task<class UKN(task_a)>(tm,
			                              [&](handler& cgh) { buf.get_access<mode::discard_write>(cgh, access::one_to_one<1>()); }, cl::sycl::range<1>{100}));
			CHECK(inspector.get_commands(tid_a, boost::none, command::COMPUTE).size() == 1);
			const auto computes = inspector.get_commands(tid_a, node_id(1), command::COMPUTE);
			CHECK(computes.size() == 1);

			const auto tid_b = build_and_flush(ggen, test_utils::add_master_access_task(tm, [&](handler& cgh) { buf.get_access<mode::read>(cgh, 100); }));
			CHECK(inspector.get_commands(tid_b, boost::none, command::PUSH).size() == 1);
			const auto pushes = inspector.get_commands(tid_b, node_id(1), command::PUSH);
			CHECK(pushes.size() == 1);

			REQUIRE(inspector.has_dependency(*pushes.cbegin(), *computes.cbegin()));

			maybe_print_graph(tm);
			maybe_print_graph(ggen);
		}

		SECTION("if data is produced by an AWAIT_PUSH command") {
			// There currently is no good way of reliably testing this because the source node for a PUSH is currently
			// selected "randomly" (i.e. the first in an unordered_set is used, ordering depends on STL implementation)
			// TODO: Revisit in the future
		}
	}

	TEST_CASE("graph_generator generates anti-dependencies for AWAIT_PUSH commands", "[graph_generator][command-graph]") {
		using namespace cl::sycl::access;

		task_manager tm{true};
		cdag_inspector inspector;
		graph_generator ggen(2, tm, inspector.get_cb());
		test_utils::mock_buffer_factory mbf(&tm, &ggen);
		auto buf = mbf.create_buffer(cl::sycl::range<1>(100), true);

		SECTION("if writing to region used by execution command") {
			// The master node starts by reading from buf (which is host-initialized)
			const auto tid_a = build_and_flush(ggen, test_utils::add_master_access_task(tm, [&](handler& cgh) { buf.get_access<mode::read>(cgh, 100); }));

			CHECK(inspector.get_commands(tid_a, boost::none, command::MASTER_ACCESS).size() == 1);
			CHECK(inspector.get_commands(tid_a, node_id(0), command::MASTER_ACCESS).size() == 1);

			// Meanwhile, the worker node writes to buf
			build_and_flush(ggen, test_utils::add_compute_task<class UKN(task_b)>(
			                          tm, [&](handler& cgh) { buf.get_access<mode::write>(cgh, access::one_to_one<1>()); }, cl::sycl::range<1>{100}));

			// Finally the master node reads again from buf, which is now the version written to by the worker node.
			// The AWAIT_PUSH command has to wait until tid_a is complete.
			const auto tid_c = build_and_flush(ggen, test_utils::add_master_access_task(tm, [&](handler& cgh) { buf.get_access<mode::read>(cgh, 100); }));

			CHECK(inspector.get_commands(tid_c, boost::none, command::AWAIT_PUSH).size() == 1);
			const auto await_pushes = inspector.get_commands(tid_c, node_id(0), command::AWAIT_PUSH);
			CHECK(await_pushes.size() == 1);
			const auto master_accesses = inspector.get_commands(tid_c, node_id(0), command::MASTER_ACCESS);
			CHECK(master_accesses.size() == 1);

			REQUIRE(inspector.has_dependency(*master_accesses.cbegin(), *await_pushes.cbegin()));

			maybe_print_graph(tm);
			maybe_print_graph(ggen);
		}

		SECTION("if writing to region used by PUSH command") {
			// Worker node writes to buf
			build_and_flush(ggen, test_utils::add_compute_task<class UKN(task_a)>(
			                          tm, [&](handler& cgh) { buf.get_access<mode::write>(cgh, access::one_to_one<1>()); }, cl::sycl::range<1>{100}));

			// Master node reads from buf, requiring a PUSH, while also writing to it
			const auto tid_b = build_and_flush(ggen, test_utils::add_master_access_task(tm, [&](handler& cgh) { buf.get_access<mode::read_write>(cgh, 100); }));

			CHECK(inspector.get_commands(tid_b, boost::none, command::PUSH).size() == 1);
			const auto pushes = inspector.get_commands(tid_b, node_id(1), command::PUSH);
			CHECK(pushes.size() == 1);

			// Finally, the worker node reads buf again, requiring an AWAIT_PUSH
			// Note that in this example the AWAIT_PUSH can never occur during the PUSH to master, as they are effectively
			// in a distributed dependency relationship, however more complex examples could give rise to situations where this can happen.
			const auto tid_c =
			    build_and_flush(ggen, test_utils::add_compute_task<class UKN(task_c)>(
			                              tm, [&](handler& cgh) { buf.get_access<mode::read>(cgh, access::one_to_one<1>()); }, cl::sycl::range<1>{100}));

			CHECK(inspector.get_commands(tid_c, boost::none, command::AWAIT_PUSH).size() == 1);
			const auto await_pushes = inspector.get_commands(tid_c, node_id(1), command::AWAIT_PUSH);
			CHECK(await_pushes.size() == 1);

			REQUIRE(inspector.has_dependency(*await_pushes.cbegin(), *pushes.cbegin()));

			maybe_print_graph(tm);
			maybe_print_graph(ggen);
		}

		SECTION("if writing to region used by another AWAIT_PUSH command") {
			build_and_flush(ggen, test_utils::add_compute_task<class UKN(task_a)>(
			                          tm, [&](handler& cgh) { buf.get_access<mode::discard_write>(cgh, access::one_to_one<1>()); }, cl::sycl::range<1>{100}));

			const auto tid_b = build_and_flush(ggen, test_utils::add_master_access_task(tm, [&](handler& cgh) { buf.get_access<mode::read>(cgh, 100); }));
			const auto await_pushes_b = inspector.get_commands(tid_b, node_id(0), command::AWAIT_PUSH);
			CHECK(await_pushes_b.size() == 1);
			const auto master_accesses_b = inspector.get_commands(tid_b, node_id(0), command::MASTER_ACCESS);
			CHECK(master_accesses_b.size() == 1);

			build_and_flush(ggen, test_utils::add_compute_task<class UKN(task_c)>(
			                          tm, [&](handler& cgh) { buf.get_access<mode::discard_write>(cgh, access::one_to_one<1>()); }, cl::sycl::range<1>{100}));

			const auto tid_d = build_and_flush(ggen, test_utils::add_master_access_task(tm, [&](handler& cgh) { buf.get_access<mode::read>(cgh, 100); }));
			const auto await_pushes_d = inspector.get_commands(tid_d, node_id(0), command::AWAIT_PUSH);
			CHECK(await_pushes_d.size() == 1);

			// The anti-dependency is delegated to the reader (i.e. the master_access)
			REQUIRE_FALSE(inspector.has_dependency(*await_pushes_d.cbegin(), *await_pushes_b.cbegin()));
			REQUIRE(inspector.has_dependency(*await_pushes_d.cbegin(), *master_accesses_b.cbegin()));

			maybe_print_graph(tm);
			maybe_print_graph(ggen);
		}
	}

	TEST_CASE("graph_generator generates anti-dependencies with subrange precision", "[graph_generator][command-graph]") {
		using namespace cl::sycl::access;

		task_manager tm{true};
		cdag_inspector inspector;
		graph_generator ggen(2, tm, inspector.get_cb());
		test_utils::mock_buffer_factory mbf(&tm, &ggen);
		auto buf = mbf.create_buffer(cl::sycl::range<1>(100));

		SECTION("for execution commands") {
			// task_a writes the first half
			const auto tid_a = build_and_flush(ggen, test_utils::add_compute_task<class UKN(task_a)>(tm,
			                                             [&](handler& cgh) {
				                                             buf.get_access<mode::discard_write>(cgh, access::fixed<1, 1>({0, 50}));
			                                             },
			                                             cl::sycl::range<1>{100}));
			const auto computes_a = inspector.get_commands(tid_a, node_id(1), command::COMPUTE);
			CHECK(computes_a.size() == 1);

			// task_b writes the second half
			const auto tid_b = build_and_flush(ggen, test_utils::add_compute_task<class UKN(task_b)>(tm,
			                                             [&](handler& cgh) {
				                                             buf.get_access<mode::discard_write>(cgh, access::fixed<1, 1>({50, 50}));
			                                             },
			                                             cl::sycl::range<1>{100}));
			const auto computes_b = inspector.get_commands(tid_b, node_id(1), command::COMPUTE);
			CHECK(computes_b.size() == 1);

			// task_c reads the first half
			const auto tid_c = build_and_flush(ggen, test_utils::add_compute_task<class UKN(task_c)>(tm,
			                                             [&](handler& cgh) {
				                                             buf.get_access<mode::read>(cgh, access::fixed<1, 1>({0, 50}));
			                                             },
			                                             cl::sycl::range<1>{100}));
			const auto computes_c = inspector.get_commands(tid_c, node_id(1), command::COMPUTE);
			CHECK(computes_c.size() == 1);
			CHECK(inspector.has_dependency(*computes_c.cbegin(), *computes_a.cbegin()));
			CHECK_FALSE(inspector.has_dependency(*computes_c.cbegin(), *computes_b.cbegin()));

			// task_d reads the second half
			const auto tid_d = build_and_flush(ggen, test_utils::add_compute_task<class UKN(task_d)>(tm,
			                                             [&](handler& cgh) {
				                                             buf.get_access<mode::read>(cgh, access::fixed<1, 1>({50, 50}));
			                                             },
			                                             cl::sycl::range<1>{100}));
			const auto computes_d = inspector.get_commands(tid_d, node_id(1), command::COMPUTE);
			CHECK(computes_d.size() == 1);
			REQUIRE(inspector.has_dependency(*computes_d.cbegin(), *computes_b.cbegin()));
			REQUIRE_FALSE(inspector.has_dependency(*computes_d.cbegin(), *computes_a.cbegin()));

			maybe_print_graph(tm);
			maybe_print_graph(ggen);
		}

		SECTION("for AWAIT_PUSH commands") {
			// task_a writes the full buffer
			const auto tid_a =
			    build_and_flush(ggen, test_utils::add_master_access_task(tm, [&](handler& cgh) { buf.get_access<mode::discard_write>(cgh, 100, 0); }));
			const auto master_accesses_a = inspector.get_commands(tid_a, node_id(0), command::MASTER_ACCESS);
			CHECK(master_accesses_a.size() == 1);

			// task_b only reads the second half
			const auto tid_b = build_and_flush(ggen, test_utils::add_master_access_task(tm, [&](handler& cgh) { buf.get_access<mode::read>(cgh, 50, 50); }));
			const auto master_accesses_b = inspector.get_commands(tid_b, node_id(0), command::MASTER_ACCESS);
			CHECK(master_accesses_b.size() == 1);

			// task_c writes to the first half
			build_and_flush(ggen, test_utils::add_compute_task<class UKN(task_c)>(
			                          tm, [&](handler& cgh) { buf.get_access<mode::discard_write>(cgh, access::one_to_one<1>()); }, cl::sycl::range<1>{50}));

			// task_d reads the first half
			const auto tid_d = build_and_flush(ggen, test_utils::add_master_access_task(tm, [&](handler& cgh) { buf.get_access<mode::read>(cgh, 50, 0); }));

			// This should generate an AWAIT_PUSH command that does NOT have an anti-dependency onto task_b, only task_a
			CHECK(inspector.get_commands(tid_d, boost::none, command::AWAIT_PUSH).size() == 1);
			const auto await_pushes = inspector.get_commands(tid_d, node_id(0), command::AWAIT_PUSH);
			REQUIRE(inspector.has_dependency(*await_pushes.cbegin(), *master_accesses_a.cbegin()));
			REQUIRE_FALSE(inspector.has_dependency(*await_pushes.cbegin(), *master_accesses_b.cbegin()));

			maybe_print_graph(tm);
			maybe_print_graph(ggen);
		}
	}

	TEST_CASE("graph_generator generates dependencies for execution commands", "[graph_generator][command-graph]") {
		using namespace cl::sycl::access;

		task_manager tm{true};
		cdag_inspector inspector;
		graph_generator ggen(2, tm, inspector.get_cb());
		test_utils::mock_buffer_factory mbf(&tm, &ggen);
		auto buf_a = mbf.create_buffer(cl::sycl::range<1>(100));
		auto buf_b = mbf.create_buffer(cl::sycl::range<1>(100));

		SECTION("if data is produced remotely") {
			build_and_flush(ggen, test_utils::add_compute_task<class UKN(task_a)>(
			                          tm, [&](handler& cgh) { buf_a.get_access<mode::discard_write>(cgh, access::one_to_one<1>()); }, cl::sycl::range<1>{100}));
			build_and_flush(ggen, test_utils::add_compute_task<class UKN(task_b)>(
			                          tm, [&](handler& cgh) { buf_b.get_access<mode::discard_write>(cgh, access::one_to_one<1>()); }, cl::sycl::range<1>{100}));
			const auto tid_c = build_and_flush(ggen, test_utils::add_master_access_task(tm, [&](handler& cgh) {
				buf_a.get_access<mode::read>(cgh, 100);
				buf_b.get_access<mode::read>(cgh, 100);
			}));
			const auto await_pushes = inspector.get_commands(tid_c, node_id(0), command::AWAIT_PUSH);
			REQUIRE(await_pushes.size() == 2);
			const auto master_accesses = inspector.get_commands(tid_c, node_id(0), command::MASTER_ACCESS);
			CHECK(master_accesses.size() == 1);
			REQUIRE(inspector.has_dependency(*master_accesses.cbegin(), *await_pushes.cbegin()));
			REQUIRE(inspector.has_dependency(*master_accesses.cbegin(), *(await_pushes.cbegin()++)));

			maybe_print_graph(tm);
			maybe_print_graph(ggen);
		}

		SECTION("if data is produced remotely but already available from an earlier task") {
			build_and_flush(ggen, test_utils::add_compute_task<class UKN(task_a)>(
			                          tm, [&](handler& cgh) { buf_a.get_access<mode::discard_write>(cgh, access::one_to_one<1>()); }, cl::sycl::range<1>{100}));
			const auto tid_b = build_and_flush(ggen, test_utils::add_master_access_task(tm, [&](handler& cgh) { buf_a.get_access<mode::read>(cgh, 100); }));
			const auto await_pushes_b = inspector.get_commands(tid_b, node_id(0), command::AWAIT_PUSH);
			REQUIRE(await_pushes_b.size() == 1);
			const auto tid_c = build_and_flush(ggen, test_utils::add_master_access_task(tm, [&](handler& cgh) { buf_a.get_access<mode::read>(cgh, 100); }));
			const auto await_pushes_c = inspector.get_commands(tid_c, node_id(0), command::AWAIT_PUSH);
			REQUIRE(await_pushes_c.size() == 0);
			const auto master_accesses = inspector.get_commands(tid_c, node_id(0), command::MASTER_ACCESS);
			REQUIRE(master_accesses.size() == 1);
			REQUIRE(inspector.has_dependency(*master_accesses.cbegin(), *await_pushes_b.cbegin()));

			maybe_print_graph(tm);
			maybe_print_graph(ggen);
		}

		SECTION("if data is produced locally") {
			const auto tid_a =
			    build_and_flush(ggen, test_utils::add_compute_task<class UKN(task_a)>(tm,
			                              [&](handler& cgh) { buf_a.get_access<mode::discard_write>(cgh, access::one_to_one<1>()); }, cl::sycl::range<1>{100}));
			const auto computes_a = inspector.get_commands(tid_a, node_id(1), command::COMPUTE);
			const auto tid_b =
			    build_and_flush(ggen, test_utils::add_compute_task<class UKN(task_b)>(tm,
			                              [&](handler& cgh) { buf_b.get_access<mode::discard_write>(cgh, access::one_to_one<1>()); }, cl::sycl::range<1>{100}));
			const auto computes_b = inspector.get_commands(tid_b, node_id(1), command::COMPUTE);
			const auto tid_c = build_and_flush(ggen, test_utils::add_compute_task<class UKN(task_c)>(tm,
			                                             [&](handler& cgh) {
				                                             buf_a.get_access<mode::read>(cgh, access::one_to_one<1>());
				                                             buf_b.get_access<mode::read>(cgh, access::one_to_one<1>());
			                                             },
			                                             cl::sycl::range<1>{100}));
			const auto computes_c = inspector.get_commands(tid_c, node_id(1), command::COMPUTE);
			REQUIRE(inspector.has_dependency(*computes_c.cbegin(), *computes_a.cbegin()));
			REQUIRE(inspector.has_dependency(*computes_c.cbegin(), *computes_b.cbegin()));

			maybe_print_graph(tm);
			maybe_print_graph(ggen);
		}
	}

	// This test case currently fails and exists for documentation purposes:
	//	- Having fixed write access to a buffer results in unclear semantics when it comes to splitting the task into chunks.
	//  - We could check for write access when using the built-in access::fixed range mapper and warn / throw.
	//		- But of course this is the easy case; the user could just as well write the same by hand.
	//
	// Really the most sensible thing to do might be to check whether chunks write to overlapping regions and abort if so.
	TEST_CASE("graph_generator handles fixed write access", "[graph_generator][command-graph][!shouldfail]") {
		using namespace cl::sycl::access;

		task_manager tm{true};
		cdag_inspector inspector;
		graph_generator ggen(3, tm, inspector.get_cb());
		test_utils::mock_buffer_factory mbf(&tm, &ggen);
		auto buf = mbf.create_buffer(cl::sycl::range<1>(100), true);

		const auto tid_a = build_and_flush(ggen, test_utils::add_compute_task<class UKN(task_a)>(tm,
		                                             [&](handler& cgh) {
			                                             buf.get_access<mode::write>(cgh, access::fixed<1, 1>({0, 100}));
		                                             },
		                                             cl::sycl::range<1>{100}));

		// Another solution could be to not split the task at all
		CHECK(inspector.get_commands(tid_a, boost::none, command::COMPUTE).size() == 1);

		const auto tid_b = build_and_flush(ggen, test_utils::add_compute_task<class UKN(task_b)>(tm,
		                                             [&](handler& cgh) {
			                                             buf.get_access<mode::read>(cgh, access::fixed<1, 1>({0, 100}));
		                                             },
		                                             cl::sycl::range<1>{100}));

		// Right now this generates a push command from the second node to the first, which also doesn't make much sense
		CHECK(inspector.get_commands(tid_b, boost::none, command::PUSH).empty());

		maybe_print_graph(tm);
		maybe_print_graph(ggen);
	}

	TEST_CASE("graph_generator allows chunks to require empty buffer ranges", "[graph_generator][command-graph]") {
		using namespace cl::sycl::access;

		task_manager tm{true};
		cdag_inspector inspector;
		graph_generator ggen(3, tm, inspector.get_cb());
		test_utils::mock_buffer_factory mbf(&tm, &ggen);
		auto buf_a = mbf.create_buffer(cl::sycl::range<1>(100));
		auto buf_b = mbf.create_buffer(cl::sycl::range<1>(100));

		build_and_flush(ggen, test_utils::add_master_access_task(tm, [&](handler& cgh) {
			buf_a.get_access<mode::discard_write>(cgh, 100);
			buf_b.get_access<mode::discard_write>(cgh, 100);
		}));
		const auto tid_b = build_and_flush(ggen, test_utils::add_compute_task<class UKN(task_b)>(tm,
		                                             [&](handler& cgh) {
			                                             // NOTE: It's important to construct range-mappers in such a way that passing the
			                                             // global size (during tdag generation) still returns the correct result!
			                                             buf_a.get_access<mode::read>(cgh, [&](chunk<1> chnk) -> subrange<1> {
				                                             if(chnk.offset[0] < 50) { return chnk; }
				                                             return {0, 0};
			                                             });
			                                             buf_b.get_access<mode::read>(cgh, [&](chunk<1> chnk) -> subrange<1> {
				                                             if(chnk.offset[0] + chnk.range[0] <= 50) return {0, 0};
				                                             return chnk;
			                                             });
		                                             },
		                                             cl::sycl::range<1>{100}));

		CHECK(inspector.get_commands(tid_b, boost::none, command::COMPUTE).size() == 2);
		const auto computes_node1 = inspector.get_commands(tid_b, node_id(1), command::COMPUTE);
		CHECK(computes_node1.size() == 1);
		const auto computes_node2 = inspector.get_commands(tid_b, node_id(2), command::COMPUTE);
		CHECK(computes_node2.size() == 1);
		const auto await_pushes_node1 = inspector.get_commands(tid_b, node_id(1), command::AWAIT_PUSH);
		REQUIRE(await_pushes_node1.size() == 1);
		CHECK(inspector.has_dependency(*computes_node1.cbegin(), *await_pushes_node1.cbegin()));
		const auto await_pushes_node2 = inspector.get_commands(tid_b, node_id(2), command::AWAIT_PUSH);
		REQUIRE(await_pushes_node2.size() == 1);
		CHECK(inspector.has_dependency(*computes_node2.cbegin(), *await_pushes_node2.cbegin()));

		maybe_print_graph(tm);
		maybe_print_graph(ggen);
	}

	// This is a highly constructed and unrealistic example, but we'd still like the behavior to be clearly defined.
	TEST_CASE("graph_generator generates anti-dependencies for execution commands that have a task-level true dependency", "[graph_generator][command-graph]") {
		using namespace cl::sycl::access;
		task_manager tm{true};
		cdag_inspector inspector;
		graph_generator ggen(3, tm, inspector.get_cb());
		test_utils::mock_buffer_factory mbf(&tm, &ggen);
		auto buf_a = mbf.create_buffer(cl::sycl::range<1>(100));
		auto buf_b = mbf.create_buffer(cl::sycl::range<1>(100));

		// Initialize both buffers
		const auto tid_a = build_and_flush(ggen, test_utils::add_compute_task<class UKN(task_a)>(tm,
		                                             [&](handler& cgh) {
			                                             buf_a.get_access<mode::discard_write>(cgh, access::one_to_one<1>());
			                                             buf_b.get_access<mode::discard_write>(cgh, access::one_to_one<1>());
		                                             },
		                                             cl::sycl::range<1>{100}));
		const auto computes_a_node1 = inspector.get_commands(tid_a, node_id(1), command::COMPUTE);
		CHECK(computes_a_node1.size() == 1);
		const auto computes_a_node2 = inspector.get_commands(tid_a, node_id(2), command::COMPUTE);
		CHECK(computes_a_node2.size() == 1);

		// Read from buf_a but overwrite buf_b
		// Importantly, we only read on the first node node, making it so the second node node does not have a true dependency on the previous task.
		const auto tid_b = build_and_flush(ggen, test_utils::add_compute_task<class UKN(task_b)>(tm,
		                                             [&](handler& cgh) {
			                                             buf_a.get_access<mode::read>(cgh, [&](chunk<1> chnk) -> subrange<1> {
				                                             if(chnk.offset[0] < 50) return chnk;
				                                             return {0, 0};
			                                             });
			                                             buf_b.get_access<mode::discard_write>(cgh, access::one_to_one<1>());
		                                             },
		                                             cl::sycl::range<1>{100}));
		const auto computes_b_node1 = inspector.get_commands(tid_b, node_id(1), command::COMPUTE);
		CHECK(computes_b_node1.size() == 1);
		const auto computes_b_node2 = inspector.get_commands(tid_b, node_id(2), command::COMPUTE);
		CHECK(computes_b_node2.size() == 1);

		CHECK(inspector.has_dependency(*computes_b_node1.cbegin(), *computes_a_node1.cbegin()));
		REQUIRE(inspector.has_dependency(*computes_b_node2.cbegin(), *computes_a_node2.cbegin()));

		maybe_print_graph(tm);
		maybe_print_graph(ggen);
	}

	TEST_CASE("graph_generator generates anti-dependencies onto the original producer if no consumer exists in between", "[graph_generator][command-graph]") {
		using namespace cl::sycl::access;
		task_manager tm{true};
		cdag_inspector inspector;
		graph_generator ggen(3, tm, inspector.get_cb());
		test_utils::mock_buffer_factory mbf(&tm, &ggen);
		auto buf = mbf.create_buffer(cl::sycl::range<1>(100));

		const auto tid_a = build_and_flush(ggen, test_utils::add_master_access_task(tm, [&](handler& cgh) { buf.get_access<mode::discard_write>(cgh, 100); }));
		const auto master_accesses_a = inspector.get_commands(tid_a, node_id(0), command::MASTER_ACCESS);
		const auto tid_b = build_and_flush(ggen, test_utils::add_master_access_task(tm, [&](handler& cgh) { buf.get_access<mode::discard_write>(cgh, 100); }));
		const auto master_accesses_b = inspector.get_commands(tid_b, node_id(0), command::MASTER_ACCESS);
		CHECK(master_accesses_b.size() == 1);
		REQUIRE(inspector.has_dependency(*master_accesses_b.cbegin(), *master_accesses_a.cbegin()));

		maybe_print_graph(tm);
		maybe_print_graph(ggen);
	}

	TEST_CASE("graph_generator generates anti-dependencies for execution commands onto PUSHes within the same task", "[graph_generator][command-graph]") {
		using namespace cl::sycl::access;
		task_manager tm{true};
		cdag_inspector inspector;
		graph_generator ggen(3, tm, inspector.get_cb());
		test_utils::mock_buffer_factory mbf(&tm, &ggen);
		auto buf = mbf.create_buffer(cl::sycl::range<1>(100));

		build_and_flush(ggen, test_utils::add_compute_task<class UKN(task_a)>(
		                          tm, [&](handler& cgh) { buf.get_access<mode::discard_write>(cgh, access::one_to_one<1>()); }, cl::sycl::range<1>{100}));
		const auto tid_b = build_and_flush(ggen, test_utils::add_compute_task<class UKN(task_b)>(tm,
		                                             [&](handler& cgh) {
			                                             // Both workers read the full buffer
			                                             buf.get_access<mode::read>(cgh, access::fixed<1, 1>({0, 100}));

			                                             // Only the second worker also writes to the buffer
			                                             buf.get_access<mode::read_write>(cgh, [](chunk<1> chnk) -> subrange<1> {
				                                             if(chnk.offset[0] + chnk.range[0] > 50) return chnk;
				                                             return {0, 0};
			                                             });
		                                             },
		                                             cl::sycl::range<1>{100}));
		CHECK(inspector.get_commands(tid_b, boost::none, command::PUSH).size() == 2);
		CHECK(inspector.get_commands(tid_b, boost::none, command::AWAIT_PUSH).size() == 2);
		CHECK(inspector.get_commands(tid_b, boost::none, command::COMPUTE).size() == 2);
		const auto pushes_node2 = inspector.get_commands(tid_b, node_id(2), command::PUSH);
		CHECK(pushes_node2.size() == 1);
		const auto computes_node2 = inspector.get_commands(tid_b, node_id(2), command::COMPUTE);
		CHECK(computes_node2.size() == 1);
		REQUIRE(inspector.has_dependency(*computes_node2.cbegin(), *pushes_node2.cbegin()));

		maybe_print_graph(tm);
		maybe_print_graph(ggen);
	}

} // namespace detail
} // namespace celerity

/**
 * This test suite uses a custom main function to add additional CLI flags.
 */
int main(int argc, char* argv[]) {
	Catch::Session session;

	using namespace Catch::clara;
	const auto cli = session.cli() | Opt(print_graphs)["--print-graphs"]("print graphs (GraphViz)");

	session.cli(cli);

	const int returnCode = session.applyCommandLine(argc, argv);
	if(returnCode != 0) { return returnCode; }

	return session.run();
}
