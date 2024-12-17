#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>

#include "command_graph_generator_test_utils.h"
#include "instruction_graph_test_utils.h"
#include "task_graph_test_utils.h"


using namespace celerity;
using namespace celerity::detail;

namespace acc = celerity::access;

// NOCOMMIT That doesnt work unfortunately - the plan was to submit different tasks after the template is ready,
//          and observe that the original task is still being created. That is not possible though because
//          the task object itself will still be recorded as-is. We could only achieve a different graph by
//          having different dependencies, but we cannot change dependencies without having different buffer
//          accesses, which in turn would again result in different task records.
//
//          => For CDAG/IDAG we could maybe do it. But it's a hack in any case.
//
// TEST_CASE("NOCOMMIT loop template works kinda illegal") {
// 	test_utils::allow_max_log_level(log_level::critical); // NOCOMMIT Just for debug output

// 	test_utils::tdag_test_context normal_tctx(1);
// 	normal_tctx.set_horizon_step(1000); // NOCOMMIT Figure out horizons

// 	test_utils::tdag_test_context loop_tctx(1);
// 	loop_tctx.set_horizon_step(1000); // NOCOMMIT Figure out horizons
// 	loop_template templ;
// 	loop_tctx.set_active_loop_template(&templ);

// 	auto normal_buf = normal_tctx.create_buffer(range<1>(128), true /* host initialized */);
// 	auto loop_buf = loop_tctx.create_buffer(range<1>(128), true /* host initialized */);

// 	normal_tctx.device_compute(range<1>(128)).read_write(normal_buf, acc::one_to_one()).submit();
// 	normal_tctx.device_compute(range<1>(128)).read_write(normal_buf, acc::one_to_one()).submit();

// 	loop_tctx.device_compute(range<1>(128)).read_write(loop_buf, acc::one_to_one()).submit();
// 	loop_tctx.device_compute(range<1>(128)).read_write(loop_buf, acc::one_to_one()).submit();
// }

TEST_CASE("task_manager does not automatically create horizon tasks when loop templates are enabled") {
	const bool use_loop_templates = GENERATE(true, false);

	test_utils::tdag_test_context tctx(1);
	tctx.set_horizon_step(1);
	auto buf = tctx.create_buffer(range<1>(128), true /* host initialized */);
	loop_template templ;
	tctx.set_active_loop_template(use_loop_templates ? &templ : nullptr);
	const size_t iterations = 10;
	for(size_t i = 0; i < iterations; ++i) {
		tctx.device_compute(buf.get_range()).read_write(buf, acc::one_to_one()).submit();
	}
	const size_t expected = 1 /* init epoch */ + iterations + (use_loop_templates ? 0ull : iterations /* horizons */);
	CHECK(tctx.query_tasks().count() == expected);
}

template <typename BuilderCallback>
void compare_task_graphs(const BuilderCallback& cb) {
	test_utils::tdag_test_context normal_ctx(1);
	auto normal_loop_wrapper = [&](const size_t n, auto loop_fn) {
		// To emulate the horizon behavior of loop templates, we disable automatic
		// horizon generation and instead create one horizon at the start of each iteration.
		const int previous_horizon_step = task_manager_testspy::get_horizon_step(normal_ctx.get_task_manager());
		normal_ctx.set_horizon_step(INT_MAX);
		for(size_t i = 0; i < n; ++i) {
			task_manager_testspy::generate_horizon(normal_ctx.get_task_manager());
			loop_fn();
		}
		normal_ctx.set_horizon_step(previous_horizon_step);
	};
	cb(normal_ctx, normal_loop_wrapper);

	test_utils::tdag_test_context loop_ctx(1);
	auto loop_wrapper = [&](const size_t n, auto loop_fn) {
		loop_template templ;
		loop_ctx.set_active_loop_template(&templ);
		for(size_t i = 0; i < n; ++i) {
			loop_ctx.get_task_manager().begin_loop_template_iteration(templ); // NOCOMMIT Ugh
			loop_fn();
			templ.tdag.complete_iteration();
		}
		CHECK((templ.tdag.is_primed && templ.tdag.is_verified)); // Sanity check
		loop_ctx.set_active_loop_template(nullptr);
	};
	cb(loop_ctx, loop_wrapper);

	auto normal_records = normal_ctx.query_tasks().raw();
	auto loop_records = loop_ctx.query_tasks().raw();
	REQUIRE(normal_records.size() == loop_records.size());

	for(size_t i = 0; i < normal_records.size(); ++i) {
		// This also compares dependencies, which are stored as part of the task record
		REQUIRE_LOOP(*normal_records[i] == *loop_records[i]);
	}
}

TEST_CASE("loop templates allow a single task to be instantiated several times without having to re-compute dependencies") {
	test_utils::allow_max_log_level(log_level::critical); // NOCOMMIT Just for debug output

	// NOCOMMIT TODO: How can we check that the template was actually applied? If we just do the default path twice it would also pass
	SECTION("read/write") {
		compare_task_graphs([](test_utils::tdag_test_context& tctx, auto loop) {
			auto buf = tctx.create_buffer(range<1>(128), true /* host initialized */);

			loop(5, [&]() { //
				tctx.device_compute(range<1>(128)).read_write(buf, acc::one_to_one()).submit();
			});
		});
	}

	// Repeated read accesses are interesting because they need to be anchored to the last applied horizon
	// (as opposed to read/write, which only depend on the previous iteration).
	SECTION("only reads - correct horizon dependencies") {
		compare_task_graphs([](test_utils::tdag_test_context& tctx, auto loop) {
			auto buf = tctx.create_buffer(range<1>(128), true /* host initialized */);

			loop(5, [&]() { //
				tctx.device_compute(range<1>(128)).read(buf, acc::one_to_one()).submit();
			});
		});
	}
}

TEST_CASE("loop templates allow a set of tasks to be instantiated several times without having to re-compute dependencies") {
	test_utils::allow_max_log_level(log_level::critical); // NOCOMMIT Just for debug output

	// NOCOMMIT TODO: How can we check that the template was actually applied? If we just do the default path twice it would also pass
	compare_task_graphs([](test_utils::tdag_test_context& tctx, auto loop) {
		auto buf = tctx.create_buffer(range<1>(128), true /* host initialized */);

		loop(5, [&]() {
			tctx.device_compute(range<1>(128)).read(buf, acc::all()).submit();
			tctx.device_compute(range<1>(128)).read_write(buf, acc::one_to_one()).submit();
		});
	});
}

TEST_CASE("NOCOMMIT loop template TDAG phases") {
	test_utils::allow_max_log_level(log_level::critical); // NOCOMMIT Just for debug output

	test_utils::tdag_test_context tctx(1);

	tctx.set_horizon_step(1000); // NOCOMMIT Figure out horizons

	auto buf = tctx.create_buffer(range<1>(128), true /* host initialized */);

	loop_template templ;
	tctx.set_active_loop_template(&templ);
	CHECK(!templ.tdag.is_primed);
	CHECK(!templ.tdag.is_verified);
	tctx.device_compute(range<1>(128)).read_write(buf, acc::one_to_one()).submit();
	templ.tdag.complete_iteration();
	CHECK(!templ.tdag.is_primed);
	CHECK(!templ.tdag.is_verified);
	tctx.device_compute(range<1>(128)).read_write(buf, acc::one_to_one()).submit();
	templ.tdag.complete_iteration();
	CHECK(templ.tdag.is_primed);
	CHECK(!templ.tdag.is_verified);
	tctx.device_compute(range<1>(128)).read_write(buf, acc::one_to_one()).submit();
	templ.tdag.complete_iteration();
	CHECK(templ.tdag.is_primed);
	CHECK(templ.tdag.is_verified);
}

TEST_CASE("NOCOMMIT TDAG sanity checks kitchen sink") {
	test_utils::allow_max_log_level(log_level::critical); // NOCOMMIT Just for debug output

	test_utils::tdag_test_context tctx(1);

	tctx.set_horizon_step(1000); // NOCOMMIT Figure out horizons

	auto buf0 = tctx.create_buffer(range<1>(128));
	auto buf1 = tctx.create_buffer(range<1>(128));

	tctx.device_compute(buf0.get_range()).name("init buf0").discard_write(buf0, acc::one_to_one()).submit();
	tctx.device_compute(buf1.get_range()).name("init buf1").discard_write(buf1, acc::one_to_one()).submit();

	loop_template templ;
	tctx.set_active_loop_template(&templ);

	// NOCOMMIT These no longer work. Need to do something with horizons and or epochs to change number of dependencies..? Not sure if even possible w/o hacks.

	SECTION("task has different dependency") {
		tctx.device_compute(range<1>(128)).read_write(buf0, acc::one_to_one()).submit();
		templ.tdag.complete_iteration(); // NOCOMMIT Come up with a proper way
		tctx.device_compute(range<1>(128)).read_write(buf0, acc::one_to_one()).submit();
		templ.tdag.complete_iteration(); // NOCOMMIT Come up with a proper way
		CHECK_THROWS_WITH(tctx.device_compute(range<1>(128)).read_write(buf1, acc::one_to_one()).submit(), "TDAG dependency mismatch");
	}

	SECTION("task has fewer dependencies") {
		tctx.device_compute(range<1>(128)).read(buf0, acc::one_to_one()).read(buf1, acc::one_to_one()).submit();
		templ.tdag.complete_iteration(); // NOCOMMIT Come up with a proper way
		tctx.device_compute(range<1>(128)).read(buf0, acc::one_to_one()).read(buf1, acc::one_to_one()).submit();
		templ.tdag.complete_iteration(); // NOCOMMIT Come up with a proper way
		CHECK_THROWS_WITH(tctx.device_compute(range<1>(128)).read(buf0, acc::one_to_one()).submit(), "Different number of dependencies in task");
	}

	SECTION("task has more dependencies") {
		tctx.device_compute(range<1>(128)).read_write(buf0, acc::one_to_one()).submit();
		templ.tdag.complete_iteration(); // NOCOMMIT Come up with a proper way
		tctx.device_compute(range<1>(128)).read_write(buf0, acc::one_to_one()).submit();
		templ.tdag.complete_iteration(); // NOCOMMIT Come up with a proper way
		CHECK_THROWS_WITH(tctx.device_compute(range<1>(128)).read_write(buf0, acc::one_to_one()).read_write(buf1, acc::one_to_one()).submit(),
		    "Different number of dependencies in task");
	}
}

// NOCOMMIT TODO: Test all other task types (host_compute, collective, fence, master_node, ...)

// ==================== CDAG


namespace celerity::detail {

// NOCOMMIT TODO: Do we care about ODR violations?
struct command_graph_generator_testspy {
	static auto create_batch() { return command_graph_generator::batch{}; }
	static command* clone_command(command_graph_generator& cggen, command_graph_generator::batch& batch, const command* cmd, const task& tsk) {
		return cggen.clone_command(batch, cmd, tsk);
	}
	static void add_dependency(command_graph_generator& cggen, command* from, command* to, dependency_kind kind, dependency_origin origin) {
		cggen.add_dependency(from, to, kind, origin);
	}
};

struct cdag_loop_template_testspy {
	static std::vector<const command*> get_previous_batch(const cdag_loop_template& templ) { return templ.m_previous_batch; }
};

} // namespace celerity::detail

TEST_CASE("NOCOMMIT CDAG whitebox") {
	test_utils::allow_max_log_level(log_level::critical); // NOCOMMIT Just for debug output

	struct tm_delegate : public task_manager::delegate {
		std::vector<const task*> tasks;
		void task_created(const task* tsk) override { tasks.push_back(tsk); }
	};

	// TODO: Call instantiate manually and observe callbacks
	// => Make clone function in CGGEN accessible through testspy
	task_graph tdag;
	tm_delegate tm_del;
	task_manager tm(1, tdag, nullptr, &tm_del);
	tm.set_horizon_step(1000); // NOCOMMIT Figure out horizons

	command_graph cdag;
	command_graph_generator cggen(1, 0, cdag, nullptr);

	tm.generate_epoch_task(epoch_action::init);
	cggen.build_task(*tm_del.tasks[0], nullptr);

	test_utils::mock_buffer_factory mbf{tm, cggen};
	auto buf = mbf.create_buffer(range<1>(128), true /* host initialized */);

	loop_template templ;

	for(size_t i = 0; i < 6; ++i) {
		auto cg = invoke_command_group_function([&buf](handler& cgh) {
			buf.get_access<access_mode::read_write>(cgh, acc::one_to_one());
			cgh.parallel_for(range<1>(128), [](item<1>) {});
		});
		tm.generate_command_group_task(std::move(cg), &templ); // NOCOMMIT Pass templ in here? We don't actually need it? Or also test TDAG interaction here?
	}

	const auto has_dependency = [](const command* const from, const command* const to) {
		return std::any_of(from->get_dependencies().begin(), from->get_dependencies().end(), [to](const auto& dep) { return dep.node == to; });
	};

	CHECK(!templ.cdag.is_primed);
	CHECK(!templ.cdag.is_verified);
	const auto batch1 = cggen.build_task(*tm_del.tasks.at(1), &templ);
	templ.cdag.complete_iteration();

	CHECK(!templ.cdag.is_primed);
	CHECK(!templ.cdag.is_verified);
	// NOCOMMIT TODO UNRELATED: I accidentally had tasks.at(1) here again (i.e. building the same task twice)
	// 	=> This did NOT produce an error w/ CGGEN after task geometries, only after moving this onto current master CGGEN. Why? Missing assertion?
	const auto batch2 = cggen.build_task(*tm_del.tasks.at(2), &templ);
	templ.cdag.complete_iteration();
	CHECK(has_dependency(batch2.at(0), batch1.at(0)));

	CHECK(!templ.cdag.is_primed); // NOCOMMIT We currently have to prime thrice - revisit (make dynamic based on DAG; up to 3 times)
	CHECK(!templ.cdag.is_verified);
	const auto batch3 = cggen.build_task(*tm_del.tasks.at(3), &templ);
	templ.cdag.complete_iteration();
	CHECK(has_dependency(batch3.at(0), batch2.at(0)));

	CHECK(templ.cdag.is_primed);
	CHECK(!templ.cdag.is_verified);
	const auto batch4 = cggen.build_task(*tm_del.tasks.at(4), &templ);
	templ.cdag.complete_iteration();
	CHECK(has_dependency(batch4.at(0), batch3.at(0)));

	CHECK(templ.cdag.is_primed);
	CHECK(templ.cdag.is_verified);

	// We now instantiate the template manually
	auto manual_batch = command_graph_generator_testspy::create_batch();
	const auto clone_cmd = [&](const command& cmd) { return command_graph_generator_testspy::clone_command(cggen, manual_batch, &cmd, *tm_del.tasks.at(5)); };
	const auto add_dependency = [&](command* from, command* to, dependency_kind kind, dependency_origin origin) {
		command_graph_generator_testspy::add_dependency(cggen, from, to, kind, origin);
	};
	templ.cdag.instantiate(clone_cmd, add_dependency);
	templ.cdag.complete_iteration();
	REQUIRE(manual_batch.size() == 1);

	const auto& batch4_cmd = dynamic_cast<const execution_command&>(*batch4[0]);
	const auto& manual_batch_cmd = dynamic_cast<const execution_command&>(*manual_batch[0]);
	CHECK(manual_batch_cmd.get_execution_spec() == batch4_cmd.get_execution_spec());
	CHECK(batch4_cmd.get_task() == tm_del.tasks.at(4));
	CHECK(manual_batch_cmd.get_task() == tm_del.tasks.at(5));
	CHECK(has_dependency(&manual_batch_cmd, &batch4_cmd));

	// Now build through CGGEN again
	const auto batch6 = cggen.build_task(*tm_del.tasks.at(6), &templ);
	templ.cdag.complete_iteration();
	CHECK(has_dependency(batch6.at(0), manual_batch[0]));
	// Verify that the template was actually used
	CHECK(cdag_loop_template_testspy::get_previous_batch(templ.cdag) == batch6);
}

template <typename BuilderCallback>
void compare_command_graphs(const size_t num_nodes, const BuilderCallback& cb) {
	test_utils::cdag_test_context normal_ctx(num_nodes);
	auto normal_loop_wrapper = [&](const size_t n, auto loop_fn) {
		// To emulate the horizon behavior of loop templates, we disable automatic horizon generation
		// and instead create one horizon at the start of each iteration.
		const int previous_horizon_step = task_manager_testspy::get_horizon_step(normal_ctx.get_task_manager());
		normal_ctx.set_horizon_step(INT_MAX);
		for(size_t i = 0; i < n; ++i) {
			task_manager_testspy::generate_horizon(normal_ctx.get_task_manager());
			loop_fn();
		}
		normal_ctx.set_horizon_step(previous_horizon_step);
	};
	cb(normal_ctx, normal_loop_wrapper);

	test_utils::cdag_test_context loop_ctx(num_nodes);
	auto loop_wrapper = [&](const size_t n, auto loop_fn) {
		std::vector<loop_template> templs(num_nodes);
		std::vector<loop_template*> templ_ptrs;
		for(auto& templ : templs) {
			templ_ptrs.push_back(&templ); // NOCOMMIT Ugh, figure this out
		}
		loop_ctx.set_active_loop_templates(templ_ptrs);
		for(size_t i = 0; i < n; ++i) {
			loop_ctx.get_task_manager().begin_loop_template_iteration(templs[0]); // NOCOMMIT ULTRA HACK: Relying on the fact that context uses [0] for TM
			loop_fn();
			templs[0].tdag.complete_iteration(); // NOCOMMIT ULTRA HACK: Relying on the fact that context uses [0] for TM
			for(auto& templ : templs) {
				templ.cdag.complete_iteration();
			}
		}
		loop_ctx.set_active_loop_templates(std::vector<loop_template*>(num_nodes, nullptr));
		// Sanity checks
		CHECK((templs[0].tdag.is_primed && templs[0].tdag.is_verified)); // TDAG NOCOMMIT ULTRA HACK
		CHECK(std::all_of(templs.begin(), templs.end(), [](const loop_template& templ) { return templ.cdag.is_primed && templ.cdag.is_verified; }));
		CHECK(std::all_of(templs.begin(), templs.end(), [](const loop_template& templ) { return templ.cdag.loop_instantiations > 0; }));
	};
	cb(loop_ctx, loop_wrapper);

	for(size_t n = 0; n < num_nodes; ++n) {
		INFO("Node " << n);
		const auto normal_records = normal_ctx.query().on(n).raw();
		const auto loop_records = loop_ctx.query().on(n).raw();
		REQUIRE(normal_records.size() == loop_records.size());

		for(size_t i = 0; i < normal_records.size(); ++i) {
			matchbox::match(*normal_records[i], [&](const auto& nrec) {
				const auto lrec = dynamic_cast<const decltype(nrec)&>(*loop_records[i]); // cast will throw if types don't match
				REQUIRE_LOOP(nrec == lrec);
			});
		}

		const auto& normal_rec = normal_ctx.get_command_recorder(n);
		const auto& loop_rec = loop_ctx.get_command_recorder(n);

		// Instantiating the template does not reproduce anti-dependencies that are subsequently subsumed by true dependencies
		// We therefore just check the dependency kind (and origin) from loop records to normal records, but not the other way round
		CHECK(std::all_of(loop_rec.get_dependencies().begin(), loop_rec.get_dependencies().end(), [&normal_rec](const command_dependency_record& ldep) {
			return std::any_of(normal_rec.get_dependencies().begin(), normal_rec.get_dependencies().end(), [&ldep](const command_dependency_record& ndep) {
				return ldep.predecessor == ndep.predecessor && ldep.successor == ndep.successor && ldep.kind == ndep.kind && ldep.origin == ndep.origin;
			});
		}));
		CHECK(std::all_of(normal_rec.get_dependencies().begin(), normal_rec.get_dependencies().end(), [&loop_rec](const command_dependency_record& ndep) {
			return std::any_of(loop_rec.get_dependencies().begin(), loop_rec.get_dependencies().end(),
			    [&ndep](const command_dependency_record& ldep) { return ldep.predecessor == ndep.predecessor && ldep.successor == ndep.successor; });
		}));
	}
}

TEST_CASE("loop templates allow commands for a single task to be instantiated several times without having to re-compute dependencies") {
	test_utils::allow_max_log_level(log_level::critical); // NOCOMMIT Just for debug output

	const size_t num_nodes = GENERATE(1, 2, 4);
	CAPTURE(num_nodes);

	SECTION("read/write") {
		compare_command_graphs(num_nodes, [](test_utils::cdag_test_context& cctx, auto loop) {
			auto buf = cctx.create_buffer(range<1>(128), true /* host initialized */);

			loop(5, [&]() { //
				cctx.device_compute(range<1>(128)).read_write(buf, acc::one_to_one()).read(buf, acc::all{}).submit();
			});
		});
	}

	SECTION("only reads - correct horizon dependencies") {
		compare_command_graphs(num_nodes, [](test_utils::cdag_test_context& cctx, auto loop) {
			auto buf = cctx.create_buffer(range<1>(128), true /* host initialized */);

			loop(5, [&]() { //
				cctx.device_compute(range<1>(128)).read(buf, acc::one_to_one()).submit();
			});
		});
	}
}

// NOCOMMIT TODO: Create issue in development board: The CDAG generated here is wrong. C11 should also anti-depend on C8.
//                It doesn't because we only look at successors of the last writer, which at this point is horizon C7.
//                To properly handle this we'd have to do something like the read access front in IGGEN.
TEST_CASE("finalizing loop template correctly updates reader data structures for anti-dependency handling") {
	const size_t num_nodes = GENERATE(1, 2, 4);
	CAPTURE(num_nodes);

	compare_command_graphs(num_nodes, [](test_utils::cdag_test_context& cctx, auto loop) {
		auto buf = cctx.create_buffer(range<1>(128), true /* host initialized */);

		loop(5, [&]() { //
			cctx.device_compute(range<1>(128)).read(buf, acc::one_to_one()).submit();
		});

		cctx.device_compute(range<1>(128)).discard_write(buf, acc::one_to_one()).submit();
	});
}

// TODO: Do we need the same test for TDAG/IGGEN?
TEST_CASE("finalizing loop template correctly updates epoch for new commands") {
	const size_t num_nodes = GENERATE(1, 2, 4);
	CAPTURE(num_nodes);

	compare_command_graphs(num_nodes, [](test_utils::cdag_test_context& cctx, auto loop) {
		auto buf0 = cctx.create_buffer(range<1>(128));
		auto buf1 = cctx.create_buffer(range<1>(128));

		// Create and apply several horizons
		// NOCOMMIT TODO: This must work for 5 and 6 iterations (i.e., 1 instantiation - TODO: Ensure that it's only one)
		loop(5, [&]() { cctx.device_compute(range<1>(128)).discard_write(buf0, acc::one_to_one()).submit(); });

		// Now submit independent task operating on buffer 1. It should be anchored to the latest horizon (effective epoch).
		cctx.device_compute(range<1>(128)).discard_write(buf1, acc::one_to_one()).submit();
	});
}

TEST_CASE("loop templates allow commands for a set of tasks to be instantiated several times without having to re-compute dependencies") {
	test_utils::allow_max_log_level(log_level::critical); // NOCOMMIT Just for debug output

	const size_t num_nodes = GENERATE(1, 2, 4);
	CAPTURE(num_nodes);

	compare_command_graphs(num_nodes, [](test_utils::cdag_test_context& cctx, auto loop) {
		auto buf = cctx.create_buffer(range<1>(128), true /* host initialized */);

		loop(5, [&]() {
			cctx.device_compute(range<1>(128)).read(buf, acc::all{}).submit();
			cctx.device_compute(range<1>(128)).read_write(buf, acc::one_to_one()).submit();
		});
	});
}

TEST_CASE("CDAG stencil pattern") {
	test_utils::allow_max_log_level(log_level::critical); // NOCOMMIT Just for debug output

	const size_t num_nodes = GENERATE(1, 2, 4);
	CAPTURE(num_nodes);

	compare_command_graphs(num_nodes, [](test_utils::cdag_test_context& cctx, auto loop) {
		auto buf0 = cctx.create_buffer(range<1>(128), true /* host initialized */);
		auto buf1 = cctx.create_buffer(range<1>(128), true);

		loop(5, [&]() {
			cctx.device_compute(range<1>(128)).name("ping").read(buf0, acc::neighborhood{{1}}).read_write(buf1, acc::one_to_one{}).submit();
			cctx.device_compute(range<1>(128)).name("pong").read(buf1, acc::neighborhood{{1}}).read_write(buf0, acc::one_to_one{}).submit();
		});
	});
}

TEST_CASE("CDAG detects non-idempotent loops") {
	const size_t num_nodes = 128;

	constexpr int N = 512;
	constexpr float dt = 0.25f;

	compare_command_graphs(num_nodes, [](test_utils::cdag_test_context& cctx, auto loop) {
		const auto fill = [&](test_utils::mock_buffer<2> u) { cctx.device_compute(u.get_range()).discard_write(u, celerity::access::one_to_one{}).submit(); };
		const auto step = [&](test_utils::mock_buffer<2> up, test_utils::mock_buffer<2> u) {
			cctx.device_compute(up.get_range())
			    .read_write(up, celerity::access::one_to_one{})
			    .read(u, celerity::access::neighborhood{{1, 1}, celerity::neighborhood_shape::along_axes})
			    .submit();
		};

		auto up = cctx.create_buffer(range<2>(N, N));
		auto u = cctx.create_buffer(range<2>(N, N));

		fill(u);
		fill(up);
		step(up, u);

		loop(10, [&]() {
			step(up, u);
			std::swap(u, up);
			FAIL("TODO: This bit me in DAG benchmarks - can we detect that we are not idempotent here?");
		});
	});
}

// NOCOMMIT TEST: Horizon commands don't store complete reductions yet -> or can we make it work..?

// ==================== IDAG

struct iteration_count {
	size_t num_iterations_after_calibration = 0; // NOCOMMIT TODO Terminology?
	iteration_count operator+(const size_t n) const {
		iteration_count copy = *this;
		copy.num_iterations_after_calibration += n;
		return copy;
	}
};

constexpr iteration_count until_calibrated;

template <typename BuilderCallback>
void compare_instruction_graphs(const size_t num_nodes, const BuilderCallback& cb) {
	// TODO: Make these configurable?
	constexpr node_id local_nid = 0;
	constexpr size_t devices_per_node = 2; // Two devices so graphs are smaller and easier to debug

	// We cannot compare graphs with staging copies because they are not the same: The IGGEN re-uses staging allocations
	// after the last use is behind an epoch (i.e., two horizons) to avoid introducing unnecessary dependencies between
	// copies. Loop templates do not replicate this behavior, because they always clone the previous batch of instructions,
	// i.e., staging allocations are re-used after one iteration. This is correct in the case of loop templates, because
	// there is no way two staged copies from two consecutive iterations could run concurrently:
	// If the copy is for a write access, the second copy will have an anti-dependency on the write.
	// If the copy is for a pure read access, then there will be no second copy, because the data is still up-to-date.
	constexpr bool supports_d2d_copies = true;

	auto iterate_dynamic = [iteration_counts = std::vector<size_t>(), loop_idx = size_t(0), normal = false](
	                           const auto& specifier, const auto& loop_fn) mutable {
		if constexpr(!std::is_same_v<std::decay_t<decltype(specifier)>, iteration_count>) { // Static iteration count
			for(size_t i = 0; i < size_t(specifier); ++i) {
				loop_fn();
			}
		} else {                                                           // Dynamic iteration count
			if constexpr(std::is_invocable_r_v<bool, decltype(loop_fn)>) { // This is a loop template invocation
				size_t calibration_count = 1;
				while(!loop_fn()) {
					calibration_count++;
				}
				for(size_t i = 0; i < specifier.num_iterations_after_calibration; ++i) {
					loop_fn();
				}
				iteration_counts.push_back(calibration_count + specifier.num_iterations_after_calibration);
			} else { // This is a normal invocation
				if(!normal) {
					normal = true;
					loop_idx = 0;
				}
				REQUIRE(loop_idx < iteration_counts.size());
				for(size_t i = 0; i < iteration_counts[loop_idx]; ++i) {
					loop_fn();
				}
				loop_idx++;
			}
		}
	};

	test_utils::idag_test_context loop_ctx(num_nodes, local_nid, devices_per_node, supports_d2d_copies);
	auto loop_wrapper = [&](auto n, auto loop_fn) {
		loop_template templ;
		loop_ctx.set_active_loop_template(&templ);
		iterate_dynamic(n, [&] {
			loop_ctx.get_task_manager().begin_loop_template_iteration(templ);
			loop_fn();
			templ.tdag.complete_iteration();
			// NOCOMMIT TODO: Do we even want to use template for CDAG?
			templ.cdag.complete_iteration();
			templ.idag.complete_iteration();
			return templ.idag.is_verified;
		});
		loop_ctx.set_active_loop_template(nullptr);
		// Sanity checks
		CHECK((templ.tdag.is_primed && templ.tdag.is_verified));
		CHECK((templ.cdag.is_primed && templ.cdag.is_verified));
		CHECK(templ.cdag.loop_instantiations > 0);
		CHECK((templ.idag.is_primed && templ.idag.is_verified));
		CHECK(templ.idag.loop_instantiations > 0);
	};
	cb(loop_ctx, loop_wrapper);

	test_utils::idag_test_context normal_ctx(num_nodes, local_nid, devices_per_node, supports_d2d_copies);
	auto normal_loop_wrapper = [&](auto n, auto loop_fn) {
		// To emulate the horizon behavior of loop templates, we disable automatic horizon generation
		// and instead create one horizon at the start of each iteration.
		const int previous_horizon_step = task_manager_testspy::get_horizon_step(normal_ctx.get_task_manager());
		normal_ctx.set_horizon_step(INT_MAX);
		iterate_dynamic(n, [&] {
			task_manager_testspy::generate_horizon(normal_ctx.get_task_manager());
			loop_fn();
		});
		normal_ctx.set_horizon_step(previous_horizon_step);
	};
	cb(normal_ctx, normal_loop_wrapper);

	// NOCOMMIT Just hacking
	{
		const auto getoption = [](const std::string_view name) -> std::optional<size_t> {
			const auto cstr = getenv(name.data());
			if(cstr == nullptr) return std::nullopt;
			return atol(cstr);
		};

		// TODO: In a proper query language we would probably like to specify whether we want only true dependencies or all
		const auto filter_by_tid = getoption("GRAPH_QUERY_TID");
		const auto before = getoption("GRAPH_QUERY_BEFORE");
		const auto after = getoption("GRAPH_QUERY_AFTER");

		const auto& normal_rec = normal_ctx.get_instruction_recorder();
		const auto& loop_rec = loop_ctx.get_instruction_recorder();
		const_cast<instruction_recorder&>(normal_rec).filter_by_task_id(*filter_by_tid, before.value_or(0), after.value_or(0));
		const_cast<instruction_recorder&>(loop_rec).filter_by_task_id(*filter_by_tid, before.value_or(0), after.value_or(0));
	}

	// NOCOMMIT Here and for TDAG/CDAG: We don't really need raw() when we have get_*_recorder
	const auto normal_records = normal_ctx.query_instructions().raw();
	const auto loop_records = loop_ctx.query_instructions().raw();
	REQUIRE(normal_records.size() == loop_records.size());

	for(size_t i = 0; i < normal_records.size(); ++i) {
		matchbox::match(*normal_records[i], [&](const auto& nrec) {
			const auto lrec = dynamic_cast<const decltype(nrec)&>(*loop_records[i]); // cast will throw if types don't match
			INFO("Instruction " << nrec.id << " / " << lrec.id);
			REQUIRE_LOOP(nrec == lrec);
		});
	}

	const auto& normal_rec = normal_ctx.get_instruction_recorder();
	const auto& loop_rec = loop_ctx.get_instruction_recorder();

	// Instantiating the template does not reproduce dependency origins, as those are only stored in recorder.
	CHECK(std::all_of(loop_rec.get_dependencies().begin(), loop_rec.get_dependencies().end(), [&normal_rec](const instruction_dependency_record& ldep) {
		return std::any_of(normal_rec.get_dependencies().begin(), normal_rec.get_dependencies().end(),
		    [&ldep](const instruction_dependency_record& ndep) { return ldep.predecessor == ndep.predecessor && ldep.successor == ndep.successor; });
	}));
	CHECK(std::all_of(normal_rec.get_dependencies().begin(), normal_rec.get_dependencies().end(), [&loop_rec](const instruction_dependency_record& ndep) {
		return std::any_of(loop_rec.get_dependencies().begin(), loop_rec.get_dependencies().end(),
		    [&ndep](const instruction_dependency_record& ldep) { return ldep.predecessor == ndep.predecessor && ldep.successor == ndep.successor; });
	}));
}

// NOCOMMIT TODO: We need to DRY this up - same test for all three graphs
TEST_CASE("loop templates allow instructions for a single task to be instantiated several times without having to re-compute dependencies") {
	test_utils::allow_max_log_level(log_level::critical); // NOCOMMIT Just for debug output

	const size_t num_nodes = GENERATE(1, 2, 4);
	const size_t iterations = GENERATE(1, 5);
	CAPTURE(num_nodes);
	CAPTURE(iterations);

	SECTION("read/write") {
		compare_instruction_graphs(num_nodes, [=](test_utils::idag_test_context& ictx, auto loop) {
			auto buf = ictx.create_buffer(range<1>(128), true /* host initialized */);

			loop(until_calibrated + iterations, [&]() { //
				ictx.device_compute(range<1>(128)).read_write(buf, acc::one_to_one()).read(buf, acc::all{}).submit();
			});

			ictx.finish();
		});
	}

	SECTION("only reads - correct horizon dependencies") {
		compare_instruction_graphs(num_nodes, [=](test_utils::idag_test_context& ictx, auto loop) {
			auto buf = ictx.create_buffer(range<1>(128), true /* host initialized */);

			loop(until_calibrated + iterations, [&]() { //
				ictx.device_compute(range<1>(128)).read(buf, acc::one_to_one()).submit();
			});

			ictx.finish();
		});
	}
}

TEST_CASE("loop templates allow instructions for a set of tasks to be instantiated several times without having to re-compute dependencies") {
	test_utils::allow_max_log_level(log_level::critical); // NOCOMMIT Just for debug output

	const size_t num_nodes = GENERATE(1, 2, 4);
	const size_t iterations = GENERATE(1, 5);
	CAPTURE(num_nodes);

	compare_instruction_graphs(num_nodes, [=](test_utils::idag_test_context& ictx, auto loop) {
		auto buf = ictx.create_buffer(range<1>(128), true /* host initialized */);

		loop(until_calibrated + iterations, [&]() {
			ictx.device_compute(range<1>(128)).read(buf, acc::all{}).submit();
			ictx.device_compute(range<1>(128)).read_write(buf, acc::one_to_one()).submit();
		});

		ictx.finish();
	});
}

TEST_CASE("IDAG stencil pattern") {
	test_utils::allow_max_log_level(log_level::critical); // NOCOMMIT Just for debug output
	enum class test_variant { loop_only, with_preamble, with_epilogue, with_both };

	const size_t num_nodes = GENERATE(1, 2, 4);
	const test_variant variant = GENERATE(test_variant::loop_only, test_variant::with_preamble, test_variant::with_epilogue, test_variant::with_both);
	CAPTURE(num_nodes, variant);

	compare_instruction_graphs(num_nodes, [variant](test_utils::idag_test_context& ictx, auto loop) {
		auto buf0 = ictx.create_buffer(range<1>(128), true /* host initialized */);
		auto buf1 = ictx.create_buffer(range<1>(128), true);

		const auto loop_body = [&]() {
			ictx.device_compute(range<1>(128)).name("ping").read(buf0, acc::neighborhood{{1}}).read_write(buf1, acc::one_to_one{}).submit();
			ictx.device_compute(range<1>(128)).name("pong").read(buf1, acc::neighborhood{{1}}).read_write(buf0, acc::one_to_one{}).submit();
		};

		if(variant == test_variant::with_preamble || variant == test_variant::with_both) {
			for(int i = 0; i < 10; ++i) {
				loop_body();
			}
		}

		loop(7, [&]() { loop_body(); });

		if(variant == test_variant::with_epilogue || variant == test_variant::with_both) {
			for(int i = 0; i < 10; ++i) {
				loop_body();
			}
		}

		ictx.finish();
	});
}

#include "geometry_builder.h" // NOCOMMIT

TEST_CASE("IDAG outset stencil") {
	test_utils::allow_max_log_level(log_level::critical); // NOCOMMIT Just for debug output

	// const size_t num_nodes = GENERATE(1, 2, 4);
	// const size_t num_nodes = 16;
	const size_t num_nodes = 128;
	CAPTURE(num_nodes);

	compare_instruction_graphs(num_nodes, [](test_utils::idag_test_context& ictx, auto loop) {
		auto buf0 = ictx.create_buffer(range<2>(128, 128), true /* host initialized */);
		auto buf1 = ictx.create_buffer(range<2>(128, 128), true);

		size_t i = 0;
		loop(7, [&]() {
			constexpr size_t outset = 2;
			const size_t inner_iterations = outset % 2 == 0 ? 2 * (outset + 1) : outset + 1;

			for(size_t j = 0; j < inner_iterations; ++j) {
				{
					const size_t current_outset = outset - i % (outset + 1);
					celerity::geometry_builder<2> gb{buf0.get_range()};
					gb.split_2d_but_recursive_and_only_for_local_chunks_v2_electric_boogaloo(16, 2, 0);
					gb.outset(current_outset);
					const auto geo = gb.make();
					ictx.device_compute(geo).name("ping").read(buf0, acc::neighborhood{{1, 1}}).read_write_replicated(buf1, acc::one_to_one{}).submit();
				}
				i++;

				{
					const size_t current_outset = outset - i % (outset + 1);
					celerity::geometry_builder<2> gb{buf0.get_range()};
					gb.split_2d_but_recursive_and_only_for_local_chunks_v2_electric_boogaloo(16, 2, 0);
					gb.outset(current_outset);
					const auto geo = gb.make();
					ictx.device_compute(geo).name("pong").read(buf1, acc::neighborhood{{1, 1}}).read_write_replicated(buf0, acc::one_to_one{}).submit();
				}
				i++;
			}
		});

		ictx.finish();
	});
}

// NOCOMMIT TEST: IDAG priming is skipped if batch contains allocations

// TODO: Buffer / task debug names are copied correctly between TDAG/CDAG/IDAG records

// TODO: Also add an end-to-end runtime test with a call to queue.loop()
// TODO: Test horizon handling
// TODO: Test diverging tasks - if we want to do some kind of detection here. Maybe do a policy set that is only enabled in debug builds and tests
// TODO: Test replacement list
// TODO: Test that graphs are also correctly generated AFTER loop template (i.e., template is correctly finalized)
// TODO: Epochs should print warning and disable template (we don't want to hard-fail here, because one might need to sync for debugging)
// => Actually we don't have to disable the template, it just isn't as effective because we can't overlap instantiations into the future. But
//    latency is still reduced, so maybe epochs are fine..? We already have the warning for excessive epochs anyway.
// TODO: Test what happens if we do stencil ping-pong in alternating iterations - hopefully we catch that as being wrong (if not we have to look at BAM!)
// TODO: On all tests: Add tags!
// TODO: For all variants: Empty loop bodies are handled gracefully (nothing blows up)

// TODO: CGGEN/IGGEN clone commands/instructions into new epochs (in CDAG/IDAG data structure - we create a new epoch for each cloned horizon)

// TODO: Can/should we support epochs.. Fences?
