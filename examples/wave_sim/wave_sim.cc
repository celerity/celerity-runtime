#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>

#include <celerity.h>

// TODO: Export from celerity.h
#include "geometry_builder.h"

#include "../fvm/hash.h"

// Optionally build simpler version of the stencil (which is not physically correct anymore),
// that only reads from one buffer instead of both in each time step.
#ifndef WAVE_SIMPLE
#define WAVE_SIMPLE 0
#endif

#ifndef DATA_T
#define DATA_T float
#endif

#define _DATA_T_STRING3(x) #x
#define _DATA_T_STRING2(x) _DATA_T_STRING3(x)
#define DATA_T_STRING _DATA_T_STRING2(DATA_T)

using DataT = DATA_T;

struct wave_sim_config {
	size_t N = 512; // Grid size
	double T = 100; // Time at end of simulation
	double dt = 0.25f;
	DataT dx = 1.f;
	DataT dy = 1.f;

	// "Sample" a frame every X iterations
	// (0 = don't produce any output)
	unsigned output_sample_rate = 0;

	unsigned outset = 0;
	unsigned oversub = 1;
	bool tiled = false;
};

void setup_wave(
    celerity::queue& queue, celerity::buffer<DataT, 2> u, sycl::vec<DataT, 2> center, DataT amplitude, sycl::vec<DataT, 2> sigma, const wave_sim_config& cfg) {
	queue.submit([&](celerity::handler& cgh) {
		celerity::accessor dw_u{u, cgh, celerity::access::one_to_one{}, celerity::write_only, celerity::no_init};
		if(cfg.oversub > 1) celerity::experimental::hint(cgh, celerity::experimental::hints::oversubscribe(cfg.oversub));
		if(cfg.tiled) celerity::experimental::hint(cgh, celerity::experimental::hints::split_2d());
		cgh.parallel_for<class setup_wave>(u.get_range(), [=, c = center, a = amplitude, s = sigma](celerity::item<2> item) {
			const DataT dx = item[1] - c.x();
			const DataT dy = item[0] - c.y();
			dw_u[item] = a * sycl::exp(-(dx * dx / (2.f * s.x() * s.x()) + dy * dy / (2.f * s.y() * s.y())));
		});
	});
}

void zero(celerity::queue& queue, celerity::buffer<DataT, 2> buf, const wave_sim_config& cfg) {
	queue.submit([&](celerity::handler& cgh) {
		celerity::accessor dw_buf{buf, cgh, celerity::access::one_to_one{}, celerity::write_only, celerity::no_init};
		if(cfg.oversub > 1) celerity::experimental::hint(cgh, celerity::experimental::hints::oversubscribe(cfg.oversub));
		if(cfg.tiled) celerity::experimental::hint(cgh, celerity::experimental::hints::split_2d());
		cgh.parallel_for<class zero>(buf.get_range(), [=](celerity::item<2> item) { dw_buf[item] = 0.f; });
	});
}

struct init_config {
	static constexpr DataT a = 0.5f;
	static constexpr DataT b = 0.0f;
	static constexpr DataT c = 0.5f;
};

struct update_config {
	static constexpr DataT a = 1.f;
	static constexpr DataT b = 1.f;
	static constexpr DataT c = 1.f;
};

template <typename T, typename Config, typename KernelName>
void step(celerity::queue& queue, celerity::buffer<T, 2> up, celerity::buffer<T, 2> u, const wave_sim_config& cfg, const size_t current_outset) {
	celerity::geometry_builder<2> gb{u.get_range()};
	if(cfg.tiled) {
		gb.split_2d_but_recursive_and_only_for_local_chunks();
	} else {
		gb.split_1d();
	}
	gb.outset(current_outset); // TODO: Should this only be along axes??
	// mgm.replicate();

	if(cfg.outset > 0 && cfg.oversub == 3 && current_outset == cfg.outset) {
		celerity::geometry_builder<2> gb2000{u.get_range()};
		if(cfg.tiled) {
			gb2000.split_2d();
		} else {
			gb2000.split_1d();
		}
		gb.splice(gb2000);
	}

	auto geo = gb.make();

	if(cfg.outset > 0 && (cfg.oversub > 1 && cfg.oversub != 3)) {
		CELERITY_CRITICAL("Custom geometry only supports oversub 3\n");
		std::exit(1);
	}

	queue.submit([&](celerity::handler& cgh) {
#if WAVE_SIMPLE
		celerity::accessor w_up{up, cgh, celerity::access::one_to_one{}, celerity::write_only_replicated, celerity::no_init};
#else
		celerity::accessor rw_up{up, cgh, celerity::access::one_to_one{}, celerity::read_write_replicated};
#endif
		celerity::accessor r_u{u, cgh, celerity::access::neighborhood{{1, 1}, celerity::neighborhood_shape::along_axes}, celerity::read_only};

		if(cfg.outset == 0 && cfg.oversub > 1) celerity::experimental::hint(cgh, celerity::experimental::hints::oversubscribe(cfg.oversub));
		if(cfg.outset == 0 && cfg.tiled) celerity::experimental::hint(cgh, celerity::experimental::hints::split_2d());

		const auto size = up.get_range();
		const DataT step_y = (cfg.dt / cfg.dy) * (cfg.dt / cfg.dy);
		const DataT step_x = (cfg.dt / cfg.dx) * (cfg.dt / cfg.dx);
#if !WAVE_SIMPLE
		const auto a2 = Config::a * 2;
#endif

		// TODO API: Should be behind unified perf assertions namespace or something. Also optionally scope to node/device.
		// => Maybe also have the reverse, assert that a data movement is required?
		if(cfg.outset > 0 && current_outset != cfg.outset) { cgh.assert_no_data_movement(); }

		const auto kernel = [=](celerity::item<2> item) {
			const size_t py = item[0] < size[0] - 1 ? item[0] + 1 : item[0];
			const size_t my = item[0] > 0 ? item[0] - 1 : item[0];
			const size_t px = item[1] < size[1] - 1 ? item[1] + 1 : item[1];
			const size_t mx = item[1] > 0 ? item[1] - 1 : item[1];

			const DataT cur = r_u[item];

			DataT lap = 0.f;
			lap += step_y * (r_u[{py, item[1]}] - cur);
			lap -= step_y * (cur - r_u[{my, item[1]}]);
			lap += step_x * (r_u[{item[0], px}] - cur);
			lap -= step_x * (cur - r_u[{item[0], mx}]);

#if WAVE_SIMPLE
			w_up[item] = Config::c * lap;
#else
			rw_up[item] = a2 * cur - Config::b * rw_up[item] + Config::c * lap;
#endif
		};

		if(cfg.outset > 0) {
			cgh.parallel_for<KernelName>(geo, kernel);
		} else {
			cgh.parallel_for<KernelName>(size, kernel);
		}
	});
}

void initialize(celerity::queue& queue, celerity::buffer<DataT, 2> up, celerity::buffer<DataT, 2> u, const wave_sim_config& cfg) {
	step<DataT, init_config, class initialize>(queue, up, u, cfg, 0);
}

void update(celerity::queue& queue, celerity::buffer<DataT, 2> up, celerity::buffer<DataT, 2> u, const wave_sim_config& cfg, const size_t current_outset) {
	step<DataT, update_config, class update>(queue, up, u, cfg, current_outset);
}

void stream_open(celerity::queue& queue, size_t N, size_t num_samples, celerity::experimental::host_object<std::ofstream> os) {
	queue.submit([&](celerity::handler& cgh) {
		celerity::experimental::side_effect os_eff{os, cgh};
		// Using `on_master_node` on all host tasks instead of `once` guarantees that all execute on the same cluster node and access the same file handle
		cgh.host_task(celerity::on_master_node, [=] {
			os_eff->open("wave_sim_result.bin", std::ios_base::out | std::ios_base::binary);
			const struct {
				uint64_t n, t;
			} header{N, num_samples};
			os_eff->write(reinterpret_cast<const char*>(&header), sizeof(header));
		});
	});
}

template <typename T>
void stream_append(celerity::queue& queue, celerity::buffer<T, 2> up, celerity::experimental::host_object<std::ofstream> os) {
	const auto range = up.get_range();
	queue.submit([&](celerity::handler& cgh) {
		celerity::accessor up_r{up, cgh, celerity::access::all{}, celerity::read_only_host_task};
		celerity::experimental::side_effect os_eff{os, cgh};
		cgh.host_task(celerity::on_master_node, [=] { os_eff->write(reinterpret_cast<const char*>(up_r.get_pointer()), range.size() * sizeof(T)); });
	});
}

void stream_close(celerity::queue& queue, celerity::experimental::host_object<std::ofstream> os) {
	queue.submit([&](celerity::handler& cgh) {
		celerity::experimental::side_effect os_eff{os, cgh};
		cgh.host_task(celerity::on_master_node, [=] { os_eff->close(); });
	});
}

using arg_vector = std::vector<const char*>;

template <typename ArgFn, typename Result>
bool get_cli_arg(const arg_vector& args, const arg_vector::const_iterator& it, const std::string& argstr, Result& result, ArgFn fn) {
	if(argstr == *it) {
		if(it + 1 == args.cend()) { throw std::runtime_error("Invalid argument"); }
		result = fn(*(it + 1));
		return true;
	}
	return false;
}

int main(int argc, char* argv[]) {
	// Parse command line arguments
	const wave_sim_config cfg = ([&]() {
		wave_sim_config result;
		const arg_vector args{argv + 1, argv + argc};
		for(auto it = args.cbegin(); it != args.cend(); ++it) {
			if(get_cli_arg(args, it, "-N", result.N, atoi) || get_cli_arg(args, it, "-T", result.T, atoi) || get_cli_arg(args, it, "--dt", result.dt, atof)
			    || get_cli_arg(args, it, "--sample-rate", result.output_sample_rate, atoi) || get_cli_arg(args, it, "--outset", result.outset, atoi)
			    || get_cli_arg(args, it, "--oversub", result.oversub, atoi) || get_cli_arg(args, it, "--tiled", result.tiled, atoi)) {
				++it;
				continue;
			}
			std::cerr << "Unknown argument: " << *it << std::endl;
			std::exit(1);
		}
		return result;
	})(); // IIFE

	if(cfg.oversub == 0) {
		std::cerr << "Oversubscription cannot be 0" << std::endl;
		return EXIT_FAILURE;
	}

#if WAVE_SIMPLE
	puts("This is WaveSIMPLE");
#else
	puts("This is WaveSim");
#endif

	puts("DataT is " DATA_T_STRING);

	const size_t num_steps = cfg.T / cfg.dt;
	// Sample (if enabled) every n-th frame, +1 for initial state
	const size_t num_samples = cfg.output_sample_rate != 0 ? num_steps / cfg.output_sample_rate + 1 : 0;
	if(cfg.output_sample_rate != 0 && num_steps % cfg.output_sample_rate != 0) {
		std::cerr << "Warning: Number of time steps (" << num_steps << ") is not a multiple of the output sample rate (wasted frames)" << std::endl;
	}

	celerity::queue queue;

	celerity::buffer<DataT, 2> up{celerity::range<2>(cfg.N, cfg.N)}; // next
	celerity::buffer<DataT, 2> u{celerity::range<2>(cfg.N, cfg.N)};  // current

	setup_wave(queue, u, {cfg.N / 4.f, cfg.N / 4.f}, 1, {cfg.N / 8.f, cfg.N / 8.f}, cfg);
	zero(queue, up, cfg);
	initialize(queue, up, u, cfg);

	const celerity::experimental::host_object<std::ofstream> os;
	if(cfg.output_sample_rate > 0) {
		stream_open(queue, cfg.N, num_samples, os);
		stream_append(queue, u, os); // Store initial state
	}

	auto t = 0.0;
	size_t i = 0;
	const size_t warmup = 10;
	std::chrono::steady_clock::time_point start;
	while(t < cfg.T) {
		if(i == warmup - 1) {
			queue.wait(celerity::experimental::barrier);
			start = std::chrono::steady_clock::now();
		}
		const size_t current_outset = cfg.outset - i % (cfg.outset + 1);
		update(queue, up, u, cfg, current_outset);
		if(cfg.output_sample_rate > 0) {
			if(i % cfg.output_sample_rate == 0) { stream_append(queue, u, os); }
		}
		std::swap(u, up);
		t += cfg.dt;
		i++;
	}
	queue.wait(celerity::experimental::barrier);
	const auto end = std::chrono::steady_clock::now();

	const auto computation_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	const size_t iterations = (cfg.T / cfg.dt) - warmup;
	const double bytes = cfg.N * cfg.N * sizeof(DataT) * iterations;
	const double gbps = bytes / computation_time / 1000.0;
	fprintf(stderr, "Computation time: %8.2lf ms (%.2lf GB/s) (%.2lf GigaCells/s)\n", computation_time / 1000.0, gbps,
	    (cfg.N * cfg.N * iterations / 1000.0) / computation_time);

	if(cfg.output_sample_rate > 0) { stream_close(queue, os); }

	if(cfg.N * cfg.N * sizeof(DataT) < 4e9) {
		printf("Computing hash for matrix of size %zu\n", cfg.N * cfg.N * sizeof(DataT));
		queue.submit([&](celerity::handler& cgh) {
			celerity::accessor read_u{u, cgh, celerity::access::all{}, celerity::read_only_host_task};
			cgh.host_task(celerity::once, [=, r = u.get_range()]() {
				hash hsh;
				celerity::experimental::for_each_item(r, [&](auto idx) { hsh.add(read_u[idx]); });
				fmt::print("Hash: {:x}\n", hsh.get());
			});
		});
	} else {
		puts("Skipping hash (matrix too large)");
	}

	return EXIT_SUCCESS;
}
