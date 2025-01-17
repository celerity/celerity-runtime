#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>

#include <celerity.h>

#include <fmt/ranges.h>

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
	bool use_loop = false;
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
void step(celerity::queue& queue, celerity::buffer<T, 2> up, celerity::buffer<T, 2> u, const wave_sim_config& cfg, const size_t current_outset,
    const bool is_warmup = false) {
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
		if(!is_warmup) {
			cgh.assert_no_allocations();
			if(cfg.outset > 0 && current_outset != cfg.outset) { cgh.assert_no_data_movement(); }
		}

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
	step<DataT, init_config, class initialize>(queue, up, u, cfg, 0, true);
}

template <typename Name = class update>
void update(celerity::queue& queue, celerity::buffer<DataT, 2> up, celerity::buffer<DataT, 2> u, const wave_sim_config& cfg, const size_t current_outset,
    const bool is_warmup = false) //
{
	step<DataT, update_config, Name>(queue, up, u, cfg, current_outset, is_warmup);
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
#if 0

	const int num_nodes = argc > 1 ? atoi(argv[1]) : 2;
	const int num_elements = argc > 2 ? atoi(argv[2]) : 1000;
	const int max_value = argc > 3 ? atoi(argv[3]) : 99;
	int seed = argc > 4 ? atoi(argv[4]) : -1;

	if(seed == -1) {
		std::random_device rd;
		seed = rd();
		fmt::print("Using random seed {}\n", seed);
	}
	std::mt19937 gen(seed);
	std::uniform_int_distribution<> jitter_dist(0, num_elements * 0.1);
	std::uniform_int_distribution<> num_dist(0, max_value); // TODO: Also try normal distribution

	struct node_state {
		std::vector<int> all_my_numbers;
		std::vector<int>::iterator begin;
		std::vector<int>::iterator pivot;
		std::vector<int>::iterator end;
	};

	std::vector<node_state> nodes(num_nodes);
	std::vector<int> all_numbers_global;
	for(int i = 0; i < num_nodes; ++i) {
		int num_elements_local = num_elements / num_nodes + jitter_dist(gen);
		auto& node = nodes[i];
		for(int j = 0; j < num_elements_local; ++j) {
			const int number = num_dist(gen);
			node.all_my_numbers.push_back(number);
			all_numbers_global.push_back(number);
		}
		node.begin = node.all_my_numbers.begin();
		node.end = node.all_my_numbers.end();
	}

	fmt::print("All numbers: {}\n", fmt::join(all_numbers_global, ","));
	std::ranges::sort(all_numbers_global);
	fmt::print("Sorted: {}\n", fmt::join(all_numbers_global, ","));
	const int true_median = all_numbers_global[all_numbers_global.size() / 2];
	const int other_true_median = all_numbers_global.size() % 2 == 0 ? all_numbers_global[all_numbers_global.size() / 2 - 1] : true_median;
	fmt::print("True median is {} (or {})\n", true_median, other_true_median);

	bool use_median_of_medians = true;
	int previous_pivot = -1;
	const auto select_pivot = [&]() {
		// NOTE: Using the median of means can lead to "livelock" situations, where the chosen pivot no longer guarantees progress.
		// For example on 3 nodes, with elements 58,59 / 57 / 57, resulting in median of medians 57.
		// This pivot causes all elements to be moved to the right, resulting in the same situation again.
		// If e.g. k = 1, nothing changes and we're stuck in an endless loop.

		if(use_median_of_medians) { // Use median of medians
			std::vector<int> medians;
			for(const auto& node : nodes) {
				const auto num_elements = std::distance(node.begin, node.end);
				if(num_elements == 0) continue;
				auto median_it = node.begin + num_elements / 2;
				std::nth_element(node.begin, median_it, node.end);
				medians.push_back(*median_it);
			}
			fmt::print("Computing median of medians: {} => ", fmt::join(medians, ","));
			std::nth_element(medians.begin(), medians.begin() + medians.size() / 2, medians.end());
			const auto pivot = medians[medians.size() / 2];
			fmt::print("{}\n", pivot);

			if(pivot == previous_pivot) {
				fmt::print("Pivot is the same as last time, switching to random\n");
				use_median_of_medians = false;
			}
			previous_pivot = pivot;

			return pivot;
		} else { // Random
			while(true) {
				std::uniform_int_distribution<> node_dist(0, nodes.size() - 1);
				const auto& node = nodes[node_dist(gen)];
				const auto remaining = std::distance(node.begin, node.end);
				if(remaining == 0) continue;
				std::uniform_int_distribution<> element_dist(0, remaining - 1);
				return *(node.begin + element_dist(gen));
			}
		}
	};

	// Search k-largest element (k=1 finds the maximum)
	int k = std::round(all_numbers_global.size() / 2.f);
	fmt::print("Total number of elements is {}. Looking for {}-th element from the right\n", all_numbers_global.size(), k);
	int pivot = -1;
	for(int s = 1; true; ++s) {
		pivot = select_pivot();
		fmt::print("\nPivot is {}, k={}\n", pivot, k);

		uint64_t sum_right = 0;
		uint64_t equal_to_pivot = 0;
		for(int i = 0; i < num_nodes; ++i) {
			auto& node = nodes[i];
			// move all elements that are larger or equal to pivot to one side
			node.pivot = std::partition(node.begin, node.end, [&](int x) { return x < pivot; });
			const auto etp = std::count(node.pivot, node.end, pivot);
			fmt::print(
			    "Node {} left: {}, right: {}, equal to pivot: {}\n", i, fmt::join(node.begin, node.pivot, ","), fmt::join(node.pivot, node.end, ","), etp);
			equal_to_pivot += etp;
			sum_right += std::distance(node.pivot, node.end);
		}

		// if(sum_right == 0 && k == (sum_left / nodes_with_nonzero_left)) {
		// 	fmt::print("SPECIAL SAUCE: All remaining nodes have the same value?!\n");
		// 	fmt::print("Pivot should be {} => {}\n", pivot, (pivot == true_median || pivot == other_true_median) ? "YAY" : "NAY");
		// 	break;
		// }

		if(sum_right - equal_to_pivot + 1 > k) {
			fmt::print("Sum of larger elements w/o duplicates of pivot ({}) is {} > k={}. dropping left side.\n", equal_to_pivot - 1,
			    sum_right - equal_to_pivot + 1, k);
			for(int i = 0; i < num_nodes; ++i) {
				auto& node = nodes[i];
				node.begin = node.pivot;
			}
		} else if(sum_right < k) {
			fmt::print("Sum of larger elements including duplicates of pivot ({}) is {} < k={}. dropping right side. New k={}\n", equal_to_pivot - 1, sum_right,
			    k, k - sum_right);
			k -= sum_right;
			for(int i = 0; i < num_nodes; ++i) {
				auto& node = nodes[i];
				node.end = node.pivot;
			}
		} else /* sum_larger == k */ {
			fmt::print("Found median after {} steps: {}\n", s, pivot);
			if(pivot != true_median && pivot != other_true_median) {
				fmt::print("ERROR: Found median is incorrect\n");
			} else {
				fmt::print("ITS A MATCH!\n");
			}
			break;
		}

		if(s > 30) {
			fmt::print("Aborting after {} steps\n", s);
			break;
		}
	}


	return 0;
#endif


	// Parse command line arguments
	const wave_sim_config cfg = ([&]() {
		wave_sim_config result;
		const arg_vector args{argv + 1, argv + argc};
		for(auto it = args.cbegin(); it != args.cend(); ++it) {
			if(get_cli_arg(args, it, "-N", result.N, atoi) || get_cli_arg(args, it, "-T", result.T, atoi) || get_cli_arg(args, it, "--dt", result.dt, atof)
			    || get_cli_arg(args, it, "--sample-rate", result.output_sample_rate, atoi) || get_cli_arg(args, it, "--outset", result.outset, atoi)
			    || get_cli_arg(args, it, "--oversub", result.oversub, atoi) || get_cli_arg(args, it, "--tiled", result.tiled, atoi)
			    || get_cli_arg(args, it, "--use-loop", result.use_loop, atoi)) {
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

	const auto init = [&]() {
		setup_wave(queue, u, {cfg.N / 4.f, cfg.N / 4.f}, 1, {cfg.N / 8.f, cfg.N / 8.f}, cfg);
		zero(queue, up, cfg);
		initialize(queue, up, u, cfg);
	};

	init();

	if(cfg.use_loop == 0) {
		// TODO: Actual number of required iterations depends on horizon step size
		// 2 * (outset + 1) covers two full "outset cycles", but that may not be enough
		// Multiply by 4 to be on the safe side for default horizon step size of 4
		const size_t warmup = 4 * std::max<size_t>(5, 2ull * (cfg.outset + 1));
		fprintf(stderr, "Doing %zu warmup iterations\n", warmup);
		for(size_t i = 0; i < warmup; ++i) {
			const size_t current_outset = cfg.outset - i % (cfg.outset + 1);
			update<class warmup>(queue, up, u, cfg, current_outset, true /*	is_warmup */);
			std::swap(u, up);
		}
	} else {
		fprintf(stderr, "Doing loop warmup\n");
		// TODO: What's the minimum number of iterations we need..?
		const size_t warmup = 10;
		size_t w = 0;
		queue.loop([&]() {
			// We have to do two iterations at a time to ensure that loop is the same
			size_t inner_iterations = 2;
			if(cfg.outset % 2 == 0) {
				inner_iterations = 2 * (cfg.outset + 1);
			} else {
				inner_iterations = cfg.outset + 1;
			}

			for(size_t j = 0; j < inner_iterations; ++j) {
				const size_t current_outset = cfg.outset - j % (cfg.outset + 1);
				update<class warmup>(queue, up, u, cfg, current_outset, true /*	is_warmup */);
				std::swap(u, up);
			}
			return w++ < warmup;
		});
	}

	init();

	const celerity::experimental::host_object<std::ofstream> os;
	if(cfg.output_sample_rate > 0) {
		stream_open(queue, cfg.N, num_samples, os);
		stream_append(queue, u, os); // Store initial state
	}

	auto t = 0.0;
	size_t i = 0;
	queue.wait(celerity::experimental::barrier);
	std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

	const auto loop_body = [&]() {
		const size_t current_outset = cfg.outset - i % (cfg.outset + 1);
		update(queue, up, u, cfg, current_outset);
		if(cfg.output_sample_rate > 0) {
			if(i % cfg.output_sample_rate == 0) { stream_append(queue, u, os); }
		}
		std::swap(u, up);
		t += cfg.dt;
		i++;
	};

	if(cfg.use_loop) {
		if(cfg.output_sample_rate > 0) throw std::runtime_error("loop + sampling are not supported");
		queue.loop([&]() {
			// We have to do two iterations at a time to ensure that loop is the same
			size_t inner_iterations = 2;

			// If we use an outset, we have to do more
			if(cfg.outset != 0) {
				if(cfg.outset % 2 == 0) {
					inner_iterations = 2 * (cfg.outset + 1);
				} else {
					inner_iterations = cfg.outset + 1;
				}
			}
			for(size_t j = 0; j < inner_iterations; ++j) {
				loop_body();
			}

			if(t + inner_iterations * cfg.dt > cfg.T) {
				CELERITY_CRITICAL("Require epilogue");
				return false;
			}

			return t < cfg.T;
		});
		// Process remaining iterations (if any)
		while(t < cfg.T) {
			loop_body();
		}
	} else {
		while(t < cfg.T) {
			loop_body();
		}
	}

	queue.wait(celerity::experimental::barrier);
	const auto end = std::chrono::steady_clock::now();

	const auto computation_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	const size_t iterations = cfg.T / cfg.dt;
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
