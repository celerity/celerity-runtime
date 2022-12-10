#include <array>
#include <cmath>
#include <fstream>
#include <vector>

#include <celerity.h>

using DataT = double;

void setup_wave(celerity::distr_queue& queue, celerity::buffer<DataT, 2> u, sycl::float2 center, DataT amplitude, sycl::float2 sigma,
    const size_t oversub_factor, const bool tiled_split) {
	queue.submit([&](celerity::handler& cgh) {
		celerity::accessor dw_u{u, cgh, celerity::access::one_to_one{}, celerity::write_only, celerity::no_init};
		if(tiled_split) { celerity::experimental::hint(cgh, celerity::experimental::hints::tiled_split{}); }
		if(oversub_factor > 1) { celerity::experimental::hint(cgh, celerity::experimental::hints::oversubscribe{oversub_factor}); }
		cgh.parallel_for<class setup_wave>(u.get_range(), [=, c = center, a = amplitude, s = sigma](celerity::item<2> item) {
			const DataT dx = item[1] - c.x();
			const DataT dy = item[0] - c.y();
			dw_u[item] = a * sycl::exp(-(dx * dx / (2.f * s.x() * s.x()) + dy * dy / (2.f * s.y() * s.y())));
		});
	});
}

void zero(celerity::distr_queue& queue, celerity::buffer<DataT, 2> buf, const size_t oversub_factor, const bool tiled_split) {
	queue.submit([&](celerity::handler& cgh) {
		celerity::accessor dw_buf{buf, cgh, celerity::access::one_to_one{}, celerity::write_only, celerity::no_init};
		if(tiled_split) { celerity::experimental::hint(cgh, celerity::experimental::hints::tiled_split{}); }
		if(oversub_factor > 1) { celerity::experimental::hint(cgh, celerity::experimental::hints::oversubscribe{oversub_factor}); }
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
void step(celerity::distr_queue& queue, celerity::buffer<T, 2> up, celerity::buffer<T, 2> u, DataT dt, sycl::float2 delta, const size_t oversub_factor,
    const bool tiled_split) {
	queue.submit([&](celerity::handler& cgh) {
		celerity::accessor rw_up{up, cgh, celerity::access::one_to_one{}, celerity::read_write};
		celerity::accessor r_u{u, cgh, celerity::access::neighborhood{1, 1}, celerity::read_only};

		if(tiled_split) { celerity::experimental::hint(cgh, celerity::experimental::hints::tiled_split{}); }
		if(oversub_factor > 1) { celerity::experimental::hint(cgh, celerity::experimental::hints::oversubscribe{oversub_factor}); }

		const auto size = up.get_range();
		const auto step_y = (dt / delta.y()) * (dt / delta.y());
		const auto step_x = (dt / delta.x()) * (dt / delta.x());
		const auto a2 = Config::a * 2;

#if 0
		const celerity::range<2> local_size{32, 32};
		const celerity::range<2> global_size{((size[0] + local_size[0] - 1) / local_size[0]) * local_size[0], ((size[1] + local_size[1] - 1) / local_size[1]) * local_size[1]};
		celerity::local_accessor<T, 2> aux{local_size + celerity::range<2>{2, 2}, cgh};
		cgh.parallel_for<KernelName>(celerity::nd_range<2>{global_size, local_size}, [=](celerity::nd_item<2> itm) {
			const auto gid = itm.get_global_id();
			if(gid[0] >= size[0] || gid[1] >= size[1]) return;
			const auto lid = itm.get_local_id();
			const auto aux_id = lid + celerity::id<2>{1, 1};

			aux[aux_id] = r_u[gid];

			const size_t my = gid[0] > 0 ? gid[0] - 1 : gid[0];
			const size_t mx = gid[1] > 0 ? gid[1] - 1 : gid[1];
			const size_t py = gid[0] < size[0] - 1 ? gid[0] + 1 : gid[0];
			const size_t px = gid[1] < size[1] - 1 ? gid[1] + 1 : gid[1];

			// NOCOMMIT TODO: This is correct but slow. Improve loading strategy (group by warp!).

			if(lid[0] == 0 || lid[1] == 0) {
				aux[lid] = r_u[{my, mx}];
				if(lid[1] == local_size[1] - 1) {
					aux[lid + celerity::id{0, 1}] = r_u[{my, gid[1]}];
					aux[lid + celerity::id{0, 2}] = r_u[{my, px}];
					aux[lid + celerity::id{1, 2}] = r_u[{gid[0], px}];
				}
			}

			if(lid[0] == local_size[0] - 1 || lid[1] == local_size[1] - 1) {
				aux[lid + celerity::id<2>{2, 2}] = r_u[{py, px}];
				if(lid[0] == local_size[0] - 1) {
					aux[lid + celerity::id<2>{1, 0}] = r_u[{gid[0], mx}];
					aux[lid + celerity::id<2>{2, 0}] = r_u[{py, mx}];
					aux[lid + celerity::id<2>{2, 1}] = r_u[{py, gid[1]}];
				}
			}

			celerity::group_barrier(itm.get_group());

			// // Compute stencil

			const DataT cur = aux[aux_id];

			DataT lap = 0.f;
			// // NOCOMMIT TODO Why doesn't plain initializer list work for local accessor?
			lap += step_y * (aux[celerity::id<2>{aux_id[0] + 1, aux_id[1]}] - cur);
			lap -= step_y * (cur - aux[celerity::id<2>{aux_id[0] - 1, aux_id[1]}]);
			lap += step_x * (aux[celerity::id<2>{aux_id[0], aux_id[1] + 1}] - cur);
			lap -= step_x * (cur - aux[celerity::id<2>{aux_id[0], aux_id[1] - 1}]);

			rw_up[gid] = a2 * cur - Config::b * rw_up[gid] + Config::c * lap;
		});
#else
		cgh.parallel_for<KernelName>(size, [=](celerity::item<2> item) {
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

			// const DataT lap = step_y * ((r_u[{py, item[1]}] - cur) - (cur - r_u[{my, item[1]}]))
			//                   + step_x * ((r_u[{item[0], px}] - cur) - (cur - r_u[{item[0], mx}]));
			rw_up[item] = a2 * cur - Config::b * rw_up[item] + Config::c * lap;
		});
#endif
	});
}

void initialize(celerity::distr_queue& queue, celerity::buffer<DataT, 2> up, celerity::buffer<DataT, 2> u, DataT dt, sycl::float2 delta,
    const size_t oversub_factor, const bool tiled_split) {
	step<DataT, init_config, class initialize>(queue, up, u, dt, delta, oversub_factor, tiled_split);
}

void update(celerity::distr_queue& queue, celerity::buffer<DataT, 2> up, celerity::buffer<DataT, 2> u, DataT dt, sycl::float2 delta,
    const size_t oversub_factor, const bool tiled_split) {
	step<DataT, update_config, class update>(queue, up, u, dt, delta, oversub_factor, tiled_split);
}

void stream_open(celerity::distr_queue& queue, size_t N, size_t num_samples, celerity::experimental::host_object<std::ofstream> os) {
	queue.submit([&](celerity::handler& cgh) {
		celerity::experimental::side_effect os_eff{os, cgh};
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
void stream_append(celerity::distr_queue& queue, celerity::buffer<T, 2> up, celerity::experimental::host_object<std::ofstream> os) {
	const auto range = up.get_range();
	queue.submit([&](celerity::handler& cgh) {
		celerity::accessor up_r{up, cgh, celerity::access::all{}, celerity::read_only_host_task};
		celerity::experimental::side_effect os_eff{os, cgh};
		cgh.host_task(celerity::on_master_node, [=] { os_eff->write(reinterpret_cast<const char*>(up_r.get_pointer()), range.size() * sizeof(T)); });
	});
}

void stream_close(celerity::distr_queue& queue, celerity::experimental::host_object<std::ofstream> os) {
	queue.submit([&](celerity::handler& cgh) {
		celerity::experimental::side_effect os_eff{os, cgh};
		cgh.host_task(celerity::on_master_node, [=] { os_eff->close(); });
	});
}

struct wave_sim_config {
	size_t N = 512; // Grid size
	DataT T = 100;  // Time at end of simulation
	DataT dt = 0.25f;
	DataT dx = 1.f;
	DataT dy = 1.f;

	// "Sample" a frame every X iterations
	// (0 = don't produce any output)
	unsigned output_sample_rate = 0;

	size_t oversub_factor = 1;
	bool tiled_split = false;
};

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

// FNV-1a hash, 64 bit length
class hasher {
  public:
	using digest = uint64_t;
	template <typename T>
	void hash(const T& value) {
		const unsigned char* bytes = reinterpret_cast<const unsigned char*>(&value);
		for(size_t i = 0; i < sizeof(T); ++i) {
			d = (d ^ bytes[i]) * 0x100000001b3ull;
		}
	}
	digest get() const { return d; }

  private:
	digest d = 0xcbf29ce484222325ull;
};

int main(int argc, char* argv[]) {
	// Parse command line arguments
	const wave_sim_config cfg = ([&]() {
		wave_sim_config result;
		const arg_vector args{argv + 1, argv + argc};
		for(auto it = args.cbegin(); it != args.cend(); ++it) {
			if(get_cli_arg(args, it, "-N", result.N, atol) || get_cli_arg(args, it, "-T", result.T, atoi) || get_cli_arg(args, it, "--dt", result.dt, atof)
			    || get_cli_arg(args, it, "--sample-rate", result.output_sample_rate, atoi) || get_cli_arg(args, it, "--oversub", result.oversub_factor, atoi)
			    || get_cli_arg(args, it, "--tiled", result.tiled_split, atoi)) {
				++it;
				continue;
			}
			std::cerr << "Unknown argument: " << *it << std::endl;
		}
		return result;
	})(); // IIFE

	const size_t num_steps = cfg.T / cfg.dt;
	// Sample (if enabled) every n-th frame, +1 for initial state
	const size_t num_samples = cfg.output_sample_rate != 0 ? num_steps / cfg.output_sample_rate + 1 : 0;
	if(cfg.output_sample_rate != 0 && num_steps % cfg.output_sample_rate != 0) {
		std::cerr << "Warning: Number of time steps (" << num_steps << ") is not a multiple of the output sample rate (wasted frames)" << std::endl;
	}

	fmt::print("N={}, T={}, dt={}, oversub={}, tiled={}\n", cfg.N, (double)cfg.T, (double)cfg.dt, cfg.oversub_factor, cfg.tiled_split);
	fmt::print("Data type is {}.\n", std::is_same_v<DataT, float> ? "float" : "double");

	celerity::distr_queue queue;

	celerity::buffer<DataT, 2> up{celerity::range<2>(cfg.N, cfg.N)}; // next
	celerity::buffer<DataT, 2> u{celerity::range<2>(cfg.N, cfg.N)};  // current

	setup_wave(queue, u, {cfg.N / 4.f, cfg.N / 4.f}, 1, {cfg.N / 8.f, cfg.N / 8.f}, cfg.oversub_factor, cfg.tiled_split);
	zero(queue, up, cfg.oversub_factor, cfg.tiled_split);
	initialize(queue, up, u, cfg.dt, {cfg.dx, cfg.dy}, cfg.oversub_factor, cfg.tiled_split);

	{
		printf("With warmup.\n");
		update(queue, up, u, cfg.dt, {cfg.dx, cfg.dy}, cfg.oversub_factor, cfg.tiled_split);
		std::swap(u, up);
		update(queue, up, u, cfg.dt, {cfg.dx, cfg.dy}, cfg.oversub_factor, cfg.tiled_split);
		std::swap(u, up);
		setup_wave(queue, u, {cfg.N / 4.f, cfg.N / 4.f}, 1, {cfg.N / 8.f, cfg.N / 8.f}, cfg.oversub_factor, cfg.tiled_split);
		zero(queue, up, cfg.oversub_factor, cfg.tiled_split);
		initialize(queue, up, u, cfg.dt, {cfg.dx, cfg.dy}, cfg.oversub_factor, cfg.tiled_split);
	}

	const celerity::experimental::host_object<std::ofstream> os;
	if(cfg.output_sample_rate > 0) {
		stream_open(queue, cfg.N, num_samples, os);
		stream_append(queue, u, os); // Store initial state
	}

	queue.slow_full_sync();
	printf("Run starts now\n");
	celerity::detail::runtime::get_instance().get_buffer_manager().NOMERGE_warn_on_device_buffer_resize = true;
	const auto before = std::chrono::steady_clock::now();

	auto t = 0.0;
	size_t i = 0;
	while(t < cfg.T) {
		update(queue, up, u, cfg.dt, {cfg.dx, cfg.dy}, cfg.oversub_factor, cfg.tiled_split);
		if(cfg.output_sample_rate > 0) {
			if(++i % cfg.output_sample_rate == 0) { stream_append(queue, u, os); }
		}
		std::swap(u, up);
		t += cfg.dt;
	}

	queue.slow_full_sync();
	const auto after = std::chrono::steady_clock::now();

	// const double flops = cfg.N * cfg.N * 14.0 * (cfg.T / cfg.dt); // NOCOMMIT Is 14 right...?
	// const double gflops = flops / std::chrono::duration_cast<std::chrono::microseconds>(after - before).count() / 1000.0;
	const double bytes = cfg.N * cfg.N * (cfg.T / cfg.dt) * sizeof(DataT); // * (6 + 1); // 6 reads, 1 write
	const auto dt = std::chrono::duration_cast<std::chrono::microseconds>(after - before).count();
	const double gbs = bytes / (dt * 1000.0);
	fmt::print("Time: {}ms ({:.2f} GB/s)\n", dt / 1000, gbs);

	if(cfg.N * cfg.N * sizeof(DataT) < 10ull * 1024 * 1024 * 1024) {
		queue.submit([=](celerity::handler& cgh) {
			celerity::accessor acc{u, cgh, celerity::access::all{}, celerity::read_only_host_task};
			cgh.host_task(celerity::on_master_node, [=]() {
				hasher hsh;
				for(size_t j = 0; j < u.get_range()[0]; ++j) {
					for(size_t i = 0; i < u.get_range()[1]; ++i) {
						hsh.hash(acc[{j, i}]);
					}
				}
				fmt::print("Digest: {:x}\n", hsh.get());
			});
		});
	} else {
		printf("Skipping hash (domain too large).\n");
	}

	if(cfg.output_sample_rate > 0) { stream_close(queue, os); }

	queue.slow_full_sync();

	return EXIT_SUCCESS;
}
