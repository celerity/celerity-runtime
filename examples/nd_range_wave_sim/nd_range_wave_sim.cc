#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>

#include <celerity.h>

constexpr uint32_t LOCAL_RANGE_X = 8;
constexpr uint32_t LOCAL_RANGE_Y = 32;

void setup_wave(celerity::queue& queue, celerity::buffer<float, 2> u, sycl::float2 center, float amplitude, sycl::float2 sigma) {
	queue.submit([&](celerity::handler& cgh) {
		celerity::accessor dw_u{u, cgh, celerity::access::one_to_one{}, celerity::write_only, celerity::no_init};
		const auto range = u.get_range();
		const celerity::range<2> local_range{LOCAL_RANGE_X, LOCAL_RANGE_Y};
		const celerity::range<2> global_range{
		    (range[0] + LOCAL_RANGE_X - 1) / LOCAL_RANGE_X * LOCAL_RANGE_X, (range[1] + LOCAL_RANGE_Y - 1) / LOCAL_RANGE_Y * LOCAL_RANGE_Y};
		cgh.parallel_for<class setup_wave>(
		    celerity::nd_range<2>(global_range, local_range), [=, c = center, a = amplitude, s = sigma](celerity::nd_item<2> item) {
			    const auto id = item.get_global_id();
			    if(id[0] < range[0] && id[1] < range[1]) {
				    const float dx = id[1] - c.x();
				    const float dy = id[0] - c.y();
				    dw_u[id] = a * sycl::exp(-(dx * dx / (2.f * s.x() * s.x()) + dy * dy / (2.f * s.y() * s.y())));
			    }
		    });
	});
}

// void setup_wave(celerity::queue& queue, celerity::buffer<float, 2> u, sycl::float2 center, float amplitude, sycl::float2 sigma) {
// 	queue.submit([&](celerity::handler& cgh) {
// 		celerity::accessor dw_u{u, cgh, celerity::access::one_to_one{}, celerity::write_only, celerity::no_init};
// 		cgh.parallel_for<class setup_wave>(u.get_range(), [=, c = center, a = amplitude, s = sigma](celerity::item<2> item) {
// 			const float dx = item[1] - c.x();
// 			const float dy = item[0] - c.y();
// 			dw_u[item] = a * sycl::exp(-(dx * dx / (2.f * s.x() * s.x()) + dy * dy / (2.f * s.y() * s.y())));
// 		});
// 	});
// }

void zero(celerity::queue& queue, celerity::buffer<float, 2> buf) {
	queue.submit([&](celerity::handler& cgh) {
		celerity::accessor dw_buf{buf, cgh, celerity::access::one_to_one{}, celerity::write_only, celerity::no_init};
		const auto range = buf.get_range();
		const celerity::range<2> local_range{LOCAL_RANGE_X, LOCAL_RANGE_Y};
		const celerity::range<2> global_range{
		    (range[0], range[1] + LOCAL_RANGE_Y - 1) / LOCAL_RANGE_Y * LOCAL_RANGE_Y, (range[1] + LOCAL_RANGE_Y - 1) / LOCAL_RANGE_Y * LOCAL_RANGE_Y};
		cgh.parallel_for<class zero>(celerity::nd_range<2>(global_range, local_range), [=](celerity::nd_item<2> item) {
			const auto id = item.get_global_id();
			if(id[0] < range[0] && id[1] < range[1]) { dw_buf[id] = 0.f; }
		});
	});
}

// void zero(celerity::queue& queue, celerity::buffer<float, 2> buf) {
// 	queue.submit([&](celerity::handler& cgh) {
// 		celerity::accessor dw_buf{buf, cgh, celerity::access::one_to_one{}, celerity::write_only, celerity::no_init};
// 		cgh.parallel_for<class zero>(buf.get_range(), [=](celerity::item<2> item) { dw_buf[item] = 0.f; });
// 	});
// }

struct init_config {
	static constexpr float a = 0.5f;
	static constexpr float b = 0.0f;
	static constexpr float c = 0.5f;
};

struct update_config {
	static constexpr float a = 1.f;
	static constexpr float b = 1.f;
	static constexpr float c = 1.f;
};

template <typename T, typename Config, typename KernelName>
void step(celerity::queue& queue, celerity::buffer<T, 2> up, celerity::buffer<T, 2> u, float dt, sycl::float2 delta) {
	queue.submit([&](celerity::handler& cgh) {
		celerity::accessor rw_up{up, cgh, celerity::access::one_to_one{}, celerity::read_write};
		celerity::accessor r_u{u, cgh, celerity::access::neighborhood{{1, 1}, celerity::neighborhood_shape::along_axes}, celerity::read_only};
		celerity::local_accessor<float, 2> lap{celerity::range<2>{LOCAL_RANGE_X + 2, LOCAL_RANGE_Y + 2}, cgh};

		const auto range = up.get_range();
		const celerity::range<2> local_range{LOCAL_RANGE_X, LOCAL_RANGE_Y};
		const celerity::range<2> global_range{
		    (range[0] + LOCAL_RANGE_X - 1) / LOCAL_RANGE_X * LOCAL_RANGE_X, (range[1] + LOCAL_RANGE_Y - 1) / LOCAL_RANGE_Y * LOCAL_RANGE_Y};
		cgh.parallel_for<KernelName>(celerity::nd_range<2>(global_range, local_range), [=](celerity::nd_item<2> item) {
			const auto id = item.get_global_id();
			// celerity::group_barrier(item.get_group());
			if(id[0] < range[0] && id[1] < range[1]) {
				// Load the neighborhood into local memory
				const size_t py = id[0] < range[0] - 1 ? id[0] + 1 : id[0];
				const size_t my = id[0] > 0 ? id[0] - 1 : id[0];
				const size_t px = id[1] < range[1] - 1 ? id[1] + 1 : id[1];
				const size_t mx = id[1] > 0 ? id[1] - 1 : id[1];

				const size_t local_x = item.get_local_id()[0] + 1;
				const size_t local_y = item.get_local_id()[1] + 1;
				lap[local_x][local_y] = r_u[id];
				lap[local_x + 1][local_y] = r_u[{py, id[1]}];
				lap[local_x - 1][local_y] = r_u[{my, id[1]}];
				lap[local_x][local_y + 1] = r_u[{id[0], px}];
				lap[local_x][local_y - 1] = r_u[{id[0], mx}];
			}

			celerity::group_barrier(item.get_group());

			if(id[0] < range[0] && id[1] < range[1]) {
				const size_t local_x = item.get_local_id()[0] + 1;
				const size_t local_y = item.get_local_id()[1] + 1;
				// const float lap = (dt / delta.y()) * (dt / delta.y()) * ((r_u[{py, id[1]}] - r_u[id]) - (r_u[id] - r_u[{my, id[1]}]))
				//                   + (dt / delta.x()) * (dt / delta.x()) * ((r_u[{id[0], px}] - r_u[id]) - (r_u[id] - r_u[{id[0], mx}]));

				const float lap_num = (dt / delta.y()) * (dt / delta.y())
				                          * ((lap[local_x + 1][local_y] - lap[local_x][local_y]) - (lap[local_x][local_y] - lap[local_x - 1][local_y]))
				                      + (dt / delta.x()) * (dt / delta.x())
				                            * ((lap[local_x][local_y + 1] - lap[local_x][local_y]) - (lap[local_x][local_y] - lap[local_x][local_y - 1]));
				rw_up[id] = Config::a * 2 * r_u[id] - Config::b * rw_up[id] + Config::c * lap_num;
			}
		});
	});
}

// template <typename T, typename Config, typename KernelName>
// void step(celerity::queue& queue, celerity::buffer<T, 2> up, celerity::buffer<T, 2> u, float dt, sycl::float2 delta) {
// 	queue.submit([&](celerity::handler& cgh) {
// 		celerity::accessor rw_up{up, cgh, celerity::access::one_to_one{}, celerity::read_write};
// 		celerity::accessor r_u{u, cgh, celerity::access::neighborhood{{1, 1}, celerity::neighborhood_shape::along_axes}, celerity::read_only};

// 		const auto size = up.get_range();
// 		cgh.parallel_for<KernelName>(size, [=](celerity::item<2> item) {
// 			const size_t py = item[0] < size[0] - 1 ? item[0] + 1 : item[0];
// 			const size_t my = item[0] > 0 ? item[0] - 1 : item[0];
// 			const size_t px = item[1] < size[1] - 1 ? item[1] + 1 : item[1];
// 			const size_t mx = item[1] > 0 ? item[1] - 1 : item[1];

// 			const float lap = (dt / delta.y()) * (dt / delta.y()) * ((r_u[{py, item[1]}] - r_u[item]) - (r_u[item] - r_u[{my, item[1]}]))
// 			                  + (dt / delta.x()) * (dt / delta.x()) * ((r_u[{item[0], px}] - r_u[item]) - (r_u[item] - r_u[{item[0], mx}]));
// 			rw_up[item] = Config::a * 2 * r_u[item] - Config::b * rw_up[item] + Config::c * lap;
// 		});
// 	});
// }

void initialize(celerity::queue& queue, celerity::buffer<float, 2> up, celerity::buffer<float, 2> u, float dt, sycl::float2 delta) {
	step<float, init_config, class initialize>(queue, up, u, dt, delta);
}

void update(celerity::queue& queue, celerity::buffer<float, 2> up, celerity::buffer<float, 2> u, float dt, sycl::float2 delta) {
	step<float, update_config, class update>(queue, up, u, dt, delta);
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

struct wave_sim_config {
	int N = 512;   // Grid size
	float T = 100; // Time at end of simulation
	float dt = 0.25f;
	float dx = 1.f;
	float dy = 1.f;

	// "Sample" a frame every X iterations
	// (0 = don't produce any output)
	unsigned output_sample_rate = 0;
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

int main(int argc, char* argv[]) {
	// Parse command line arguments
	const wave_sim_config cfg = ([&]() {
		wave_sim_config result;
		const arg_vector args{argv + 1, argv + argc};
		for(auto it = args.cbegin(); it != args.cend(); ++it) {
			if(get_cli_arg(args, it, "-N", result.N, atoi) || get_cli_arg(args, it, "-T", result.T, atoi) || get_cli_arg(args, it, "--dt", result.dt, atof)
			    || get_cli_arg(args, it, "--sample-rate", result.output_sample_rate, atoi)) {
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

	celerity::queue queue;

	celerity::buffer<float, 2> up{celerity::range<2>(cfg.N, cfg.N)}; // next
	celerity::buffer<float, 2> u{celerity::range<2>(cfg.N, cfg.N)};  // current

	setup_wave(queue, u, {cfg.N / 2.f, cfg.N / 2.f}, 1, {cfg.N / 2.f, cfg.N / 2.f});
	zero(queue, up);
	initialize(queue, up, u, cfg.dt, {cfg.dx, cfg.dy});

	const celerity::experimental::host_object<std::ofstream> os;
	if(cfg.output_sample_rate > 0) {
		stream_open(queue, cfg.N, num_samples, os);
		stream_append(queue, u, os); // Store initial state
	}

	auto t = 0.0;
	size_t i = 0;

	queue.wait(celerity::experimental::barrier);
	// time loop
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

	while(t < cfg.T) {
		update(queue, up, u, cfg.dt, {cfg.dx, cfg.dy});
		if(cfg.output_sample_rate > 0) {
			if(++i % cfg.output_sample_rate == 0) { stream_append(queue, u, os); }
		}
		std::swap(u, up);
		t += cfg.dt;
	}

	queue.wait(celerity::experimental::barrier);

	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

	std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms" << std::endl;

	if(cfg.output_sample_rate > 0) { stream_close(queue, os); }

	return EXIT_SUCCESS;
}
