#include <array>
#include <cmath>
#include <fstream>
#include <vector>

#include <celerity.h>

void setup_wave(celerity::distr_queue& queue, celerity::buffer<float, 2> u, sycl::float2 center, float amplitude, sycl::float2 sigma) {
	queue.submit([=](celerity::handler& cgh) {
		celerity::accessor dw_u{u, cgh, celerity::access::one_to_one{}, celerity::write_only, celerity::no_init};
		cgh.parallel_for<class setup_wave>(u.get_range(), [=, c = center, a = amplitude, s = sigma](celerity::item<2> item) {
			const float dx = item[1] - c.x();
			const float dy = item[0] - c.y();
			dw_u[item] = a * sycl::exp(-(dx * dx / (2.f * s.x() * s.x()) + dy * dy / (2.f * s.y() * s.y())));
		});
	});
}

void zero(celerity::distr_queue& queue, celerity::buffer<float, 2> buf) {
	queue.submit([=](celerity::handler& cgh) {
		celerity::accessor dw_buf{buf, cgh, celerity::access::one_to_one{}, celerity::write_only, celerity::no_init};
		cgh.parallel_for<class zero>(buf.get_range(), [=](celerity::item<2> item) { dw_buf[item] = 0.f; });
	});
}

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
void step(celerity::distr_queue& queue, celerity::buffer<T, 2> up, celerity::buffer<T, 2> u, float dt, sycl::float2 delta) {
	queue.submit([=](celerity::handler& cgh) {
		celerity::accessor rw_up{up, cgh, celerity::access::one_to_one{}, celerity::read_write};
		celerity::accessor r_u{u, cgh, celerity::access::neighborhood{1, 1}, celerity::read_only};

		const auto size = up.get_range();
		cgh.parallel_for<KernelName>(size, [=](celerity::item<2> item) {
			const size_t py = item[0] < size[0] - 1 ? item[0] + 1 : item[0];
			const size_t my = item[0] > 0 ? item[0] - 1 : item[0];
			const size_t px = item[1] < size[1] - 1 ? item[1] + 1 : item[1];
			const size_t mx = item[1] > 0 ? item[1] - 1 : item[1];

			const float lap = (dt / delta.y()) * (dt / delta.y()) * ((r_u[{py, item[1]}] - r_u[item]) - (r_u[item] - r_u[{my, item[1]}]))
			                  + (dt / delta.x()) * (dt / delta.x()) * ((r_u[{item[0], px}] - r_u[item]) - (r_u[item] - r_u[{item[0], mx}]));
			rw_up[item] = Config::a * 2 * r_u[item] - Config::b * rw_up[item] + Config::c * lap;
		});
	});
}

void initialize(celerity::distr_queue& queue, celerity::buffer<float, 2> up, celerity::buffer<float, 2> u, float dt, sycl::float2 delta) {
	step<float, init_config, class initialize>(queue, up, u, dt, delta);
}

void update(celerity::distr_queue& queue, celerity::buffer<float, 2> up, celerity::buffer<float, 2> u, float dt, sycl::float2 delta) {
	step<float, update_config, class update>(queue, up, u, dt, delta);
}

template <typename T>
void store(celerity::distr_queue& queue, celerity::buffer<T, 2> up, celerity::host_object<std::vector<std::vector<float>>>& result_frames) {
	const auto range = up.get_range();
	queue.submit([=](celerity::handler& cgh) {
		celerity::accessor up_r{up, cgh, celerity::access::all{}, celerity::read_only_host_task};
		celerity::side_effect store_frames{result_frames, cgh};
		cgh.host_task(celerity::on_master_node, [=](celerity::partition<0> p) {
			store_frames->emplace_back();
			auto& frame = *store_frames->rbegin();
			frame.resize(range.size());
			memcpy(frame.data(), up_r.get_pointer(), range[0] * range[1] * sizeof(float));
		});
	});
}

void write_bin(size_t N, const std::vector<std::vector<float>>& result_frames) {
	std::ofstream os("wave_sim_result.bin", std::ios_base::out | std::ios_base::binary);

	const struct { uint64_t n, t; } header{N, result_frames.size()};
	os.write(reinterpret_cast<const char*>(&header), sizeof(header));

	for(const auto& frame : result_frames) {
		os.write(reinterpret_cast<const char*>(frame.data()), sizeof(float) * N * N);
	}
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
		const arg_vector args(argv + 1, argv + argc);
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

	const int num_steps = cfg.T / cfg.dt;
	if(cfg.output_sample_rate != 0 && num_steps % cfg.output_sample_rate != 0) {
		std::cerr << "Warning: Number of time steps (" << num_steps << ") is not a multiple of the output sample rate (wasted frames)" << std::endl;
	}

	// TODO: We could allocate the required size at the beginning
	celerity::host_object<std::vector<std::vector<float>>> result_frames;

	celerity::distr_queue queue;

	celerity::buffer<float, 2> up(nullptr, celerity::range<2>(cfg.N, cfg.N)); // next
	celerity::buffer<float, 2> u(nullptr, celerity::range<2>(cfg.N, cfg.N));  // current

	setup_wave(queue, u, {cfg.N / 4.f, cfg.N / 4.f}, 1, {cfg.N / 8.f, cfg.N / 8.f});
	zero(queue, up);
	initialize(queue, up, u, cfg.dt, {cfg.dx, cfg.dy});

	// Store initial state
	if(cfg.output_sample_rate > 0) { store(queue, u, result_frames); }

	auto t = 0.0;
	size_t i = 0;
	while(t < cfg.T) {
		update(queue, up, u, cfg.dt, {cfg.dx, cfg.dy});
		if(cfg.output_sample_rate != 0 && ++i % cfg.output_sample_rate == 0) { store(queue, u, result_frames); }
		std::swap(u, up);
		t += cfg.dt;
	}

	queue.slow_full_sync();

	if(cfg.output_sample_rate > 0) {
		queue.submit([=](celerity::handler& cgh) {
			celerity::side_effect load_frames{result_frames, cgh, celerity::read_only};
			cgh.host_task(celerity::on_master_node, [=](celerity::partition<0> p) {
				// TODO: Consider writing results to disk as they're coming in, instead of just at the end
				write_bin(cfg.N, *load_frames);
			});
		});
	}

	return EXIT_SUCCESS;
}
