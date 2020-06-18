#include <array>
#include <cmath>
#include <fstream>
#include <sstream>
#include <vector>

#include <CL/sycl.hpp>
#include <celerity.h>

void setup_wave(celerity::distr_queue& queue, celerity::buffer<float, 2> u, cl::sycl::float2 center, float amplitude, cl::sycl::float2 sigma) {
	queue.submit([=](celerity::handler& cgh) {
		auto dw_u = u.get_access<cl::sycl::access::mode::discard_write>(cgh, celerity::access::one_to_one<2>());
		cgh.parallel_for<class setup_wave>(u.get_range(), [=, c = center, a = amplitude, s = sigma](cl::sycl::item<2> item) {
			const float dx = item[1] - c.x();
			const float dy = item[0] - c.y();
			dw_u[item] = a * cl::sycl::exp(-(dx * dx / (2.f * s.x() * s.x()) + dy * dy / (2.f * s.y() * s.y())));
		});
	});
}

void zero(celerity::distr_queue& queue, celerity::buffer<float, 2> buf) {
	queue.submit([=](celerity::handler& cgh) {
		auto dw_buf = buf.get_access<cl::sycl::access::mode::discard_write>(cgh, celerity::access::one_to_one<2>());
		cgh.parallel_for<class zero>(buf.get_range(), [=](cl::sycl::item<2> item) { dw_buf[item] = 0.f; });
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
void step(celerity::distr_queue& queue, celerity::buffer<T, 2> up, celerity::buffer<T, 2> u, float dt, cl::sycl::float2 delta) {
	queue.submit([=](celerity::handler& cgh) {
		auto rw_up = up.template get_access<cl::sycl::access::mode::read_write>(cgh, celerity::access::one_to_one<2>());
		auto r_u = u.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::neighborhood<2>(1, 1));

		const auto size = up.get_range();
		cgh.parallel_for<KernelName>(size, [=](cl::sycl::item<2> item) {
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

void initialize(celerity::distr_queue& queue, celerity::buffer<float, 2> up, celerity::buffer<float, 2> u, float dt, cl::sycl::float2 delta) {
	step<float, init_config, class initialize>(queue, up, u, dt, delta);
}

void update(celerity::distr_queue& queue, celerity::buffer<float, 2> up, celerity::buffer<float, 2> u, float dt, cl::sycl::float2 delta) {
	step<float, update_config, class update>(queue, up, u, dt, delta);
}

template <typename T>
void store(celerity::distr_queue& queue, celerity::buffer<T, 2> up, std::vector<std::vector<float>>& result_frames) {
	const auto range = up.get_range();
	queue.submit(celerity::allow_by_ref, [=, &result_frames](celerity::handler& cgh) {
		auto up_r = up.template get_access<cl::sycl::access::mode::read, cl::sycl::access::target::host_buffer>(cgh, celerity::access::fixed<2>{{{}, range}});
		cgh.host_task(celerity::on_master_node, [=, &result_frames] {
			result_frames.emplace_back();
			auto& frame = *result_frames.rbegin();
			frame.resize(range.size());
			memcpy(frame.data(), up_r.get_pointer(), range[0] * range[1] * sizeof(float));
		});
	});
}

void write_csv(size_t N, std::vector<std::vector<float>>& result_frames) {
	std::ofstream os;
	os.open("wave_sim_result.csv", std::ios_base::out | std::ios_base::binary);

	os << "t";
	for(size_t y = 0; y < N; ++y) {
		for(size_t x = 0; x < N; ++x) {
			os << "," << y << ":" << x;
		}
	}
	os << "\n";

	size_t i = 0;
	for(auto& frame : result_frames) {
		os << i++;
		for(size_t y = 0; y < N; ++y) {
			for(size_t x = 0; x < N; ++x) {
				auto v = frame[y * N + x];
				os << "," << v;
			}
		}
		os << "\n";
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
	// Explicitly initialize here so we can use MPI functions right away
	celerity::runtime::init(&argc, &argv);
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	const bool is_master = world_rank == 0;

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
	if(is_master && cfg.output_sample_rate != 0 && num_steps % cfg.output_sample_rate != 0) {
		std::cerr << "Warning: Number of time steps (" << num_steps << ") is not a multiple of the output sample rate (wasted frames)" << std::endl;
	}

	celerity::experimental::bench::log_user_config({{"N", std::to_string(cfg.N)}, {"T", std::to_string(cfg.T)}, {"dt", std::to_string(cfg.dt)},
	    {"dx", std::to_string(cfg.dx)}, {"dy", std::to_string(cfg.dy)}, {"outputSampleRate", std::to_string(cfg.output_sample_rate)}});

	// TODO: We could allocate the required size at the beginning
	std::vector<std::vector<float>> result_frames;
	{
		celerity::distr_queue queue;

		celerity::buffer<float, 2> up(nullptr, cl::sycl::range<2>(cfg.N, cfg.N)); // next
		celerity::buffer<float, 2> u(nullptr, cl::sycl::range<2>(cfg.N, cfg.N));  // current

		MPI_Barrier(MPI_COMM_WORLD);
		celerity::experimental::bench::begin("main program");

		setup_wave(queue, u, {cfg.N / 4.f, cfg.N / 4.f}, 1, {cfg.N / 8.f, cfg.N / 8.f});
		zero(queue, up);
		initialize(queue, up, u, cfg.dt, {cfg.dx, cfg.dy});

		// We need to rotate buffers. Since we cannot swap them directly, we use pointers instead.
		// TODO: Make buffers swappable
		auto up_ref = &up;
		auto u_ref = &u;

		// Store initial state
		if(cfg.output_sample_rate > 0) { store(queue, *u_ref, result_frames); }

		auto t = 0.0;
		size_t i = 0;
		while(t < cfg.T) {
			update(queue, *up_ref, *u_ref, cfg.dt, {cfg.dx, cfg.dy});
			if(cfg.output_sample_rate != 0 && ++i % cfg.output_sample_rate == 0) { store(queue, *up_ref, result_frames); }
			std::swap(u_ref, up_ref);
			t += cfg.dt;
		}
	}

	if(is_master) {
		if(cfg.output_sample_rate > 0) {
			// TODO: Consider writing results to disk as they're coming in, instead of just at the end
			write_csv(cfg.N, result_frames);
		}
	}

	return EXIT_SUCCESS;
}
