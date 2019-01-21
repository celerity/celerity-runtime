#include <array>
#include <cmath>
#include <fstream>
#include <sstream>
#include <vector>

#include <CL/sycl.hpp>
#include <celerity.h>

// We have to provide the STL implementation over SYCL on Linux,
// as our POCL SPIR -> PTX translation seems to have issues with the latter.
inline float my_exp(float x) {
#ifdef _MSC_VER
	return cl::sycl::exp(x);
#else
	return std::exp(x);
#endif
}

// NOTE: We have to make amplitude a double to avoid some weird ComputeCpp behavior - possibly a device compiler bug.
// See https://codeplay.atlassian.net/servicedesk/customer/portal/1/CPPB-94 (psalz)
void setup_wave(celerity::distr_queue& queue, celerity::buffer<float, 2>& u, const cl::sycl::float2& center, double amplitude, cl::sycl::float2 sigma) {
	queue.submit([&, center, amplitude, sigma](auto& cgh) {
		auto dw_u = u.get_access<cl::sycl::access::mode::discard_write>(cgh, celerity::access::one_to_one<2>());
		cgh.template parallel_for<class setup_wave>(u.get_range(), [=, c = center, a = amplitude, s = sigma](cl::sycl::item<2> item) {
			const float dx = item[1] - c.x();
			const float dy = item[0] - c.y();
			dw_u[item] = a * my_exp(-(dx * dx / (2.f * s.x() * s.x()) + dy * dy / (2.f * s.y() * s.y())));
		});
	});
}

void zero(celerity::distr_queue& queue, celerity::buffer<float, 2>& buf) {
	queue.submit([&](auto& cgh) {
		auto dw_buf = buf.get_access<cl::sycl::access::mode::discard_write>(cgh, celerity::access::one_to_one<2>());
		cgh.template parallel_for<class zero>(buf.get_range(), [=](cl::sycl::item<2> item) { dw_buf[item] = 0.f; });
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
// TODO: See if we can make buffers u and um const refs here
void step(celerity::distr_queue& queue, celerity::buffer<T, 2>& up, celerity::buffer<T, 2>& u, celerity::buffer<T, 2>& um, float dt, cl::sycl::float2 delta) {
	queue.submit([&, dt, delta](auto& cgh) {
		auto dw_up = up.template get_access<cl::sycl::access::mode::discard_write>(cgh, celerity::access::one_to_one<2>());
		auto r_u = u.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::neighborhood<2>(1, 1));
		auto r_um = um.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::neighborhood<2>(1, 1));

		const auto size = up.get_range();
		cgh.template parallel_for<KernelName>(size, [=](cl::sycl::item<2> item) {
			// NOTE: We have to do some casting due to some weird ComputeCpp behavior - possibly a device compiler bug.
			// See https://codeplay.atlassian.net/servicedesk/customer/portal/1/CPPB-94 (psalz)
			const size_t py = item[0] < size[0] - 1 ? item[0] + 1 : item[0];
			const size_t my = item[0] > 0 ? static_cast<size_t>(static_cast<int>(item[0]) - 1) : item[0];
			const size_t px = item[1] < size[1] - 1 ? item[1] + 1 : item[1];
			const size_t mx = item[1] > 0 ? static_cast<size_t>(static_cast<int>(item[1]) - 1) : item[1];

			// NOTE: We have to copy delta here, again to avoid some ComputeCpp weirdness.
			cl::sycl::float2 delta2 = delta;
			const float lap = (dt / delta2.y()) * (dt / delta2.y()) * ((r_u[{py, item[1]}] - r_u[item]) - (r_u[item] - r_u[{my, item[1]}]))
			                  + (dt / delta2.x()) * (dt / delta2.x()) * ((r_u[{item[0], px}] - r_u[item]) - (r_u[item] - r_u[{item[0], mx}]));
			dw_up[item] = Config::a * 2 * r_u[item] - Config::b * r_um[item] + Config::c * lap;
		});
	});
}

void initialize(celerity::distr_queue& queue, celerity::buffer<float, 2>& up, celerity::buffer<float, 2>& u, celerity::buffer<float, 2>& um, float dt,
    cl::sycl::float2 delta) {
	step<float, init_config, class initialize>(queue, up, u, um, dt, delta);
}

void update(celerity::distr_queue& queue, celerity::buffer<float, 2>& up, celerity::buffer<float, 2>& u, celerity::buffer<float, 2>& um, float dt,
    cl::sycl::float2 delta) {
	step<float, update_config, class update>(queue, up, u, um, dt, delta);
}

template <typename T, size_t Size>
void store(celerity::distr_queue& queue, celerity::buffer<T, 2>& up, std::vector<std::array<float, Size>>& result_frames) {
	const auto range = up.get_range();
	celerity::with_master_access([&, range](auto& mah) {
		auto up_r = up.template get_access<cl::sycl::access::mode::read>(mah, range);
		mah.run([&]() {
			result_frames.push_back({});
			auto& frame = *result_frames.rbegin();
			memcpy(frame.data(), up_r.get_pointer(), range[0] * range[1] * sizeof(float));
		});
	});
}

template <size_t Size>
void print_ascii_plot(size_t columns, size_t rows, std::vector<std::array<float, Size>>& result_frames) {
	// Scale the plot down
	const size_t scale = std::max({columns / 25ull, rows / 25ull, 1ull});
	std::stringstream ss;

	int i = 1;
	for(auto& frame : result_frames) {
		ss << "============= " << i++ << " / " << result_frames.size() << " ============\n";
		for(size_t y = 0; y < rows; y += scale) {
			for(size_t x = 0; x < columns; x += scale) {
				auto v = frame[y * columns + x];
				ss << ((v > 0.3) ? 'X' : (v > 0.1) ? '+' : (v > -0.1) ? '-' : (v > -0.3) ? '.' : ' ') << ' ';
			}
			ss << "\n";
		}
		ss << "\n\n";
		std::cout << ss.str();
		std::flush(std::cout);
		std::this_thread::sleep_for(std::chrono::milliseconds(500));
	}
}

template <size_t Size>
void write_csv(size_t columns, size_t rows, std::vector<std::array<float, Size>>& result_frames) {
	std::ofstream os;
	os.open("wave_sim_result.csv", std::ios_base::out | std::ios_base::binary);

	os << "t";
	for(size_t y = 0; y < rows; ++y) {
		for(size_t x = 0; x < columns; ++x) {
			os << "," << y << ":" << x;
		}
	}
	os << "\n";

	size_t i = 0;
	for(auto& frame : result_frames) {
		os << i++;
		for(size_t y = 0; y < rows; ++y) {
			for(size_t x = 0; x < columns; ++x) {
				auto v = frame[y * columns + x];
				os << "," << v;
			}
		}
		os << "\n";
	}
}

enum class output_mode { CSV, ASCII_PLOT };

int main(int argc, char* argv[]) {
	// Explicitly initialize here so we can use MPI functions below
	celerity::runtime::init(&argc, &argv);
	output_mode out_mode = output_mode::CSV;
	if(argc > 1 && std::string(argv[1]) == "--ascii-plot") { out_mode = output_mode::ASCII_PLOT; }

	const int N = 400;   // Grid size
	const float T = 400; // Time at end of simulation

	const float dt = 0.25f;
	const float dx = 1.f;
	const float dy = 1.f;

	const int rows = N;
	const int columns = N;

	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	const bool is_master = world_rank == 0;
	std::vector<std::array<float, rows * columns>> result_frames;

	try {
		celerity::distr_queue queue;

		celerity::buffer<float, 2> up(nullptr, cl::sycl::range<2>(rows, columns)); // next
		celerity::buffer<float, 2> u(nullptr, cl::sycl::range<2>(rows, columns));  // current
		celerity::buffer<float, 2> um(nullptr, cl::sycl::range<2>(rows, columns)); // previous

		setup_wave(queue, u, {N / 4.f, N / 4.f}, 1, {N / 8.f, N / 8.f});
		zero(queue, up);
		initialize(queue, um, u, up, dt, {dx, dy});

		// We need to rotate buffers. Since we cannot swap them directly, we use pointers instead.
		auto up_ref = &up;
		auto u_ref = &u;
		auto um_ref = &um;

		auto t = 0.0;
		size_t i = 0;
		while(t < T) {
			update(queue, *up_ref, *u_ref, *um_ref, dt, {dx, dy});

			if(i++ % 10 == 0) { store(queue, *up_ref, result_frames); }

			std::swap(um_ref, u_ref);
			std::swap(u_ref, up_ref);

			t += dt;
		}
	} catch(std::exception& e) {
		std::cerr << "Exception: " << e.what() << std::endl;
		return EXIT_FAILURE;
	} catch(cl::sycl::exception& e) {
		std::cerr << "SYCL Exception: " << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	if(is_master) {
		if(out_mode == output_mode::CSV) {
			write_csv(columns, rows, result_frames);
		} else {
			print_ascii_plot(columns, rows, result_frames);
		}
	}

	return EXIT_SUCCESS;
}
