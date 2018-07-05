#include <chrono>
#include <cstdlib>
#include <random>

#ifdef _MSC_VER
#include <process.h>
#else
#include <unistd.h>
#endif

#include <SYCL/sycl.hpp>
#include <celerity.h>
#include <spdlog/fmt/fmt.h>

#define MAT_SIZE 256
#define ENABLE_VERIFICATION (MAT_SIZE < 2048)

void print_pid() {
	std::cout << "PID: ";
#ifdef _MSC_VER
	std::cout << _getpid();
#else
	std::cout << getpid();
#endif
	std::cout << std::endl;
}

// TODO: See if we can make buffers a and b const refs here
template <typename T>
void multiply(celerity::distr_queue& queue, celerity::buffer<T, 2>& mat_a, celerity::buffer<T, 2>& mat_b, celerity::buffer<T, 2>& mat_c) {
	queue.submit([&](auto& cgh) {
		auto a = mat_a.template get_access<cl::sycl::access::mode::read>(cgh, [=](celerity::chunk<2> chnk) {
			auto rows = chnk;
			rows.offset[1] = 0;
			rows.range[1] = MAT_SIZE;
			return rows;
		});
		auto b = mat_b.template get_access<cl::sycl::access::mode::read>(cgh, [=](celerity::chunk<2> chnk) {
			auto cols = chnk;
			cols.offset[0] = 0;
			cols.range[0] = MAT_SIZE;
			return cols;
		});
		auto c = mat_c.template get_access<cl::sycl::access::mode::write>(cgh, celerity::access::one_to_one<2>());

		cgh.template parallel_for<class mat_mul>(cl::sycl::range<2>(MAT_SIZE, MAT_SIZE), [=](cl::sycl::item<2> item) {
			auto sum = 0.f;
			for(auto k = 0ull; k < MAT_SIZE; ++k) {
				const auto a_ik = a[{item[0], k}];
				const auto b_kj = b[{k, item[1]}];
				sum += a_ik * b_kj;
			}
			c[item] = sum;
		});
	});
}

int main(int argc, char* argv[]) {
	celerity::runtime::init(&argc, &argv);
	print_pid();
	bool verification_passed = true;

	std::vector<float> mat_a(MAT_SIZE * MAT_SIZE);
	std::vector<float> mat_b(MAT_SIZE * MAT_SIZE);
	std::vector<float> mat_c(MAT_SIZE * MAT_SIZE);

	std::mt19937 gen(1337);
	std::uniform_real_distribution<float> dis(0.f, 10.f);

	// Initialize matrices a and b with random values
	for(auto i = 0; i < MAT_SIZE; ++i) {
		for(auto j = 0; j < MAT_SIZE; ++j) {
			mat_a[i * MAT_SIZE + j] = dis(gen);
			mat_b[i * MAT_SIZE + j] = dis(gen);
		}
	}

	std::vector<float> result_host(MAT_SIZE * MAT_SIZE);
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	const bool is_master = world_rank == 0;

#if ENABLE_VERIFICATION
	if(is_master) {
		std::cout << "Computing ground-truth." << std::endl;

		std::vector<float> tmp_host(MAT_SIZE * MAT_SIZE);
		for(auto i = 0; i < MAT_SIZE; ++i) {
			for(auto j = 0; j < MAT_SIZE; ++j) {
				auto sum = 0.f;
				for(auto k = 0; k < MAT_SIZE; ++k) {
					const auto a_ik = mat_a[i * MAT_SIZE + k];
					const auto b_kj = mat_b[k * MAT_SIZE + j];
					sum += a_ik * b_kj;
				}
				tmp_host[i * MAT_SIZE + j] = sum;
			}
		}
		for(auto i = 0; i < MAT_SIZE; ++i) {
			for(auto j = 0; j < MAT_SIZE; ++j) {
				auto sum = 0.f;
				for(auto k = 0; k < MAT_SIZE; ++k) {
					const auto b_ik = mat_b[i * MAT_SIZE + k];
					const auto t_kj = tmp_host[k * MAT_SIZE + j];
					sum += b_ik * t_kj;
				}
				result_host[i * MAT_SIZE + j] = sum;
			}
		}

		std::cout << "Done computing ground-truth." << std::endl;
	}
#endif

	try {
		std::chrono::high_resolution_clock bench_clock;
		const auto bench_start = bench_clock.now();

		celerity::distr_queue queue;

		celerity::buffer<float, 2> mat_a_buf(mat_a.data(), cl::sycl::range<2>(MAT_SIZE, MAT_SIZE));
		celerity::buffer<float, 2> mat_b_buf(mat_b.data(), cl::sycl::range<2>(MAT_SIZE, MAT_SIZE));
		celerity::buffer<float, 2> mat_c_buf(mat_c.data(), cl::sycl::range<2>(MAT_SIZE, MAT_SIZE));

		multiply(queue, mat_a_buf, mat_b_buf, mat_c_buf);
		multiply(queue, mat_b_buf, mat_c_buf, mat_a_buf);

		celerity::with_master_access([&](auto& mah) {
			auto result = mat_a_buf.get_access<cl::sycl::access::mode::read>(mah, cl::sycl::range<2>(MAT_SIZE, MAT_SIZE));

			mah.run([=, &verification_passed]() {
				std::cout << fmt::format(
				    "Execution time: {}ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(bench_clock.now() - bench_start).count());

#if ENABLE_VERIFICATION
#define EPSILON 1e-5
				for(auto i = 0ull; i < MAT_SIZE; ++i) {
					for(auto j = 0ull; j < MAT_SIZE; ++j) {
						const auto kernel_value = result[{i, j}];
						const auto host_value = result_host[i * MAT_SIZE + j];
						if(std::abs(kernel_value - host_value) > EPSILON) {
							std::cerr << fmt::format("VERIFICATION FAILED for element {},{}: {} != {}", i, j, kernel_value, host_value) << std::endl;
							verification_passed = false;
							break;
						}
					}
					if(!verification_passed) { break; }
				}
				if(verification_passed) { std::cout << "VERIFICATION PASSED!" << std::endl; }
#else
				std::cout << "(VERIFICATION IS DISABLED)" << std::endl;
#endif
			});
		});

		celerity::runtime::get_instance().TEST_do_work();

	} catch(std::exception& e) {
		std::cerr << "Exception: " << e.what() << std::endl;
		return EXIT_FAILURE;
	} catch(cl::sycl::exception& e) {
		std::cerr << "SYCL Exception: " << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return verification_passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
