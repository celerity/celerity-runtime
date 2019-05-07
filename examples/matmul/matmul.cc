#include <cstdlib>
#include <random>

#include <celerity.h>

#define MAT_SIZE 1024

// TODO: See if we can make buffers a and b const refs here
template <typename T>
void multiply(celerity::distr_queue queue, celerity::buffer<T, 2> mat_a, celerity::buffer<T, 2> mat_b, celerity::buffer<T, 2> mat_c) {
	queue.submit([=](celerity::handler& cgh) {
		auto a = mat_a.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::slice<2>(1));
		auto b = mat_b.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::slice<2>(0));
		auto c = mat_c.template get_access<cl::sycl::access::mode::discard_write>(cgh, celerity::access::one_to_one<2>());

		cgh.parallel_for<class mat_mul>(cl::sycl::range<2>(MAT_SIZE, MAT_SIZE), [=](cl::sycl::item<2> item) {
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
	// Explicitly initialize here so we can use MPI functions below
	celerity::runtime::init(&argc, &argv);
	bool verification_passed = true;

	celerity::experimental::bench::log_user_config({{"matSize", std::to_string(MAT_SIZE)}});

	std::vector<float> mat_a(MAT_SIZE * MAT_SIZE);
	std::vector<float> mat_b(MAT_SIZE * MAT_SIZE);

	// Initialize matrices a and b to the identity
	for(auto i = 0; i < MAT_SIZE; ++i) {
		for(auto j = 0; j < MAT_SIZE; ++j) {
			mat_a[i * MAT_SIZE + j] = i == j;
			mat_b[i * MAT_SIZE + j] = i == j;
		}
	}

	try {
		celerity::distr_queue queue;

		celerity::buffer<float, 2> mat_a_buf(mat_a.data(), cl::sycl::range<2>(MAT_SIZE, MAT_SIZE));
		celerity::buffer<float, 2> mat_b_buf(mat_b.data(), cl::sycl::range<2>(MAT_SIZE, MAT_SIZE));
		celerity::buffer<float, 2> mat_c_buf(cl::sycl::range<2>(MAT_SIZE, MAT_SIZE));

		MPI_Barrier(MPI_COMM_WORLD);
		celerity::experimental::bench::begin("main program");

		multiply(queue, mat_a_buf, mat_b_buf, mat_c_buf);
		multiply(queue, mat_b_buf, mat_c_buf, mat_a_buf);

		queue.with_master_access([&](celerity::handler& cgh) {
			auto result = mat_a_buf.get_access<cl::sycl::access::mode::read>(cgh, cl::sycl::range<2>(MAT_SIZE, MAT_SIZE));

			cgh.run([=, &verification_passed]() {
				celerity::experimental::bench::end("main program");

				for(auto i = 0ull; i < MAT_SIZE; ++i) {
					for(auto j = 0ull; j < MAT_SIZE; ++j) {
						const float kernel_value = result[{i, j}];
						const float host_value = i == j;
						if(kernel_value != host_value) {
							std::cerr << fmt::format(
							                 "VERIFICATION FAILED for element {},{}: {} != {}", i, j, std::to_string(kernel_value), std::to_string(host_value))
							          << std::endl;
							verification_passed = false;
							break;
						}
					}
					if(!verification_passed) { break; }
				}
				if(verification_passed) { std::cout << "VERIFICATION PASSED!" << std::endl; }
			});
		});
	} catch(std::exception& e) {
		std::cerr << "Exception: " << e.what() << std::endl;
		return EXIT_FAILURE;
	} catch(cl::sycl::exception& e) {
		std::cerr << "SYCL Exception: " << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return verification_passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
