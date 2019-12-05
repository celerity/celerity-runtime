#include <cstdio>
#include <vector>

#include <celerity.h>

template <typename T>
void multiply(celerity::distr_queue queue, celerity::buffer<T, 2> mat_a, celerity::buffer<T, 2> mat_b, celerity::buffer<T, 2> mat_c, const size_t mat_size) {
	queue.submit([=](celerity::handler& cgh) {
		auto a = mat_a.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::slice<2>(1));
		auto b = mat_b.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::slice<2>(0));
		auto c = mat_c.template get_access<cl::sycl::access::mode::discard_write>(cgh, celerity::access::one_to_one<2>());

		cgh.parallel_for<class mat_mul>(cl::sycl::range<2>(mat_size, mat_size), [=](cl::sycl::item<2> item) {
			auto sum = 0.f;
			for(size_t k = 0; k < mat_size; ++k) {
				const auto a_ik = a[{item[0], k}];
				const auto b_kj = b[{k, item[1]}];
				sum += a_ik * b_kj;
			}
			c[item] = sum;
		});
	});
}

int main(int argc, char* argv[]) {
	if(argc <= 1) {
		std::cout << "Usage: ./matmulChain size\n";
		return 1;
	}
	const size_t mat_size = std::stoi(argv[1]);
	// Explicitly initialize here so we can use MPI functions below
	celerity::runtime::init(&argc, &argv);
	bool verification_passed = true;

	celerity::experimental::bench::log_user_config({{"matSize", std::to_string(mat_size)}});

	std::vector<float> mat_a(mat_size * mat_size);
	std::vector<float> mat_b(mat_size * mat_size);
	std::vector<float> mat_c(mat_size * mat_size);
	std::vector<float> mat_d(mat_size * mat_size);

	// Initialize matrices a and b to the identity
	for(size_t i = 0; i < mat_size; ++i) {
		for(size_t j = 0; j < mat_size; ++j) {
			mat_a[i * mat_size + j] = i == j;
			mat_b[i * mat_size + j] = i == j;
			mat_c[i * mat_size + j] = i == j;
			mat_d[i * mat_size + j] = i == j;
		}
	}

	{
		celerity::distr_queue queue;

		celerity::buffer<float, 2> mat_a_buf(mat_a.data(), cl::sycl::range<2>(mat_size, mat_size));
		celerity::buffer<float, 2> mat_b_buf(mat_b.data(), cl::sycl::range<2>(mat_size, mat_size));
		celerity::buffer<float, 2> mat_c_buf(mat_b.data(), cl::sycl::range<2>(mat_size, mat_size));
		celerity::buffer<float, 2> mat_d_buf(mat_b.data(), cl::sycl::range<2>(mat_size, mat_size));
		celerity::buffer<float, 2> mat_r_buf(cl::sycl::range<2>(mat_size, mat_size));
		celerity::buffer<float, 2> mat_p_buf(cl::sycl::range<2>(mat_size, mat_size));
		celerity::buffer<float, 2> mat_q_buf(cl::sycl::range<2>(mat_size, mat_size));

		MPI_Barrier(MPI_COMM_WORLD);
		celerity::experimental::bench::begin("main program");

		multiply(queue, mat_a_buf, mat_b_buf, mat_p_buf, mat_size);
		multiply(queue, mat_c_buf, mat_d_buf, mat_q_buf, mat_size);
		multiply(queue, mat_p_buf, mat_q_buf, mat_r_buf, mat_size);

		queue.with_master_access([&](celerity::handler& cgh) {
			auto result = mat_r_buf.get_access<cl::sycl::access::mode::read>(cgh, cl::sycl::range<2>(mat_size, mat_size));

			cgh.run([=, &verification_passed]() {
				celerity::experimental::bench::end("main program");

				for(size_t i = 0; i < mat_size; ++i) {
					for(size_t j = 0; j < mat_size; ++j) {
						const float kernel_value = result[{i, j}];
						const float host_value = i == j;
						if(kernel_value != host_value) {
							fprintf(stderr, "VERIFICATION FAILED for element %ld,%ld: %f != %f\n", i, j, kernel_value, host_value);
							verification_passed = false;
							break;
						}
					}
					if(!verification_passed) { break; }
				}
				if(verification_passed) { printf("VERIFICATION PASSED!\n"); }
			});
		});
	}

	return verification_passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
