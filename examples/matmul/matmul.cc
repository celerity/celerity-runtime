#include <cstdio>

#include <celerity.h>

constexpr size_t MAT_SIZE = 1024;

template <typename T>
void set_identity(celerity::distr_queue queue, celerity::buffer<T, 2> mat) {
	queue.submit([=](celerity::handler& cgh) {
		auto dw = mat.template get_access<cl::sycl::access::mode::discard_write>(cgh, celerity::access::one_to_one<2>());
		cgh.parallel_for<class set_identity_kernel>(mat.get_range(), [=](cl::sycl::item<2> item) { dw[item] = item[0] == item[1]; });
	});
}

template <typename T>
void multiply(celerity::distr_queue queue, celerity::buffer<T, 2> mat_a, celerity::buffer<T, 2> mat_b, celerity::buffer<T, 2> mat_c) {
	queue.submit([=](celerity::handler& cgh) {
		auto a = mat_a.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::slice<2>(1));
		auto b = mat_b.template get_access<cl::sycl::access::mode::read>(cgh, celerity::access::slice<2>(0));
		auto c = mat_c.template get_access<cl::sycl::access::mode::discard_write>(cgh, celerity::access::one_to_one<2>());

		cgh.parallel_for<class mat_mul>(cl::sycl::range<2>(MAT_SIZE, MAT_SIZE), [=](cl::sycl::item<2> item) {
			auto sum = 0.f;
			for(size_t k = 0; k < MAT_SIZE; ++k) {
				const auto a_ik = a[{item[0], k}];
				const auto b_kj = b[{k, item[1]}];
				sum += a_ik * b_kj;
			}
			c[item] = sum;
		});
	});
}

int main(int argc, char* argv[]) {
	bool verification_passed = true;

	celerity::runtime::init(&argc, &argv);

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	celerity::experimental::bench::log_user_config({{"matSize", std::to_string(MAT_SIZE)}});

	{
		celerity::distr_queue queue;

		auto range = cl::sycl::range<2>(MAT_SIZE, MAT_SIZE);
		celerity::buffer<float, 2> mat_a_buf(range);
		celerity::buffer<float, 2> mat_b_buf(range);
		celerity::buffer<float, 2> mat_c_buf(range);

		set_identity(queue, mat_a_buf);
		set_identity(queue, mat_b_buf);

		celerity::experimental::bench::begin("main program");

		multiply(queue, mat_a_buf, mat_b_buf, mat_c_buf);
		multiply(queue, mat_b_buf, mat_c_buf, mat_a_buf);

		queue.submit(celerity::allow_by_ref, [&](celerity::handler& cgh) {
			auto result = mat_a_buf.get_access<cl::sycl::access::mode::read, cl::sycl::access::target::host_buffer>(cgh, celerity::access::one_to_one<2>());

			cgh.host_task(range, [=, &verification_passed](celerity::partition<2> part) {
				celerity::experimental::bench::end("main program");

				auto sr = part.get_subrange();
				for(size_t i = sr.offset[0]; i < sr.offset[0] + sr.range[0]; ++i) {
					for(size_t j = sr.offset[0]; j < sr.offset[0] + sr.range[0]; ++j) {
						const float kernel_value = result[{i, j}];
						const float host_value = i == j;
						if(kernel_value != host_value) {
							fprintf(stderr, "rank %d: VERIFICATION FAILED for element %zu,%zu: %f (received) != %f (expected)\n", rank, i, j, kernel_value,
							    host_value);
							verification_passed = false;
							break;
						}
					}
					if(!verification_passed) { break; }
				}
				if(verification_passed) { printf("rank %d: VERIFICATION PASSED!\n", rank); }
			});
		});
	}

	return verification_passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
