#include <cstdio>
#include <vector>

#include <celerity.h>

constexpr size_t MAT_SIZE = 2048;
constexpr size_t GROUP_SIZE = 8;

void multiply(celerity::distr_queue queue, celerity::buffer<float, 2> mat_a, celerity::buffer<float, 2> mat_b, celerity::buffer<float, 2> mat_c) {
	assert(MAT_SIZE % GROUP_SIZE == 0);

	queue.submit([=](celerity::handler& cgh) {
		auto a = mat_a.get_access<cl::sycl::access::mode::read>(cgh, celerity::access::slice<2>{1});
		auto b = mat_b.get_access<cl::sycl::access::mode::read>(cgh, celerity::access::slice<2>{0});
		auto c = mat_c.get_access<cl::sycl::access::mode::discard_write>(cgh, celerity::access::one_to_one<2>{});

		cgh.parallel_for_work_group<class matmul_tiled>(
		    cl::sycl::range<2>(MAT_SIZE / GROUP_SIZE, MAT_SIZE / GROUP_SIZE), cl::sycl::range<2>(GROUP_SIZE, GROUP_SIZE), [=](cl::sycl::group<2> group) {
#if 1
			    // Create local scratch buffers to hold a tile of a matrix
			    float scratch_a[GROUP_SIZE * GROUP_SIZE];
			    float scratch_b[GROUP_SIZE * GROUP_SIZE];

#define USE_RESULT_SCRATCH 1
#if USE_RESULT_SCRATCH
			    // float scratch_c[GROUP_SIZE * GROUP_SIZE];
			    cl::sycl::private_memory<float, 2> scratch_c(group);
			    group.parallel_for_work_item([&](cl::sycl::h_item<2> h_item) {
				    // scratch_c[h_item.get_local().get_linear_id()] = 0.f;
				    scratch_c(h_item) = 0.f;
			    });
#endif

			    // Iterate over all tiles
			    for(size_t K = 0; K < MAT_SIZE; K += GROUP_SIZE) {
				    // Copy data into scratch buffers in parallel
				    group.parallel_for_work_item([&](cl::sycl::h_item<2> h_item) {
					    auto lid = h_item.get_local_id();
					    // TODO: One of these 2D indices should likely be swapped for coalescing access
					    scratch_a[h_item.get_local().get_linear_id()] = a[{group.get_id()[0] * GROUP_SIZE + lid[0], K + lid[1]}];
					    scratch_b[h_item.get_local().get_linear_id()] = b[{K + lid[0], group.get_id()[1] * GROUP_SIZE + lid[1]}];
				    });

				    // Compute partial result in current tile
				    group.parallel_for_work_item([&](cl::sycl::h_item<2> h_item) {
					    float sum = 0.f;
					    for(size_t k = 0; k < GROUP_SIZE; ++k) {
						    auto lid = h_item.get_local_id();
						    const auto a_ik = scratch_a[lid[0] * GROUP_SIZE + k];
						    const auto b_kj = scratch_b[k * GROUP_SIZE + lid[1]];
						    sum += a_ik * b_kj;
					    }

#if USE_RESULT_SCRATCH
					    // scratch_c[h_item.get_local().get_linear_id()] += sum;
					    scratch_c(h_item) += sum;
#else
					    c[h_item.get_global()] += sum;
#endif
				    });
			    }

#if USE_RESULT_SCRATCH
			    // Write final values back to global memory
			    group.parallel_for_work_item([&](cl::sycl::h_item<2> h_item) {
				    // c[h_item.get_global()] = scratch_c[h_item.get_local().get_linear_id()];
				    c[h_item.get_global()] = scratch_c(h_item);
			    });
#endif
#else
			    group.parallel_for_work_item([&](cl::sycl::h_item<2> h_item) {
				    auto item = h_item.get_global();
				    auto sum = 0.f;
				    for(size_t k = 0; k < MAT_SIZE; ++k) {
					    const auto a_ik = a[{item[0], k}];
					    const auto b_kj = b[{k, item[1]}];
					    sum += a_ik * b_kj;
				    }
				    c[item] = sum;
			    });
#endif
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
	for(size_t i = 0; i < MAT_SIZE; ++i) {
		for(size_t j = 0; j < MAT_SIZE; ++j) {
			mat_a[i * MAT_SIZE + j] = i == j;
			mat_b[i * MAT_SIZE + j] = i == j;
		}
	}

	{
		celerity::distr_queue queue;

		celerity::buffer<float, 2> mat_a_buf(mat_a.data(), cl::sycl::range<2>(MAT_SIZE, MAT_SIZE));
		celerity::buffer<float, 2> mat_b_buf(mat_b.data(), cl::sycl::range<2>(MAT_SIZE, MAT_SIZE));
		celerity::buffer<float, 2> mat_c_buf(cl::sycl::range<2>(MAT_SIZE, MAT_SIZE));
		celerity::buffer<float, 2> mat_d_buf(cl::sycl::range<2>(MAT_SIZE, MAT_SIZE));

		MPI_Barrier(MPI_COMM_WORLD);
		celerity::experimental::bench::begin("main program");

		multiply(queue, mat_a_buf, mat_b_buf, mat_c_buf);
		multiply(queue, mat_b_buf, mat_c_buf, mat_d_buf);

		queue.with_master_access([&, mat_d_buf](celerity::handler& cgh) {
			auto result = mat_d_buf.get_access<cl::sycl::access::mode::read>(cgh, cl::sycl::range<2>(MAT_SIZE, MAT_SIZE));

			cgh.run([=, &verification_passed]() {
				celerity::experimental::bench::end("main program");

				for(size_t i = 0; i < MAT_SIZE; ++i) {
					for(size_t j = 0; j < MAT_SIZE; ++j) {
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
