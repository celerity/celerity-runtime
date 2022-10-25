#include <cstdio>

#include <celerity.h>

const size_t MAT_SIZE = 1024;

template <typename T>
void set_identity(celerity::distr_queue queue, celerity::buffer<T, 2> mat) {
	queue.submit([&](celerity::handler& cgh) {
		celerity::accessor dw{mat, cgh, celerity::access::one_to_one{}, celerity::write_only, celerity::no_init};
		cgh.parallel_for<class set_identity_kernel>(mat.get_range(), [=](celerity::item<2> item) { dw[item] = item[0] == item[1]; });
	});
}

template <typename T>
void multiply(celerity::distr_queue queue, celerity::buffer<T, 2> mat_a, celerity::buffer<T, 2> mat_b, celerity::buffer<T, 2> mat_c) {
	queue.submit([&](celerity::handler& cgh) {
		celerity::accessor a{mat_a, cgh, celerity::access::slice<2>(1), celerity::read_only};
		celerity::accessor b{mat_b, cgh, celerity::access::slice<2>(0), celerity::read_only};
		celerity::accessor c{mat_c, cgh, celerity::access::one_to_one{}, celerity::write_only, celerity::no_init};

		// Use local-memory tiling to avoid waiting on global memory too often
		// Note: We assume a local range size of 64 here, this should be supported by most devices.
		const size_t group_size = 8;
		celerity::local_accessor<T, 2> scratch_a{{group_size, group_size}, cgh};
		celerity::local_accessor<T, 2> scratch_b{{group_size, group_size}, cgh};

		cgh.parallel_for<class mat_mul>(celerity::nd_range<2>{{MAT_SIZE, MAT_SIZE}, {group_size, group_size}}, [=](celerity::nd_item<2> item) {
			T sum{};
			const auto lid = item.get_local_id();
			for(size_t j = 0; j < MAT_SIZE; j += group_size) {
				scratch_a[lid] = a[item.get_group(0) * group_size + lid[0]][j + lid[1]];
				scratch_b[lid] = b[j + lid[0]][item.get_group(1) * group_size + lid[1]];
				celerity::group_barrier(item.get_group());

				for(size_t k = 0; k < group_size; ++k) {
					const auto a_ik = scratch_a[lid[0]][k];
					const auto b_kj = scratch_b[k][lid[1]];
					sum += a_ik * b_kj;
				}
				celerity::group_barrier(item.get_group());
			}
			c[item.get_global_id()] = sum;
		});
	});
}

template <typename T>
void verify(celerity::distr_queue queue, celerity::buffer<T, 2> mat_a_buf, bool& verification_passed) {
	// capturing by reference is safe here as long as the caller of verify() ensures that verification_passed lives until the next synchronization point
	queue.submit([&](celerity::handler& cgh) {
		celerity::accessor result{mat_a_buf, cgh, celerity::access::one_to_one{}, celerity::read_only_host_task};

		cgh.host_task(mat_a_buf.get_range(), [=, &verification_passed](celerity::partition<2> part) {
			auto sr = part.get_subrange();
			for(size_t i = sr.offset[0]; i < sr.offset[0] + sr.range[0]; ++i) {
				for(size_t j = sr.offset[0]; j < sr.offset[0] + sr.range[0]; ++j) {
					const float received = result[{i, j}];
					const float expected = i == j;
					if(expected != received) {
						fprintf(stderr, "VERIFICATION FAILED for element %zu,%zu: %f (received) != %f (expected)\n", i, j, received, expected);
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

int main() {
	celerity::distr_queue queue;

	const auto range = celerity::range<2>(MAT_SIZE, MAT_SIZE);
	celerity::buffer<float, 2> mat_a_buf(range);
	celerity::buffer<float, 2> mat_b_buf(range);
	celerity::buffer<float, 2> mat_c_buf(range);

	celerity::debug::set_buffer_name(mat_a_buf, "mat_a");
	celerity::debug::set_buffer_name(mat_b_buf, "mat_b");

	set_identity(queue, mat_a_buf);
	set_identity(queue, mat_b_buf);

	multiply(queue, mat_a_buf, mat_b_buf, mat_c_buf);
	multiply(queue, mat_b_buf, mat_c_buf, mat_a_buf);

	bool verification_passed = true;
	verify(queue, mat_a_buf, verification_passed);
	queue.slow_full_sync(); // Wait for verification_passed to become available

	return verification_passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
