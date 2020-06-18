#include <cstdio>
#include <vector>

#include <celerity.h>

using namespace celerity;

int main(int argc, char* argv[]) {
	constexpr int N = 10;

	celerity::distr_queue q;
	celerity::buffer<int, 1> buff(N);
	std::vector<int> host_buff(N);

	q.submit([=](handler& cgh) {
		auto b = buff.get_access<cl::sycl::access::mode::discard_write>(cgh, access::one_to_one<1>());
		cgh.parallel_for<class mat_mul>(cl::sycl::range<1>(N), [=](cl::sycl::item<1> item) { b[item] = item.get_linear_id(); });
	});

	q.submit(celerity::allow_by_ref, [=, &host_buff](handler& cgh) {
		auto b = buff.get_access<cl::sycl::access::mode::read, cl::sycl::access::target::host_buffer>(cgh, access::fixed<1>({0, N}));
		cgh.host_task(on_master_node, [=, &host_buff] {
			std::this_thread::sleep_for(std::chrono::milliseconds(10)); // give the synchronization more time to fail
			for(int i = 0; i < N; i++) {
				host_buff[i] = b[i];
			}
		});
	});

	q.slow_full_sync();

	int rank = 1;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	bool valid = true;
	if(rank == 0) {
		for(int i = 0; i < N; i++) {
			if(host_buff[i] != i) valid = false;
		}
	}

	return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}
