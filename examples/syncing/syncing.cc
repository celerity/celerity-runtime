#include <vector>

#include <celerity.h>

int main(int argc, char* argv[]) {
	constexpr size_t buf_size = 512;

	celerity::distr_queue queue;
	celerity::buffer<size_t, 1> buf(buf_size);

	// Initialize buffer in a distributed device kernel
	queue.submit([&](celerity::handler& cgh) {
		celerity::accessor b{buf, cgh, celerity::access::one_to_one{}, celerity::write_only, celerity::no_init};
		cgh.parallel_for<class write_linear_id>(buf.get_range(), [=](celerity::item<1> item) { b[item] = item.get_linear_id(); });
	});

	// Process values on the host
	std::vector<size_t> host_buf(buf_size);
	queue.submit([&](celerity::handler& cgh) {
		celerity::accessor b{buf, cgh, celerity::access::all{}, celerity::read_only_host_task};
		cgh.host_task(celerity::experimental::collective, [=, &host_buf](celerity::experimental::collective_partition) {
			std::this_thread::sleep_for(std::chrono::milliseconds(100)); // give the synchronization more time to fail
			for(size_t i = 0; i < buf_size; i++) {
				host_buf[i] = 2 * b[i];
			}
		});
	});

	// Wait until both tasks have completed
	queue.slow_full_sync();

	// At this point we can safely interact with host_buf from within the main thread
	bool valid = true;
	for(size_t i = 0; i < buf_size; i++) {
		if(host_buf[i] != 2 * i) {
			CELERITY_ERROR("got {}, expected {}", host_buf[i], 2 * i);
			valid = false;
			break;
		}
	}

	return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}
