#include <vector>

#include <celerity.h>

int main(int argc, char* argv[]) {
	constexpr size_t buf_size = 512;

	celerity::queue queue;
	celerity::buffer<size_t, 1> buf(buf_size);

#if 1 // simple case, initialize everything replicated
	queue.submit([&](celerity::handler& cgh) {
		celerity::accessor b{buf, cgh, celerity::access::all{}, celerity::write_only_replicated, celerity::no_init};
		celerity::debug::set_task_name(cgh, "write replicated");
		cgh.parallel_for(buf.get_range() * 4, [=](celerity::item<1> item) { b[item[0] % buf_size] = item.get_linear_id() % buf_size; });
	});

#else // more complex case - initialize half one-to-one, other half replicated
	// Initialize buffer in a distributed device kernel
	queue.submit([&](celerity::handler& cgh) {
		celerity::accessor b{buf, cgh, celerity::access::one_to_one{}, celerity::write_only, celerity::no_init};
		cgh.parallel_for<class write_linear_id>(buf.get_range() / 2, [=](celerity::item<1> item) { b[item[0]] = item.get_linear_id(); });
	});

	// TODO NEXT STEP: Make it work for IDAG as well!
	queue.submit([&](celerity::handler& cgh) {
		celerity::accessor b{
		    buf, cgh, celerity::access::fixed{celerity::subrange<1>{buf_size / 2, buf_size / 2}}, celerity::write_only_replicated, celerity::no_init};
		cgh.parallel_for<class write_linear_id_two>(buf.get_range() * 2, celerity::id<1>(buf_size / 2),
		    [=](celerity::item<1> item) { b[buf_size / 2 + (item[0] % (buf_size / 2))] = buf_size / 2 + (item.get_linear_id() % (buf_size / 2)); });
	});
#endif

	// Read/write the whole thing (NOP) to test whether IDAG does any copies
	queue.submit([&](celerity::handler& cgh) {
		celerity::accessor b{buf, cgh, celerity::access::one_to_one{}, celerity::read_write};
		celerity::debug::set_task_name(cgh, "dummy read");
		cgh.parallel_for<class write_linear_id>(buf.get_range(), [=](celerity::item<1> item) { (void)b; });
	});

	// Process values on the host
	std::vector<size_t> host_buf(buf_size);
	queue.submit([&](celerity::handler& cgh) {
		celerity::accessor b{buf, cgh, celerity::access::all{}, celerity::read_only_host_task};
		celerity::debug::set_task_name(cgh, "verify");
		cgh.host_task(celerity::experimental::collective, [=, &host_buf](celerity::experimental::collective_partition) {
			std::this_thread::sleep_for(std::chrono::milliseconds(100)); // give the synchronization more time to fail
			for(size_t i = 0; i < buf_size; i++) {
				host_buf[i] = 2 * b[i];
			}
		});
	});

	// Wait until both tasks have completed
	queue.wait();

	// At this point we can safely interact with host_buf from within the application thread
	bool valid = true;
	for(size_t i = 0; i < buf_size; i++) {
		if(host_buf[i] != 2 * i) {
			CELERITY_CRITICAL("Validation failed: host_buf[{}] = {} (expected {})", i, host_buf[i], 2 * i);
			valid = false;
			break;
		}
	}

	return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}
