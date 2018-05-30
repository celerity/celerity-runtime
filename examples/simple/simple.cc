#include <cstdlib>

#ifdef _MSC_VER
#include <process.h>
#else
#include <unistd.h>
#endif

#include <SYCL/sycl.hpp>
#include <celerity.h>

// Use define instead of constexpr as MSVC seems to have some trouble getting it into nested closures
#define DEMO_DATA_SIZE (1024)

void print_pid() {
	std::cout << "PID: ";
#ifdef _MSC_VER
	std::cout << _getpid();
#else
	std::cout << getpid();
#endif
	std::cout << std::endl;
}

int main(int argc, char* argv[]) {
	celerity::runtime::init(&argc, &argv);
	print_pid();
	bool verification_passed = true;

	std::vector<float> host_data_a(DEMO_DATA_SIZE);
	std::vector<float> host_data_b(DEMO_DATA_SIZE);
	std::vector<float> host_data_c(DEMO_DATA_SIZE);
	std::vector<float> host_data_d(DEMO_DATA_SIZE);

	try {
		celerity::distr_queue queue;

		// TODO: Do we support SYCL sub-buffers & images? Section 4.7.2
		celerity::buffer<float, 1> buf_a(host_data_a.data(), cl::sycl::range<1>(DEMO_DATA_SIZE));
		celerity::buffer<float, 1> buf_b(host_data_b.data(), cl::sycl::range<1>(DEMO_DATA_SIZE));
		celerity::buffer<float, 1> buf_c(host_data_c.data(), cl::sycl::range<1>(DEMO_DATA_SIZE));
		celerity::buffer<float, 1> buf_d(host_data_d.data(), cl::sycl::range<1>(DEMO_DATA_SIZE));

		queue.submit([&](auto& cgh) {
			auto a = buf_a.get_access<cl::sycl::access::mode::write>(cgh, [](celerity::subrange<1> range) -> celerity::subrange<1> {
				celerity::subrange<1> sr(range);
				// Write the opposite subrange
				// This is useful to demonstrate that the nodes are assigned to
				// chunks somewhat intelligently in order to minimize buffer
				// transfers. Remove this line and the node assignment in the
				// command graph should be flipped.
				sr.start = range.global_size - range.start - range.range;
				return sr;
			});

			cgh.template parallel_for<class produce_a>(cl::sycl::range<1>(DEMO_DATA_SIZE), [=](cl::sycl::item<1> item) {
				auto id = item.get_id()[0];
				a[DEMO_DATA_SIZE - 1 - id] = 1.f;
			});
		});

		queue.submit([&](auto& cgh) {
			auto a = buf_a.get_access<cl::sycl::access::mode::read>(cgh, [](celerity::subrange<1> range) -> celerity::subrange<1> {
				celerity::subrange<1> sr(range);
				// Add some overlap so we can generate pull commands
				// NOTE: JUST A DEMO. NOT HONORED IN KERNEL.
				if(range.start[0] > 10) { sr.start -= 10; }
				sr.range += 20;
				return sr;
			});

			auto b = buf_b.get_access<cl::sycl::access::mode::write>(cgh, celerity::access::one_to_one<1>());
			cgh.template parallel_for<class compute_b>(cl::sycl::range<1>(DEMO_DATA_SIZE), [=](cl::sycl::item<1> item) {
				auto i = item.get_id();
				b[i] = a[i] * 2.f;
			});
		});

#define COMPUTE_C_ON_MASTER 1
#if COMPUTE_C_ON_MASTER
		celerity::with_master_access([&](auto& mah) {
			auto a = buf_a.get_access<cl::sycl::access::mode::read>(mah, cl::sycl::range<1>(DEMO_DATA_SIZE));
			auto c = buf_c.get_access<cl::sycl::access::mode::write>(mah, cl::sycl::range<1>(DEMO_DATA_SIZE));

			mah.run([=]() {
				for(int i = 0; i < DEMO_DATA_SIZE; ++i) {
					c[i] = 2.f - a[i];
				}
			});
		});
#else
		queue.submit([&](auto& cgh) {
			auto a = buf_a.get_access<cl::sycl::access::mode::read>(cgh, celerity::access::one_to_one<1>());
			auto c = buf_c.get_access<cl::sycl::access::mode::write>(cgh, celerity::access::one_to_one<1>());
			cgh.template parallel_for<class compute_c>(cl::sycl::range<1>(DEMO_DATA_SIZE), [=](cl::sycl::item<1> item) {
				auto i = item.get_id();
				c[i] = 2.f - a[i];
			});
		});

#endif

		queue.submit([&](auto& cgh) {
			auto b = buf_b.get_access<cl::sycl::access::mode::read>(cgh, celerity::access::one_to_one<1>());
			auto c = buf_c.get_access<cl::sycl::access::mode::read>(cgh, celerity::access::one_to_one<1>());
			auto d = buf_d.get_access<cl::sycl::access::mode::write>(cgh, celerity::access::one_to_one<1>());
			cgh.template parallel_for<class compute_d>(cl::sycl::range<1>(DEMO_DATA_SIZE), [=](cl::sycl::item<1> item) {
				auto i = item.get_id();
				d[i] = b[i] + c[i];
			});
		});

		celerity::with_master_access([&](auto& mah) {
			auto d = buf_d.get_access<cl::sycl::access::mode::read>(mah, cl::sycl::range<1>(DEMO_DATA_SIZE));

			mah.run([=]() {
				size_t sum = 0;
				for(int i = 0; i < DEMO_DATA_SIZE; ++i) {
					sum += (size_t)d[i];
				}

				std::cout << "## RESULT: ";
				if(sum == 3 * DEMO_DATA_SIZE) {
					std::cout << "Success! Correct value was computed." << std::endl;
				} else {
					std::cout << "Fail! Value is " << sum << std::endl;
					verification_passed = false;
				}
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
