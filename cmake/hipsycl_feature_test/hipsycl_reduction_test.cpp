#include <CL/sycl.hpp>

int main() {
	cl::sycl::buffer<int> b{{1}};
	cl::sycl::queue{}.submit([&](cl::sycl::handler& cgh) {
		cgh.parallel_for(cl::sycl::range<1>{1}, cl::sycl::reduction(b, cgh, cl::sycl::plus<int>{}), [](cl::sycl::item<1>, auto& r) { r += 1; });
	});
}
