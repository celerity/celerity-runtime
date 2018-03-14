#include "handler.h"

#include "distr_queue.h"

namespace celerity {

template <>
void handler<is_prepass::true_t>::require(prepass_accessor<cl::sycl::access::mode::read> a, buffer_id bid, std::unique_ptr<detail::range_mapper_base> rm) {
	queue.add_requirement(tid, bid, cl::sycl::access::mode::read, std::move(rm));
}

template <>
void handler<is_prepass::true_t>::require(prepass_accessor<cl::sycl::access::mode::write> a, buffer_id bid, std::unique_ptr<detail::range_mapper_base> rm) {
	queue.add_requirement(tid, bid, cl::sycl::access::mode::write, std::move(rm));
}

handler<is_prepass::true_t>::~handler() {
	const int dimensions = global_size.which() + 1;
	switch(dimensions) {
	case 1: queue.set_task_data(tid, boost::get<cl::sycl::range<1>>(global_size), debug_name); break;
	case 2: queue.set_task_data(tid, boost::get<cl::sycl::range<2>>(global_size), debug_name); break;
	case 3: queue.set_task_data(tid, boost::get<cl::sycl::range<3>>(global_size), debug_name); break;
	default:
		// Can't happen
		assert(false);
		break;
	}
}

} // namespace celerity
