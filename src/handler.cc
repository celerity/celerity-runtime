#include "handler.h"

#include "distr_queue.h"

namespace celerity {

void compute_prepass_handler::require(cl::sycl::access::mode mode, buffer_id bid, std::unique_ptr<detail::range_mapper_base> rm) {
	queue.add_requirement(tid, bid, mode, std::move(rm));
}

compute_prepass_handler::~compute_prepass_handler() {
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

void master_access_prepass_handler::require(cl::sycl::access::mode mode, buffer_id bid, cl::sycl::range<3> range, cl::sycl::id<3> offset) const {
	queue.add_requirement(tid, bid, mode, range, offset);
}

} // namespace celerity
