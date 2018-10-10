#include "handler.h"

#include "task_manager.h"

namespace celerity {

void compute_prepass_handler::require(cl::sycl::access::mode mode, buffer_id bid, std::unique_ptr<detail::range_mapper_base> rm) {
	task_mngr.add_requirement(tid, bid, mode, std::move(rm));
}

compute_prepass_handler::~compute_prepass_handler() {
	task_mngr.set_compute_task_data(tid, dimensions, global_size, debug_name);
}

void master_access_prepass_handler::require(cl::sycl::access::mode mode, buffer_id bid, cl::sycl::range<3> range, cl::sycl::id<3> offset) const {
	task_mngr.add_requirement(tid, bid, mode, range, offset);
}

} // namespace celerity
