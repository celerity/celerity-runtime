#include "handler.h"

#include "task_manager.h"

namespace celerity {

void compute_prepass_handler::require(buffer_id bid, std::unique_ptr<detail::range_mapper_base> rm) const {
	task.add_range_mapper(bid, std::move(rm));
}

void master_access_prepass_handler::require(cl::sycl::access::mode mode, buffer_id bid, cl::sycl::range<3> range, cl::sycl::id<3> offset) const {
	task.add_buffer_access(bid, mode, subrange<3>(offset, range));
}

} // namespace celerity
