#include "buffer.h"

#include "runtime.h"

namespace celerity {

buffer_id buffer_base::register_with_runtime(
    cl::sycl::range<3> range, std::shared_ptr<detail::buffer_storage_base> buffer_storage, bool host_initialized) const {
	return runtime::get_instance().register_buffer(range, buffer_storage, host_initialized);
}

void buffer_base::unregister_with_runtime(buffer_id id) const {
	runtime::get_instance().unregister_buffer(id);
}

} // namespace celerity
