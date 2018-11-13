#include "buffer.h"

#include "runtime.h"

namespace celerity {
namespace detail {

	buffer_type buffer_base::get_type() const {
		// By making the master node always use HOST_BUFFERs we currently prevent it from running as a single node
		// TODO: Look into running on a single node (for development / debugging)
		return runtime::get_instance().is_master_node() ? buffer_type::HOST_BUFFER : buffer_type::DEVICE_BUFFER;
	}

	buffer_id buffer_base::register_with_runtime(
	    cl::sycl::range<3> range, std::shared_ptr<detail::buffer_storage_base> buffer_storage, bool host_initialized) const {
		return runtime::get_instance().register_buffer(range, buffer_storage, host_initialized);
	}

	void buffer_base::unregister_with_runtime(buffer_id id) const { runtime::get_instance().unregister_buffer(id); }

} // namespace detail
} // namespace celerity
