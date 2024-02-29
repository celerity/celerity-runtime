#include "backend/generic_backend.h"

#include "nd_memory.h"
#include "ranges.h"
#include "types.h"


namespace celerity::detail::backend {

std::vector<sycl::device> get_device_vector(const std::vector<std::pair<device_id, sycl::device>>& devices) {
	std::vector<sycl::device> vector;
	vector.reserve(devices.size());
	for(auto& [did, dev] : devices) {
		vector.push_back(dev);
	}
	return vector;
}

generic_queue::generic_queue(const std::vector<device_config>& devices) {
	m_memory_queues.emplace(host_memory_id, sycl::queue());

	for(const auto& config : devices) {
		assert(m_device_queues.count(config.device_id) == 0);
		assert(m_memory_queues.count(config.native_memory) == 0); // TODO handle devices that share memory

		sycl::queue queue(config.sycl_device, backend::handle_sycl_errors);
		m_device_queues.emplace(config.device_id, queue);
		m_memory_queues.emplace(config.native_memory, queue);
	}
}

void* generic_queue::alloc(const memory_id where, const size_t size, [[maybe_unused]] const size_t alignment) {
	assert(where != user_memory_id);
	auto& queue = m_memory_queues.at(where);
	void* ptr;
	if(where == host_memory_id) {
		ptr = sycl::aligned_alloc_host(alignment, size, queue);
#if CELERITY_DETAIL_ENABLE_DEBUG
		memset(ptr, static_cast<int>(uninitialized_memory_pattern), size);
#endif
	} else {
		ptr = sycl::aligned_alloc_device(alignment, size, queue);
#if CELERITY_DETAIL_ENABLE_DEBUG
		queue.memset(ptr, static_cast<int>(uninitialized_memory_pattern), size).wait();
#endif
	}
	return ptr;
}

void generic_queue::free(const memory_id where, void* const allocation) {
	assert(where != user_memory_id);
	sycl::free(allocation, m_memory_queues.at(where));
}

async_event generic_queue::copy_region(const memory_id source_mid, memory_id dest_mid, const void* const source_base, void* const dest_base,
    const box<3>& source_box, const box<3>& dest_box, const region<3>& copy_region, const size_t elem_size) //
{
	assert(source_mid != user_memory_id);
	assert(dest_mid != user_memory_id);
	assert(source_mid != host_memory_id || dest_mid != host_memory_id);

	auto& queue = m_memory_queues.at(source_mid == host_memory_id ? dest_mid : source_mid);

	std::vector<sycl::event> wait_list;
	for(const auto& copy_box : copy_region.get_boxes()) {
		assert(source_box.covers(copy_box));
		assert(dest_box.covers(copy_box));
		for_each_linear_slice_in_nd_copy(source_box.get_range(), dest_box.get_range(), copy_box.get_offset() - source_box.get_offset(),
		    copy_box.get_offset() - dest_box.get_offset(), copy_box.get_range(),
		    [&](const size_t linear_offset_in_source, const size_t linear_offset_in_dest, const size_t linear_size) {
			    wait_list.push_back(queue.memcpy(static_cast<std::byte*>(dest_base) + linear_offset_in_dest * elem_size,
			        static_cast<const std::byte*>(source_base) + linear_offset_in_source * elem_size, linear_size * elem_size));
		    });
	}
	flush_sycl_queue(queue);
	return make_async_event<sycl_event>(std::move(wait_list));
}

async_event generic_queue::launch_kernel(
    device_id did, const device_kernel_launcher& launcher, const box<3>& execution_range, const std::vector<void*>& reduction_ptrs) //
{
	return launch_sycl_kernel(m_device_queues.at(did), launcher, execution_range, reduction_ptrs);
}

} // namespace celerity::detail::backend
