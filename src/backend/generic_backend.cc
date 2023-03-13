#include "backend/generic_backend.h"

#include "ranges.h"

namespace celerity::detail::backend_detail {

class sycl_event_wrapper final : public native_event_wrapper {
  public:
	sycl_event_wrapper(sycl::event evt) : m_event(std::move(evt)) {}

	bool is_done() const override { return m_event.get_info<sycl::info::event::command_execution_status>() == sycl::info::event_command_status::complete; }
	// void wait() override { m_event.wait(); }

  private:
	sycl::event m_event;
};

backend::async_event memcpy_strided_device_generic(sycl::queue& queue, const void* source_base_ptr, void* target_base_ptr, size_t elem_size,
    const sycl::range<1>& source_range, const sycl::id<1>& source_offset, const sycl::range<1>& target_range, const sycl::id<1>& target_offset,
    const sycl::range<1>& copy_range, void* HACK_backend_context) {
	const size_t line_size = elem_size * copy_range[0];
	auto evt = queue.memcpy(static_cast<char*>(target_base_ptr) + elem_size * get_linear_index(target_range, target_offset),
	    static_cast<const char*>(source_base_ptr) + elem_size * get_linear_index(source_range, source_offset), line_size);
	return backend::async_event{std::make_shared<sycl_event_wrapper>(evt)};
}

// TODO Optimize for contiguous copies?
backend::async_event memcpy_strided_device_generic(sycl::queue& queue, const void* source_base_ptr, void* target_base_ptr, size_t elem_size,
    const sycl::range<2>& source_range, const sycl::id<2>& source_offset, const sycl::range<2>& target_range, const sycl::id<2>& target_offset,
    const sycl::range<2>& copy_range, void* HACK_backend_context) {
	const auto source_base_offset = get_linear_index(source_range, source_offset);
	const auto target_base_offset = get_linear_index(target_range, target_offset);
	const size_t line_size = elem_size * copy_range[1];
	std::vector<sycl::event> wait_list;
	wait_list.reserve(copy_range[0]);
	for(size_t i = 0; i < copy_range[0]; ++i) {
		auto e = queue.memcpy(static_cast<char*>(target_base_ptr) + elem_size * (target_base_offset + i * target_range[1]),
		    static_cast<const char*>(source_base_ptr) + elem_size * (source_base_offset + i * source_range[1]), line_size);
		wait_list.push_back(e);
	}
	sycl::event::wait(wait_list);
	// NOCOMMIT FIXME: Make above async
	return backend::async_event{};
}

// TODO Optimize for contiguous copies?
backend::async_event memcpy_strided_device_generic(sycl::queue& queue, const void* source_base_ptr, void* target_base_ptr, size_t elem_size,
    const sycl::range<3>& source_range, const sycl::id<3>& source_offset, const sycl::range<3>& target_range, const sycl::id<3>& target_offset,
    const sycl::range<3>& copy_range, void* HACK_backend_context) {
	// We simply decompose this into a bunch of 2D copies. Subtract offset on the copy plane, as it will be added again during the 2D copy.
	const auto source_base_offset =
	    get_linear_index(source_range, source_offset) - get_linear_index({source_range[1], source_range[2]}, {source_offset[1], source_offset[2]});
	const auto target_base_offset =
	    get_linear_index(target_range, target_offset) - get_linear_index({target_range[1], target_range[2]}, {target_offset[1], target_offset[2]});

	for(size_t i = 0; i < copy_range[0]; ++i) {
		const auto* const source_ptr = static_cast<const char*>(source_base_ptr) + elem_size * (source_base_offset + i * (source_range[1] * source_range[2]));
		auto* const target_ptr = static_cast<char*>(target_base_ptr) + elem_size * (target_base_offset + i * (target_range[1] * target_range[2]));
		auto e =
		    memcpy_strided_device_generic(queue, source_ptr, target_ptr, elem_size, {source_range[1], source_range[2]}, {source_offset[1], source_offset[2]},
		        {target_range[1], target_range[2]}, {target_offset[1], target_offset[2]}, {copy_range[1], copy_range[2]}, HACK_backend_context);
		e.wait();
	}
	// NOCOMMIT FIXME: Return aggregate event
	return backend::async_event{};
}

} // namespace celerity::detail::backend_detail