#include "backend/cuda_backend.h"

#include <cuda_runtime.h>

#include "ranges.h"

namespace celerity::detail::backend_detail {

inline cudaEvent_t create_and_record_cuda_event(cudaStream_t stream = 0) {
	// TODO: Perf considerations - we should probably have an event pool
	cudaEvent_t result;
	cudaEventCreateWithFlags(&result, cudaEventDisableTiming);
	cudaEventRecord(result, stream);
	return result;
}

class cuda_event_wrapper final : public native_event_wrapper {
  public:
	cuda_event_wrapper(cudaEvent_t evt) : m_event(evt) {}
	~cuda_event_wrapper() override { cudaEventDestroy(m_event); }

	bool is_done() const override {
		const auto ret = cudaEventQuery(m_event);
		assert(ret == cudaSuccess || ret == cudaErrorNotReady);
		return ret == cudaSuccess;
	}

  private:
	cudaEvent_t m_event;
};

backend::async_event memcpy_strided_device_cuda(sycl::queue& queue, const void* source_base_ptr, void* target_base_ptr, size_t elem_size,
    const sycl::range<1>& source_range, const sycl::id<1>& source_offset, const sycl::range<1>& target_range, const sycl::id<1>& target_offset,
    const sycl::range<1>& copy_range) {
	(void)queue;
	const size_t line_size = elem_size * copy_range[0];
	const auto ret = cudaMemcpyAsync(static_cast<char*>(target_base_ptr) + elem_size * get_linear_index(target_range, target_offset),
	    static_cast<const char*>(source_base_ptr) + elem_size * get_linear_index(source_range, source_offset), line_size, cudaMemcpyDefault);
	if(ret != cudaSuccess) throw std::runtime_error("cudaMemcpyAsync failed");
	return backend::async_event{std::make_shared<cuda_event_wrapper>(create_and_record_cuda_event(0))};
}

backend::async_event memcpy_strided_device_cuda(sycl::queue& queue, const void* source_base_ptr, void* target_base_ptr, size_t elem_size,
    const sycl::range<2>& source_range, const sycl::id<2>& source_offset, const sycl::range<2>& target_range, const sycl::id<2>& target_offset,
    const sycl::range<2>& copy_range) {
	(void)queue;
	const auto source_base_offset = get_linear_index(source_range, source_offset);
	const auto target_base_offset = get_linear_index(target_range, target_offset);
	const auto ret = cudaMemcpy2DAsync(static_cast<char*>(target_base_ptr) + elem_size * target_base_offset, target_range[1] * elem_size,
	    static_cast<const char*>(source_base_ptr) + elem_size * source_base_offset, source_range[1] * elem_size, copy_range[1] * elem_size, copy_range[0],
	    cudaMemcpyDefault);
	if(ret != cudaSuccess) throw std::runtime_error("cudaMemcpy2DAsync failed");
	return backend::async_event{std::make_shared<cuda_event_wrapper>(create_and_record_cuda_event(0))};
}

backend::async_event memcpy_strided_device_cuda(sycl::queue& queue, const void* source_base_ptr, void* target_base_ptr, size_t elem_size,
    const sycl::range<3>& source_range, const sycl::id<3>& source_offset, const sycl::range<3>& target_range, const sycl::id<3>& target_offset,
    const sycl::range<3>& copy_range) {
	cudaMemcpy3DParms parms = {};
	parms.srcPos = make_cudaPos(source_offset[2] * elem_size, source_offset[1], source_offset[0]);
	parms.srcPtr = make_cudaPitchedPtr(
	    const_cast<void*>(source_base_ptr), source_range[2] * elem_size, source_range[2], source_range[1]); // NOLINT cppcoreguidelines-pro-type-const-cast
	parms.dstPos = make_cudaPos(target_offset[2] * elem_size, target_offset[1], target_offset[0]);
	parms.dstPtr = make_cudaPitchedPtr(target_base_ptr, target_range[2] * elem_size, target_range[2], target_range[1]);
	parms.extent = {copy_range[2] * elem_size, copy_range[1], copy_range[0]};
	parms.kind = cudaMemcpyDefault;
	const auto ret = cudaMemcpy3DAsync(&parms);
	if(ret != cudaSuccess) throw std::runtime_error("cudaMemcpy3DAsync failed");
	return backend::async_event{std::make_shared<cuda_event_wrapper>(create_and_record_cuda_event(0))};
}

} // namespace celerity::detail::backend_detail