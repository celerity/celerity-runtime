#include "backend/cuda_backend.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include "ranges.h"

#define STRINGIFY2(f) #f
#define STRINGIFY(f) STRINGIFY2(f)
#define CUDA_CHECK(f, ...)                                                                                                                                     \
	if(const auto cuda_check_result = (f)(__VA_ARGS__); cuda_check_result != cudaSuccess) {                                                                    \
		CELERITY_CRITICAL(STRINGIFY(f) ": {}", cudaGetErrorString(cuda_check_result));                                                                         \
		abort();                                                                                                                                               \
	}
#define CUDA_DRV_CHECK(f, ...)                                                                                                                                 \
	if(const auto check_result = (f)(__VA_ARGS__); check_result != CUDA_SUCCESS) {                                                                             \
		const char* err_str = NULL;                                                                                                                            \
		(void)cuGetErrorString(check_result, &err_str);                                                                                                        \
		CELERITY_CRITICAL(STRINGIFY(f) ": {}", err_str);                                                                                                       \
		abort();                                                                                                                                               \
	}

namespace {

inline cudaEvent_t create_and_record_cuda_event(cudaStream_t stream) {
	// TODO: Perf considerations - we should probably have an event pool
	cudaEvent_t result = 0;
	CUDA_CHECK(cudaEventCreateWithFlags, &result, cudaEventDisableTiming);
	CUDA_CHECK(cudaEventRecord, result, stream);
	return result;
}

class cuda_event_wrapper final : public celerity::detail::backend_detail::native_event_wrapper {
  public:
	cuda_event_wrapper(cudaEvent_t evt) : m_event(evt) {}
	~cuda_event_wrapper() override { CUDA_CHECK(cudaEventDestroy, m_event); }

	bool is_done() const override {
		const auto ret = cudaEventQuery(m_event);
		if(ret != cudaSuccess && ret != cudaErrorNotReady) {
			CELERITY_CRITICAL("cudaEventQuery: {}", cudaGetErrorString(ret));
			abort();
		}
		return ret == cudaSuccess;
	}

  private:
	cudaEvent_t m_event;
};

// When creating an event the correct device needs to be set
// TODO: It might be sufficient if we just set the context - investigate
class set_cuda_device_scoped {
  public:
	set_cuda_device_scoped(cudaStream_t stream) {
		CUDA_CHECK(cudaGetDevice, &m_previous_device);
		CUcontext ctx;
		CUDA_DRV_CHECK(cuStreamGetCtx, stream, &ctx);
		CUDA_DRV_CHECK(cuCtxPushCurrent, ctx);
		CUdevice dev;
		CUDA_DRV_CHECK(cuCtxGetDevice, &dev);
		CUDA_DRV_CHECK(cuCtxPopCurrent, nullptr);
		CUDA_CHECK(cudaSetDevice, dev);
	}

	~set_cuda_device_scoped() { CUDA_CHECK(cudaSetDevice, m_previous_device); }

  private:
	int m_previous_device = -1;
};

} // namespace

namespace celerity::detail::backend_detail {

backend::async_event memcpy_strided_device_cuda(sycl::queue& queue, const void* source_base_ptr, void* target_base_ptr, size_t elem_size,
    const sycl::range<1>& source_range, const sycl::id<1>& source_offset, const sycl::range<1>& target_range, const sycl::id<1>& target_offset,
    const sycl::range<1>& copy_range, void* HACK_backend_context) {
	(void)queue;
	cudaStream_t stream = static_cast<cudaStream_t>(HACK_backend_context);
	set_cuda_device_scoped scds{stream};
	const size_t line_size = elem_size * copy_range[0];
	const auto ret = cudaMemcpyAsync(static_cast<char*>(target_base_ptr) + elem_size * get_linear_index(target_range, target_offset),
	    static_cast<const char*>(source_base_ptr) + elem_size * get_linear_index(source_range, source_offset), line_size, cudaMemcpyDefault, stream);
	if(ret != cudaSuccess) throw std::runtime_error("cudaMemcpyAsync failed");
	return backend::async_event{std::make_shared<cuda_event_wrapper>(create_and_record_cuda_event(stream))};
}

backend::async_event memcpy_strided_device_cuda(sycl::queue& queue, const void* source_base_ptr, void* target_base_ptr, size_t elem_size,
    const sycl::range<2>& source_range, const sycl::id<2>& source_offset, const sycl::range<2>& target_range, const sycl::id<2>& target_offset,
    const sycl::range<2>& copy_range, void* HACK_backend_context) {
	(void)queue;
	cudaStream_t stream = static_cast<cudaStream_t>(HACK_backend_context);
	set_cuda_device_scoped scds{stream};
	const auto source_base_offset = get_linear_index(source_range, source_offset);
	const auto target_base_offset = get_linear_index(target_range, target_offset);
	const auto ret = cudaMemcpy2DAsync(static_cast<char*>(target_base_ptr) + elem_size * target_base_offset, target_range[1] * elem_size,
	    static_cast<const char*>(source_base_ptr) + elem_size * source_base_offset, source_range[1] * elem_size, copy_range[1] * elem_size, copy_range[0],
	    cudaMemcpyDefault, stream);
	if(ret != cudaSuccess) throw std::runtime_error("cudaMemcpy2DAsync failed");
	return backend::async_event{std::make_shared<cuda_event_wrapper>(create_and_record_cuda_event(stream))};
}

backend::async_event memcpy_strided_device_cuda(sycl::queue& queue, const void* source_base_ptr, void* target_base_ptr, size_t elem_size,
    const sycl::range<3>& source_range, const sycl::id<3>& source_offset, const sycl::range<3>& target_range, const sycl::id<3>& target_offset,
    const sycl::range<3>& copy_range, void* HACK_backend_context) {
	(void)queue;
	cudaStream_t stream = static_cast<cudaStream_t>(HACK_backend_context);
	set_cuda_device_scoped scds{stream};
	cudaMemcpy3DParms parms = {};
	parms.srcPos = make_cudaPos(source_offset[2] * elem_size, source_offset[1], source_offset[0]);
	parms.srcPtr = make_cudaPitchedPtr(
	    const_cast<void*>(source_base_ptr), source_range[2] * elem_size, source_range[2], source_range[1]); // NOLINT cppcoreguidelines-pro-type-const-cast
	parms.dstPos = make_cudaPos(target_offset[2] * elem_size, target_offset[1], target_offset[0]);
	parms.dstPtr = make_cudaPitchedPtr(target_base_ptr, target_range[2] * elem_size, target_range[2], target_range[1]);
	parms.extent = {copy_range[2] * elem_size, copy_range[1], copy_range[0]};
	parms.kind = cudaMemcpyDefault;
	const auto ret = cudaMemcpy3DAsync(&parms, stream);
	if(ret != cudaSuccess) throw std::runtime_error("cudaMemcpy3DAsync failed");
	return backend::async_event{std::make_shared<cuda_event_wrapper>(create_and_record_cuda_event(stream))};
}

} // namespace celerity::detail::backend_detail