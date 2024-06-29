#include "legacy_backend/cuda_backend.h"

#include <cuda_runtime.h>

#include "log.h"
#include "ranges.h"

#define CELERITY_STRINGIFY2(f) #f
#define CELERITY_STRINGIFY(f) CELERITY_STRINGIFY2(f)
#define CELERITY_CUDA_CHECK(f, ...)                                                                                                                            \
	if(const auto cuda_check_result = (f)(__VA_ARGS__); cuda_check_result != cudaSuccess) {                                                                    \
		CELERITY_CRITICAL(CELERITY_STRINGIFY(f) ": {}", cudaGetErrorString(cuda_check_result));                                                                \
		abort();                                                                                                                                               \
	}

namespace celerity::detail::legacy_backend_detail {

void memcpy_strided_device_cuda(sycl::queue& queue, const void* source_base_ptr, void* target_base_ptr, size_t elem_size, const range<0>& /* source_range */,
    const id<0>& /* source_offset */, const range<0>& /* target_range */, const id<0>& /* target_offset */, const range<0>& /* copy_range */) {
	(void)queue;
	const auto ret = cudaMemcpy(target_base_ptr, source_base_ptr, elem_size, cudaMemcpyDefault);
	if(ret != cudaSuccess) throw std::runtime_error("cudaMemcpy failed");
	// Classic CUDA footgun: Memcpy is not always synchronous (e.g. for D2D)
	cudaStreamSynchronize(0);
}

void memcpy_strided_device_cuda(sycl::queue& queue, const void* source_base_ptr, void* target_base_ptr, size_t elem_size, const range<1>& source_range,
    const id<1>& source_offset, const range<1>& target_range, const id<1>& target_offset, const range<1>& copy_range) {
	(void)queue;
	const size_t line_size = elem_size * copy_range[0];
	CELERITY_CUDA_CHECK(cudaMemcpy, static_cast<char*>(target_base_ptr) + elem_size * get_linear_index(target_range, target_offset),
	    static_cast<const char*>(source_base_ptr) + elem_size * get_linear_index(source_range, source_offset), line_size, cudaMemcpyDefault);
	// Classic CUDA footgun: Memcpy is not always synchronous (e.g. for D2D)
	CELERITY_CUDA_CHECK(cudaStreamSynchronize, 0);
}

void memcpy_strided_device_cuda(sycl::queue& queue, const void* source_base_ptr, void* target_base_ptr, size_t elem_size, const range<2>& source_range,
    const id<2>& source_offset, const range<2>& target_range, const id<2>& target_offset, const range<2>& copy_range) {
	(void)queue;
	const auto source_base_offset = get_linear_index(source_range, source_offset);
	const auto target_base_offset = get_linear_index(target_range, target_offset);
	CELERITY_CUDA_CHECK(cudaMemcpy2D, static_cast<char*>(target_base_ptr) + elem_size * target_base_offset, target_range[1] * elem_size,
	    static_cast<const char*>(source_base_ptr) + elem_size * source_base_offset, source_range[1] * elem_size, copy_range[1] * elem_size, copy_range[0],
	    cudaMemcpyDefault);
	// Classic CUDA footgun: Memcpy is not always synchronous (e.g. for D2D)
	CELERITY_CUDA_CHECK(cudaStreamSynchronize, 0);
}

void memcpy_strided_device_cuda(sycl::queue& queue, const void* source_base_ptr, void* target_base_ptr, size_t elem_size, const range<3>& source_range,
    const id<3>& source_offset, const range<3>& target_range, const id<3>& target_offset, const range<3>& copy_range) {
	cudaMemcpy3DParms parms = {};
	parms.srcPos = make_cudaPos(source_offset[2] * elem_size, source_offset[1], source_offset[0]);
	parms.srcPtr = make_cudaPitchedPtr(
	    const_cast<void*>(source_base_ptr), source_range[2] * elem_size, source_range[2], source_range[1]); // NOLINT cppcoreguidelines-pro-type-const-cast
	parms.dstPos = make_cudaPos(target_offset[2] * elem_size, target_offset[1], target_offset[0]);
	parms.dstPtr = make_cudaPitchedPtr(target_base_ptr, target_range[2] * elem_size, target_range[2], target_range[1]);
	parms.extent = {copy_range[2] * elem_size, copy_range[1], copy_range[0]};
	parms.kind = cudaMemcpyDefault;
	CELERITY_CUDA_CHECK(cudaMemcpy3D, &parms);
	// Classic CUDA footgun: Memcpy is not always synchronous (e.g. for D2D)
	CELERITY_CUDA_CHECK(cudaStreamSynchronize, 0);
}

} // namespace celerity::detail::legacy_backend_detail
