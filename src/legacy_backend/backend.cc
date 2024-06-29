#include "legacy_backend/backend.h"

namespace celerity::detail::legacy_backend {

type get_type(const sycl::device& device) {
#if defined(__HIPSYCL__) && defined(SYCL_EXT_HIPSYCL_BACKEND_CUDA)
	if(device.get_backend() == sycl::backend::cuda) { return type::cuda; }
#endif
#if defined(__SYCL_COMPILER_VERSION) // DPC++ (TODO: This may break when using OpenSYCL w/ DPC++ as compiler)
	if(device.get_backend() == sycl::backend::ext_oneapi_cuda) { return type::cuda; }
#endif
	return type::unknown;
}

type get_effective_type(const sycl::device& device) {
	[[maybe_unused]] const auto b = get_type(device);

#if defined(CELERITY_DETAIL_BACKEND_CUDA_ENABLED)
	if(b == type::cuda) return b;
#endif

	return type::generic;
}

} // namespace celerity::detail::legacy_backend
