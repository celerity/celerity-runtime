#include "backend/cuda_backend.h"

#include <cuda_runtime.h>

#include "../tracy.h"
#include "ranges.h"
#include "utils.h"
#include "workaround.h"


#define CELERITY_STRINGIFY2(f) #f
#define CELERITY_STRINGIFY(f) CELERITY_STRINGIFY2(f)
#define CELERITY_CUDA_CHECK(f, ...)                                                                                                                            \
	if(const auto cuda_check_result = (f)(__VA_ARGS__); cuda_check_result != cudaSuccess) {                                                                    \
		utils::panic(CELERITY_STRINGIFY(f) ": {}", cudaGetErrorString(cuda_check_result));                                                                     \
	}

namespace celerity::detail::backend_detail {

void nd_copy_cuda(const void* const source_base, void* const dest_base, const range<3>& source_range, const range<3>& dest_range, const id<3>& offset_in_source,
    const id<3>& offset_in_dest, const range<3>& copy_range, const size_t elem_size, const cudaStream_t stream) {
	assert(all_true(offset_in_source + copy_range <= source_range));
	assert(all_true(offset_in_dest + copy_range <= dest_range));

	if(copy_range.size() == 0) return;

	// TODO copied from nd_copy_host - this works but is not optimal, it will still do a 2D copy on a [1,1,1] range if there is a dim1 offset
	int linear_dim = 0;
	for(int d = 1; d < 3; ++d) {
		if(source_range[d] != copy_range[d] || dest_range[d] != copy_range[d]) { linear_dim = d; }
	}

	const auto first_source_elem = static_cast<const std::byte*>(source_base) + get_linear_index(source_range, offset_in_source) * elem_size;
	const auto first_dest_elem = static_cast<std::byte*>(dest_base) + get_linear_index(dest_range, offset_in_dest) * elem_size;

	switch(linear_dim) {
	case 0: {
		const auto copy_bytes = (copy_range[0] * copy_range[1] * copy_range[2]) * elem_size;
		CELERITY_DETAIL_TRACY_SCOPED_ZONE("cuda::memcpy_1d", ForestGreen, "cudaMemcpyAsync")
		CELERITY_CUDA_CHECK(cudaMemcpyAsync, first_dest_elem, first_source_elem, copy_bytes, cudaMemcpyDefault, stream);
		break;
	}
	case 1: {
		CELERITY_DETAIL_TRACY_SCOPED_ZONE("cuda::memcpy_2d", ForestGreen, "cudaMemcpy2DAsync")
		CELERITY_CUDA_CHECK(cudaMemcpy2DAsync, first_dest_elem, dest_range[1] * dest_range[2] * elem_size, first_source_elem,
		    source_range[1] * source_range[2] * elem_size, copy_range[1] * copy_range[2] * elem_size, copy_range[0], cudaMemcpyDefault, stream);
		break;
	}
	case 2: {
		cudaMemcpy3DParms parms = {};
		parms.srcPos = make_cudaPos(offset_in_source[2] * elem_size, offset_in_source[1], offset_in_source[0]);
		parms.srcPtr = make_cudaPitchedPtr(
		    const_cast<void*>(source_base), source_range[2] * elem_size, source_range[2], source_range[1]); // NOLINT cppcoreguidelines-pro-type-const-cast
		parms.dstPos = make_cudaPos(offset_in_dest[2] * elem_size, offset_in_dest[1], offset_in_dest[0]);
		parms.dstPtr = make_cudaPitchedPtr(dest_base, dest_range[2] * elem_size, dest_range[2], dest_range[1]);
		parms.extent = {copy_range[2] * elem_size, copy_range[1], copy_range[0]};
		parms.kind = cudaMemcpyDefault;
		CELERITY_DETAIL_TRACY_SCOPED_ZONE("cuda::memcpy_3d", ForestGreen, "cudaMemcpy3DAsync")
		CELERITY_CUDA_CHECK(cudaMemcpy3DAsync, &parms, stream);
		break;
	}
	default: assert(!"unreachable");
	}
}

using cuda_device_id = celerity::detail::backend::cuda_queue::cuda_device_id;

struct cuda_set_device_guard {
	explicit cuda_set_device_guard(cuda_device_id cudid) {
		CELERITY_CUDA_CHECK(cudaGetDevice, &cudid_before);
		CELERITY_CUDA_CHECK(cudaSetDevice, cudid);
	}
	cuda_set_device_guard(const cuda_set_device_guard&) = delete;
	cuda_set_device_guard(cuda_set_device_guard&&) = delete;
	cuda_set_device_guard& operator=(const cuda_set_device_guard&) = delete;
	cuda_set_device_guard& operator=(cuda_set_device_guard&&) = delete;
	~cuda_set_device_guard() { CELERITY_CUDA_CHECK(cudaSetDevice, cudid_before); }

	cuda_device_id cudid_before = -1;
};

struct cuda_stream_deleter {
	void operator()(const cudaStream_t stream) const { CELERITY_CUDA_CHECK(cudaStreamDestroy, stream); }
};

using unique_cuda_stream = std::unique_ptr<std::remove_pointer_t<cudaStream_t>, cuda_stream_deleter>;

unique_cuda_stream make_cuda_stream(const cuda_device_id id) {
	cudaStream_t stream;
	CELERITY_CUDA_CHECK(cudaStreamCreateWithFlags, &stream, cudaStreamNonBlocking);
	return unique_cuda_stream(stream);
}

struct cuda_event_deleter {
	void operator()(const cudaEvent_t evt) const { CELERITY_CUDA_CHECK(cudaEventDestroy, evt); }
};

using unique_cuda_event = std::unique_ptr<std::remove_pointer_t<cudaEvent_t>, cuda_event_deleter>;

unique_cuda_event make_cuda_event() {
	cudaEvent_t event;
	CELERITY_CUDA_CHECK(cudaEventCreateWithFlags, &event, cudaEventDisableTiming);
	return backend_detail::unique_cuda_event(event);
}

} // namespace celerity::detail::backend_detail

namespace celerity::detail::backend {

class cuda_event final : public async_event_base {
  public:
	cuda_event(backend_detail::unique_cuda_event evt) : m_evt(std::move(evt)) {}

	static async_event record(const cudaStream_t stream) {
		auto event = backend_detail::make_cuda_event();
		CELERITY_CUDA_CHECK(cudaEventRecord, event.get(), stream);
		return make_async_event<cuda_event>(std::move(event));
	}

	bool is_complete() const override {
		switch(const auto result = cudaEventQuery(m_evt.get())) {
		case cudaSuccess: return true;
		case cudaErrorNotReady: return false;
		default: utils::panic("cudaEventQuery: {}", cudaGetErrorString(result));
		}
	}

  private:
	backend_detail::unique_cuda_event m_evt;
};

// TODO dispatch "host" operations to thread queue and replace this type's implementation with a std::future
class cuda_host_event final : public async_event_base {
  public:
	bool is_complete() const override { return true; }
};

cuda_queue::cuda_device_id get_cuda_device_id(const sycl::device& device) {
#if CELERITY_WORKAROUND(HIPSYCL)
	return sycl::get_native<sycl::backend::cuda>(device);
#else
	return sycl::get_native<sycl::backend::ext_oneapi_cuda>(device);
#endif
}

struct cuda_queue::impl {
	struct device {
		cuda_device_id cuda_id;
		sycl::queue sycl_queue;
	};
	struct memory {
		cuda_device_id cuda_id;
		backend_detail::unique_cuda_stream copy_from_host_stream;
		backend_detail::unique_cuda_stream copy_to_host_stream;
		std::unordered_map<memory_id, backend_detail::unique_cuda_stream> copy_from_peer_stream;
	};

	std::unordered_map<device_id, device> devices;
	std::unordered_map<memory_id, memory> memories;
};

cuda_queue::cuda_queue(const std::vector<device_config>& devices) : m_impl(std::make_unique<impl>()) {
	for(const auto& config : devices) {
		assert(m_impl->devices.count(config.device_id) == 0);
		assert(m_impl->memories.count(config.native_memory) == 0); // TODO handle devices that share memory

		const cuda_device_id cuda_id = get_cuda_device_id(config.sycl_device);
		backend_detail::cuda_set_device_guard set_device(cuda_id);

		impl::device dev{cuda_id, sycl::queue(config.sycl_device, backend::handle_sycl_errors)};
		m_impl->devices.emplace(config.device_id, std::move(dev));

		impl::memory mem;
		mem.cuda_id = cuda_id;
		mem.copy_from_host_stream = backend_detail::make_cuda_stream(cuda_id);
		mem.copy_to_host_stream = backend_detail::make_cuda_stream(cuda_id);
		for(const auto& other_config : devices) {
			// device can be its own "peer" - buffer resizes need to copy within the device's memory
			mem.copy_from_peer_stream.emplace(other_config.native_memory, backend_detail::make_cuda_stream(cuda_id));
		}
		m_impl->memories.emplace(config.native_memory, std::move(mem));
	}
}

cuda_queue::~cuda_queue() = default;

void cuda_queue::init() {
	CELERITY_DETAIL_TRACY_SCOPED_ZONE("cuda::init", ForestGreen, "cudaInit")
	for(const auto& [_, dev] : m_impl->devices) {
		backend_detail::cuda_set_device_guard set_device(dev.cuda_id);
		CELERITY_CUDA_CHECK(cudaFree, 0);
	}
}

void* cuda_queue::alloc(const memory_id where, const size_t size, [[maybe_unused]] const size_t alignment) {
	assert(where != user_memory_id);
	void* ptr;
	if(where == host_memory_id) {
		{
			CELERITY_DETAIL_TRACY_SCOPED_ZONE("cuda::malloc_host", ForestGreen, "cudaMallocHost")
			CELERITY_CUDA_CHECK(cudaMallocHost, &ptr, size, cudaHostAllocDefault);
		}
#if CELERITY_DETAIL_ENABLE_DEBUG
		memset(ptr, static_cast<int>(uninitialized_memory_pattern), size);
#endif
	} else {
		const auto& mem = m_impl->memories.at(where);
		backend_detail::cuda_set_device_guard set_device(mem.cuda_id);
		// We _want_ to use cudaMallocAsync / cudaMallocFromPoolAsync for asynchronicity and stream ordering here, but according to
		// https://developer.nvidia.com/blog/using-cuda-stream-ordered-memory-allocator-part-2 memory allocated through that API cannot be used with GPUDirect
		// RDMA (although NVIDIA plans to support this at an unspecified time in the future).
		// When we eventually switch to cudaMallocAsync, remember to call cudaMemPoolSetAccess to allow d2d copies (see the same article).
		{
			CELERITY_DETAIL_TRACY_SCOPED_ZONE("cuda::malloc_device", ForestGreen, "cudaMalloc")
			CELERITY_CUDA_CHECK(cudaMalloc, &ptr, size);
		}
#if CELERITY_DETAIL_ENABLE_DEBUG
		CELERITY_CUDA_CHECK(cudaMemset, ptr, static_cast<int>(uninitialized_memory_pattern), size);
		CELERITY_CUDA_CHECK(cudaDeviceSynchronize);
#endif
	}

	assert(reinterpret_cast<uintptr_t>(ptr) % alignment == 0);
	return ptr;
}

void cuda_queue::free(const memory_id where, void* const allocation) {
	assert(where != user_memory_id);
	if(where == host_memory_id) {
		CELERITY_DETAIL_TRACY_SCOPED_ZONE("cuda::free_host", ForestGreen, "cudaFreeHost");
		CELERITY_CUDA_CHECK(cudaFreeHost, allocation);
	} else {
		const auto& mem = m_impl->memories.at(where);
		backend_detail::cuda_set_device_guard set_device(mem.cuda_id);
		CELERITY_DETAIL_TRACY_SCOPED_ZONE("cuda::free_device", ForestGreen, "cudaFree");
		CELERITY_CUDA_CHECK(cudaFree, allocation);
	}
}

async_event cuda_queue::copy_region(const memory_id source_mid, memory_id dest_mid, const void* const source_base, void* const dest_base,
    const box<3>& source_box, const box<3>& dest_box, const region<3>& copy_region, const size_t elem_size) //
{
	assert(source_mid != user_memory_id);
	assert(dest_mid != user_memory_id);

	const impl::memory* memory = nullptr;
	cudaStream_t stream = nullptr;
	if(source_mid == host_memory_id) {
		assert(dest_mid != host_memory_id);
		memory = &m_impl->memories.at(dest_mid);
		stream = memory->copy_from_host_stream.get();
	} else if(dest_mid == host_memory_id) {
		assert(source_mid != host_memory_id);
		memory = &m_impl->memories.at(source_mid);
		stream = memory->copy_from_host_stream.get();
	} else {
		memory = &m_impl->memories.at(dest_mid);
		stream = memory->copy_from_peer_stream.at(source_mid).get();
	}

	backend_detail::cuda_set_device_guard set_device(memory->cuda_id);
	for(const auto& copy_box : copy_region.get_boxes()) {
		assert(source_box.covers(copy_box));
		assert(dest_box.covers(copy_box));
		backend_detail::nd_copy_cuda(source_base, dest_base, source_box.get_range(), dest_box.get_range(), copy_box.get_offset() - source_box.get_offset(),
		    copy_box.get_offset() - dest_box.get_offset(), copy_box.get_range(), elem_size, stream);
	}
	return cuda_event::record(stream);
}

async_event cuda_queue::launch_kernel(
    device_id did, const device_kernel_launcher& launcher, const box<3>& execution_range, const std::vector<void*>& reduction_ptrs) //
{
	// TODO, with special hipSYCL patch: Post the kernel to the same stream as its last incomplete dependency to avoid round-trip latency
	return launch_sycl_kernel(m_impl->devices.at(did).sycl_queue, launcher, execution_range, reduction_ptrs);
}

} // namespace celerity::detail::backend
