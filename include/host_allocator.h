#pragma once

#include <foonathan/memory/fallback_allocator.hpp>
#include <foonathan/memory/memory_pool.hpp>
#include <foonathan/memory/segregator.hpp>

#include <sycl/sycl.hpp>

#if defined(__HIPSYCL__) && defined(SYCL_EXT_HIPSYCL_BACKEND_CUDA)
#include <cuda_runtime.h>
#endif

#include "log.h"

namespace celerity::detail {

// FIXME: Provide only for CUDA backends
class cuda_pinned_memory_allocator {
  public:
	using is_stateful = std::integral_constant<bool, true>;

	cuda_pinned_memory_allocator(const size_t max_allocation) : m_max_allocation(max_allocation) {}

	void* try_allocate_node(const std::size_t size, const std::size_t alignment) {
		if(!can_allocate(size)) {
			CELERITY_TRACE(
			    "try_allocate_node: Allocation of {} plus current allocation {} would exceed limit of {}", size, m_current_allocation, m_max_allocation);
			return nullptr;
		}
		CELERITY_TRACE("Attempting to allocate {} bytes of pinned memory", size);
		void* ptr;
		// FIXME: How do we ensure alignment is satisfied?
		// FIXME: We just assume that we're in a compatible CUDA context
		const auto ret = cudaHostAlloc(&ptr, size, cudaHostAllocDefault | cudaHostAllocPortable);
		if(ret != cudaSuccess) {
			if(ret != cudaErrorMemoryAllocation) {
				CELERITY_CRITICAL("cudaHostAlloc: {}", cudaGetErrorString(ret));
				abort();
			}
			return nullptr;
		}
		assert(reinterpret_cast<uintptr_t>(ptr) % alignment == 0);
		m_current_allocation += size;
		return ptr;
	}

	bool try_deallocate_node(void* node, const std::size_t size, const std::size_t alignment) {
		if(const auto ret = cudaFreeHost(node); ret != cudaSuccess) {
			fprintf(stderr, "cudaFreeHost: %s\n", cudaGetErrorString(ret));
			return false; // This might not have been a pinned allocation
		}
		m_current_allocation -= size;
		return true;
	}

	void* allocate_node(const std::size_t size, const std::size_t alignment) {
		if(!can_allocate(size)) {
			CELERITY_TRACE("allocate_note: Allocation of {} plus current allocation {} exceeds limit of {}", size, m_current_allocation, m_max_allocation);
			throw std::bad_alloc();
		}
		CELERITY_TRACE("Allocating {} bytes of pinned memory", size);
		void* ptr;
		// FIXME: How do we ensure alignment is satisfied?
		// FIXME: We just assume that we're in a compatible CUDA context
		const auto ret = cudaHostAlloc(&ptr, size, cudaHostAllocDefault | cudaHostAllocPortable);
		if(ret != cudaSuccess) {
			if(ret != cudaErrorMemoryAllocation) {
				CELERITY_CRITICAL("cudaHostAlloc: {}", cudaGetErrorString(ret));
				abort();
			}
			CELERITY_ERROR("cudaHostAlloc", cudaGetErrorString(ret));
			throw std::bad_alloc();
		}
		assert(reinterpret_cast<uintptr_t>(ptr) % alignment == 0);
		m_current_allocation += size;
		return ptr;
	}

	void deallocate_node(void* node, const std::size_t size, const std::size_t alignment) noexcept {
		// FIXME: Can't log here because of static destruction order (spdlog no longer exists). => Don't use singleton for this.
		// CELERITY_TRACE("Freeing {} bytes of pinned memory", size);
		if(const auto ret = cudaFreeHost(node); ret != cudaSuccess) {
			// CELERITY_ERROR("Call to cudaFreeHost failed");
			fprintf(stderr, "cudaFreeHost: %s\n", cudaGetErrorString(ret));
			abort();
		} else {
			m_current_allocation -= size;
		}
	}

	// If these are not implemented, the library automatically forwards to the *_node functions.
	// TODO: Does that mean they get called in a loop (bad?) or simply with a larger size (size * count)?
	// void* allocate_array(std::size_t count, std::size_t size, std::size_t alignment);
	// void deallocate_array(void* ptr, std::size_t count, std::size_t size, std::size_t alignment) noexcept;

	// std::size_t max_node_size() const;
	// std::size_t max_array_size() const;
	// std::size_t max_alignment() const;
  private:
	size_t m_max_allocation;
	size_t m_current_allocation = 0;

	bool can_allocate(const size_t size) const { return m_current_allocation + size <= m_max_allocation; }
};
static_assert(foonathan::memory::is_raw_allocator<cuda_pinned_memory_allocator>::value);
static_assert(foonathan::memory::is_composable_allocator<cuda_pinned_memory_allocator>::value);

// FIXME: Ideally this shouldn't be a singleton; should probably be somehow tied to host_queue and/or buffer_manager. Works for now.
class host_allocator {
  public:
	static host_allocator& get_instance() {
		if(instance == nullptr) { instance = new host_allocator(); }
		return *instance;
	}

	void* allocate(const size_t size) {
		CELERITY_TRACE("allocating {} bytes on pinned memory pool, capacity left {} bytes", size, m_allocator.capacity_left());
		// TODO: Make this templated to get alignof(T)?
		// const auto alignment = 1;
		const auto node_size = m_allocator.node_size();
		const auto count = (size + node_size - 1) / node_size;
		return m_allocator.allocate_array(count);
	}

	void free(void* ptr, const size_t size) {
		// const auto alignment = 1;
		const auto node_size = m_allocator.node_size();
		const auto count = (size + node_size - 1) / node_size;
		m_allocator.deallocate_array(ptr, count);
		CELERITY_TRACE("deallocated {} bytes from pinned memory pool, capacity left {} bytes", size, m_allocator.capacity_left());
	}

  private:
	using memory_pool_t = foonathan::memory::memory_pool<foonathan::memory::array_pool, cuda_pinned_memory_allocator>;
	// TODO fallback should use the cuda_pinned_memory_allocator directly for sizes > block_size, but we must not hide allocator bugs that way
	// using fallback_allocator_t = foonathan::memory::fallback_allocator<memory_pool_t, foonathan::memory::null_allocator>;

	inline static host_allocator* instance = nullptr; // singleton - leak on purpose to avoid static destruction order issues

	host_allocator() = default;

	static memory_pool_t construct_allocator() {
		using namespace foonathan::memory::literals;
		const auto pool_node_size = 1_MiB;      // pool returns memory in this granularity
		const auto block_size = 256_MiB;          // pool allocates backing memory in this granularity (will be the maximum allocation size!)
		const auto max_pinned_memory = 256_GiB; // FIXME: This should be a configurable percentage of total host memory
		return memory_pool_t(pool_node_size, block_size, max_pinned_memory);
	}

  private:
	memory_pool_t m_allocator = construct_allocator();
};

} // namespace celerity::detail
