#pragma once

#include <foonathan/memory/fallback_allocator.hpp>
#include <foonathan/memory/memory_pool.hpp>

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
			if(ret != cudaErrorMemoryAllocation) { CELERITY_ERROR("Call to cudaAllocHost failed due to unknown error"); }
			return nullptr;
		}
		assert(reinterpret_cast<uintptr_t>(ptr) % alignment == 0);
		m_current_allocation += size;
		return ptr;
	}

	bool try_deallocate_node(void* node, const std::size_t size, const std::size_t alignment) {
		const auto ret = cudaFreeHost(node);
		if(ret != cudaSuccess) return false; // This might not have been a pinned allocation
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
			CELERITY_ERROR("Call to cudaAllocHost failed ({})", ret == cudaErrorMemoryAllocation ? "out of memory" : "unknown error");
			throw std::bad_alloc();
		}
		assert(reinterpret_cast<uintptr_t>(ptr) % alignment == 0);
		m_current_allocation += size;
		return ptr;
	}

	void deallocate_node(void* node, const std::size_t size, const std::size_t alignment) noexcept {
		// FIXME: Can't log here because of static destruction order (spdlog no longer exists). => Don't use singleton for this.
		// CELERITY_TRACE("Freeing {} bytes of pinned memory", size);
		const auto ret = cudaFreeHost(node);
		if(ret != cudaSuccess) {
			// CELERITY_ERROR("Call to cudaFreeHost failed");
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
		if(instance == nullptr) { instance = std::make_unique<host_allocator>(); }
		return *instance;
	}

	void* allocate(const size_t size) {
		// TODO: Make this templated to get alignof(T)?
		const auto alignment = 1;
		const auto node_size = get_pool_node_size();
		const auto count = (size + node_size - 1) / node_size;
		return m_allocator.allocate_array(count, node_size, alignment);
	}

	void free(void* ptr, const size_t size) {
		const auto alignment = 1;
		const auto node_size = get_pool_node_size();
		const auto count = (size + node_size - 1) / node_size;
		m_allocator.deallocate_array(ptr, count, node_size, alignment);
	}

  private:
	using memory_pool_t = foonathan::memory::memory_pool<foonathan::memory::node_pool, cuda_pinned_memory_allocator>;
	using fallback_allocator_t = foonathan::memory::fallback_allocator<memory_pool_t, foonathan::memory::default_allocator>;

	inline static std::unique_ptr<host_allocator> instance;

	static size_t get_pool_node_size() {
		using namespace foonathan::memory::literals;
		return 10_MiB; // TODO: Figure out a good tradeoff here
	}

	static fallback_allocator_t construct_allocator() {
		using namespace foonathan::memory::literals;
		const auto max_pinned_memory = 8_GiB; // FIXME: This should be a configurable percentage of total host memory
		auto pool = memory_pool_t(get_pool_node_size(), 1_GiB, max_pinned_memory);
		return fallback_allocator_t{std::move(pool), foonathan::memory::default_allocator{}};
	}

  private:
	fallback_allocator_t m_allocator = construct_allocator();
};

} // namespace celerity::detail