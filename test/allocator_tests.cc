#include "test_utils.h"

#include <foonathan/memory/fallback_allocator.hpp>
#include <foonathan/memory/memory_pool.hpp>

#include "frame.h"

using namespace celerity;
using namespace celerity::detail;

TEST_CASE("various (TODO: cleanup, split)") {
	spdlog::set_level(spdlog::level::trace); // we don't load config so need to set manually (FIXME just for debugging)

	namespace memory = foonathan::memory;
	using namespace memory::literals;

	// Current thinking:
	// Provide a frontend that internally uses allocate_array to allocate arbitrary sizes.
	// Use a fallback_allocator with the pinned memory pool and a default allocator.
	// The pinned memory pool stops working once a certain limit of host memory is reached.

	// Basic
	{
		memory::memory_pool<memory::node_pool, cuda_pinned_memory_allocator> my_pool(10_MiB, 1_GiB, 2_GiB);
		auto ptr = my_pool.allocate_array(10);
		CHECK(ptr != nullptr);
		cudaPointerAttributes attrs;
		cudaPointerGetAttributes(&attrs, ptr);
		CHECK(attrs.type == cudaMemoryTypeHost);
		my_pool.deallocate_node(ptr);
	}

	// Cannot exceed pinned memory limit
	{
		memory::memory_pool<memory::node_pool, cuda_pinned_memory_allocator> my_pool(10_MiB, 100_MiB, 200_MiB);
		auto ptr = my_pool.allocate_array(5);
		CHECK_THROWS(my_pool.allocate_array(10));
		my_pool.deallocate_array(ptr, 5);
	}

	// Fall back to default allocator if pinned memory limit is exceeded
	{
		memory::memory_pool<memory::node_pool, cuda_pinned_memory_allocator> my_pool(10_MiB, 100_MiB, 200_MiB);
		memory::fallback_allocator<decltype(my_pool), memory::default_allocator> my_allocator{std::move(my_pool), memory::default_allocator{}};

		cudaPointerAttributes attrs;
		const auto pinned_ptr = my_allocator.allocate_array(5, 10_MiB, 1);
		cudaPointerGetAttributes(&attrs, pinned_ptr);
		CHECK(attrs.type == cudaMemoryTypeHost);
		const auto paged_ptr = my_allocator.allocate_array(10, 10_MiB, 1);
		cudaPointerGetAttributes(&attrs, paged_ptr);
		CHECK(attrs.type == cudaMemoryTypeUnregistered);
		my_allocator.deallocate_array(pinned_ptr, 5, 10_MiB, 1);
		my_allocator.deallocate_array(paged_ptr, 10, 10_MiB, 1);
	}

	// TODO: What might also be interesting is to use a memory_pool_collection.
	// Unfortunately there is a bug that prevents us from allocating large nodes.
	// See https://github.com/foonathan/memory/issues/148.
#if 0
	const size_t max_node_size = 512 * 1024 * 1024;
	const size_t block_size = 1024 * 1024 * 1024;
	memory::memory_pool_collection<memory::node_pool, memory::log2_buckets> my_pool(max_node_size, block_size);
	for(size_t alloc_size = 1; alloc_size <= max_node_size; alloc_size <<= 1) {
		fmt::print(stderr, "ALLOCATING {} bytes\n", alloc_size);
		auto ptr = my_pool.allocate_node(alloc_size);
		REQUIRE_LOOP(ptr != nullptr);
		my_pool.deallocate_node(ptr, alloc_size);
	}
#endif
}