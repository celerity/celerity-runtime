#pragma once

#include "compression.h"
#include "compression_impl.h"

#include <cstdint>

namespace celerity::detail {

struct compression_testspy {
	template <typename ChunkDataState, typename WorkgroupCompressionState>
	static int32_t get_raw_status_and_count(
	    const global_compression_state_tracker<ChunkDataState, WorkgroupCompressionState>& tracker, const celerity::id<1>& linearized_idx) {
		const auto allocation_offset = tracker.m_chunk_data_state_counter_global_acc.get_allocation_offset()[0];
		return tracker.m_chunk_data_state_counter_global_acc[{allocation_offset, linearized_idx[0]}];
	}

	template <typename Tracker>
	static int32_t extract_status(const Tracker& tracker, int32_t combined_status) {
		return tracker.extract_status(combined_status);
	}

	template <typename Tracker>
	static int32_t extract_count(const Tracker& tracker, int32_t combined_status) {
		return tracker.extract_count(combined_status);
	}

	template <typename Tracker>
	static constexpr int status_is_compressed(const Tracker&) {
		return Tracker::is_compressed;
	}
	template <typename Tracker>
	static constexpr int status_decompressing(const Tracker&) {
		return Tracker::decompressing;
	}
	template <typename Tracker>
	static constexpr int status_is_decompressed(const Tracker&) {
		return Tracker::is_decompressed;
	}
	template <typename Tracker>
	static constexpr int status_compressing(const Tracker&) {
		return Tracker::compressing;
	}

	template <typename DecompressedDataAccessor>
	static bool is_moved(const DecompressedDataAccessor& acc) {
		return acc.m_is_moved;
	}

	template <typename CompressedAccessor>
	static decltype(auto) get_raw_storage_accessor(const CompressedAccessor& acc) {
		return acc.m_base_data_acc;
	}

	template <typename CompressedAccessor>
	static decltype(auto) get_compressed_storage_accessor(const CompressedAccessor& acc) {
		return acc.m_compressed_data_acc;
	}

	template <typename CompressedAccessor>
	static decltype(auto) get_compression_object(const CompressedAccessor& acc) {
		return acc.m_compression;
	}
};

} // namespace celerity::detail
