#pragma once

#include <cstdint>
#include <type_traits>

namespace celerity {
namespace detail {
	namespace utils {

		template <typename BitMaskT>
		constexpr inline uint32_t popcount(const BitMaskT bit_mask) noexcept {
			static_assert(std::is_integral_v<BitMaskT> && std::is_unsigned_v<BitMaskT>, "popcount argument needs to be an unsigned integer type.");

			uint32_t counter = 0;
			for(auto b = bit_mask; b; b >>= 1) {
				counter += b & 1;
			}
			return counter;
		}

	} // namespace utils
} // namespace detail
} // namespace celerity
