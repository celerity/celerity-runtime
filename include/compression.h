#pragma once

namespace celerity {
// shift operator to combine multiple compression categories for automatic detection of the wanted compression
enum class compression_category { none = 0, element_wise = 1 << 0, local_memory = 1 << 1, global_memory = 1 << 2, kernel = 1 << 3, automatic = 1 << 4 };

inline constexpr compression_category operator|(compression_category a, compression_category b) {
	return static_cast<compression_category>(static_cast<int>(a) | static_cast<int>(b));
}

inline constexpr compression_category operator&(compression_category a, compression_category b) {
	return static_cast<compression_category>(static_cast<int>(a) & static_cast<int>(b));
}

template <typename T, compression_category CompressionCategory>
class compressed {};
} // namespace celerity

namespace celerity::compression {
// compression tag
struct uncompressed {};

// template <typename T, typename Q>
// struct quantization {
// 	using value_type = T;
// 	using quant_type = Q;
// };

// template <typename T, typename Q>
// struct point_cloud {
// 	using value_type = T;
// 	using compression_type = Q;
// };

} // namespace celerity::compression
