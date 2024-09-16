#pragma once

namespace celerity {
template <typename T>
class compressed {};
} // namespace celerity

namespace celerity::compression {
// compression tag
struct uncompressed {};

template <typename T, typename Q>
struct quantization {
	using value_type = T;
	using quant_type = Q;
};

template <typename T, typename Q>
struct point_cloud {
	using value_type = T;
	using compression_type = Q;
};

} // namespace celerity::compression
