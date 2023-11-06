#pragma once

namespace celerity::compression {

// compression tag
struct uncompressed {};

template <typename T, typename Q>
struct quantization {
	using value_type = T;
	using quant_type = Q;
};

} // namespace celerity::compression