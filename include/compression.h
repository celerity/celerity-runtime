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

template <compression_category Category>
struct compression_tags {
	static constexpr compression_category category = Category;
};

namespace detail {
	template <typename T>
	struct compression_checker {
		static constexpr bool value = false;
	};

	template <template <typename, typename> typename T, typename C, typename V, compression_category Category>
	struct compression_checker<compressed<T<C, V>, Category>> {
		static constexpr bool value = true;
	};

	// get the category of the compression algorithm
	template <typename T>
	struct compression_category_extractor {
		static constexpr compression_category value = compression_category::none; // default
	};
	template <template <typename, typename> typename T, typename C, typename V, compression_category Category>
	struct compression_category_extractor<compressed<T<C, V>, Category>> {
		static constexpr compression_category value = Category;
	};

	template <typename T>
	concept compression_algorithm = requires(T a) {
		detail::compression_checker<T>::value;
		{ T::category } -> std::convertible_to<compression_category>;
	};

	template <compression_category Current, compression_category Wanted>
	concept contains_same_category = (Current & Wanted) == Wanted;
} // namespace detail

template <typename T>
concept check_provided_type_with_algorithm =
    detail::compression_algorithm<T> && detail::contains_same_category<T::category, detail::compression_category_extractor<T>::value>;

namespace compression {
	// compression tag
	struct uncompressed {};

} // namespace compression
} // namespace celerity
