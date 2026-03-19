#pragma once
#include "accessor.h"
#include "buffer.h"

#include "compression.h"

#include <utility>


namespace celerity {
namespace detail {
	template <typename Algorithm>
	struct algorithm_skeleton {
		using compression_object_type = std::remove_cvref_t<decltype(std::declval<const Algorithm&>().get_compression_object())>;
	};

	enum class range_mapper_type { one_to_one, slice, all, neighborhood, fixed, custom };

	template <typename Functor, int Dims>
	constexpr range_mapper_type deduce_range_mapper_type() {
		if constexpr(std::is_same_v<Functor, celerity::access::one_to_one>) {
			return range_mapper_type::one_to_one;
		} else if constexpr(std::is_same_v<Functor, celerity::access::all>) {
			return range_mapper_type::all;
		} else if constexpr(std::is_same_v<Functor, celerity::access::fixed<Dims>>) {
			return range_mapper_type::fixed;
		} else if constexpr(std::is_same_v<Functor, celerity::access::slice<Dims>>) {
			return range_mapper_type::slice;
		} else if constexpr(std::is_same_v<Functor, celerity::access::neighborhood<Dims>>) {
			return range_mapper_type::neighborhood;
		} else {
			return range_mapper_type::custom;
		}
	}

	// Combined descriptor for accessor extension points.
	// It currently bundles compression, range-mapper type, and the compile-time dependency description.
	template <typename Compression, detail::range_mapper_type RangeMapperType, typename ExtensionArg = std::tuple<>>
	struct accessor_extension {
		using compression_type = Compression;
		using extension_type = ExtensionArg;
		static constexpr detail::range_mapper_type range_mapper_type = RangeMapperType;
	};

	template <typename Algorithm, typename = void>
	struct compression_dependency {
		using type = std::tuple<>;
	};

	template <typename Algorithm>
	struct compression_dependency<Algorithm, std::void_t<decltype(std::declval<const Algorithm&>().get_dependencies())>> {
		using type = typename std::remove_cvref_t<decltype(std::declval<const Algorithm&>().get_dependencies())>::dependency_type;
	};
} // namespace detail


template <typename DataT, int Dims, typename Intype, typename Functor, access_mode Mode, access_mode ModeNoInit, target Target,
    template <typename, typename, compression_category> typename SelectedCompression, compression_category Category>
accessor(const buffer<Intype, Dims, SelectedCompression<Intype, DataT, Category>>& buff, handler& cgh, const Functor& rmfn,
    const detail::access_tag<Mode, ModeNoInit, Target> tag) -> accessor<DataT, Dims, Mode, Target,
    detail::accessor_extension<typename detail::algorithm_skeleton<SelectedCompression<Intype, DataT, Category>>::compression_object_type,
        detail::deduce_range_mapper_type<Functor, Dims>(), typename detail::compression_dependency<SelectedCompression<Intype, DataT, Category>>::type>>;

template <typename DataT, int Dims, typename Intype, typename Functor, access_mode Mode, access_mode TagMode, target Target,
    template <typename, typename, compression_category> typename SelectedCompression, compression_category Category>
accessor(const buffer<Intype, Dims, SelectedCompression<Intype, DataT, Category>>& buff, handler& cgh, const Functor& rmfn,
    const detail::access_tag<TagMode, Mode, Target> tag, const property::no_init& prop) -> accessor<DataT, Dims, Mode, Target,
    detail::accessor_extension<typename detail::algorithm_skeleton<SelectedCompression<Intype, DataT, Category>>::compression_object_type,
        detail::deduce_range_mapper_type<Functor, Dims>(), typename detail::compression_dependency<SelectedCompression<Intype, DataT, Category>>::type>>;

template <typename DataT, int Dims, typename Intype, access_mode TagMode, access_mode TagModeNoInit, target Target,
    template <typename, typename, compression_category> typename SelectedCompression, compression_category Category>
accessor(buffer<Intype, Dims, SelectedCompression<Intype, DataT, Category>>& buff, handler& cgh, const detail::access_tag<TagMode, TagModeNoInit, Target> tag,
    const property_list& prop_list) -> accessor<DataT, Dims, TagModeNoInit, Target,
    detail::accessor_extension<typename detail::algorithm_skeleton<SelectedCompression<Intype, DataT, Category>>::compression_object_type,
        detail::range_mapper_type::all, typename detail::compression_dependency<SelectedCompression<Intype, DataT, Category>>::type>>;

} // namespace celerity