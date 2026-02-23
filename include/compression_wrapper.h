#pragma once
#include "accessor.h"
#include "buffer.h"

#include "compression.h"


namespace celerity {
namespace detail {
	template <typename Algorithm>
	struct algorithm_skeleton {
		using compression_object_type = decltype(std::declval<Algorithm>().get_compression_object());
	};
} // namespace detail


template <typename DataT, int Dims, typename Intype, typename Functor, access_mode Mode, access_mode ModeNoInit, target Target,
    template <typename, typename, compression_category> typename SelectedCompression, compression_category Category>
accessor(const buffer<Intype, Dims, SelectedCompression<Intype, DataT, Category>>& buff, handler& cgh, const Functor& rmfn,
    const detail::access_tag<Mode, ModeNoInit, Target> tag)
    -> accessor<DataT, Dims, Mode, Target, typename detail::algorithm_skeleton<SelectedCompression<Intype, DataT, Category>>::compression_object_type>;

template <typename DataT, int Dims, typename Intype, typename Functor, access_mode Mode, access_mode TagMode, target Target,
    template <typename, typename, compression_category> typename SelectedCompression, compression_category Category>
accessor(const buffer<Intype, Dims, SelectedCompression<Intype, DataT, Category>>& buff, handler& cgh, const Functor& rmfn,
    const detail::access_tag<TagMode, Mode, Target> tag, const property::no_init& prop)
    -> accessor<DataT, Dims, Mode, Target, typename detail::algorithm_skeleton<SelectedCompression<Intype, DataT, Category>>::compression_object_type>;

// TODO: create a template extraction for this
template <typename DataT, int Dims, typename Intype, access_mode TagMode, access_mode TagModeNoInit, target Target,
    template <typename, typename, compression_category> typename SelectedCompression, compression_category Category>
accessor(buffer<Intype, Dims, SelectedCompression<Intype, DataT, Category>>& buff, handler& cgh, const detail::access_tag<TagMode, TagModeNoInit, Target> tag,
    const property_list& prop_list)
    -> accessor<DataT, Dims, TagModeNoInit, Target, typename detail::algorithm_skeleton<SelectedCompression<Intype, DataT, Category>>::compression_object_type>;
} // namespace celerity