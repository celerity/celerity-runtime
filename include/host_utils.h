#pragma once

#include "item.h"
#include "partition.h"

namespace celerity::experimental {

template <int Dims, typename Functor>
void for_each_item(const partition<Dims> part, Functor func) {
	static_assert(std::is_invocable_v<Functor, item<Dims>>, "for_each_item functor needs to be invocable with celerity::item<Dims>.");

	if constexpr(Dims == 0) { //
		func(detail::make_item<0>(id<0>(), id<0>(), range<0>()));
	} else if constexpr(Dims == 1) {
		for(size_t d0 = 0; d0 < part.get_subrange().range[0]; ++d0) {
			func(detail::make_item<1>(id<1>(d0) + part.get_subrange().offset, part.get_subrange().offset, part.get_global_size()));
		}
	} else if constexpr(Dims == 2) {
		for(size_t d0 = 0; d0 < part.get_subrange().range[0]; ++d0) {
			for(size_t d1 = 0; d1 < part.get_subrange().range[1]; ++d1) {
				func(detail::make_item<2>(id<2>(d0, d1) + part.get_subrange().offset, part.get_subrange().offset, part.get_global_size()));
			}
		}
	} else if constexpr(Dims == 3) {
		for(size_t d0 = 0; d0 < part.get_subrange().range[0]; ++d0) {
			for(size_t d1 = 0; d1 < part.get_subrange().range[1]; ++d1) {
				for(size_t d2 = 0; d2 < part.get_subrange().range[2]; ++d2) {
					func(detail::make_item<3>(id<3>(d0, d1, d2) + part.get_subrange().offset, part.get_subrange().offset, part.get_global_size()));
				}
			}
		}
	}
}

template <int Dims, typename Functor>
void for_each_item(const range<Dims> range, Functor func) {
	const auto partition = detail::make_partition(range, subrange<Dims>(id<Dims>(detail::zeros), range));
	for_each_item(partition, func);
}

} // namespace celerity::experimental
