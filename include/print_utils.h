#pragma once

#include "ranges.h"

namespace celerity {

namespace detail {
	std::ostream& print_chunk3(std::ostream& os, chunk<3> chnk3);
	std::ostream& print_subrange3(std::ostream& os, subrange<3> subr3);
} // namespace detail

template <int Dims>
std::ostream& operator<<(std::ostream& os, chunk<Dims> chnk) {
	return detail::print_chunk3(os, chnk);
}

template <int Dims>
std::ostream& operator<<(std::ostream& os, subrange<Dims> subr) {
	return detail::print_subrange3(os, subr);
}

} // namespace celerity
