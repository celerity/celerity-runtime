#pragma once

#include "ranges.h"
#include "types.h"

#include <fmt/format.h>
#include <fmt/ostream.h>

namespace celerity {

namespace detail {
	std::ostream& print_chunk3(std::ostream& os, chunk<3> chnk3);
	std::ostream& print_subrange3(std::ostream& os, subrange<3> subr3);
} // namespace detail

template <int Dims>
std::ostream& operator<<(std::ostream& os, chunk<Dims> chnk) {
	return detail::print_chunk3(os, detail::chunk_cast<3>(chnk));
}

template <int Dims>
std::ostream& operator<<(std::ostream& os, subrange<Dims> subr) {
	return detail::print_subrange3(os, detail::subrange_cast<3>(subr));
}

} // namespace celerity

// TODO prefix type aliases like in GDB pretty_printers (requires removing explicit prefixes elsewhere in the code)
#define CELERITY_DETAIL_IMPLEMENT_FMT_FORMATTER_FOR_PHANTOM_TYPE(TYPE_ALIAS)                                                                              \
	template <>                                                                                                                                                \
	struct fmt::formatter<celerity::detail::TYPE_ALIAS> : fmt::formatter<celerity::detail::TYPE_ALIAS::underlying_t> {};

CELERITY_DETAIL_IMPLEMENT_FMT_FORMATTER_FOR_PHANTOM_TYPE(task_id)
CELERITY_DETAIL_IMPLEMENT_FMT_FORMATTER_FOR_PHANTOM_TYPE(buffer_id)
CELERITY_DETAIL_IMPLEMENT_FMT_FORMATTER_FOR_PHANTOM_TYPE(node_id)
CELERITY_DETAIL_IMPLEMENT_FMT_FORMATTER_FOR_PHANTOM_TYPE(command_id)
CELERITY_DETAIL_IMPLEMENT_FMT_FORMATTER_FOR_PHANTOM_TYPE(collective_group_id)
CELERITY_DETAIL_IMPLEMENT_FMT_FORMATTER_FOR_PHANTOM_TYPE(reduction_id)
CELERITY_DETAIL_IMPLEMENT_FMT_FORMATTER_FOR_PHANTOM_TYPE(host_object_id)
CELERITY_DETAIL_IMPLEMENT_FMT_FORMATTER_FOR_PHANTOM_TYPE(hydration_id)
CELERITY_DETAIL_IMPLEMENT_FMT_FORMATTER_FOR_PHANTOM_TYPE(memory_id)
CELERITY_DETAIL_IMPLEMENT_FMT_FORMATTER_FOR_PHANTOM_TYPE(device_id)
CELERITY_DETAIL_IMPLEMENT_FMT_FORMATTER_FOR_PHANTOM_TYPE(transfer_id)

#undef CELERITY_DETAIL_IMPLEMENT_FMT_FORMATTER_FOR_PHANTOM_TYPE


template <int Dims> struct fmt::formatter<celerity::chunk<Dims>> : ostream_formatter {};
template <int Dims> struct fmt::formatter<celerity::subrange<Dims>> : ostream_formatter {};
