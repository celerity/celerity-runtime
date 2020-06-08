#include "print_utils.h"

#include <spdlog/spdlog.h>

#include <spdlog/fmt/ostr.h>

namespace celerity {
namespace detail {

	std::ostream& print_chunk3(std::ostream& os, chunk<3> chnk3) {
		auto start = chnk3.offset;
		auto end = chnk3.offset + chnk3.range;
		auto size = chnk3.global_size;
		return os << fmt::format("[{},{},{}] - [{},{},{}] : {{{},{},{}}}", start[0], start[1], start[2], end[0], end[1], end[2], size[0], size[1], size[2]);
	}

	std::ostream& print_subrange3(std::ostream& os, subrange<3> subr3) {
		auto start = subr3.offset;
		auto end = subr3.offset + subr3.range;
		return os << fmt::format("[{},{},{}] - [{},{},{}]", start[0], start[1], start[2], end[0], end[1], end[2]);
	}

} // namespace detail
} // namespace celerity
