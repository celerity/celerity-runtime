#pragma once

#include <vector>

#include "ranges.h"

namespace celerity::detail {

std::vector<chunk<3>> split_1d(const chunk<3>& full_chunk, const range<3>& granularity, const size_t num_chunks);
std::vector<chunk<3>> split_2d(const chunk<3>& full_chunk, const range<3>& granularity, const size_t num_chunks);

} // namespace celerity::detail
