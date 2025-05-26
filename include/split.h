#pragma once

#include "grid.h"
#include "ranges.h"

#include <cstddef>
#include <vector>


namespace celerity::detail {

std::vector<box<3>> split_1d(const box<3>& full_box, const range<3>& granularity, const size_t num_boxs);
std::vector<box<3>> split_2d(const box<3>& full_box, const range<3>& granularity, const size_t num_boxs);

} // namespace celerity::detail
