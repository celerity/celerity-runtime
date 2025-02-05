#pragma once

#include "grid.h"
#include "ranges.h"

#include <cstddef>
#include <vector>


namespace celerity::detail {

std::vector<box<3>> split_1d(const box<3>& full_box, const range<3>& granularity, const size_t num_boxs);

std::array<size_t, 2> find_best_split_factors_2d(const box<3>& full_chunk, const range<3>& granularity, const size_t num_chunks);
std::vector<box<3>> split_2d(const box<3>& full_chunk, const range<3>& granularity, const std::array<size_t, 2>& num_chunks);
std::vector<box<3>> split_2d(const box<3>& full_box, const range<3>& granularity, const size_t num_boxs);

} // namespace celerity::detail
