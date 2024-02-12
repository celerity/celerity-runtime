#include "grid.h"

namespace celerity::test_utils {

template <int Dims>
detail::box_vector<Dims> create_random_boxes(const size_t grid_size, const size_t max_box_size, const size_t num_boxes, const uint32_t seed);

void render_boxes(const detail::box_vector<2>& boxes, const std::string_view suffix = "region");

} // namespace celerity::test_utils
