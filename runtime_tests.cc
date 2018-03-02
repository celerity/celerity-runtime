#include <gtest/gtest.h>

#include "celerity_runtime.h"

namespace celerity {
TEST(buffer_state, Basic) {
  detail::buffer_state<1> bs(cl::sycl::range<1>(256), 2);
  EXPECT_EQ(bs.get_dimensions(), 1);

  auto sn = bs.get_source_nodes(GridRegion<1>(256));
  EXPECT_EQ(sn.size(), 1);
  EXPECT_EQ(sn[0].first, GridBox<1>(256));
  EXPECT_EQ(sn[0].second.size(), 2);
  EXPECT_EQ(sn[0].second.count(0), 1);
  EXPECT_EQ(sn[0].second.count(1), 1);
}
}  // namespace celerity
