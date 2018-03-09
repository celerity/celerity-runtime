#include <gtest/gtest.h>
#include <celerity_runtime.h>

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

TEST(buffer_state, UpdateRegion) {
  detail::buffer_state<1> bs(cl::sycl::range<1>(256), 2);
  bs.update_region({0, 128}, {1});

  auto sn = bs.get_source_nodes(GridRegion<1>(32, 64));
  EXPECT_EQ(sn.size(), 1);
  EXPECT_EQ(sn[0].first, GridBox<1>(32, 64));
  EXPECT_EQ(sn[0].second.size(), 1);
  EXPECT_EQ(sn[0].second.count(1), 1);

  sn = bs.get_source_nodes(GridRegion<1>(256));
  EXPECT_EQ(sn.size(), 2);
  EXPECT_EQ(sn[0].first, GridBox<1>(128, 256));
  EXPECT_EQ(sn[0].second.size(), 2);
  EXPECT_EQ(sn[0].second.count(0), 1);
  EXPECT_EQ(sn[0].second.count(1), 1);
  EXPECT_EQ(sn[1].first, GridBox<1>(0, 128));
  EXPECT_EQ(sn[1].second.size(), 1);
  EXPECT_EQ(sn[1].second.count(1), 1);
}

TEST(buffer_state, CollapseRegions) {
  // We test buffer_state<>::collapse_regions by observing the order of the
  // returned boxes. This somewhat relies on implementation details of
  // buffer_state<>::get_source_nodes.
  // TODO: We may want to test this directly instead
  detail::buffer_state<1> bs(cl::sycl::range<1>(256), 2);
  bs.update_region({64, 128}, {1});
  bs.update_region({192, 256}, {1});

  auto sn = bs.get_source_nodes(GridRegion<1>(64, 256));
  EXPECT_EQ(sn.size(), 3);
  EXPECT_EQ(sn[0].first, GridBox<1>(64, 128));
  EXPECT_EQ(sn[0].second.size(), 1);
  EXPECT_EQ(sn[0].second.count(1), 1);

  // Since this one is returned before the [128,192) box,
  // the {[64,128), [192,256)} region must exist internally.
  EXPECT_EQ(sn[1].first, GridBox<1>(192, 256));
  EXPECT_EQ(sn[1].second.size(), 1);
  EXPECT_EQ(sn[1].second.count(1), 1);

  EXPECT_EQ(sn[2].first, GridBox<1>(128, 192));
  EXPECT_EQ(sn[2].second.size(), 2);
  EXPECT_EQ(sn[2].second.count(0), 1);
  EXPECT_EQ(sn[2].second.count(1), 1);
}
}  // namespace celerity
