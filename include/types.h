#pragma once

#include <boost/variant/variant.hpp>

namespace celerity {

using task_id = size_t;
using buffer_id = size_t;
using node_id = size_t;
using vertex = size_t;

using any_range = boost::variant<cl::sycl::range<1>, cl::sycl::range<2>, cl::sycl::range<3>>;

} // namespace celerity
