#ifndef CELERITY_RUNTIME
#define CELERITY_RUNTIME

#define CELERITY_NUM_WORKER_NODES 2

#include <cassert>
#include <functional>
#include <regex>
#include <set>
#include <string>
#include <unordered_map>

#include <allscale/api/user/data/grid.h>
#include <SYCL/sycl.hpp>
#include <boost/format.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/type_index.hpp>

using namespace allscale::api::user::data;

namespace boost_graph {
using boost::adjacency_list;
using boost::directedS;
using boost::vecS;

struct vertex_properties {
  std::string label;
};

struct graph_properties {
  std::string name;
};

using Graph = adjacency_list<vecS, vecS, directedS, vertex_properties,
                             boost::no_property, graph_properties>;
}  // namespace boost_graph

using namespace boost_graph;

namespace celerity {

class buffer;

using task_id = size_t;
using buffer_id = size_t;
using node_id = size_t;
using region_version = size_t;

// TODO: Naming. Check again what difference between "range" and a "nd_range" is
// in SYCL
struct nd_point {};
struct nd_subrange {
  nd_point start;
  nd_point range;
  // FIXME Dimensions
  cl::sycl::nd_range<1> global_size;
};
using range_mapper = std::function<nd_subrange(nd_subrange range)>;

// Convenience range mappers
namespace access {
struct one_to_one {
  nd_subrange operator()(nd_subrange range) const { return range; }
};
}  // namespace access

class distr_queue;

// FIXME: Type, dimensions
template <cl::sycl::access::mode Mode>
using accessor =
    cl::sycl::accessor<float, 1, Mode, cl::sycl::access::target::global_buffer>;

// TODO: Looks like we will have to provide the full accessor API
template <cl::sycl::access::mode Mode>
class prepass_accessor {
 public:
  float& operator[](cl::sycl::id<1> index) const { return value; }

 private:
  mutable float value = 0.f;
};

enum class is_prepass { true_t, false_t };

template <is_prepass IsPrepass>
class handler {};

template <>
class handler<is_prepass::true_t> {
 public:
  template <typename name, typename functorT>
  void parallel_for(size_t count, const functorT& kernel) {
    // DEBUG: Find nice name for kernel (regex is probably not super portable)
    auto qualified_name = boost::typeindex::type_id<name*>().pretty_name();
    std::regex name_regex(R"(.*?(?:::)?([\w_]+)\s?\*.*)");
    std::smatch matches;
    std::regex_search(qualified_name, matches, name_regex);
    debug_name = matches.size() > 0 ? matches[1] : qualified_name;
  }

  template <cl::sycl::access::mode Mode>
  void require(prepass_accessor<Mode> a, size_t buffer_id, range_mapper rm);

  std::string get_debug_name() const { return debug_name; };

 private:
  friend class distr_queue;
  distr_queue& queue;
  size_t task_id;
  std::string debug_name;

  handler(distr_queue& q, size_t task_id) : queue(q), task_id(task_id) {
    debug_name = (boost::format("task%d") % task_id).str();
  }
};

template <>
class handler<is_prepass::false_t> {
 public:
  template <typename name, typename functorT>
  void parallel_for(size_t count, const functorT& kernel) {
    sycl_handler->parallel_for<name>(
        cl::sycl::nd_range<1>(cl::sycl::range<1>(count), cl::sycl::range<1>(1)),
        kernel);
  }

  template <cl::sycl::access::mode Mode>
  void require(accessor<Mode> a, size_t buffer_id);

  cl::sycl::handler& get_sycl_handler() { return *sycl_handler; }

 private:
  friend class distr_queue;
  distr_queue& queue;
  cl::sycl::handler* sycl_handler;
  size_t task_id;

  // The handler does not take ownership of the sycl_handler, but expects it to
  // exist for the duration of it's lifetime.
  handler(distr_queue& q, size_t task_id, cl::sycl::handler* sycl_handler)
      : queue(q), task_id(task_id), sycl_handler(sycl_handler) {
    this->sycl_handler = sycl_handler;
  }
};

// TODO: Templatize (data type, dimensions) - see sycl::buffer
class buffer {
 public:
  template <cl::sycl::access::mode Mode>
  prepass_accessor<Mode> get_access(handler<is_prepass::true_t> handler,
                                    range_mapper rm) {
    prepass_accessor<Mode> a;
    handler.require(a, id, rm);
    return a;
  }

  template <cl::sycl::access::mode Mode>
  accessor<Mode> get_access(handler<is_prepass::false_t> handler,
                            range_mapper) {
    auto a = accessor<Mode>(sycl_buffer, handler.get_sycl_handler());
    handler.require(a, id);
    return a;
  }

  size_t get_id() { return id; }

  float operator[](size_t idx) { return 1.f; }

 private:
  friend distr_queue;
  buffer_id id;
  size_t size;

  cl::sycl::buffer<float, 1> sycl_buffer;

  buffer(float* host_ptr, size_t size, buffer_id bid)
      : id(bid), size(size), sycl_buffer(host_ptr, cl::sycl::range<1>(size)){};
};

class branch_handle {
 public:
  template <size_t idx>
  void get(buffer){};
};

namespace detail {
// This is a workaround that let's us store a command group functor with auto&
// parameter, which we require in order to be able to pass different
// celerity::handlers (celerity::is_prepass::true_t/false_t) for prepass and
// live invocations.
struct cgf_storage_base {
  virtual void operator()(handler<is_prepass::true_t>) = 0;
  virtual void operator()(handler<is_prepass::false_t>) = 0;
  virtual ~cgf_storage_base(){};
};

template <typename CGF>
struct cgf_storage : cgf_storage_base {
  CGF cgf;

  cgf_storage(CGF cgf) : cgf(cgf) {}

  void operator()(handler<is_prepass::true_t> cgh) override { cgf(cgh); }
  void operator()(handler<is_prepass::false_t> cgh) override { cgf(cgh); }
};

inline GridPoint<1> sycl_range_to_grid_point(cl::sycl::range<1> range) {
  return GridPoint<1>(range.get(0));
}

inline GridPoint<2> sycl_range_to_grid_point(cl::sycl::range<2> range) {
  return GridPoint<2>(range.get(0), range.get(1));
}

inline GridPoint<3> sycl_range_to_grid_point(cl::sycl::range<3> range) {
  return GridPoint<3>(range.get(0), range.get(1), range.get(2));
}

class buffer_state_base {
 public:
  virtual size_t get_dimensions() const = 0;
  virtual ~buffer_state_base(){};
};

template <size_t Dims>
class buffer_state : public buffer_state_base {
 public:
  buffer_state(cl::sycl::range<Dims> size) {
    static_assert(Dims >= 1 && Dims <= 3, "Unsupported dimensionality");
    region_versions.insert(
        std::make_pair(0, GridRegion<Dims>(sycl_range_to_grid_point(size))));
  }
  size_t get_dimensions() const override { return Dims; }

 private:
  region_version latest = 0;
  std::unordered_multimap<region_version, GridRegion<Dims>> region_versions;
};

class node {
 public:
  template <size_t Dims>
  void bump_buffer_state(buffer_id bid, GridRegion<Dims> updated_region,
                         bool has_updated_region) {
    assert(Dims == buffer_states[bid]->get_dimensions());
    // TODO
  };

  template <size_t Dims>
  void add_buffer(buffer_id bid, cl::sycl::range<Dims> size) {
    buffer_states[bid] = std::make_unique<buffer_state<Dims>>(size);
  }

 private:
  std::unordered_map<buffer_id, std::unique_ptr<buffer_state_base>>
      buffer_states;
};
}  // namespace detail

class distr_queue {
 public:
  // TODO: Device should be selected transparently
  distr_queue(cl::sycl::device device);

  template <typename CGF>
  void submit(CGF cgf) {
    // Task ids start at 1
    task_id tid = ++task_count;
    handler<is_prepass::true_t> h(*this, tid);
    cgf(h);
    task_names[tid] = h.get_debug_name();
    task_command_groups[tid] = std::make_unique<detail::cgf_storage<CGF>>(cgf);
  }

  buffer create_buffer(float* host_ptr, size_t size) {
    buffer_id bid = buffer_count++;
    for (auto& it : nodes) {
      // FIXME: Dimensions
      it.second.add_buffer<1>(bid, cl::sycl::range<1>(size));
    }
    return buffer(host_ptr, size, bid);
  }

  // experimental
  // TODO: Can we derive 2nd lambdas args from requested values in 1st?
  void branch(std::function<void(branch_handle& bh)>,
              std::function<void(float)>){};

  void debug_print_task_graph();
  void TEST_execute_deferred();
  void build_command_graph();

 private:
  friend handler<is_prepass::true_t>;
  // TODO: We may want to move all these task maps into a dedicated struct
  std::unordered_map<task_id, std::string> task_names;
  std::unordered_map<task_id, std::unique_ptr<detail::cgf_storage_base>>
      task_command_groups;
  std::unordered_map<task_id, std::unordered_multimap<buffer_id, range_mapper>>
      task_range_mappers;
  std::unordered_map<buffer_id, task_id> buffer_last_writer;
  size_t task_count = 0;
  size_t buffer_count = 0;
  Graph task_graph;
  std::set<task_id> active_tasks;
  std::unordered_map<node_id, detail::node> nodes;

  cl::sycl::queue sycl_queue;

  void add_requirement(task_id tid, buffer_id bid, cl::sycl::access::mode mode,
                       range_mapper rm);
};

}  // namespace celerity

#endif
