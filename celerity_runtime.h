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

using task_id = size_t;
using buffer_id = size_t;
using node_id = size_t;
using region_version = size_t;

template <int Dims>
struct subrange {
  cl::sycl::id<Dims> start;
  cl::sycl::range<Dims> range;
  cl::sycl::range<Dims> global_size;
};

namespace detail {
template <int Dims>
using range_mapper_fn = std::function<subrange<Dims>(subrange<Dims> range)>;

class range_mapper_base {
 public:
  virtual size_t get_dimensions() const = 0;
  virtual subrange<1> operator()(subrange<1> range) { return subrange<1>(); }
  virtual subrange<2> operator()(subrange<2> range) { return subrange<2>(); }
  virtual subrange<3> operator()(subrange<3> range) { return subrange<3>(); }
  virtual ~range_mapper_base() {}
};

template <int Dims>
class range_mapper : public range_mapper_base {
 public:
  range_mapper(range_mapper_fn<Dims> fn) : rmfn(fn) {}
  size_t get_dimensions() const override { return Dims; }
  subrange<Dims> operator()(subrange<Dims> range) override {
    return rmfn(range);
  }

 private:
  range_mapper_fn<Dims> rmfn;
};
}  // namespace detail

// Convenience range mappers
namespace access {
template <int Dims>
struct one_to_one {
  subrange<Dims> operator()(subrange<Dims> range) const { return range; }
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
  template <typename name, typename functorT, int Dims>
  void parallel_for(cl::sycl::range<Dims> range, const functorT& kernel) {
    // DEBUG: Find nice name for kernel (regex is probably not super portable)
    auto qualified_name = boost::typeindex::type_id<name*>().pretty_name();
    std::regex name_regex(R"(.*?(?:::)?([\w_]+)\s?\*.*)");
    std::smatch matches;
    std::regex_search(qualified_name, matches, name_regex);
    debug_name = matches.size() > 0 ? matches[1] : qualified_name;
  }

  template <cl::sycl::access::mode Mode>
  void require(prepass_accessor<Mode> a, size_t buffer_id,
               std::unique_ptr<detail::range_mapper_base> rm);

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
  template <typename name, typename functorT, int Dims>
  void parallel_for(cl::sycl::range<Dims> range, const functorT& kernel) {
    sycl_handler->parallel_for<name>(range, kernel);
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

template <typename DataT, int Dims>
class buffer {
 public:
  template <cl::sycl::access::mode Mode>
  prepass_accessor<Mode> get_access(handler<is_prepass::true_t> handler,
                                    detail::range_mapper_fn<Dims> rmfn) {
    prepass_accessor<Mode> a;
    handler.require(a, id, std::make_unique<detail::range_mapper<Dims>>(rmfn));
    return a;
  }

  template <cl::sycl::access::mode Mode>
  accessor<Mode> get_access(handler<is_prepass::false_t> handler,
                            detail::range_mapper_fn<Dims> rmfn) {
    auto a = accessor<Mode>(sycl_buffer, handler.get_sycl_handler());
    handler.require(a, id);
    return a;
  }

  size_t get_id() { return id; }

  // FIXME Host-size access should block
  DataT operator[](size_t idx) { return 1.f; }

 private:
  friend distr_queue;
  buffer_id id;
  cl::sycl::range<Dims> size;
  cl::sycl::buffer<float, 1> sycl_buffer;

  buffer(DataT* host_ptr, cl::sycl::range<Dims> size, buffer_id bid)
      : id(bid), size(size), sycl_buffer(host_ptr, size){};
};

class branch_handle {
 public:
  template <typename DataT, int Dims>
  void get(buffer<DataT, Dims>, cl::sycl::range<Dims>){};
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
  return GridPoint<1>(range[0]);
}

inline GridPoint<2> sycl_range_to_grid_point(cl::sycl::range<2> range) {
  return GridPoint<2>(range[0], range[1]);
}

inline GridPoint<3> sycl_range_to_grid_point(cl::sycl::range<3> range) {
  return GridPoint<3>(range[0], range[1], range[2]);
}

class buffer_state_base {
 public:
  virtual size_t get_dimensions() const = 0;
  virtual ~buffer_state_base(){};
};

template <int Dims>
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
  template <int Dims>
  void bump_buffer_state(buffer_id bid, GridRegion<Dims> updated_region,
                         bool has_updated_region) {
    assert(Dims == buffer_states[bid]->get_dimensions());
    // TODO
  };

  template <int Dims>
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

  template <typename DataT, int Dims>
  buffer<DataT, Dims> create_buffer(DataT* host_ptr,
                                    cl::sycl::range<Dims> size) {
    buffer_id bid = buffer_count++;
    for (auto& it : nodes) {
      it.second.add_buffer<Dims>(bid, size);
    }
    return buffer<DataT, Dims>(host_ptr, size, bid);
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
  std::unordered_map<
      task_id,
      std::unordered_map<
          buffer_id, std::vector<std::unique_ptr<detail::range_mapper_base>>>>
      task_range_mappers;
  std::unordered_map<buffer_id, task_id> buffer_last_writer;
  size_t task_count = 0;
  size_t buffer_count = 0;
  Graph task_graph;
  std::set<task_id> active_tasks;
  std::unordered_map<node_id, detail::node> nodes;

  cl::sycl::queue sycl_queue;

  void add_requirement(task_id tid, buffer_id bid, cl::sycl::access::mode mode,
                       std::unique_ptr<detail::range_mapper_base> rm);
};

}  // namespace celerity

#endif
