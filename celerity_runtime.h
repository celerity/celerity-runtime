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
#include <boost/variant.hpp>

using namespace allscale::api::user::data;

namespace celerity {

using task_id = size_t;
using buffer_id = size_t;
using node_id = size_t;
using region_version = size_t;

// Graphs

using vertex = size_t;

struct tdag_vertex_properties {
  std::string label;

  // Whether this task has been processed into the command dag
  bool processed = false;

  // The number of unsatisfied (= unprocessed) dependencies this task has
  size_t num_unsatisfied = 0;
};

struct tdag_graph_properties {
  std::string name;
};

using task_dag =
    boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS,
                          tdag_vertex_properties, boost::no_property,
                          tdag_graph_properties>;

struct cdag_vertex_properties {
  std::string label;
};

struct cdag_graph_properties {
  std::string name;
  std::map<task_id, vertex> task_complete_vertices;
};

using command_dag =
    boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS,
                          cdag_vertex_properties, boost::no_property,
                          cdag_graph_properties>;

// FIXME: Naming; could be clearer
template <int Dims>
struct subrange {
  // TODO: Should "start" be a cl::sycl::id instead? (What's the difference?)
  // We'll leave it a range for now so we don't have to provide conversion
  // overloads below
  cl::sycl::range<Dims> start;
  cl::sycl::range<Dims> range;
  cl::sycl::range<Dims> global_size;
};

namespace detail {
template <int Dims>
using range_mapper_fn = std::function<subrange<Dims>(subrange<Dims> range)>;

class range_mapper_base {
 public:
  range_mapper_base(cl::sycl::access::mode am) : access_mode(am) {}
  cl::sycl::access::mode get_access_mode() const { return access_mode; }

  virtual size_t get_dimensions() const = 0;
  virtual subrange<1> operator()(subrange<1> range) { return subrange<1>(); }
  virtual subrange<2> operator()(subrange<2> range) { return subrange<2>(); }
  virtual subrange<3> operator()(subrange<3> range) { return subrange<3>(); }
  virtual ~range_mapper_base() {}

 private:
  cl::sycl::access::mode access_mode;
};

template <int Dims>
class range_mapper : public range_mapper_base {
 public:
  range_mapper(range_mapper_fn<Dims> fn, cl::sycl::access::mode am)
      : range_mapper_base(am), rmfn(fn) {}
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
  void parallel_for(cl::sycl::range<Dims> global_size, const functorT& kernel) {
    this->global_size = global_size;
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

  ~handler();

 private:
  friend class distr_queue;
  distr_queue& queue;
  size_t task_id;
  std::string debug_name;
  boost::variant<cl::sycl::range<1>, cl::sycl::range<2>, cl::sycl::range<3>>
      global_size;

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
    handler.require(a, id,
                    std::make_unique<detail::range_mapper<Dims>>(rmfn, Mode));
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

inline GridRegion<1> subrange_to_grid_region(const subrange<1>& sr) {
  return GridRegion<1>(sycl_range_to_grid_point(sr.start),
                       sycl_range_to_grid_point(sr.start + sr.range));
}

inline GridRegion<2> subrange_to_grid_region(const subrange<2>& sr) {
  return GridRegion<2>(sycl_range_to_grid_point(sr.start),
                       sycl_range_to_grid_point(sr.start + sr.range));
}

inline GridRegion<3> subrange_to_grid_region(const subrange<3>& sr) {
  return GridRegion<3>(sycl_range_to_grid_point(sr.start),
                       sycl_range_to_grid_point(sr.start + sr.range));
}
class buffer_state_base {
 public:
  virtual size_t get_dimensions() const = 0;
  virtual ~buffer_state_base(){};
};

template <int Dims>
class buffer_state : public buffer_state_base {
  static_assert(Dims >= 1 && Dims <= 3, "Unsupported dimensionality");

 public:
  buffer_state(cl::sycl::range<Dims> size)
      : region(GridRegion<Dims>(sycl_range_to_grid_point(size))) {}
  size_t get_dimensions() const override { return Dims; }

 private:
  GridRegion<Dims> region;
};
}  // namespace detail

class distr_queue {
 public:
  // TODO: Device should be selected transparently
  distr_queue(cl::sycl::device device);

  template <typename CGF>
  void submit(CGF cgf) {
    task_id tid = task_count++;
    boost::add_vertex(task_graph);
    handler<is_prepass::true_t> h(*this, tid);
    cgf(h);
    task_command_groups[tid] = std::make_unique<detail::cgf_storage<CGF>>(cgf);
  }

  template <typename DataT, int Dims>
  buffer<DataT, Dims> create_buffer(DataT* host_ptr,
                                    cl::sycl::range<Dims> size) {
    buffer_id bid = buffer_count++;
    valid_buffer_regions[bid] =
        std::make_unique<detail::buffer_state<Dims>>(size);
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
  std::unordered_map<task_id, std::unique_ptr<detail::cgf_storage_base>>
      task_command_groups;
  std::unordered_map<task_id,
                     boost::variant<cl::sycl::range<1>, cl::sycl::range<2>,
                                    cl::sycl::range<3>>>
      task_global_sizes;
  std::unordered_map<
      task_id,
      std::unordered_map<
          buffer_id, std::vector<std::unique_ptr<detail::range_mapper_base>>>>
      task_range_mappers;

  // This is a high-level view on buffer writers, for creating the task graph
  // NOTE: This represents the state after the latest performed pre-pass, i.e.
  // it corresponds to the leaf nodes of the current task graph.
  std::unordered_map<buffer_id, task_id> buffer_last_writer;

  // This is a more granular view which encodes where (= on which node) valid
  // regions of a buffer can be found. A valid region is any region that has not
  // been written to on another node.
  // NOTE: This represents the buffer regions after all commands in the current
  // command graph have been completed.
  std::unordered_map<buffer_id, std::unique_ptr<detail::buffer_state_base>>
      valid_buffer_regions;

  size_t task_count = 0;
  size_t buffer_count = 0;
  task_dag task_graph;
  command_dag command_graph;
  std::set<task_id> active_tasks;

  // For now we don't store any additional data on nodes
  const size_t num_nodes;

  cl::sycl::queue sycl_queue;

  void add_requirement(task_id tid, buffer_id bid, cl::sycl::access::mode mode,
                       std::unique_ptr<detail::range_mapper_base> rm);

  template <int Dims>
  void set_task_data(task_id tid, cl::sycl::range<Dims> global_size,
                     std::string debug_name) {
    task_global_sizes[tid] = global_size;
    task_graph[tid].label =
        (boost::format("Task %d (%s)") % tid % debug_name).str();
  }
};

}  // namespace celerity

#endif
