#ifndef CELERITY_RUNTIME
#define CELERITY_RUNTIME

#include <functional>
#include <memory>
#include <regex>
#include <string>
#include <unordered_map>

#include <SYCL/sycl.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/type_index.hpp>

namespace boost_graph {
using boost::adjacency_list;
using boost::bidirectionalS;
using boost::vecS;

typedef adjacency_list<vecS, vecS, bidirectionalS> Graph;
typedef std::pair<int, int> Edge;
}  // namespace boost_graph

using namespace boost_graph;

namespace celerity {

class buffer;

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

class handler {
 public:
  // TODO naming: execution_handle vs handler?
  template <typename name = class unnamed_task, typename functorT>
  void parallel_for(size_t count, const functorT& kernel) {
    // TODO: Handle access ranges

    sycl_handler.parallel_for<name>(
        cl::sycl::nd_range<1>(cl::sycl::range<1>(count), cl::sycl::range<1>(1)),
        kernel);

    // DEBUG: Find nice name for kernel (regex is probably not super portable)
    auto qualified_name = boost::typeindex::type_id<name*>().pretty_name();
    std::regex name_regex(R"(.*?(?:::)?([\w_]+)\s?\*.*)");
    std::smatch matches;
    std::regex_search(qualified_name, matches, name_regex);
    debug_name = matches.size() > 0 ? matches[1] : qualified_name;
  }

  template <cl::sycl::access::mode Mode>
  void require(accessor<Mode> a, size_t buffer_id);

  size_t get_id() { return id; }
  cl::sycl::handler& get_sycl_handler() { return sycl_handler; }
  std::string get_debug_name() const { return debug_name; };

 private:
  friend class distr_queue;
  distr_queue& queue;
  cl::sycl::handler& sycl_handler;

  static size_t instance_count;
  size_t id;

  std::string debug_name;

  handler(distr_queue& q, cl::sycl::handler& sycl_handler);
};

// TODO: Templatize (data type, dimensions) - see sycl::buffer
class buffer {
 public:
  buffer(float* host_ptr, size_t size)
      : id(instance_count++),
        size(size),
        sycl_buffer(host_ptr, cl::sycl::range<1>(size)){};

  template <cl::sycl::access::mode Mode>
  accessor<Mode> get_access(celerity::handler handler, range_mapper) {
    auto a = accessor<Mode>(sycl_buffer, handler.get_sycl_handler());
    handler.require(a, id);
    return a;
  };

  size_t get_id() { return id; }

  float operator[](size_t idx) { return 1.f; }

 private:
  static size_t instance_count;
  size_t id;
  size_t size;

  cl::sycl::buffer<float, 1> sycl_buffer;
};

class branch_handle {
 public:
  template <size_t idx>
  void get(buffer){};
};

class distr_queue {
 public:
  // TODO: Device should be selected transparently
  distr_queue(cl::sycl::device device);
  void submit(std::function<void(handler& cgh)> cgf);

  // experimental
  // TODO: Can we derive 2nd lambdas args from requested values in 1st?
  void branch(std::function<void(branch_handle& bh)>,
              std::function<void(float)>){};

  void debug_print_task_graph();

 private:
  friend handler;
  std::unordered_map<size_t, std::string> task_names;
  std::unordered_map<size_t, size_t> buffer_last_writer;
  Graph task_graph;

  cl::sycl::queue sycl_queue;

  void add_requirement(size_t task_id, size_t buffer_id,
                       cl::sycl::access::mode mode);
};

}  // namespace celerity

#endif
