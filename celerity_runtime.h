#ifndef CELERITY_RUNTIME
#define CELERITY_RUNTIME

#include <functional>
#include <memory>
#include <unordered_map>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/type_index.hpp>
#include <regex>

namespace boost_graph {
using boost::adjacency_list;
using boost::bidirectionalS;
using boost::vecS;

typedef adjacency_list<vecS, vecS, bidirectionalS> Graph;
typedef std::pair<int, int> Edge;
}  // namespace boost_graph

using namespace boost_graph;

namespace cl {
namespace sycl {

struct nd_range {};

struct nd_item {
  size_t get_global() { return 0; }
};

namespace access {

enum class mode { read, write };
}
}  // namespace sycl
}  // namespace cl

namespace celerity {

class buffer;

struct nd_point {};
struct nd_subrange {
  nd_point start;
  nd_point range;
  cl::sycl::nd_range global_size;
};
using range_mapper = std::function<nd_subrange(nd_subrange range)>;

// Convenience range mappers
namespace access {
struct one_to_one {
  nd_subrange operator()(nd_subrange range) const { return range; }
};
}  // namespace access

using kernel_functor = std::function<void(cl::sycl::nd_item)>;

template <cl::sycl::access::mode Mode>
class accessor {
 public:
  accessor(buffer& buf) : buf(buf) {}

  void operator=(float value) const {};
  float& operator[](size_t i) const { return somefloat; }

  buffer& get_buffer() { return buf; }

 private:
  mutable float somefloat = 13.37f;
  buffer& buf;
};

// We have to wrap the SYCL handler to support our count/kernel syntax
// (as opposed to work group size/kernel)
class handler {
 public:
  // TODO naming: execution_handle vs handler?
  template <typename name = class unnamed_task>
  void parallel_for(size_t count, kernel_functor) {
    // TODO: Handle access ranges

    // DEBUG: Find nice name for kernel
    auto qualified_name = boost::typeindex::type_id<name*>().pretty_name();
    std::regex name_regex(R"(.*::(\w+)[\s\*]*)");
    std::smatch matches;
    std::regex_search(qualified_name, matches, name_regex);
    debug_name = matches.size() > 0 ? matches[1] : qualified_name;
  }

  template <cl::sycl::access::mode Mode>
  void require(accessor<Mode> a);

  size_t get_id() { return id; }
  std::string get_debug_name() const { return debug_name; };

 private:
  friend class distr_queue;
  distr_queue& queue;

  static inline size_t instance_count = 0;
  size_t id;

  std::string debug_name;

  handler(distr_queue& q);
};

// Presumably we will have to wrap SYCL buffers as well (as opposed to the code
// samples given in the proposal):
// - We have to assign buffers a unique ID to identify them accross nodes
// - We have to return custom accessors to support the CELERITY range specifiers
class buffer {
 public:
  explicit buffer(size_t size) : size(size), id(instance_count++){};

  template <cl::sycl::access::mode Mode>
  accessor<Mode> get_access(celerity::handler handler, range_mapper) {
    auto a = accessor<Mode>(*this);
    handler.require(a);
    return a;
  };

  size_t get_id() { return id; }

  float operator[](size_t idx) { return 1.f; }

 private:
  static inline size_t instance_count = 0;
  size_t id;
  size_t size;
};

class branch_handle {
 public:
  template <size_t idx>
  void get(buffer){};
};

class distr_queue {
 public:
  void submit(std::function<void(handler& cgh)> cgf);

  // experimental
  // TODO: Can we derive 2nd lambdas args from requested values in 1st?
  void branch(std::function<void(branch_handle& bh)>,
              std::function<void(float)>){};

  void debug_print_task_graph();

 private:
  friend handler;
  // We keep the handlers around to retrieve their name when debug printing
  std::unordered_map<size_t, std::unique_ptr<handler>> handlers;
  std::unordered_map<size_t, size_t> buffer_last_writer;
  Graph task_graph;

  void add_requirement(size_t task_id, size_t buffer_id,
                       cl::sycl::access::mode mode);
};

}  // namespace celerity

#endif
