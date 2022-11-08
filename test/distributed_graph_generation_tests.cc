#include <algorithm>
#include <deque>
#include <functional>
#include <memory>
#include <tuple>
#include <type_traits>
#include <unordered_set>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include <celerity.h>

#include "cool_region_map.h"
#include "distributed_graph_generator.h"
#include "print_graph.h"
#include "test_utils.h"

using namespace celerity;
using namespace celerity::detail;

namespace acc = celerity::access;

class dist_cdag_test_context;

class task_builder {
	friend class dist_cdag_test_context;

	using action = std::function<void(handler&)>;

	class step {
	  public:
		step(dist_cdag_test_context& dctx, std::deque<action> actions) : m_dctx(dctx), m_actions(std::move(actions)) {}
		virtual ~step() noexcept(false);

		task_id submit();

		step(const step&) = delete;

	  private:
		dist_cdag_test_context& m_dctx;
		std::deque<action> m_actions;

	  protected:
		template <typename StepT>
		StepT chain(action a) {
			static_assert(std::is_base_of_v<step, StepT>);
			m_actions.push_front(a);
			return StepT{m_dctx, std::move(m_actions)};
		}
	};

	class buffer_access_step : public step {
	  public:
		buffer_access_step(dist_cdag_test_context& dctx, std::deque<action> actions) : step(dctx, std::move(actions)) {}

		buffer_access_step(const buffer_access_step&) = delete;

		template <typename BufferT, typename RangeMapper>
		buffer_access_step read(BufferT& buf, RangeMapper rmfn) {
			return chain<buffer_access_step>([&buf, rmfn](handler& cgh) { buf.template get_access<access_mode::read>(cgh, rmfn); });
		}

		template <typename BufferT, typename RangeMapper>
		buffer_access_step read_write(BufferT& buf, RangeMapper rmfn) {
			return chain<buffer_access_step>([&buf, rmfn](handler& cgh) { buf.template get_access<access_mode::read_write>(cgh, rmfn); });
		}

		template <typename BufferT, typename RangeMapper>
		buffer_access_step discard_write(BufferT& buf, RangeMapper rmfn) {
			return chain<buffer_access_step>([&buf, rmfn](handler& cgh) { buf.template get_access<access_mode::discard_write>(cgh, rmfn); });
		}

		// FIXME: Misnomer (not a "buffer access")
		template <typename HostObjT>
		buffer_access_step affect(HostObjT& host_obj) {
			return chain<buffer_access_step>([&host_obj](handler& cgh) { host_obj.add_side_effect(cgh, experimental::side_effect_order::sequential); });
		}
	};

  public:
	template <typename Name, int Dims>
	buffer_access_step device_compute(const range<Dims>& global_size) {
		std::deque<action> actions;
		actions.push_front([global_size](handler& cgh) { cgh.parallel_for<Name>(global_size, [](id<Dims>) {}); });
		return buffer_access_step(m_dctx, std::move(actions));
	}

	template <typename Name, int Dims>
	buffer_access_step device_compute(const nd_range<Dims>& nd_range) {
		std::deque<action> actions;
		actions.push_front([nd_range](handler& cgh) { cgh.parallel_for<Name>(nd_range, [](nd_item<Dims>) {}); });
		return buffer_access_step(m_dctx, std::move(actions));
	}

	template <int Dims>
	buffer_access_step host_task(const range<Dims>& global_size) {
		std::deque<action> actions;
		actions.push_front([global_size](handler& cgh) { cgh.host_task(global_size, [](partition<Dims>) {}); });
		return buffer_access_step(m_dctx, std::move(actions));
	}

  private:
	dist_cdag_test_context& m_dctx;

	task_builder(dist_cdag_test_context& dctx) : m_dctx(dctx) {}
};

class command_query {
	friend class dist_cdag_test_context;

	class query_exception : public std::runtime_error {
		using std::runtime_error::runtime_error;
	};

  public:
	// TODO Other ideas:
	// - remove(filters...)						=> Remove all commands matching the filters
	// - executes(global_size)					=> Check that commands execute a given global size (exactly? at least? ...)
	// - writes(buffer_id, subrange) 			=> Check that commands write a given buffer subrange (exactly? at least? ...)
	// - find_one(filters...)					=> Throws if result set contains more than 1 (per node?)

	template <typename... Filters>
	command_query find_all(Filters... filters) const {
		static_assert(((std::is_same_v<node_id, Filters> || std::is_same_v<task_id, Filters> || std::is_same_v<command_type, Filters>
		                  || std::is_same_v<command_id, Filters>)&&...),
		    "Unsupported filter");

		const auto node_filter = get_optional<node_id>(filters...);
		const auto task_filter = get_optional<task_id>(filters...);
		const auto type_filter = get_optional<command_type>(filters...);
		// Note that command ids are not unique across nodes!
		const auto id_filter = get_optional<command_id>(filters...);

		std::vector<std::unordered_set<const abstract_command*>> filtered(m_commands_by_node.size());
		for(node_id nid = 0; nid < m_commands_by_node.size(); ++nid) {
			if(node_filter.has_value() && *node_filter != nid) continue;
			for(const auto* cmd : m_commands_by_node[nid]) {
				if(task_filter.has_value()) {
					if(!isa<task_command>(cmd)) continue;
					if(static_cast<const task_command*>(cmd)->get_tid() != *task_filter) continue;
				}
				if(type_filter.has_value()) {
					if(get_type(cmd) != *type_filter) continue;
				}
				if(id_filter.has_value()) {
					if(cmd->get_cid() != id_filter) continue;
				}
				filtered[nid].insert(cmd);
			}
		}

		return command_query{std::move(filtered)};
	}

	template <typename... Filters>
	command_query find_predecessors(Filters... filters) const {
		return find_adjacent(true, filters...);
	}

	template <typename... Filters>
	command_query find_successors(Filters... filters) const {
		return find_adjacent(false, filters...);
	}

	size_t count() const {
		return std::accumulate(
		    m_commands_by_node.begin(), m_commands_by_node.end(), size_t(0), [](size_t current, auto& cmds) { return current + cmds.size(); });
	}

	bool empty() const { return count() == 0; }

	command_query subtract(const command_query& other) const {
		assert(m_commands_by_node.size() == other.m_commands_by_node.size());
		std::vector<std::unordered_set<const abstract_command*>> result(m_commands_by_node.size());
		for(node_id nid = 0; nid < m_commands_by_node.size(); ++nid) {
			std::copy_if(m_commands_by_node[nid].cbegin(), m_commands_by_node[nid].cend(), std::inserter(result[nid], result[nid].begin()),
			    [&other, nid](const abstract_command* cmd) { return other.m_commands_by_node[nid].count(cmd) == 0; });
		}
		return command_query{std::move(result)};
	}

	// Call the provided function once for each node, with a subquery containing commands only for that node.
	template <typename PerNodeCallback>
	void for_each_node(PerNodeCallback&& cb) const {
		for(node_id nid = 0; nid < m_commands_by_node.size(); ++nid) {
			UNSCOPED_INFO(fmt::format("On node {}", nid));
			cb(find_all(nid));
		}
	}

	// Call the provided function once for each command, with a subquery only containing that command.
	template <typename PerCmdCallback>
	void for_each_command(PerCmdCallback&& cb) const {
		for(node_id nid = 0; nid < m_commands_by_node.size(); ++nid) {
			for(auto* cmd : m_commands_by_node[nid]) {
				UNSCOPED_INFO(fmt::format("Command {} on node {}", nid, cmd->get_cid()));
				// We also need to filter by node here, as command ids are not globally unique!
				cb(find_all(nid, cmd->get_cid()));
			}
		}
	}

	// TODO: Use plural 'have_type'? Have both but singular throws if count > 1?
	bool has_type(const command_type expected) const {
		return for_all_commands([expected](const node_id nid, const abstract_command* cmd) {
			const auto received = get_type(cmd);
			if(received != expected) {
				UNSCOPED_INFO(fmt::format("Expected command {} on node {} to have type '{}' but found type '{}'", cmd->get_nid(), nid, get_type_name(expected),
				    get_type_name(received)));
				return false;
			}
			return true;
		});
	}

	bool has_successor(const command_query& successors, const std::optional<dependency_kind>& kind = std::nullopt) const {
		return for_all_commands([&successors, &kind](const node_id nid, const abstract_command* cmd) {
			for(const auto* expected : successors.m_commands_by_node[nid]) {
				bool found = false;
				for(const auto received : cmd->get_dependents()) {
					if(received.node == expected) {
						found = true;
						if(kind.has_value() && received.kind != *kind) {
							UNSCOPED_INFO(fmt::format("Expected command {} on node {} to have successor {} with kind {}, but found kind {}", cmd->get_cid(),
							    nid, expected->get_cid(), *kind, received.kind));
							return false;
						}
					}
				}
				if(!found) {
					UNSCOPED_INFO(fmt::format("Expected command {} on node {} to have successor {}", cmd->get_cid(), nid, expected->get_cid()));
					return false;
				}
			}
			return true;
		});
	}

	std::vector<const abstract_command*> get_raw(const node_id nid) const {
		std::vector<const abstract_command*> result;
		std::copy(m_commands_by_node.at(nid).cbegin(), m_commands_by_node.at(nid).cend(), std::back_inserter(result));
		return result;
	}

  private:
	std::vector<std::unordered_set<const abstract_command*>> m_commands_by_node;

	// Constructor for initial top-level query (containing all commands)
	command_query(const std::vector<std::unique_ptr<command_graph>>& cdags) {
		for(auto& cdag : cdags) {
			m_commands_by_node.push_back({cdag->all_commands().begin(), cdag->all_commands().end()});
		}
	}

	// Constructor for narrowed-down queries
	command_query(std::vector<std::unordered_set<const abstract_command*>> commands_by_node) : m_commands_by_node(std::move(commands_by_node)) {}

	template <typename Callback>
	bool for_all_commands(Callback&& cb) const {
		bool cont = true;
		for(node_id nid = 0; cont && nid < m_commands_by_node.size(); ++nid) {
			for(const auto* cmd : m_commands_by_node[nid]) {
				if constexpr(std::is_invocable_r_v<bool, Callback, node_id, decltype(cmd)>) {
					cont &= cb(nid, cmd);
				} else {
					cb(nid, cmd);
				}
				if(!cont) break;
			}
		}
		return cont;
	}

	template <typename... Filters>
	command_query find_adjacent(const bool find_predecessors, Filters... filters) const {
		const auto kind_filter = get_optional<dependency_kind>(filters...);

		std::vector<std::unordered_set<const abstract_command*>> adjacent(m_commands_by_node.size());
		for_all_commands([&adjacent, find_predecessors, kind_filter](const node_id nid, const abstract_command* cmd) {
			const auto iterable = find_predecessors ? cmd->get_dependencies() : cmd->get_dependents();
			for(auto it = iterable.begin(); it != iterable.end(); ++it) {
				if(kind_filter.has_value() && it->kind != *kind_filter) continue;
				adjacent[nid].insert(it->node);
			}
		});

		const auto query = command_query{std::move(adjacent)};
		// Filter resulting set of commands, but remove dependency_kind filter (if present)
		// TODO: Refactor into generic utility
		const auto filters_tuple = std::tuple{filters...};
		constexpr auto idx = get_index_of<dependency_kind>(filters_tuple);
		if constexpr(idx != -1) {
			const auto filters_without_kind = tuple_splice<size_t(idx), 1>(filters_tuple);
			return std::apply([&query](auto... fs) { return query.find_all(fs...); }, filters_without_kind);
		} else {
			return query.find_all(filters...);
		}
	}

	template <typename T, typename... Ts>
	static constexpr std::optional<T> get_optional(const std::tuple<Ts...>& tuple) {
		if constexpr((std::is_same_v<T, Ts> || ...)) { return std::get<T>(tuple); }
		return std::nullopt;
	}

	template <typename T, typename... Ts>
	static constexpr std::optional<T> get_optional(Ts... ts) {
		return get_optional<T>(std::tuple(ts...));
	}

	// TODO: Move to utils header?

	template <typename T, size_t I = 0, typename Tuple>
	static constexpr int64_t get_index_of(const Tuple& t) {
		if constexpr(I >= std::tuple_size_v<Tuple>) {
			return -1;
		} else if(std::is_same_v<T, std::tuple_element_t<I, Tuple>>) {
			return I;
		} else {
			return get_index_of<T, I + 1>(t);
		}
	}

	template <size_t Offset, size_t Count, size_t... Prefix, size_t... Suffix, typename Tuple>
	static constexpr auto tuple_splice_impl(std::index_sequence<Prefix...>, std::index_sequence<Suffix...>, const Tuple& t) {
		return std::tuple_cat(std::tuple{std::get<Prefix>(t)...}, std::tuple{std::get<Offset + Count + Suffix>(t)...});
	}

	template <size_t Offset, size_t Count, typename Tuple>
	static constexpr auto tuple_splice(const Tuple& t) {
		constexpr size_t N = std::tuple_size_v<Tuple>;
		static_assert(Offset + Count <= N);
		return tuple_splice_impl<Offset, Count>(std::make_index_sequence<Offset>{}, std::make_index_sequence<N - Count - Offset>{}, t);
	}

	static command_type get_type(const abstract_command* cmd) {
		if(isa<epoch_command>(cmd)) return command_type::epoch;
		if(isa<horizon_command>(cmd)) return command_type::horizon;
		if(isa<execution_command>(cmd)) return command_type::execution;
		if(isa<data_request_command>(cmd)) return command_type::data_request;
		if(isa<push_command>(cmd)) return command_type::push;
		if(isa<await_push_command>(cmd)) return command_type::await_push;
		if(isa<reduction_command>(cmd)) return command_type::reduction;
		throw query_exception("Unknown command type");
	}

	static std::string get_type_name(const command_type type) {
		switch(type) {
		case command_type::epoch: return "epoch";
		case command_type::horizon: return "horizon";
		case command_type::execution: return "execution";
		case command_type::data_request: return "data_request";
		case command_type::push: return "push";
		case command_type::await_push: return "await_push";
		case command_type::reduction: return "reduction";
		default: return "<unknown>";
		}
	}
};

class dist_cdag_test_context {
	friend class task_builder;

  public:
	dist_cdag_test_context(size_t num_nodes, size_t devices_per_node = 1) : m_num_nodes(num_nodes) {
		m_rm = std::make_unique<reduction_manager>();
		m_tm = std::make_unique<task_manager>(num_nodes, nullptr /* host_queue */);
		// m_gser = std::make_unique<graph_serializer>(*m_cdag, m_inspector.get_cb());
		for(node_id nid = 0; nid < num_nodes; ++nid) {
			m_cdags.emplace_back(std::make_unique<command_graph>());
			m_dggens.emplace_back(std::make_unique<distributed_graph_generator>(num_nodes, devices_per_node, nid, *m_cdags[nid], *m_tm));
		}
	}

	~dist_cdag_test_context() { maybe_print_graphs(); }

	template <int Dims>
	test_utils::mock_buffer<Dims> create_buffer(range<Dims> size, bool mark_as_host_initialized = false) {
		const buffer_id bid = m_next_buffer_id++;
		const auto buf = test_utils::mock_buffer<Dims>(bid, size);
		m_tm->add_buffer(bid, range_cast<3>(size), mark_as_host_initialized);
		for(auto& dggen : m_dggens) {
			dggen->add_buffer(bid, range_cast<3>(size), Dims);
		}
		return buf;
	}

	test_utils::mock_host_object create_host_object() { return test_utils::mock_host_object{m_next_host_object_id++}; }

	template <typename Name = unnamed_kernel, int Dims>
	auto device_compute(const range<Dims>& global_size) {
		return task_builder(*this).device_compute<Name>(global_size);
	}

	template <typename Name = unnamed_kernel, int Dims>
	auto device_compute(const nd_range<Dims>& nd_range) {
		return task_builder(*this).device_compute<Name>(nd_range);
	}

	template <int Dims>
	auto host_task(const range<Dims>& global_size) {
		return task_builder(*this).host_task(global_size);
	}

	command_query query() { return command_query(m_cdags); }

	void set_horizon_step(const int step) { m_tm->set_horizon_step(step); }

	distributed_graph_generator& get_graph_generator(node_id nid) { return *m_dggens.at(nid); }

  private:
	size_t m_num_nodes;
	buffer_id m_next_buffer_id = 0;
	host_object_id m_next_host_object_id = 0;
	std::optional<task_id> m_most_recently_built_horizon;
	std::unique_ptr<reduction_manager> m_rm;
	std::unique_ptr<task_manager> m_tm;
	std::vector<std::unique_ptr<command_graph>> m_cdags;
	std::vector<std::unique_ptr<distributed_graph_generator>> m_dggens;

	task_manager& get_task_manager() { return *m_tm; }

	void build_task(const task_id tid) {
		for(auto& dggen : m_dggens) {
			dggen->build_task(*m_tm->get_task(tid));
		}
	}

	void maybe_build_horizon() {
		const auto current_horizon = task_manager_testspy::get_current_horizon(*m_tm);
		if(m_most_recently_built_horizon != current_horizon) {
			assert(current_horizon.has_value());
			build_task(*current_horizon);
		}
		m_most_recently_built_horizon = current_horizon;
	}

	void maybe_print_graphs() {
		if(test_utils::print_graphs) {
			test_utils::maybe_print_graph(*m_tm);

			std::vector<std::string> graphs;
			for(node_id nid = 0; nid < m_num_nodes; ++nid) {
				const auto& cdag = m_cdags[nid];
				const auto graph = cdag->print_graph(nid, std::numeric_limits<size_t>::max(), *m_tm, nullptr);
				assert(graph.has_value());
				graphs.push_back(*graph);
			}
			CELERITY_INFO("Command graph:\n\n{}\n", combine_command_graphs(graphs));
		}
	}
};

task_builder::step::~step() noexcept(false) {
	if(!m_actions.empty()) { throw std::runtime_error("Found incomplete task build. Did you forget to call submit()?"); }
}

task_id task_builder::step::submit() {
	assert(!m_actions.empty());
	const auto tid = m_dctx.get_task_manager().submit_command_group([this](handler& cgh) {
		while(!m_actions.empty()) {
			auto a = m_actions.front();
			a(cgh);
			m_actions.pop_front();
		}
	});
	m_dctx.build_task(tid);
	m_dctx.maybe_build_horizon();
	m_actions.clear();
	return tid;
}

TEST_CASE("distributed push-model hello world", "[NOCOMMIT][dist-ggen]") {
	dist_cdag_test_context dctx(2);

	const range<1> test_range = {128};

	auto buf0 = dctx.create_buffer(test_range);
	auto buf1 = dctx.create_buffer(test_range);

	// FIXME: We can't use this for writing as we cannot invert it. Need higher-level mechanism.
	const auto swap_rm = [test_range](chunk<1> chnk) { return subrange<1>{{test_range[0] - chnk.range[0] - chnk.offset[0]}, chnk.range}; };

	const auto tid_a = dctx.device_compute<class UKN(task_a)>(test_range).discard_write(buf0, acc::one_to_one{}).submit();
	const auto tid_b = dctx.device_compute<class UKN(task_b)>(test_range).read(buf0, swap_rm).submit();
	const auto tid_c = dctx.device_compute<class UKN(task_c)>(test_range).discard_write(buf0, acc::one_to_one{}).submit();
	const auto tid_d = dctx.device_compute<class UKN(task_d)>(test_range).read(buf0, swap_rm).submit();

	const auto cmds_b = dctx.query().find_all(tid_b);
	CHECK(cmds_b.count() == 2);
	CHECK(cmds_b.has_type(command_type::execution));

	const auto pushes_c = dctx.query().find_all(tid_a).find_successors(dependency_kind::true_dep);
	CHECK(pushes_c.count() == 2);
	CHECK(pushes_c.has_type(command_type::push));

	const auto pushes_d = dctx.query().find_all(tid_c).find_successors(dependency_kind::true_dep);
	CHECK(pushes_d.count() == 2);
	CHECK(pushes_d.has_type(command_type::push));

	const auto cmds_d = dctx.query().find_all(tid_d);
	const auto await_pushes_d = cmds_d.find_predecessors();
	CHECK(await_pushes_d.count() == 2);
	CHECK(await_pushes_d.has_type(command_type::await_push));
	CHECK(cmds_b.has_successor(await_pushes_d, dependency_kind::anti_dep));
}

TEST_CASE("don't treat replicated data as owned", "[regression][dist-ggen]") {
	dist_cdag_test_context dctx(2);

	const range<2> test_range = {128, 128};

	auto buf0 = dctx.create_buffer(test_range);
	auto buf1 = dctx.create_buffer(test_range);

	const auto tid_a = dctx.device_compute<class UKN(task_a)>(test_range).discard_write(buf0, acc::one_to_one{}).submit();
	const auto tid_b = dctx.device_compute<class UKN(task_b)>(test_range).read(buf0, acc::slice<2>{0}).discard_write(buf1, acc::one_to_one{}).submit();

	const auto pushes = dctx.query().find_all(command_type::push);
	CHECK(pushes.count() == 2);
	// Regression: Node 0 assumed that it owned the data it got pushed by node 1 for its chunk, so it generated a push for node 1's chunk.
	pushes.for_each_node([](const auto& q) { CHECK(q.count() == 1); });
}

// NOCOMMIT TODO: Test that same transfer isn't being generated twice!!

TEST_CASE("a single await push command can await multiple pushes", "[dist-ggen]") {
	dist_cdag_test_context dctx(3);

	const range<1> test_range = {128};

	auto buf0 = dctx.create_buffer(test_range);

	const auto tid_a = dctx.device_compute<class UKN(task_a)>(test_range).discard_write(buf0, acc::one_to_one{}).submit();
	const auto tid_b = dctx.device_compute<class UKN(task_b)>(test_range).read(buf0, acc::all{}).submit();
	dctx.query().find_all(command_type::await_push).for_each_node([](const auto& q) { CHECK(q.count() == 1); });
	dctx.query().find_all(command_type::push).for_each_node([](const auto& q) { CHECK(q.count() == 2); });
}

TEST_CASE("data owners generate separate push commands for each last writer command", "[dist-ggen]") {
	// TODO: Add this test to document the current behavior. OR: Make it a !shouldfail and check for a single command?
}

// TODO: Move?
namespace celerity::detail {

// FIXME: Duplicated from graph_compaction_tests
struct region_map_testspy {
	template <typename T>
	static size_t get_num_regions(const region_map<T>& map) {
		return map.m_region_values.size();
	}
	template <typename T>
	static size_t get_num_regions(const my_cool_region_map_wrapper<T>& map) {
		switch(map.dims) {
		case 1: return std::get<1>(map.region_map).get_num_regions();
		case 2: return std::get<2>(map.region_map).get_num_regions();
		case 3: return std::get<3>(map.region_map).get_num_regions();
		};
		return -1;
	}
};

struct distributed_graph_generator_testspy {
	static size_t get_last_writer_num_regions(const distributed_graph_generator& dggen, const buffer_id bid) {
		return region_map_testspy::get_num_regions(dggen.m_buffer_states.at(bid).local_last_writer);
	}

	static size_t get_command_buffer_reads_size(const distributed_graph_generator& dggen) { return dggen.m_command_buffer_reads.size(); }
};
} // namespace celerity::detail

// This started out as a port of "horizons prevent number of regions from growing indefinitely", but was then changed (and simplified) considerably
TEST_CASE("horizons prevent tracking data structures from growing indefinitely", "[horizon][command-graph]") {
	constexpr int num_timesteps = 100;

	dist_cdag_test_context dctx(1);
	const size_t buffer_width = 300;
	auto buf_a = dctx.create_buffer(range<2>(num_timesteps, buffer_width));

	const int horizon_step_size = GENERATE(values({1, 2, 3}));
	CAPTURE(horizon_step_size);

	dctx.set_horizon_step(horizon_step_size);

	for(int t = 0; t < num_timesteps; ++t) {
		CAPTURE(t);
		const auto read_accessor = [=](celerity::chunk<1> chnk) {
			celerity::subrange<2> ret;
			ret.range = range<2>(t, buffer_width);
			ret.offset = id<2>(0, 0);
			return ret;
		};
		const auto write_accessor = [=](celerity::chunk<1> chnk) {
			celerity::subrange<2> ret;
			ret.range = range<2>(1, buffer_width);
			ret.offset = id<2>(t, 0);
			return ret;
		};
		dctx.device_compute<class UKN(timestep)>(range<1>(buffer_width)).read(buf_a, read_accessor).discard_write(buf_a, write_accessor).submit();

		auto& ggen = dctx.get_graph_generator(0);

		// Assert once we've reached steady state as to not overcomplicate things
		if(t > 2 * horizon_step_size) {
			const auto num_regions = distributed_graph_generator_testspy::get_last_writer_num_regions(ggen, buf_a.get_id());
			const size_t cmds_before_applied_horizon = 1;
			const size_t cmds_after_applied_horizon = horizon_step_size + ((t + 1) % horizon_step_size);
			REQUIRE_LOOP(num_regions == cmds_before_applied_horizon + cmds_after_applied_horizon);

			// Pruning happens with a one step delay after a horizon has been applied
			const size_t expected_reads = horizon_step_size + (t % horizon_step_size) + 1;
			const size_t reads_per_timestep = t > 0 ? 1 : 0;
			REQUIRE_LOOP(distributed_graph_generator_testspy::get_command_buffer_reads_size(ggen) == expected_reads);
		}

		REQUIRE_LOOP(dctx.query().find_all(command_type::horizon).count() <= 3);
	}
}

TEST_CASE("the same buffer range is not pushed twice") {
	dist_cdag_test_context dctx(2);
	auto buf1 = dctx.create_buffer(range<1>(128));

	dctx.device_compute<class UKN(task_a)>(buf1.get_range()).discard_write(buf1, acc::one_to_one{}).submit();
	dctx.device_compute<class UKN(task_b)>(buf1.get_range()).read(buf1, acc::fixed<1>({{0}, {32}})).submit();

	const auto pushes_b = dctx.query().find_all(command_type::push);
	CHECK(pushes_b.count() == 1);

	SECTION("when requesting the exact same range") {
		dctx.device_compute<class UKN(task_c)>(buf1.get_range()).read(buf1, acc::fixed<1>({{0}, {32}})).submit();
		const auto pushes_c = dctx.query().find_all(command_type::push).subtract(pushes_b);
		CHECK(pushes_c.empty());
	}

	SECTION("when requesting a partially overlapping range") {
		dctx.device_compute<class UKN(task_c)>(buf1.get_range()).read(buf1, acc::fixed<1>({{0}, {64}})).submit();
		const auto pushes_c = dctx.query().find_all(command_type::push).subtract(pushes_b);
		REQUIRE(pushes_c.count() == 1);
		const auto push_cmd = dynamic_cast<const push_command*>(pushes_c.get_raw(0)[0]);
		CHECK(subrange_cast<1>(push_cmd->get_range()) == subrange<1>({32}, {32}));
	}
}

// TODO: I've removed an assertion in this same commit that caused problems when determining anti-dependencies for read-write accesses.
// The difference to master-worker is that we now update the local_last_writer directly instead of using an update list when generating an await push.
// This means that the write access finds the await push command as the last writer when determining anti-dependencies, which we then simply skip.
// This should be fine as it already has a true dependency on it anyway (and the await push already has all transitive anti-dependencies it needs).
// The order of processing accesses (producer/consumer) shouldn't matter either as we do defer the update of the last writer for the actual
// execution command until all modes have been processed (i.e., we won't forget to generate the await push).
// Go through this again and see if everything works as expected (in particular with multiple chunks).
TEST_CASE("read_write access works", "[smoke-test]") {
	dist_cdag_test_context dctx(2);

	auto buf = dctx.create_buffer(range<1>(128));

	dctx.device_compute(buf.get_range()).discard_write(buf, acc::one_to_one{}).submit();
	dctx.device_compute(buf.get_range()).read_write(buf, acc::fixed<1>({{0, 64}})).submit();
}

// NOCOMMIT TODO: Test intra-task anti-dependencies

TEST_CASE("side effect dependencies") {
	dist_cdag_test_context dctx(1);
	auto hobj = dctx.create_host_object();
	const auto tid_a = dctx.host_task(range<1>(1)).affect(hobj).submit();
	const auto tid_b = dctx.host_task(range<1>(1)).affect(hobj).submit();
	CHECK(dctx.query().find_all(tid_a).has_successor(dctx.query().find_all(tid_b)));
	// NOCOMMIT TODO: Test horizon / epoch subsumption as well
}

TEST_CASE("reading from host-initialized or uninitialized buffers doesn't generate faulty await push commands") {
	const int num_nodes = GENERATE(1, 2); // (We used to generate an await push even for a single node!)
	dist_cdag_test_context dctx(num_nodes);

	const auto test_range = range<1>(128);
	auto host_init_buf = dctx.create_buffer(test_range, true);
	auto uninit_buf = dctx.create_buffer(test_range, false);
	const auto tid_a = dctx.device_compute(test_range).read(host_init_buf, acc::one_to_one{}).read(uninit_buf, acc::one_to_one{}).submit();
	CHECK(dctx.query().find_all(command_type::await_push).empty());
	CHECK(dctx.query().find_all(command_type::push).empty());
}

// Regression test
TEST_CASE("overlapping read/write access to the same buffer doesn't generate intra-task dependencies between chunks on the same worker") {
	dist_cdag_test_context dctx(1, 2);

	const auto test_range = range<1>(128);
	auto buf = dctx.create_buffer(test_range, true);
	dctx.device_compute(test_range).read(buf, acc::neighborhood<1>{1}).discard_write(buf, acc::one_to_one{}).submit();
	CHECK(dctx.query().find_all(command_type::execution).count() == 2);
	dctx.query().find_all(command_type::execution).for_each_command([](const auto& q) {
		// Both commands should only depend on initial epoch, not each other
		CHECK(q.find_predecessors().count() == 1);
	});
}

TEST_CASE("local chunks can create multiple await push commands for a single push") {
	dist_cdag_test_context dctx(2, 2);
	const auto test_range = range<1>(128);
	auto buf = dctx.create_buffer(test_range);
	const auto transpose = [](celerity::chunk<1> chnk) { return celerity::subrange<1>(chnk.global_size[0] - chnk.offset[0] - chnk.range[0], chnk.range); };

	// Since we currently create a separate push command for each last writer command, this just happens to work out.
	SECTION("this works by accident") {
		dctx.device_compute(test_range).discard_write(buf, acc::one_to_one{}).submit();
		dctx.device_compute(test_range).read(buf, transpose).submit();
		CHECK(dctx.query().find_all(command_type::push).count() == dctx.query().find_all(command_type::await_push).count());
	}

	SECTION("this is what we actually wanted") {
		// Prevent additional local chunks from being created by using nd_range
		dctx.device_compute(nd_range<1>(test_range, {64})).discard_write(buf, acc::one_to_one{}).submit();
		dctx.device_compute(test_range).read(buf, transpose).submit();

		// NOCOMMIT TODO: If would be sweet if we could somehow get the union region across all await pushes and check it against the corresponding push
		CHECK(dctx.query().find_all(node_id(0), command_type::push).count() == 1);
		CHECK(dctx.query().find_all(node_id(1), command_type::await_push).count() == 2);
		CHECK(dctx.query().find_all(node_id(1), command_type::push).count() == 1);
		CHECK(dctx.query().find_all(node_id(0), command_type::await_push).count() == 2);
	}
}