#include <algorithm>
#include <deque>
#include <functional>
#include <memory>
#include <tuple>
#include <type_traits>
#include <unordered_set>

#include <catch2/catch_test_macros.hpp>

#include <celerity.h>

#include "distributed_graph_generator.h"
#include "print_graph.h"
#include "test_utils.h"

using namespace celerity;
using namespace celerity::detail;

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
		buffer_access_step discard_write(BufferT& buf, RangeMapper rmfn) {
			return chain<buffer_access_step>([&buf, rmfn](handler& cgh) { buf.template get_access<access_mode::discard_write>(cgh, rmfn); });
		}
	};

  public:
	template <typename Name, int Dims>
	buffer_access_step device_compute(const range<Dims>& global_size) {
		std::deque<action> actions;
		actions.push_front([global_size](handler& cgh) { cgh.parallel_for<Name>(global_size, [](id<Dims>) {}); });
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
		static_assert(
		    ((std::is_same_v<node_id, Filters> || std::is_same_v<task_id, Filters> || std::is_same_v<command_type, Filters>)&&...), "Unsupported filter");

		const auto node_filter = get_optional<node_id>(filters...);
		const auto task_filter = get_optional<task_id>(filters...);
		const auto type_filter = get_optional<command_type>(filters...);
		// TODO: Do we want/need this? Currently IDs are not unique across nodes, which may be (is) confusing.
		// const auto id_filter = get_optional<command_id>(filters...);

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
	dist_cdag_test_context(size_t num_nodes) : m_num_nodes(num_nodes) {
		m_rm = std::make_unique<reduction_manager>();
		m_tm = std::make_unique<task_manager>(num_nodes, nullptr /* host_queue */);
		// m_gser = std::make_unique<graph_serializer>(*m_cdag, m_inspector.get_cb());
		for(node_id nid = 0; nid < num_nodes; ++nid) {
			m_cdags.emplace_back(std::make_unique<command_graph>());
			m_dggens.emplace_back(std::make_unique<distributed_graph_generator>(num_nodes, nid, *m_cdags[nid], *m_tm));
		}
	}

	~dist_cdag_test_context() { maybe_print_graphs(); }

	template <int Dims>
	test_utils::mock_buffer<Dims> create_buffer(cl::sycl::range<Dims> size, bool mark_as_host_initialized = false) {
		const buffer_id bid = m_next_buffer_id++;
		const auto buf = test_utils::mock_buffer<Dims>(bid, size);
		m_tm->add_buffer(bid, range_cast<3>(size), mark_as_host_initialized);
		for(auto& dggen : m_dggens) {
			dggen->add_buffer(bid, range_cast<3>(size));
		}
		return buf;
	}

	template <typename Name = unnamed_kernel, int Dims>
	auto device_compute(const range<Dims>& global_size) {
		return task_builder(*this).device_compute<Name>(global_size);
	}

	command_query query() { return command_query(m_cdags); }

  private:
	size_t m_num_nodes;
	buffer_id m_next_buffer_id = 0;
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
	m_actions.clear();
	return tid;
}

TEST_CASE("FOO", "[bar]") {
	dist_cdag_test_context dctx(2);

	const range<1> test_range = {128};

	auto buf0 = dctx.create_buffer(test_range);
	auto buf1 = dctx.create_buffer(test_range);

	// FIXME: We can't use this for writing as we cannot invert it. Need higher-level mechanism.
	const auto swap_rm = [test_range](chunk<1> chnk) { return subrange<1>{{test_range[0] - chnk.range[0] - chnk.offset[0]}, chnk.range}; };

	dctx.device_compute<class UKN(task_a)>(test_range).discard_write(buf0, ::celerity::access::one_to_one{}).submit();
	const auto tid_b = dctx.device_compute<class UKN(task_b)>(test_range).read(buf0, swap_rm).submit();
	dctx.device_compute<class UKN(task_c)>(test_range).discard_write(buf0, ::celerity::access::one_to_one{}).submit();
	const auto tid_d = dctx.device_compute<class UKN(task_d)>(test_range).read(buf0, swap_rm).submit();

	const auto cmds_b = dctx.query().find_all(tid_b);
	CHECK(cmds_b.count() == 2);
	CHECK(cmds_b.has_type(command_type::execution));

	const auto cmds_d = dctx.query().find_all(tid_d);
	const auto transfers_d = cmds_d.find_predecessors();
	CHECK(transfers_d.count() == 2);
	CHECK(cmds_b.has_successor(transfers_d, dependency_kind::anti_dep));
}
