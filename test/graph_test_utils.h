#pragma once

#include <deque>
#include <exception>
#include <functional>

#include "test_utils.h"

using namespace celerity;
using namespace celerity::detail;

namespace celerity::test_utils {

class tdag_test_context;
class cdag_test_context;
class idag_test_context;
class scheduler_test_context;

template <typename TestContext>
class task_builder {
	friend class tdag_test_context;
	friend class cdag_test_context;
	friend class idag_test_context;
	friend class scheduler_test_context;

	using action = std::function<void(handler&)>;

	class step {
	  public:
		step(TestContext& tctx, action command, std::vector<action> requirements = {})
		    : m_tctx(tctx), m_command(std::move(command)), m_requirements(std::move(requirements)), m_uncaught_exceptions_before(std::uncaught_exceptions()) {}

		~step() noexcept(false) { // NOLINT(bugprone-exception-escape)
			if(std::uncaught_exceptions() == m_uncaught_exceptions_before && (m_command || !m_requirements.empty())) {
				throw std::runtime_error("Found incomplete task build. Did you forget to call submit()?");
			}
		}

		step(const step&) = delete;
		step(step&&) = delete;
		step& operator=(const step&) = delete;
		step& operator=(step&&) = delete;

		task_id submit() {
			assert(m_command);
			const auto tid = m_tctx.submit_command_group([this](handler& cgh) {
				for(auto& a : m_requirements) {
					a(cgh);
				}
				m_command(cgh);
			});
			m_command = {};
			m_requirements = {};
			return tid;
		}

		step name(const std::string& name) {
			return chain<step>([&name](handler& cgh) { celerity::debug::set_task_name(cgh, name); });
		}

		template <typename BufferT, typename RangeMapper>
		step read(BufferT& buf, RangeMapper rmfn) {
			return chain<step>([&buf, rmfn](handler& cgh) { buf.template get_access<access_mode::read>(cgh, rmfn); });
		}

		template <typename BufferT, typename RangeMapper>
		step read_write(BufferT& buf, RangeMapper rmfn) {
			return chain<step>([&buf, rmfn](handler& cgh) { buf.template get_access<access_mode::read_write>(cgh, rmfn); });
		}

		template <typename BufferT, typename RangeMapper>
		step write(BufferT& buf, RangeMapper rmfn) {
			return chain<step>([&buf, rmfn](handler& cgh) { buf.template get_access<access_mode::write>(cgh, rmfn); });
		}

		template <typename BufferT, typename RangeMapper>
		step discard_write(BufferT& buf, RangeMapper rmfn) {
			return chain<step>([&buf, rmfn](handler& cgh) { buf.template get_access<access_mode::discard_write>(cgh, rmfn); });
		}

		template <typename BufferT>
		inline step reduce(BufferT& buf, const bool include_current_buffer_value) {
			return chain<step>([this, &buf, include_current_buffer_value](
			                       handler& cgh) { add_reduction(cgh, m_tctx.create_reduction(buf.get_id(), include_current_buffer_value)); });
		}

		template <typename HostObjT>
		step affect(HostObjT& host_obj, experimental::side_effect_order order = experimental::side_effect_order::sequential) {
			return chain<step>([&host_obj, order](handler& cgh) { host_obj.add_side_effect(cgh, order); });
		}

		template <int Dims>
		step constrain_split(const range<Dims>& constraint) {
			return chain<step>([constraint](handler& cgh) { experimental::constrain_split(cgh, constraint); });
		}

		template <typename Hint>
		step hint(Hint hint) {
			return chain<step>([&hint](handler& cgh) { experimental::hint(cgh, hint); });
		}

		template <typename Hint>
		step hint_if(const bool condition, Hint hint) {
			return chain<step>([condition, &hint](handler& cgh) {
				if(condition) { experimental::hint(cgh, hint); }
			});
		}

	  private:
		TestContext& m_tctx;
		action m_command;
		std::vector<action> m_requirements;
		int m_uncaught_exceptions_before;

		template <typename StepT>
		StepT chain(action a) {
			static_assert(std::is_base_of_v<step, StepT>);
			// move constructing a std::function doesn't guarantee that the source is empty afterwards
			auto requirements = std::move(m_requirements);
			requirements.push_back(std::move(a));
			auto command = std::move(m_command);
			m_requirements = {};
			m_command = {};
			return StepT{m_tctx, std::move(command), std::move(requirements)};
		}
	};

  public:
	template <typename Name, int Dims>
	step device_compute(const range<Dims>& global_size, const id<Dims>& global_offset) {
		return step(m_tctx, [global_size, global_offset](handler& cgh) { cgh.parallel_for<Name>(global_size, global_offset, [](id<Dims>) {}); });
	}

	template <typename Name, int Dims>
	step device_compute(const nd_range<Dims>& execution_range) {
		return step(m_tctx, [execution_range](handler& cgh) { cgh.parallel_for<Name>(execution_range, [](nd_item<Dims>) {}); });
	}

	template <int Dims>
	step host_task(const range<Dims>& global_size) {
		return step(m_tctx, [global_size](handler& cgh) { cgh.host_task(global_size, [](partition<Dims>) {}); });
	}

	step master_node_host_task() {
		return step(m_tctx, [](handler& cgh) { cgh.host_task(on_master_node, [] {}); });
	}

	step collective_host_task(experimental::collective_group group) {
		return step(m_tctx, [group](handler& cgh) { cgh.host_task(experimental::collective(group), [](const experimental::collective_partition&) {}); });
	}

  private:
	TestContext& m_tctx;

	task_builder(TestContext& cctx) : m_tctx(cctx) {}
};

// In lieu of adding yet another template parameter to graph_query, we just hard code this here.
// We'll just have to remember to update it once we inevitably add another level of graphs ¯\_(ツ)_/¯
template <typename Recorder>
constexpr static const char* gq_print_prefix = std::is_same_v<Recorder, instruction_recorder> ? "I" : "C";

/// View on a subset of graph records within a recorder. Allows selecting for subsets by various predicates, sets of successors and predecessor
/// nodes within the graph. Depending on the selectors used, returned sub-queries are automatically specialized on the record type, which will allow directly
/// de-referencing a single-element query via the `->` operator.
template <typename Record, typename BaseRecord, typename Recorder, template <typename> typename Matcher>
class graph_query {
  public:
	using record_type = Record;

	static_assert(std::is_base_of_v<BaseRecord, Record>);

	template <typename T>
	using sub_query = graph_query<T, BaseRecord, Recorder, Matcher>;

	using base_query = graph_query<BaseRecord, BaseRecord, Recorder, Matcher>;

	explicit graph_query(const Recorder& recorder) : graph_query(&recorder, non_owning_pointers(recorder.get_graph_nodes()), std::string()) {}

	// allow upcast
	template <typename SpecificRecord, std::enable_if_t<std::is_base_of_v<Record, SpecificRecord> && !std::is_same_v<Record, SpecificRecord>, int> = 0>
	graph_query(const sub_query<SpecificRecord>& other)
	    : graph_query(other.m_recorder, std::vector<const Record*>(other.m_result.begin(), other.m_result.end()), other.m_trace) {}

	/// Returns a subset of the selected records that matches all provided filters and (optionally) the provided record type.
	template <typename SpecificRecord = Record, typename... Filters>
	sub_query<SpecificRecord> select_all(const Filters&... filters) const {
		std::vector<const SpecificRecord*> filtered;
		for(const auto rec : m_result) {
			if(matches<SpecificRecord>(*rec, filters...)) { filtered.push_back(utils::as<SpecificRecord>(rec)); }
		}
		return sub_query<SpecificRecord>(m_recorder, std::move(filtered),
		    std::is_same_v<SpecificRecord, Record> && sizeof...(Filters) == 0 ? m_trace : filter_trace<Record, SpecificRecord>("select_all", filters...));
	}

	/// Like `select_all`, but asserts that the returned query holds exactly one record (and thus can be dereferenced using `->`).
	template <typename SpecificRecord = Record, typename... Filters>
	sub_query<SpecificRecord> select_unique(const Filters&... filters) const {
		auto query = select_all<SpecificRecord>(filters...).assert_unique();
		return sub_query<SpecificRecord>(m_recorder, std::move(query.m_result), filter_trace<Record, SpecificRecord>("select_unique", filters...));
	}

	/// Selects all predecessors (dependencies) of all records in the current query.
	base_query predecessors() const {
		std::vector<const BaseRecord*> predecessors;
		// find predecessors without duplicates (even when m_result.size() > 1) and keep them in recorder-ordering
		for(const auto& maybe_predecessor : m_recorder->get_graph_nodes()) {
			if(std::any_of(m_recorder->get_dependencies().begin(), m_recorder->get_dependencies().end(), [&](const auto& dep) {
				   return dep.predecessor == maybe_predecessor->id
				          && std::any_of(m_result.begin(), m_result.end(), [&](const Record* me) { return me->id == dep.successor; });
			   })) {
				predecessors.push_back(maybe_predecessor.get());
			}
		}
		return base_query(m_recorder, std::move(predecessors), m_trace + ".predecessors()");
	}

	/// Recursively selects all predecessors of all records in the current query.
	base_query transitive_predecessors() const {
		for(auto query = predecessors();;) {
			auto next = union_of(query, query.predecessors());
			if(query == next) { return base_query(m_recorder, std::move(query.m_result), m_trace + ".transitive_predecessors()"); }
			query = std::move(next);
		}
	};

	/// Recursively selects all predecessors of all records in the current query, but only walks across records that match all provided filters (and the
	/// specific record type if provided).
	template <typename SpecificRecord, typename... Filters>
	base_query transitive_predecessors_across(const Filters&... filters) const {
		for(auto query = predecessors();;) {
			auto next = union_of(query, query.template select_all<SpecificRecord>(filters...).predecessors());
			if(query == next) {
				return base_query(m_recorder, std::move(query.m_result), filter_trace<Record, SpecificRecord>("transitive_predecessors_across", filters...));
			}
			query = std::move(next);
		}
	};

	/// Selects all successors (dependers) of all records in the current query.
	base_query successors() const {
		std::vector<const BaseRecord*> successors;
		// find successors without duplicates (even when m_result.size() > 1) and keep them in recorder-ordering
		for(const auto& maybe_successor : m_recorder->get_graph_nodes()) {
			if(std::any_of(m_recorder->get_dependencies().begin(), m_recorder->get_dependencies().end(), [&](const auto& dep) {
				   return dep.successor == maybe_successor->id
				          && std::any_of(m_result.begin(), m_result.end(), [&](const Record* me) { return me->id == dep.predecessor; });
			   })) {
				successors.push_back(maybe_successor.get());
			}
		}
		return base_query(m_recorder, std::move(successors), m_trace + ".successors()");
	}

	/// Recursively selects all successors of all records in the current query.
	base_query transitive_successors() const {
		for(auto query = successors();;) {
			auto next = union_of(query, query.successors());
			if(query == next) { return base_query(m_recorder, std::move(query.m_result), m_trace + ".transitive_successors()"); }
			query = std::move(next);
		}
	};

	/// Recursively selects all successors of all records in the current query, but only walks across records that match all provided filters (and the specific
	/// record type if provided).
	template <typename SpecificRecord, typename... Filters>
	base_query transitive_successors_across(const Filters&... filters) const {
		for(auto query = successors();;) {
			auto next = union_of(query, query.template select_all<SpecificRecord>(filters...).successors());
			if(query == next) {
				return base_query(m_recorder, std::move(query.m_result), filter_trace<Record, SpecificRecord>("transitive_successors_across", filters...));
			}
			query = std::move(next);
		}
	};

	/// Check if `this` and `other` are concurrent, i.e. neither is a transitive predecessor of the other.
	bool is_concurrent_with(const base_query& other) const { return !transitive_predecessors().contains(other) && !transitive_successors().contains(other); }

	/// Returns the number of records in the query.
	size_t count() const { return m_result.size(); }

	/// Asserts that the count is equal to the provided value.
	const graph_query& assert_count(const size_t expected) const {
		INFO(fmt::format("query: ", m_trace));
		INFO(fmt::format("result: {}", *this));
		REQUIRE(count() == expected);
		return *this;
	}

	/// Returns the number of records in the query that match all provided filters (and the specified record type, if any)
	template <typename SpecificRecord = Record, typename... Filters>
	size_t count(const Filters&... filters) const {
		return static_cast<size_t>(
		    std::count_if(m_result.begin(), m_result.end(), [&](const Record* rec) { return matches<SpecificRecord>(*rec, filters...); }));
	}

	/// Returns sub-query containing exactly the `index`-th record in the current query. Nodes keep the order they were recorded in.
	graph_query operator[](const size_t index) const {
		if(index >= m_result.size()) {
			INFO(fmt::format("query: ", m_trace));
			INFO(fmt::format("result: {}", *this));
			FAIL(fmt::format("index {} out of bounds (size: {})", index, m_result.size()));
		}
		return graph_query(m_recorder, {m_result[index]}, fmt::format("{}[{}]", m_trace, index));
	}

	/// Returns a sequence of single-record queries to iterate over all entries of the current query, in the order they were recorded in.
	std::vector<graph_query> iterate() const {
		std::vector<graph_query> queries;
		for(size_t i = 0; i < m_result.size(); ++i) {
			queries.push_back(graph_query(m_recorder, {m_result[i]}, fmt::format("{}[{}]", m_trace, i)));
		}
		return queries;
	}

	/// Returns whether all records in the current query match all provided filters (and have a specific type, if provided).
	template <typename SpecificRecord = Record, typename... Filters>
	bool all_match(const Filters&... filters) const {
		std::string non_matching;
		for(const Record* rec : m_result) {
			if(!matches<SpecificRecord>(*rec, filters...)) {
				if(!non_matching.empty()) { non_matching += ", "; }
				fmt::format_to(std::back_inserter(non_matching), "{}{}", gq_print_prefix<Recorder>, rec->id);
			}
		}
		if(non_matching.empty()) return true;

		UNSCOPED_INFO(fmt::format("query: {}", filter_trace<Record, SpecificRecord>("all_match", filters...)));
		UNSCOPED_INFO(fmt::format("non-matching: {{{}}}", non_matching));
		return false;
	}

	/// Asserts that the current query does `match_all` the provided filters, and casts the query to the specific record type (if provided).
	template <typename SpecificRecord = Record, typename... Filters>
	sub_query<SpecificRecord> assert_all(const Filters&... filters) const {
		REQUIRE(all_match<SpecificRecord>(filters...));
		std::vector<const SpecificRecord*> result(m_result.size());
		std::transform(m_result.begin(), m_result.end(), result.begin(), [](const Record* rec) { return utils::as<SpecificRecord>(rec); });
		return sub_query<SpecificRecord>(m_recorder, std::move(result), filter_trace<Record, SpecificRecord>("assert_all", filters...));
	}

	/// Returns whether `is_concurrent` holds for each pair of single-record sub-queries.
	bool all_concurrent() const {
		for(size_t i = 0; i < count(); ++i) {
			for(size_t j = i + 1; j < count(); ++j) {
				if(!(*this)[i].is_concurrent_with((*this)[j])) {
					UNSCOPED_INFO(fmt::format("query: {}", m_trace));
					UNSCOPED_INFO(fmt::format("result: {}", *this));
					UNSCOPED_INFO(
					    fmt::format("{}{} and {}{} are not concurrent", gq_print_prefix<Recorder>, (*this)[i]->id, gq_print_prefix<Recorder>, (*this)[j]->id));
					return false;
				}
			}
		}
		return true;
	}

	/// Returns whether `this` contains all records that `subset` also contains.
	bool contains(const graph_query& subset) const {
		return std::all_of(subset.m_result.begin(), subset.m_result.end(), [&](const Record* their) {
			return std::any_of(m_result.begin(), m_result.end(), [&](const Record* my) { return their->id == my->id; }); //
		});
	}

	/// Returns whether this query holds exactly one record and whether that record matches all provided filters.
	template <typename SpecificRecord = Record, typename... Filters>
	bool is_unique(const Filters&... filters) const {
		if(m_result.size() != 1) {
			UNSCOPED_INFO(fmt::format("query: {}", m_trace));
			UNSCOPED_INFO(fmt::format("result: {}", *this));
		}
		return m_result.size() == 1 && all_match<SpecificRecord>(filters...);
	}

	/// Asserts that`is_unique` is true for the provided filters, and casts the result to the record type (if specified).
	template <typename SpecificRecord = Record, typename... Filters>
	sub_query<SpecificRecord> assert_unique(const Filters&... filters) const {
		REQUIRE(is_unique<SpecificRecord>(filters...));
		return sub_query<SpecificRecord>(
		    m_recorder, {utils::as<SpecificRecord>(m_result.front())}, filter_trace<Record, SpecificRecord>("assert_unique", filters...));
	}

	/// Asserts that this query contains exactly one record, and provides direct access to its record.
	const Record* operator->() const {
		REQUIRE(is_unique());
		return m_result.front();
	}

	// m_result follows m_recorder ordering, so vector-equality is enough
	friend bool operator==(const graph_query& lhs, const graph_query& rhs) { return lhs.m_result == rhs.m_result; }
	friend bool operator!=(const graph_query& lhs, const graph_query& rhs) { return lhs.m_result != rhs.m_result; }

	/// Returns a query containing the union of records from all parameters.
	template <typename... GraphQueries>
	friend base_query union_of(const graph_query& head, const GraphQueries&... tail) {
		// call through a proper member function, because GCC will not extend friendship with GraphQueries... to inline-friend functions
		return head.union_with(tail...);
	}

	/// Returns a query containing the intersection of records from all parameters.
	template <typename... GraphQueries>
	friend graph_query intersection_of(const graph_query& head, const GraphQueries&... tail) {
		// call through a proper member function, because GCC will not extend friendship with GraphQueries... to inline-friend functions
		return head.intersection_with(tail...);
	}

	/// Returns a query containing all records that are in `first` but not in `second`.
	friend graph_query difference_of(const graph_query& first, const graph_query& second) {
		// call through a proper member function, because GCC will not extend friendship with GraphQueries... to inline-friend functions
		return first.difference_with(second);
	}

  private:
	template <typename, typename, typename, template <typename> typename>
	friend class graph_query;

	template <typename, typename, typename>
	friend struct fmt::formatter;

	const Recorder* m_recorder;
	std::vector<const Record*> m_result;
	std::string m_trace;

	template <typename T>
	static std::vector<const T*> non_owning_pointers(const std::vector<std::unique_ptr<T>>& unique) {
		std::vector<const T*> ptrs(unique.size());
		std::transform(unique.begin(), unique.end(), ptrs.begin(), [](const std::unique_ptr<T>& p) { return p.get(); });
		return ptrs;
	}

	template <typename SpecificRecord, typename... Filters>
	static bool matches(const Record& rec, const Filters&... filters) {
		return utils::isa<SpecificRecord>(&rec) && (Matcher<SpecificRecord>::matches(*utils::as<SpecificRecord>(&rec), filters) && ...);
	}

	template <typename GeneralRecord, typename SpecificRecord, typename... Filters>
	std::string filter_trace(const std::string& selector, const Filters&... filters) const {
		auto trace = m_trace.empty() ? selector : fmt::format("{}.{}", m_trace, selector);
		if constexpr(!std::is_same_v<GeneralRecord, SpecificRecord>) { fmt::format_to(std::back_inserter(trace), "<{}>", kernel_debug_name<SpecificRecord>()); }

		std::string param_trace;
		// [[maybe_unused]]: GCC will emit an unused-warning if `Filters...` is empty and the comma-fold expression below expands to nothing
		[[maybe_unused]] const auto format_filter_param = [&](const auto& f) {
			if(!param_trace.empty()) { param_trace += ", "; }
			param_trace += Matcher<SpecificRecord>::print_filter(f);
		};
		(format_filter_param(filters), ...);
		fmt::format_to(std::back_inserter(trace), "({})", param_trace);
		return trace;
	}

	template <typename... GraphQueries>
	base_query union_with(const GraphQueries&... tail) const {
		assert(((m_recorder == tail.m_recorder) && ...));

		// construct union in recorder-ordering without duplicates - always returns the base type
		std::vector<const BaseRecord*> union_query;
		for(const auto& rec : m_recorder->get_graph_nodes()) {
			if(std::find(m_result.begin(), m_result.end(), rec.get()) != m_result.end()
			    || ((std::find(tail.m_result.begin(), tail.m_result.end(), rec.get()) != tail.m_result.end()) || ...)) { //
				union_query.push_back(rec.get());
			}
		}

		std::string trace = fmt::format("union_of({}", m_trace);
		(((trace += ", ") += tail.m_trace), ...);
		trace += ")";
		return base_query(m_recorder, std::move(union_query), std::move(trace));
	}

	template <typename... GraphQueries>
	graph_query intersection_with(const GraphQueries&... tail) const {
		assert(((m_recorder == tail.m_recorder) && ...));

		// construct intersection in recorder-ordering without duplicates - returns the type of `head`
		std::vector<const Record*> intersection_query;
		for(const auto& rec : m_result) {
			if(((std::find(tail.m_result.begin(), tail.m_result.end(), rec) != tail.m_result.end()) && ...)) { //
				intersection_query.push_back(rec);
			}
		}

		std::string trace = fmt::format("intersection_of({}", m_trace);
		(((trace += ", ") += tail.m_trace), ...);
		trace += ")";
		return graph_query(m_recorder, std::move(intersection_query), std::move(trace));
	}

	graph_query difference_with(const graph_query& other) const {
		assert(m_recorder == other.m_recorder);

		// construct difference in recorder-ordering without duplicates - returns the type of `this`
		std::vector<const Record*> difference_query;
		for(const auto& rec : m_result) {
			if(std::find(other.m_result.begin(), other.m_result.end(), rec) == other.m_result.end()) { difference_query.push_back(rec); }
		}

		return graph_query(m_recorder, std::move(difference_query), fmt::format("difference_of({}, {})", m_trace, other.m_trace));
	}

	graph_query(const Recorder* recorder, std::vector<const Record*> query, std::string trace)
	    : m_recorder(recorder), m_result(std::move(query)), m_trace(std::move(trace)) {}
};

} // namespace celerity::test_utils

template <typename Record, typename BaseRecord, typename Recorder, template <typename> typename Matcher>
struct fmt::formatter<celerity::test_utils::graph_query<Record, BaseRecord, Recorder, Matcher>> : fmt::formatter<size_t> {
	format_context::iterator format(const celerity::test_utils::graph_query<Record, BaseRecord, Recorder, Matcher>& gq, format_context& ctx) const {
		auto out = ctx.out();
		*out++ = '{';
		for(size_t i = 0; i < gq.m_result.size(); ++i) {
			if(i > 0) { out = std::copy_n(", ", 2, out); }
			fmt::format_to(out, "{}{}", celerity::test_utils::gq_print_prefix<Recorder>, gq.m_result[i]->id);
		}
		*out++ = '}';
		return out;
	}
};

template <typename Record, typename BaseRecord, typename Recorder, template <typename> typename Matcher>
struct Catch::StringMaker<celerity::test_utils::graph_query<Record, BaseRecord, Recorder, Matcher>> {
	static std::string convert(const celerity::test_utils::graph_query<Record, BaseRecord, Recorder, Matcher>& gq) { return fmt::format("{}", gq); }
};
