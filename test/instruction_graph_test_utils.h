#pragma once

#include "distributed_graph_generator_test_utils.h"
#include "instruction_graph_generator.h"


namespace celerity::test_utils {

/// View on a subset of instructions within an instruction_recorder. Allows selecting for subsets by various predicates, sets of successors and predecessor
/// nodes within the graph. Depending on the selectors used, returned sub-queries are automatically specialized on the record type, which will allow directly
/// de-referencing a single-element query via the `->` operator.
template <typename Record = instruction_record>
class instruction_query {
  public:
	using record_type = Record;

	template <typename R = Record, std::enable_if_t<std::is_same_v<R, instruction_record>, int> = 0>
	explicit instruction_query(const instruction_recorder& recorder)
	    : instruction_query(&recorder, non_owning_pointers(recorder.get_instructions()), std::string()) {}

	// allow upcast
	template <typename SpecificRecord, std::enable_if_t<std::is_base_of_v<Record, SpecificRecord> && !std::is_same_v<Record, SpecificRecord>, int> = 0>
	instruction_query(const instruction_query<SpecificRecord>& other)
	    : instruction_query(other.m_recorder, std::vector<const Record*>(other.m_result.begin(), other.m_result.end()), other.m_trace) {}

	/// Returns a subset of the selected instructions that matches all provided filters and (optionally) the provided record type.
	template <typename SpecificRecord = Record, typename... Filters>
	instruction_query<SpecificRecord> select_all(const Filters&... filters) const {
		std::vector<const SpecificRecord*> filtered;
		for(const auto instr : m_result) {
			if(matches<SpecificRecord>(*instr, filters...)) { filtered.push_back(utils::as<SpecificRecord>(instr)); }
		}
		return instruction_query<SpecificRecord>(m_recorder, std::move(filtered),
		    std::is_same_v<SpecificRecord, Record> && sizeof...(Filters) == 0 ? m_trace : filter_trace<Record, SpecificRecord>("select_all", filters...));
	}

	/// Like `select_all`, but asserts that the returned query holds exactly one element (and thus can be dereferenced using `->`).
	template <typename SpecificRecord = Record, typename... Filters>
	instruction_query<SpecificRecord> select_unique(const Filters&... filters) const {
		auto query = select_all<SpecificRecord>(filters...).assert_unique();
		return instruction_query<SpecificRecord>(m_recorder, std::move(query.m_result), filter_trace<Record, SpecificRecord>("select_unique", filters...));
	}

	/// Selects all predecessors (dependencies) of all nodes in the current query.
	instruction_query<instruction_record> predecessors() const {
		std::vector<const instruction_record*> predecessors;
		// find predecessors without duplicates (even when m_result.size() > 1) and keep them in recorder-ordering
		for(const auto& maybe_predecessor : m_recorder->get_instructions()) {
			if(std::any_of(m_recorder->get_dependencies().begin(), m_recorder->get_dependencies().end(), [&](const instruction_dependency_record& dep) {
				   return dep.predecessor == maybe_predecessor->id
				          && std::any_of(m_result.begin(), m_result.end(), [&](const Record* me) { return me->id == dep.successor; });
			   })) {
				predecessors.push_back(maybe_predecessor.get());
			}
		}
		return instruction_query<instruction_record>(m_recorder, std::move(predecessors), m_trace + ".predecessors()");
	}

	/// Recursively selects all predecessors of all nodes in the current query.
	instruction_query<instruction_record> transitive_predecessors() const {
		for(auto query = predecessors();;) {
			auto next = union_of(query, query.predecessors());
			if(query == next) { return instruction_query<instruction_record>(m_recorder, std::move(query.m_result), m_trace + ".transitive_predecessors()"); }
			query = std::move(next);
		}
	};

	/// Recursively selects all predecessors of all nodes in the current query, but only walks across nodes that match all provided filters (and the specific
	/// record type if provided).
	template <typename SpecificRecord, typename... Filters>
	instruction_query<instruction_record> transitive_predecessors_across(const Filters&... filters) const {
		for(auto query = predecessors();;) {
			auto next = union_of(query, query.template select_all<SpecificRecord>(filters...).predecessors());
			if(query == next) {
				return instruction_query<instruction_record>(
				    m_recorder, std::move(query.m_result), filter_trace<Record, SpecificRecord>("transitive_predecessors_across", filters...));
			}
			query = std::move(next);
		}
	};

	/// Selects all successors (dependers) of all nodes in the current query.
	instruction_query<instruction_record> successors() const {
		std::vector<const instruction_record*> successors;
		// find successors without duplicates (even when m_result.size() > 1) and keep them in recorder-ordering
		for(const auto& maybe_successor : m_recorder->get_instructions()) {
			if(std::any_of(m_recorder->get_dependencies().begin(), m_recorder->get_dependencies().end(), [&](const instruction_dependency_record& dep) {
				   return dep.successor == maybe_successor->id
				          && std::any_of(m_result.begin(), m_result.end(), [&](const Record* me) { return me->id == dep.predecessor; });
			   })) {
				successors.push_back(maybe_successor.get());
			}
		}
		return instruction_query<instruction_record>(m_recorder, std::move(successors), m_trace + ".successors()");
	}

	/// Recursively selects all successors of all nodes in the current query.
	instruction_query<instruction_record> transitive_successors() const {
		for(auto query = successors();;) {
			auto next = union_of(query, query.successors());
			if(query == next) { return instruction_query<instruction_record>(m_recorder, std::move(query.m_result), m_trace + ".transitive_successors()"); }
			query = std::move(next);
		}
	};

	/// Recursively selects all successors of all nodes in the current query, but only walks across nodes that match all provided filters (and the specific
	/// record type if provided).
	template <typename SpecificRecord, typename... Filters>
	instruction_query<instruction_record> transitive_successors_across(const Filters&... filters) const {
		for(auto query = successors();;) {
			auto next = union_of(query, query.template select_all<SpecificRecord>(filters...).successors());
			if(query == next) {
				return instruction_query<instruction_record>(
				    m_recorder, std::move(query.m_result), filter_trace<Record, SpecificRecord>("transitive_successors_across", filters...));
			}
			query = std::move(next);
		}
	};

	/// Check if `this` and `other` are concurrent, i.e. neither is a transitive predecessor of the other.
	bool is_concurrent_with(const instruction_query<>& other) const {
		return !transitive_predecessors().contains(other) && !transitive_successors().contains(other);
	}

	/// Returns the number of nodes in the query.
	size_t count() const { return m_result.size(); }

	/// Returns the number of nodes in the query that match all provided filters (and the specified record type, if any)
	template <typename SpecificRecord = Record, typename... Filters>
	size_t count(const Filters&... filters) const {
		return static_cast<size_t>(
		    std::count_if(m_result.begin(), m_result.end(), [&](const Record* instr) { return matches<SpecificRecord>(*instr, filters...); }));
	}

	/// Returns sub-query containing exactly the `index`-th node in the current query. Nodes keep the order they were recorded in.
	instruction_query operator[](const size_t index) const {
		if(index > m_result.size()) {
			INFO(fmt::format("query: ", m_trace));
			INFO(fmt::format("result: {}", *this));
			FAIL(fmt::format("index {} out of bounds (size: {})", index, m_result.size()));
		}
		return instruction_query(m_recorder, {m_result[index]}, fmt::format("{}[{}]", m_trace, index));
	}

	/// Returns a sequence of single-node queries to iterate over all entries of the current query, in the order the nodes were recorded in.
	std::vector<instruction_query> iterate() const {
		std::vector<instruction_query> queries;
		for(size_t i = 0; i < m_result.size(); ++i) {
			queries.push_back(instruction_query(m_recorder, {m_result[i]}, fmt::format("{}[{}]", m_trace, i)));
		}
		return queries;
	}

	/// Returns whether all nodes in the current query match all provided filters (and have a specific record type, if provided).
	template <typename SpecificRecord = Record, typename... Filters>
	bool all_match(const Filters&... filters) const {
		std::string non_matching;
		for(const Record* instr : m_result) {
			if(!matches<SpecificRecord>(*instr, filters...)) {
				if(!non_matching.empty()) { non_matching += ", "; }
				fmt::format_to(std::back_inserter(non_matching), "I{}", instr->id);
			}
		}
		if(non_matching.empty()) return true;

		UNSCOPED_INFO(fmt::format("query: {}", filter_trace<Record, SpecificRecord>("all_match", filters...)));
		UNSCOPED_INFO(fmt::format("non-matching: {{{}}}", non_matching));
		return false;
	}

	/// Asserts that the current query does not `match_all` the provided filters, and casts the query to the specific record type (if provided).
	template <typename SpecificRecord = Record, typename... Filters>
	instruction_query<SpecificRecord> assert_all(const Filters&... filters) const {
		REQUIRE(all_match<SpecificRecord>(filters...));
		std::vector<const SpecificRecord*> result(m_result.size());
		std::transform(m_result.begin(), m_result.end(), result.begin(), [](const Record* instr) { return utils::as<SpecificRecord>(instr); });
		return instruction_query<SpecificRecord>(m_recorder, std::move(result), filter_trace<Record, SpecificRecord>("assert_all", filters...));
	}

	/// Returns whether `is_concurrent` holds for each pair of single-node sub-queries.
	bool all_concurrent() const {
		for(size_t i = 0; i < count(); ++i) {
			for(size_t j = i + 1; j < count(); ++j) {
				if(!(*this)[i].is_concurrent_with((*this)[j])) {
					UNSCOPED_INFO(fmt::format("query: {}", m_trace));
					UNSCOPED_INFO(fmt::format("result: {}", *this));
					UNSCOPED_INFO(fmt::format("I{} and I{} are not concurrent", (*this)[i]->id, (*this)[j]->id));
					return false;
				}
			}
		}
		return true;
	}

	/// Returns whether `this` contains all nodes that `subset` also contains.
	bool contains(const instruction_query& subset) const {
		return std::all_of(subset.m_result.begin(), subset.m_result.end(), [&](const Record* their) {
			return std::any_of(m_result.begin(), m_result.end(), [&](const Record* my) { return their->id == my->id; }); //
		});
	}

	/// Returns whether this query holds exactly one node and whether that node matches all provided filters.
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
	instruction_query<SpecificRecord> assert_unique(const Filters&... filters) const {
		REQUIRE(is_unique<SpecificRecord>(filters...));
		return instruction_query<SpecificRecord>(
		    m_recorder, {utils::as<SpecificRecord>(m_result.front())}, filter_trace<Record, SpecificRecord>("assert_unique", filters...));
	}

	/// Asserts that this query contains exactly one node, and provides direct access to its record.
	const Record* operator->() const {
		REQUIRE(is_unique());
		return m_result.front();
	}

	// m_result follows m_recorder ordering, so vector-equality is enough
	friend bool operator==(const instruction_query& lhs, const instruction_query& rhs) { return lhs.m_result == rhs.m_result; }
	friend bool operator!=(const instruction_query& lhs, const instruction_query& rhs) { return lhs.m_result != rhs.m_result; }

	/// Returns a query containing the union of nodes from all parameters.
	template <typename... InstructionQueries>
	friend instruction_query<> union_of(const instruction_query& head, const InstructionQueries&... tail) {
		// call through a proper member function, because GCC will not extend friendship with InstructionQueries... to inline-friend functions
		return head.union_with(tail...);
	}

	/// Returns a query containing the intersection of nodes from all parameters.
	template <typename... InstructionQueries>
	friend instruction_query intersection_of(const instruction_query& head, const InstructionQueries&... tail) {
		// call through a proper member function, because GCC will not extend friendship with InstructionQueries... to inline-friend functions
		return head.intersection_with(tail...);
	}

  private:
	template <typename>
	friend class instruction_query;

	template <typename, typename, typename>
	friend struct fmt::formatter;

	const instruction_recorder* m_recorder;
	std::vector<const Record*> m_result;
	std::string m_trace;

	template <typename T>
	static std::vector<const T*> non_owning_pointers(const std::vector<std::unique_ptr<T>>& unique) {
		std::vector<const T*> ptrs(unique.size());
		std::transform(unique.begin(), unique.end(), ptrs.begin(), [](const std::unique_ptr<T>& p) { return p.get(); });
		return ptrs;
	}

	template <typename SpecificRecord, typename... Filters>
	static bool matches(const Record& instr, const Filters&... filters) {
		return utils::isa<SpecificRecord>(&instr) && (instruction_query<SpecificRecord>::matches(*utils::as<SpecificRecord>(&instr), filters) && ...);
	}

	static bool matches(const Record& instr, const instruction_id iid) { return instr.id == iid; }

	static bool matches(const Record& instr, const task_id tid) {
		return matchbox::match(
		    instr,                                                                                                 //
		    [=](const device_kernel_instruction_record& dkinstr) { return dkinstr.command_group_task_id == tid; }, //
		    [=](const host_task_instruction_record& htinstr) { return htinstr.command_group_task_id == tid; },     //
		    [=](const fence_instruction_record& finstr) { return finstr.tid == tid; },                             //
		    [=](const horizon_instruction_record& hinstr) { return hinstr.horizon_task_id == tid; },               //
		    [=](const epoch_instruction_record& einstr) { return einstr.epoch_task_id == tid; },                   //
		    [](const auto& /* other */) { return false; });
	}

	static bool matches(const Record& instr, const device_id did) {
		return utils::isa<device_kernel_instruction_record>(&instr) && utils::as<device_kernel_instruction_record>(&instr)->device_id == did;
	}

	static bool matches(const Record& instr, const std::string& debug_name) {
		return matchbox::match(
		    instr,                                                                                             //
		    [&](const device_kernel_instruction_record& dkinstr) { return dkinstr.debug_name == debug_name; }, //
		    [&](const host_task_instruction_record& htinstr) { return htinstr.debug_name == debug_name; },     //
		    [](const auto& /* other */) { return false; });
	}

	template <typename Predicate, std::enable_if_t<std::is_invocable_r_v<bool, const Predicate&, const Record&>, int> = 0>
	static bool matches(const Record& instr, const Predicate& predicate) {
		return predicate(instr);
	}

	static std::string print_filter(const instruction_id iid) { return fmt::format("I{}", iid); }
	static std::string print_filter(const task_id iid) { return fmt::format("T{}", iid); }
	static std::string print_filter(const device_id iid) { return fmt::format("D{}", iid); }
	static std::string print_filter(const std::string& debug_name) { return fmt::format("\"{}\"", debug_name); }

	template <typename Predicate, std::enable_if_t<std::is_invocable_r_v<bool, const Predicate&, const Record&>, int> = 0>
	static std::string print_filter(const Predicate& /* predicate */) {
		return "<lambda>";
	}

	template <typename GeneralRecord, typename SpecificRecord, typename... Filters>
	std::string filter_trace(const std::string& selector, const Filters&... filters) const {
		auto trace = m_trace.empty() ? selector : fmt::format("{}.{}", m_trace, selector);
		if constexpr(!std::is_same_v<GeneralRecord, SpecificRecord>) { fmt::format_to(std::back_inserter(trace), "<{}>", kernel_debug_name<SpecificRecord>()); }

		std::string param_trace;
		// [[maybe_unused]]: GCC will emit an unused-warning if `Filters...` is empty and the comma-fold expression below expands to nothing
		[[maybe_unused]] const auto format_filter_param = [&](const auto& f) {
			if(!param_trace.empty()) { param_trace += ", "; }
			param_trace += instruction_query<SpecificRecord>::print_filter(f);
		};
		(format_filter_param(filters), ...);
		fmt::format_to(std::back_inserter(trace), "({})", param_trace);
		return trace;
	}

	template <typename... InstructionQueries>
	instruction_query<> union_with(const InstructionQueries&... tail) const {
		assert(((m_recorder == tail.m_recorder) && ...));

		// construct union in recorder-ordering without duplicates - always returns the base type
		std::vector<const instruction_record*> union_query;
		for(const auto& instr : m_recorder->get_instructions()) {
			if(std::find(m_result.begin(), m_result.end(), instr.get()) != m_result.end()
			    || ((std::find(tail.m_result.begin(), tail.m_result.end(), instr.get()) != tail.m_result.end()) || ...)) { //
				union_query.push_back(instr.get());
			}
		}

		std::string trace = fmt::format("union_of({}", m_trace);
		(((trace += ", ") += tail.m_trace), ...);
		trace += ")";
		return instruction_query<>(m_recorder, std::move(union_query), std::move(trace));
	}

	template <typename... InstructionQueries>
	instruction_query intersection_with(const InstructionQueries&... tail) const {
		assert(((m_recorder == tail.m_recorder) && ...));

		// construct intersection in recorder-ordering without duplicates - returns the type of `head`
		std::vector<const Record*> intersection_query;
		for(const auto& instr : m_result) {
			if(((std::find(tail.m_result.begin(), tail.m_result.end(), instr) != tail.m_result.end()) && ...)) { //
				intersection_query.push_back(instr);
			}
		}

		std::string trace = fmt::format("intersection_of({}", m_trace);
		(((trace += ", ") += tail.m_trace), ...);
		trace += ")";
		return instruction_query(m_recorder, std::move(intersection_query), std::move(trace));
	}

	instruction_query(const instruction_recorder* recorder, std::vector<const Record*> query, std::string trace)
	    : m_recorder(recorder), m_result(std::move(query)), m_trace(std::move(trace)) {}
};

/// A view on an `instruction_recorder` like `instruction_query`, but for `outbound_pilot`s.
class pilot_query {
  public:
	explicit pilot_query(const instruction_recorder& recorder) : pilot_query(&recorder, recorder.get_outbound_pilots()) {}

	/// Returns the subset-query of all pilots that match the provided filters.
	template <typename... Filters>
	pilot_query select_all(const Filters&... filters) const {
		std::vector<outbound_pilot> filtered;
		std::copy_if(
		    m_result.begin(), m_result.end(), std::back_inserter(filtered), [&](const outbound_pilot& pilot) { return matches_all(pilot, filters...); });
		return pilot_query(m_recorder, std::move(filtered));
	}

	/// Like `select_all`, but asserts that the resulting query contains exactly one pilot.
	template <typename... Filters>
	pilot_query select_unique(const Filters&... filters) const {
		return select_all(filters...).assert_unique();
	}

	/// Returns the number of pilots in this query.
	size_t count() const { return m_result.size(); }

	/// Returns the number of pilots in this query that satisfy all provided filters.
	template <typename... Filters>
	size_t count(const Filters&... filters) const {
		return static_cast<size_t>(std::count_if(m_result.begin(), m_result.end(), [&](const outbound_pilot& p) { return matches_all(p, filters...); }));
	}

	/// Returns whether all pilots in this query match all provided filters.
	template <typename... Filters>
	bool all_match(const Filters&... filters) const {
		const auto num_non_matching = std::count_if(m_result.begin(), m_result.end(), [&](const outbound_pilot& p) { return !matches_all(p, filters...); });
		if(num_non_matching > 0) { UNSCOPED_INFO(fmt::format("non-matching: {} pilots", num_non_matching)); }
		return num_non_matching == 0;
	}

	/// Returns whether this query contains exactly one pilot, and whether that pilot matches all provided filters.
	template <typename... Filters>
	bool is_unique(const Filters&... filters) const {
		if(m_result.size() != 1) { UNSCOPED_INFO(fmt::format("{} pilots in query result", m_result.size())); }
		return m_result.size() == 1 && all_match(filters...);
	}

	/// Asserts that `is_unique` holds and returns the query unchanged.
	template <typename... Filters>
	pilot_query assert_unique(const Filters&... filters) const {
		REQUIRE(is_unique(filters...));
		return *this;
	}

	/// Asserts that `is_unique` holds and provides direct access to the single pilot.
	const outbound_pilot* operator->() const {
		REQUIRE(is_unique());
		return &m_result.front();
	}

  private:
	pilot_query(const instruction_recorder* recorder, std::vector<outbound_pilot> result) : m_recorder(recorder), m_result(std::move(result)) {}

	static bool matches(const outbound_pilot& pilot, const node_id nid) { return pilot.to == nid; }
	static bool matches(const outbound_pilot& pilot, const transfer_id& trid) { return pilot.message.transfer_id == trid; }
	static bool matches(const outbound_pilot& pilot, const box<3>& box) { return pilot.message.box == box; }

	template <typename Predicate, std::enable_if_t<std::is_invocable_r_v<bool, const Predicate&, const outbound_pilot&>, int> = 0>
	static bool matches(const outbound_pilot& pilot, const Predicate& predicate) {
		return predicate(pilot);
	}

	template <typename... Filters>
	static bool matches_all(const outbound_pilot& pilot, const Filters&... filters) {
		return (matches(pilot, filters) && ...);
	}

	const instruction_recorder* m_recorder;
	std::vector<outbound_pilot> m_result;
};

class mock_host_object_fence_promise : public fence_promise {
  public:
	void fulfill() override { FAIL("unimplemented"); }
	allocation_id get_user_allocation_id() override {
		FAIL("unimplemented");
		return null_allocation_id;
	}
};

class mock_buffer_fence_promise : public fence_promise {
  public:
	mock_buffer_fence_promise() = default;
	explicit mock_buffer_fence_promise(allocation_id user_allocation_id) : m_user_aid(user_allocation_id) {}

	void fulfill() override { FAIL("unimplemented"); }

	allocation_id get_user_allocation_id() override { return m_user_aid; }

  private:
	allocation_id m_user_aid;
};

class idag_test_context {
	friend class task_builder<idag_test_context>;

  private:
	class uncaught_exception_guard {
	  public:
		explicit uncaught_exception_guard(idag_test_context* ictx) : m_ictx(ictx), m_uncaught_exceptions_before(std::uncaught_exceptions()) {}

		~uncaught_exception_guard() {
			if(std::uncaught_exceptions() > m_uncaught_exceptions_before) {
				m_ictx->m_uncaught_exceptions_before = -1; // always destroy idag_test_context under the uncaught-exception condition
			}
		}

		uncaught_exception_guard(const uncaught_exception_guard&) = delete;
		uncaught_exception_guard(uncaught_exception_guard&&) = delete;
		uncaught_exception_guard& operator=(const uncaught_exception_guard&) = delete;
		uncaught_exception_guard& operator=(uncaught_exception_guard&&) = delete;

	  private:
		idag_test_context* m_ictx;
		int m_uncaught_exceptions_before;
	};

  public:
	struct policy_set {
		task_manager::policy_set tm;
		distributed_graph_generator::policy_set dggen;
		instruction_graph_generator::policy_set iggen;
	};

	idag_test_context(
	    const size_t num_nodes, const node_id local_nid, const size_t num_devices_per_node, bool supports_d2d_copies = true, const policy_set& policy = {})
	    : m_num_nodes(num_nodes), m_local_nid(local_nid), m_num_devices_per_node(num_devices_per_node),
	      m_uncaught_exceptions_before(std::uncaught_exceptions()), m_tm(num_nodes, &m_task_recorder, policy.tm), m_cmd_recorder(), m_cdag(),
	      m_dggen(num_nodes, local_nid, m_cdag, m_tm, &m_cmd_recorder, policy.dggen), m_instr_recorder(),
	      m_iggen(m_tm, num_nodes, local_nid, make_system_info(num_devices_per_node, supports_d2d_copies), m_idag, nullptr /* delegate */, &m_instr_recorder,
	          policy.iggen) //
	{
		REQUIRE(local_nid < num_nodes);
		REQUIRE(num_devices_per_node > 0);
	}

	~idag_test_context() {
		// instruction-graph-generator has no exception guarantees, so we must not call further member functions if one of them threw an exception
		if(m_uncaught_exceptions_before == std::uncaught_exceptions()) { finish(); }
		maybe_print_graphs();
	}

	idag_test_context(const idag_test_context&) = delete;
	idag_test_context(idag_test_context&&) = delete;
	idag_test_context& operator=(const idag_test_context&) = delete;
	idag_test_context& operator=(idag_test_context&&) = delete;

	static memory_id get_native_memory(const device_id did) { return first_device_memory_id + did; }

	/// Call this after issuing all submissions in order to trigger the shutdown epoch together with all cleanup instructions.
	void finish() {
		if(m_finished) return;

		for(auto iter = m_managed_objects.rbegin(); iter != m_managed_objects.rend(); ++iter) {
			matchbox::match(
			    *iter,
			    [&](const buffer_id bid) {
				    m_iggen.notify_buffer_destroyed(bid);
				    m_dggen.notify_buffer_destroyed(bid);
				    m_tm.notify_buffer_destroyed(bid);
			    },
			    [&](const host_object_id hoid) {
				    m_iggen.notify_host_object_destroyed(hoid);
				    m_dggen.notify_host_object_destroyed(hoid);
				    m_tm.notify_host_object_destroyed(hoid);
			    });
		}
		build_task(m_tm.generate_epoch_task(epoch_action::shutdown));

		m_finished = true;
	}

	template <typename DataT, int Dims>
	test_utils::mock_buffer<Dims> create_buffer(range<Dims> size, bool mark_as_host_initialized = false) {
		if(m_finished) { FAIL("idag_test_context already finish()ed"); } // if{FAIL} instead of REQUIRE so we don't count these as passed assertions
		const uncaught_exception_guard guard(this);
		const buffer_id bid = m_next_buffer_id++;
		const auto buf = test_utils::mock_buffer<Dims>(bid, size);
		m_tm.notify_buffer_created(bid, range_cast<3>(size), mark_as_host_initialized);
		m_dggen.notify_buffer_created(bid, range_cast<3>(size), mark_as_host_initialized);
		m_iggen.notify_buffer_created(bid, range_cast<3>(size), sizeof(DataT), alignof(DataT),
		    mark_as_host_initialized ? detail::allocation_id(detail::user_memory_id, m_next_user_allocation_id++) : detail::null_allocation_id);
		m_managed_objects.emplace_back(bid);
		return buf;
	}

	template <int Dims>
	test_utils::mock_buffer<Dims> create_buffer(range<Dims> size, bool mark_as_host_initialized = false) {
		return create_buffer<float, Dims>(size, mark_as_host_initialized);
	}

	test_utils::mock_host_object create_host_object(const bool owns_instance = true) {
		if(m_finished) { FAIL("idag_test_context already finish()ed"); }
		const uncaught_exception_guard guard(this);
		const host_object_id hoid = m_next_host_object_id++;
		m_tm.notify_host_object_created(hoid);
		m_dggen.notify_host_object_created(hoid);
		m_iggen.notify_host_object_created(hoid, owns_instance);
		m_managed_objects.emplace_back(hoid);
		return test_utils::mock_host_object(hoid);
	}

	template <typename Name = unnamed_kernel, int Dims>
	auto device_compute(const range<Dims>& global_size, const id<Dims>& global_offset = {}) {
		return task_builder(*this).template device_compute<Name>(global_size, global_offset);
	}

	template <typename Name = unnamed_kernel, int Dims>
	auto device_compute(const nd_range<Dims>& execution_range) {
		return task_builder(*this).template device_compute<Name>(execution_range);
	}

	template <int Dims>
	auto host_task(const range<Dims>& global_size) {
		return task_builder(*this).host_task(global_size);
	}

	auto master_node_host_task() { return task_builder(*this).master_node_host_task(); }

	auto collective_host_task(experimental::collective_group group = experimental::default_collective_group) {
		return task_builder(*this).collective_host_task(group);
	}

	task_id fence(test_utils::mock_host_object ho) {
		side_effect_map side_effects;
		side_effects.add_side_effect(ho.get_id(), experimental::side_effect_order::sequential);
		return fence({}, std::move(side_effects), std::make_unique<mock_host_object_fence_promise>());
	}

	template <int Dims>
	task_id fence(test_utils::mock_buffer<Dims> buf, subrange<Dims> sr) {
		buffer_access_map access_map;
		access_map.add_access(buf.get_id(),
		    std::make_unique<range_mapper<Dims, celerity::access::fixed<Dims>>>(celerity::access::fixed<Dims>(sr), access_mode::read, buf.get_range()));
		return fence(std::move(access_map), {}, std::make_unique<mock_buffer_fence_promise>(create_user_allocation()));
	}

	template <int Dims>
	task_id fence(test_utils::mock_buffer<Dims> buf) {
		return fence(buf, {{}, buf.get_range()});
	}

	task_id epoch(epoch_action action) {
		if(m_finished) { FAIL("idag_test_context already finish()ed"); }
		const uncaught_exception_guard guard(this);
		const auto tid = m_tm.generate_epoch_task(action);
		build_task(tid);
		return tid;
	}

	void set_horizon_step(const int step) { m_tm.set_horizon_step(step); }

	task_manager& get_task_manager() { return m_tm; }

	distributed_graph_generator& get_graph_generator() { return m_dggen; }

	instruction_query<> query_instructions() const { return instruction_query<>(m_instr_recorder); }

	pilot_query query_outbound_pilots() const { return pilot_query(m_instr_recorder); }

	[[nodiscard]] std::string print_task_graph() { //
		return detail::print_task_graph(m_task_recorder, make_test_graph_title("Task Graph"));
	}
	[[nodiscard]] std::string print_command_graph() {
		return detail::print_command_graph(m_local_nid, m_cmd_recorder, make_test_graph_title("Command Graph", m_num_nodes, m_local_nid));
	}
	[[nodiscard]] std::string print_instruction_graph() {
		return detail::print_instruction_graph(
		    m_instr_recorder, m_cmd_recorder, m_task_recorder, make_test_graph_title("Instruction Graph", m_num_nodes, m_local_nid, m_num_devices_per_node));
	}

  private:
	size_t m_num_nodes;
	node_id m_local_nid;
	size_t m_num_devices_per_node;
	int m_uncaught_exceptions_before;
	buffer_id m_next_buffer_id = 0;
	host_object_id m_next_host_object_id = 0;
	reduction_id m_next_reduction_id = no_reduction_id + 1;
	detail::raw_allocation_id m_next_user_allocation_id = 1;
	std::vector<std::variant<buffer_id, host_object_id>> m_managed_objects;
	std::optional<task_id> m_most_recently_built_horizon;
	task_recorder m_task_recorder;
	task_manager m_tm;
	command_recorder m_cmd_recorder;
	command_graph m_cdag;
	distributed_graph_generator m_dggen;
	instruction_graph m_idag;
	instruction_recorder m_instr_recorder;
	instruction_graph_generator m_iggen;
	bool m_finished = false;

	allocation_id create_user_allocation() { return detail::allocation_id(detail::user_memory_id, m_next_user_allocation_id++); }

	reduction_info create_reduction(const buffer_id bid, const bool include_current_buffer_value) {
		return reduction_info{m_next_reduction_id++, bid, include_current_buffer_value};
	}

	template <typename CGF, typename... Hints>
	task_id submit_command_group(CGF cgf, Hints... hints) {
		if(m_finished) { FAIL("idag_test_context already finish()ed"); }
		const uncaught_exception_guard guard(this);
		return m_tm.submit_command_group(cgf, hints...);
	}

	void build_task(const task_id tid) {
		if(m_finished) { FAIL("idag_test_context already finish()ed"); }
		const uncaught_exception_guard guard(this);
		const auto commands = detail::sort_topologically(m_dggen.build_task(*m_tm.get_task(tid)));
		for(const auto cmd : commands) {
			m_iggen.compile(*cmd);
		}
	}

	void maybe_build_horizon() {
		if(m_finished) { FAIL("idag_test_context already finish()ed"); }
		const uncaught_exception_guard guard(this);
		const auto current_horizon = task_manager_testspy::get_current_horizon(m_tm);
		if(m_most_recently_built_horizon != current_horizon) {
			assert(current_horizon.has_value());
			build_task(*current_horizon);
		}
		m_most_recently_built_horizon = current_horizon;
	}

	void maybe_print_graphs() {
		if(test_utils::g_print_graphs) {
			fmt::print("\n{}\n", print_task_graph());
			fmt::print("\n{}\n", print_command_graph());
			fmt::print("\n{}\n", print_instruction_graph());
		}
	}

	task_id fence(buffer_access_map access_map, side_effect_map side_effects, std::unique_ptr<fence_promise> promise) {
		if(m_finished) { FAIL("idag_test_context already finish()ed"); }
		const uncaught_exception_guard guard(this);
		const auto tid = m_tm.generate_fence_task(std::move(access_map), std::move(side_effects), std::move(promise));
		build_task(tid);
		maybe_build_horizon();
		return tid;
	}
};

} // namespace celerity::test_utils

template <typename Record>
struct fmt::formatter<celerity::test_utils::instruction_query<Record>> : fmt::formatter<size_t> {
	format_context::iterator format(const celerity::test_utils::instruction_query<Record>& iq, format_context& ctx) const {
		auto out = ctx.out();
		*out++ = '{';
		for(size_t i = 0; i < iq.m_result.size(); ++i) {
			if(i > 0) { out = std::copy_n(", ", 2, out); }
			fmt::format_to(out, "I{}", iq.m_result[i]->id);
		}
		*out++ = '}';
		return out;
	}
};

template <typename Record>
struct Catch::StringMaker<celerity::test_utils::instruction_query<Record>> {
	static std::string convert(const celerity::test_utils::instruction_query<Record>& iq) { return fmt::format("{}", iq); }
};
