#pragma once

#include <variant>

#include "intrusive_graph.h"
#include "ranges.h"
#include "task.h"
#include "types.h"
#include "utils.h"

namespace celerity {
namespace detail {

	enum class command_type { EPOCH, HORIZON, EXECUTION, PUSH, AWAIT_PUSH, REDUCTION };
	constexpr const char* command_string[] = {"EPOCH", "HORIZON", "EXECUTION", "PUSH", "AWAIT_PUSH", "REDUCTION"};

	// ----------------------------------------------------------------------------------------------------------------
	// ------------------------------------------------ COMMAND GRAPH -------------------------------------------------
	// ----------------------------------------------------------------------------------------------------------------

	// TODO: Consider using LLVM-style RTTI for better performance
	template <typename T, typename P>
	bool isa(P* p) {
		return dynamic_cast<T*>(const_cast<std::remove_const_t<P>*>(p)) != nullptr;
	}

	// TODO: Consider adding a mechanism (during debug builds?) to assert that dependencies can only exist between commands on the same node
	class abstract_command : public intrusive_graph_node<abstract_command> {
		friend class command_graph;

	  protected:
		abstract_command(command_id cid, node_id nid) : cid(cid), nid(nid) {}

	  public:
		virtual ~abstract_command() = 0;

		command_id get_cid() const { return cid; }

		node_id get_nid() const { return nid; }

		void mark_as_flushed() {
			assert(!flushed);
			flushed = true;
		}
		bool is_flushed() const { return flushed; }

		// TODO: Consider only having this in debug builds
		std::string debug_label;

	  private:
		// Should only be possible to add/remove dependencies using command_graph.
		using parent_type = intrusive_graph_node<abstract_command>;
		using parent_type::add_dependency;
		using parent_type::remove_dependency;

		command_id cid;
		node_id nid;
		bool flushed = false;
	};
	inline abstract_command::~abstract_command() {}

	class push_command final : public abstract_command {
		friend class command_graph;
		push_command(command_id cid, node_id nid, buffer_id bid, reduction_id rid, node_id target, subrange<3> push_range)
		    : abstract_command(cid, nid), bid(bid), rid(rid), target(target), push_range(push_range) {}

	  public:
		buffer_id get_bid() const { return bid; }
		reduction_id get_rid() const { return rid; }
		node_id get_target() const { return target; }
		const subrange<3>& get_range() const { return push_range; }

	  private:
		buffer_id bid;
		reduction_id rid;
		node_id target;
		subrange<3> push_range;
	};

	class await_push_command final : public abstract_command {
		friend class command_graph;
		await_push_command(command_id cid, node_id nid, push_command* source) : abstract_command(cid, nid), source(source) {}

	  public:
		push_command* get_source() const { return source; }

	  private:
		push_command* source;
	};

	class reduction_command final : public abstract_command {
		friend class command_graph;
		reduction_command(command_id cid, node_id nid, reduction_id rid) : abstract_command(cid, nid), rid(rid) {}

	  public:
		reduction_id get_rid() const { return rid; }

	  private:
		reduction_id rid;
	};

	class task_command : public abstract_command {
	  protected:
		task_command(command_id cid, node_id nid, task_id tid) : abstract_command(cid, nid), tid(tid) {}

	  public:
		task_id get_tid() const { return tid; }

	  private:
		task_id tid;
	};

	class epoch_command final : public task_command {
		friend class command_graph;
		epoch_command(const command_id& cid, const node_id& nid, const task_id& tid, epoch_action action) : task_command(cid, nid, tid), action(action) {}

	  public:
		epoch_action get_epoch_action() const { return action; }

	  private:
		epoch_action action;
	};

	class horizon_command final : public task_command {
		friend class command_graph;
		using task_command::task_command;
	};

	class execution_command final : public task_command {
		friend class command_graph;

	  protected:
		execution_command(command_id cid, node_id nid, task_id tid, subrange<3> execution_range)
		    : task_command(cid, nid, tid), execution_range(execution_range) {}

	  public:
		const subrange<3>& get_execution_range() const { return execution_range; }

		void set_is_reduction_initializer(bool is_initializer) { initialize_reductions = is_initializer; }

		bool is_reduction_initializer() const { return initialize_reductions; }

	  private:
		subrange<3> execution_range;
		bool initialize_reductions = false;
	};

	// ----------------------------------------------------------------------------------------------------------------
	// -------------------------------------------- SERIALIZED COMMANDS -----------------------------------------------
	// ----------------------------------------------------------------------------------------------------------------

	struct horizon_data {
		task_id tid;
	};

	struct epoch_data {
		task_id tid;
		epoch_action action;
	};

	struct execution_data {
		task_id tid;
		subrange<3> sr;
		bool initialize_reductions;
	};

	struct push_data {
		buffer_id bid;
		reduction_id rid;
		node_id target;
		subrange<3> sr;
	};

	struct await_push_data {
		buffer_id bid;
		reduction_id rid;
		node_id source;
		command_id source_cid;
		subrange<3> sr;
	};

	struct reduction_data {
		reduction_id rid;
	};

	using command_data = std::variant<std::monostate, horizon_data, epoch_data, execution_data, push_data, await_push_data, reduction_data>;

	/**
	 * A command package is what is actually transferred between nodes.
	 */
	struct command_pkg {
		command_id cid{};
		command_data data;

		std::optional<task_id> get_tid() const {
			// clang-format off
			return utils::match(data,
				[](const horizon_data& d) { return std::optional{d.tid}; },
				[](const epoch_data& d) { return std::optional{d.tid}; },
				[](const execution_data& d) { return std::optional{d.tid}; },
				[](const auto&) { return std::optional<task_id>{}; }
			);
			// clang-format on
		}

		command_type get_command_type() const {
			// clang-format off
			return utils::match(data,
			    [](const std::monostate&) -> command_type {
				    assert(!"calling get_command_type() on an empty command_pkg");
				    std::terminate();
			    },
			    [](const horizon_data&) { return command_type::HORIZON; },
			    [](const epoch_data&) { return command_type::EPOCH; },
			    [](const execution_data&) { return command_type::EXECUTION; },
			    [](const push_data&) { return command_type::PUSH; },
			    [](const await_push_data&) { return command_type::AWAIT_PUSH; },
			    [](const reduction_data&) { return command_type::REDUCTION; }
			);
			// clang-format on
		}

		// clang-format on
	};

} // namespace detail
} // namespace celerity
