#pragma once

#include <cstddef>
#include <variant>

#include "intrusive_graph.h"
#include "mpi_support.h"
#include "ranges.h"
#include "task.h"
#include "types.h"
#include "utils.h"

namespace celerity {
namespace detail {

	enum class command_type { epoch, horizon, execution, data_request, push, await_push, reduction };

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
		abstract_command(command_id cid, node_id nid) : m_cid(cid), m_nid(nid) {}

	  public:
		virtual ~abstract_command() = 0;

		command_id get_cid() const { return m_cid; }

		node_id get_nid() const { return m_nid; }

		void mark_as_flushed() {
			assert(!m_flushed);
			m_flushed = true;
		}
		bool is_flushed() const { return m_flushed; }

	  private:
		// Should only be possible to add/remove dependencies using command_graph.
		using parent_type = intrusive_graph_node<abstract_command>;
		using parent_type::add_dependency;
		using parent_type::remove_dependency;

		command_id m_cid;
		node_id m_nid;
		bool m_flushed = false;
	};
	inline abstract_command::~abstract_command() {}

	class push_command final : public abstract_command {
		friend class command_graph;
		push_command(command_id cid, node_id nid, buffer_id bid, reduction_id rid, node_id target, transaction_id trid, subrange<3> push_range)
		    : abstract_command(cid, nid), m_bid(bid), m_rid(rid), m_target(target), m_trid(trid), m_push_range(push_range) {}

	  public:
		buffer_id get_bid() const { return m_bid; }
		reduction_id get_rid() const { return m_rid; }
		node_id get_target() const { return m_target; }
		transaction_id get_transaction_id() const { return m_trid; }
		const subrange<3>& get_range() const { return m_push_range; }

	  private:
		buffer_id m_bid;
		reduction_id m_rid;
		node_id m_target;
		transaction_id m_trid;
		subrange<3> m_push_range;
	};

	class await_push_command final : public abstract_command {
		friend class command_graph;
		await_push_command(command_id cid, node_id nid, transaction_id trid) : abstract_command(cid, nid), m_trid(trid) {}

	  public:
		transaction_id get_transaction_id() const { return m_trid; }

	  private:
		transaction_id m_trid;
	};

	class data_request_command final : public abstract_command {
		friend class command_graph;
		data_request_command(command_id cid, node_id nid, buffer_id bid, node_id source, subrange<3> data_range)
		    : abstract_command(cid, nid), m_bid(bid), m_source(source), m_data_range(data_range) {}

	  public:
		buffer_id get_bid() const { return m_bid; }
		node_id get_source() const { return m_source; }
		const subrange<3>& get_range() const { return m_data_range; }

	  private:
		buffer_id m_bid;
		node_id m_source;
		subrange<3> m_data_range;
	};

	class reduction_command final : public abstract_command {
		friend class command_graph;
		reduction_command(command_id cid, node_id nid, const reduction_info& info) : abstract_command(cid, nid), m_info(info) {}

	  public:
		const reduction_info& get_reduction_info() const { return m_info; }

	  private:
		reduction_info m_info;
	};

	class task_command : public abstract_command {
	  protected:
		task_command(command_id cid, node_id nid, task_id tid) : abstract_command(cid, nid), m_tid(tid) {}

	  public:
		task_id get_tid() const { return m_tid; }

	  private:
		task_id m_tid;
	};

	class epoch_command final : public task_command {
		friend class command_graph;
		epoch_command(const command_id& cid, const node_id& nid, const task_id& tid, epoch_action action) : task_command(cid, nid, tid), m_action(action) {}

	  public:
		epoch_action get_epoch_action() const { return m_action; }

	  private:
		epoch_action m_action;
	};

	class horizon_command final : public task_command {
		friend class command_graph;
		using task_command::task_command;
	};

	class execution_command final : public task_command {
		friend class command_graph;

	  protected:
		execution_command(command_id cid, node_id nid, task_id tid, subrange<3> execution_range)
		    : task_command(cid, nid, tid), m_execution_range(execution_range) {}

	  public:
		const subrange<3>& get_execution_range() const { return m_execution_range; }

		void set_is_reduction_initializer(bool is_initializer) { m_initialize_reductions = is_initializer; }

		bool is_reduction_initializer() const { return m_initialize_reductions; }

	  private:
		subrange<3> m_execution_range;
		bool m_initialize_reductions = false;
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

	struct data_request_data { // ...
		buffer_id bid;
		node_id source;
		subrange<3> sr;
	};

	struct reduction_data {
		reduction_id rid;
	};

	using command_data = std::variant<std::monostate, horizon_data, epoch_data, execution_data, push_data, await_push_data, data_request_data, reduction_data>;

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
			    [](const horizon_data&) { return command_type::horizon; },
			    [](const epoch_data&) { return command_type::epoch; },
			    [](const execution_data&) { return command_type::execution; },
			    [](const push_data&) { return command_type::push; },
			    [](const await_push_data&) { return command_type::await_push; },
				[](const data_request_data&) { return command_type::data_request; },
			    [](const reduction_data&) { return command_type::reduction; }
			);
			// clang-format on
		}
	};

	struct command_frame {
		using payload_type = command_id;

		command_pkg pkg;
		size_t num_dependencies = 0;
		payload_type dependencies[];

		// variable-sized structure
		command_frame() = default;
		command_frame(const command_frame&) = delete;
		command_frame& operator=(const command_frame&) = delete;

		iterable_range<const command_id*> iter_dependencies() const { return {dependencies, dependencies + num_dependencies}; }
	};

	// unique_frame_ptr assumes that the flexible payload member begins at exactly sizeof(Frame) bytes
	static_assert(offsetof(command_frame, dependencies) == sizeof(command_frame));

} // namespace detail
} // namespace celerity
