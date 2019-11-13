#pragma once

#include <boost/variant.hpp>

#include "intrusive_graph.h"
#include "ranges.h"
#include "types.h"

namespace celerity {
namespace detail {

	enum class command_type { NOP, COMPUTE, MASTER_ACCESS, PUSH, AWAIT_PUSH, SHUTDOWN, SYNC };
	constexpr const char* command_string[] = {"NOP", "COMPUTE", "MASTER_ACCESS", "PUSH", "AWAIT_PUSH", "SHUTDOWN", "SYNC"};

	// ----------------------------------------------------------------------------------------------------------------
	// ------------------------------------------------ COMMAND GRAPH -------------------------------------------------
	// ----------------------------------------------------------------------------------------------------------------

	// TODO: Consider using LLVM-style RTTI for better performance
	template <typename T, typename P>
	bool isa(P* p) {
		return dynamic_cast<T*>(const_cast<std::remove_const_t<P>*>(p)) != nullptr;
	}

	template <typename P, typename... Types>
	struct _isany_impl {
		bool operator()(const P* p) { return false; }
	};

	template <typename P, typename T, typename... Types>
	struct _isany_impl<P, T, Types...> {
		bool operator()(const P* p) { return isa<T>(p) || _isany_impl<P, Types...>{}(p); }
	};

	template <typename... Types, typename P>
	bool isany(const P* p) {
		return _isany_impl<P, Types...>{}(p);
	}

	// TODO: Consider adding a mechanism (during debug builds?) to assert that dependencies can only exist between commands on the same node
	class abstract_command : public intrusive_graph_node<abstract_command> {
	  public:
		abstract_command(command_id cid, node_id nid) : cid(cid), nid(nid) {}

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
		command_id cid;
		node_id nid;
		bool flushed = false;
	};

	inline abstract_command::~abstract_command() {}

	// Used for the init task.
	class nop_command : public abstract_command {
	  public:
		nop_command(command_id cid, node_id nid) : abstract_command(cid, nid) {
			// There's no point in flushing NOP commands.
			mark_as_flushed();
		}
	};

	class push_command : public abstract_command {
	  public:
		push_command(command_id cid, node_id nid, buffer_id bid, node_id target, subrange<3> push_range)
		    : abstract_command(cid, nid), bid(bid), target(target), push_range(push_range) {}

		buffer_id get_bid() const { return bid; }
		node_id get_target() const { return target; }
		const subrange<3>& get_range() const { return push_range; }

	  private:
		buffer_id bid;
		node_id target;
		subrange<3> push_range;
	};

	class await_push_command : public abstract_command {
	  public:
		await_push_command(command_id cid, node_id nid, push_command* source) : abstract_command(cid, nid), source(source) {}

		push_command* get_source() const { return source; }

	  private:
		push_command* source;
	};

	class task_command : public abstract_command {
	  public:
		task_command(command_id cid, node_id nid, task_id tid) : abstract_command(cid, nid), tid(tid) {}

		task_id get_tid() const { return tid; }

	  private:
		task_id tid;
	};

	class compute_command : public task_command {
	  public:
		compute_command(command_id cid, node_id nid, task_id tid, subrange<3> execution_range)
		    : task_command(cid, nid, tid), execution_range(execution_range) {}

		const subrange<3>& get_execution_range() const { return execution_range; }

	  private:
		subrange<3> execution_range;
	};

	class master_access_command : public task_command {
	  public:
		master_access_command(command_id cid, node_id nid, task_id tid) : task_command(cid, nid, tid) {}
	};

	// ----------------------------------------------------------------------------------------------------------------
	// -------------------------------------------- SERIALIZED COMMANDS -----------------------------------------------
	// ----------------------------------------------------------------------------------------------------------------

	struct nop_data {};

	struct compute_data {
		task_id tid;
		subrange<3> sr;

		compute_data(const compute_data&) = default;
	};

	struct master_access_data {
		task_id tid;
	};

	struct push_data {
		buffer_id bid;
		node_id target;
		subrange<3> sr;
	};

	struct await_push_data {
		buffer_id bid;
		node_id source;
		command_id source_cid;
		subrange<3> sr;
	};

	struct shutdown_data {};

	struct sync_data {
		uint64_t sync_id;
	};

	using command_data = boost::variant<nop_data, compute_data, master_access_data, push_data, await_push_data, shutdown_data, sync_data>;

	/**
	 * A command package is what is actually transferred between nodes.
	 */
	struct command_pkg {
		command_id cid;
		command_type cmd;
		command_data data;

		command_pkg() = default;
		command_pkg(command_id cid, command_type cmd, command_data data) : cid(cid), cmd(cmd), data(data) {}
	};

} // namespace detail
} // namespace celerity
