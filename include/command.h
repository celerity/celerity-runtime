#pragma once

#include "intrusive_graph.h"
#include "ranges.h"
#include "task.h"
#include "types.h"

#include <matchbox.hh>

namespace celerity {
namespace detail {

	enum class command_type { epoch, horizon, execution, push, await_push, reduction, fence };

	class abstract_command : public intrusive_graph_node<abstract_command>,
	                         // Accept visitors to enable matchbox::match() on the command inheritance hierarchy
	                         public matchbox::acceptor<class epoch_command, class horizon_command, class execution_command, class push_command,
	                             class await_push_command, class reduction_command, class fence_command> {
		friend class command_graph;

	  protected:
		abstract_command(command_id cid) : m_cid(cid) {}

	  public:
		virtual command_type get_type() const = 0;

		command_id get_cid() const { return m_cid; }

	  private:
		// Should only be possible to add/remove dependencies using command_graph.
		using parent_type = intrusive_graph_node<abstract_command>;
		using parent_type::add_dependency;
		using parent_type::remove_dependency;

		command_id m_cid;
	};

	class push_command final : public matchbox::implement_acceptor<abstract_command, push_command> {
		friend class command_graph;
		push_command(const command_id cid, const node_id target, const transfer_id& trid, const subrange<3>& push_range)
		    : acceptor_base(cid), m_target(target), m_trid(trid), m_push_range(push_range) {}

		command_type get_type() const override { return command_type::push; }

	  public:
		node_id get_target() const { return m_target; }
		const transfer_id& get_transfer_id() const { return m_trid; }
		const subrange<3>& get_range() const { return m_push_range; }

	  private:
		node_id m_target;
		transfer_id m_trid;
		subrange<3> m_push_range;
	};

	class await_push_command final : public matchbox::implement_acceptor<abstract_command, await_push_command> {
		friend class command_graph;
		await_push_command(const command_id cid, const transfer_id& trid, region<3> region) : acceptor_base(cid), m_trid(trid), m_region(std::move(region)) {}

		command_type get_type() const override { return command_type::await_push; }

	  public:
		const transfer_id& get_transfer_id() const { return m_trid; }
		const region<3>& get_region() const { return m_region; }

	  private:
		transfer_id m_trid;
		region<3> m_region;
	};

	class reduction_command final : public matchbox::implement_acceptor<abstract_command, reduction_command> {
		friend class command_graph;
		reduction_command(command_id cid, const reduction_info& info, const bool has_local_contribution)
		    : acceptor_base(cid), m_info(info), m_has_local_contribution(has_local_contribution) {}

		command_type get_type() const override { return command_type::reduction; }

	  public:
		const reduction_info& get_reduction_info() const { return m_info; }
		bool has_local_contribution() const { return m_has_local_contribution; }

	  private:
		reduction_info m_info;
		bool m_has_local_contribution;
	};

	class task_command : public abstract_command {
	  protected:
		task_command(command_id cid, task_id tid) : abstract_command(cid), m_tid(tid) {}

	  public:
		task_id get_tid() const { return m_tid; }

	  private:
		task_id m_tid;
	};

	class epoch_command final : public matchbox::implement_acceptor<task_command, epoch_command> {
		friend class command_graph;
		epoch_command(const command_id cid, const task_id tid, const epoch_action action, std::vector<reduction_id> completed_reductions)
		    : acceptor_base(cid, tid), m_action(action), m_completed_reductions(std::move(completed_reductions)) {}

		command_type get_type() const override { return command_type::epoch; }

	  public:
		epoch_action get_epoch_action() const { return m_action; }
		const std::vector<reduction_id>& get_completed_reductions() const { return m_completed_reductions; }

	  private:
		epoch_action m_action;
		std::vector<reduction_id> m_completed_reductions;
	};

	class horizon_command final : public matchbox::implement_acceptor<task_command, horizon_command> {
		friend class command_graph;
		horizon_command(const command_id cid, const task_id tid, std::vector<reduction_id> completed_reductions)
		    : acceptor_base(cid, tid), m_completed_reductions(std::move(completed_reductions)) {}

		command_type get_type() const override { return command_type::horizon; }

	  public:
		const std::vector<reduction_id>& get_completed_reductions() const { return m_completed_reductions; }

	  private:
		std::vector<reduction_id> m_completed_reductions;
	};

	class execution_command final : public matchbox::implement_acceptor<task_command, execution_command> {
		friend class command_graph;

	  protected:
		execution_command(command_id cid, task_id tid, subrange<3> execution_range) : acceptor_base(cid, tid), m_execution_range(execution_range) {}

	  public:
		command_type get_type() const override { return command_type::execution; }

		const subrange<3>& get_execution_range() const { return m_execution_range; }

		void set_is_reduction_initializer(bool is_initializer) { m_initialize_reductions = is_initializer; }

		bool is_reduction_initializer() const { return m_initialize_reductions; }

	  private:
		subrange<3> m_execution_range;
		bool m_initialize_reductions = false;
	};

	class fence_command final : public matchbox::implement_acceptor<task_command, fence_command> {
		friend class command_graph;
		using acceptor_base::acceptor_base;

		command_type get_type() const override { return command_type::fence; }
	};

	/// Hash function for `unordered_sets/maps` of `command *` that is deterministic even as allocation addresses change between application runs.
	struct command_hash_by_id {
		template <typename Pointer>
		constexpr size_t operator()(const Pointer instr) const {
			return std::hash<command_id>()(instr->get_cid());
		}
	};

	using command_set = std::unordered_set<abstract_command*, command_hash_by_id>;

} // namespace detail
} // namespace celerity
