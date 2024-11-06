#pragma once

#include "graph.h"
#include "grid.h"
#include "intrusive_graph.h"
#include "ranges.h"
#include "reduction.h"
#include "types.h"

#include <cstddef>
#include <functional>
#include <unordered_set>
#include <utility>
#include <vector>

#include <matchbox.hh>


namespace celerity::detail {

class task;

class command : public intrusive_graph_node<command>,
                // Accept visitors to enable matchbox::match() on the command inheritance hierarchy
                public matchbox::acceptor<class epoch_command, class horizon_command, class execution_command, class push_command, class await_push_command,
                    class reduction_command, class fence_command> {
	friend class command_graph;

  protected:
	explicit command(const command_id cid) : m_cid(cid) {}

  public:
	command_id get_id() const { return m_cid; }

  private:
	command_id m_cid;
};

class push_command final : public matchbox::implement_acceptor<command, push_command> {
  public:
	explicit push_command(const command_id cid, const transfer_id& trid, std::vector<std::pair<node_id, region<3>>> target_regions)
	    : acceptor_base(cid), m_trid(trid), m_target_regions(std::move(target_regions)) {}

	const transfer_id& get_transfer_id() const { return m_trid; }
	const std::vector<std::pair<node_id, region<3>>>& get_target_regions() const { return m_target_regions; }

  private:
	transfer_id m_trid;
	std::vector<std::pair<node_id, region<3>>> m_target_regions;
};

class await_push_command final : public matchbox::implement_acceptor<command, await_push_command> {
  public:
	explicit await_push_command(const command_id cid, const transfer_id& trid, region<3> region)
	    : acceptor_base(cid), m_trid(trid), m_region(std::move(region)) {}

	const transfer_id& get_transfer_id() const { return m_trid; }
	const region<3>& get_region() const { return m_region; }

  private:
	transfer_id m_trid;
	region<3> m_region;
};

class reduction_command final : public matchbox::implement_acceptor<command, reduction_command> {
  public:
	explicit reduction_command(command_id cid, const reduction_info& info, const bool has_local_contribution)
	    : acceptor_base(cid), m_info(info), m_has_local_contribution(has_local_contribution) {}

	const reduction_info& get_reduction_info() const { return m_info; }
	bool has_local_contribution() const { return m_has_local_contribution; }

  private:
	reduction_info m_info;
	bool m_has_local_contribution;
};

class task_command : public command {
  protected:
	explicit task_command(const command_id cid, const task* const tsk) : command(cid), m_task(tsk) {}

  public:
	const task* get_task() const { return m_task; }

  private:
	const task* m_task;
};

class epoch_command final : public matchbox::implement_acceptor<task_command, epoch_command> {
  public:
	explicit epoch_command(const command_id cid, const task* const tsk, const epoch_action action, std::vector<reduction_id> completed_reductions)
	    : acceptor_base(cid, tsk), m_action(action), m_completed_reductions(std::move(completed_reductions)) {}

	epoch_action get_epoch_action() const { return m_action; }
	const std::vector<reduction_id>& get_completed_reductions() const { return m_completed_reductions; }

  private:
	epoch_action m_action;
	std::vector<reduction_id> m_completed_reductions;
};

class horizon_command final : public matchbox::implement_acceptor<task_command, horizon_command> {
  public:
	explicit horizon_command(const command_id cid, const task* const tsk, std::vector<reduction_id> completed_reductions)
	    : acceptor_base(cid, tsk), m_completed_reductions(std::move(completed_reductions)) {}

	const std::vector<reduction_id>& get_completed_reductions() const { return m_completed_reductions; }

  private:
	std::vector<reduction_id> m_completed_reductions;
};

class execution_command final : public matchbox::implement_acceptor<task_command, execution_command> {
  public:
	explicit execution_command(const command_id cid, const task* const tsk, subrange<3> execution_range, const bool is_reduction_initializer)
	    : acceptor_base(cid, tsk), m_execution_range(execution_range), m_initialize_reductions(is_reduction_initializer) {}

	const subrange<3>& get_execution_range() const { return m_execution_range; }
	bool is_reduction_initializer() const { return m_initialize_reductions; }

  private:
	subrange<3> m_execution_range;
	bool m_initialize_reductions = false;
};

class fence_command final : public matchbox::implement_acceptor<task_command, fence_command> {
  public:
	explicit fence_command(const command_id cid, const task* const tsk) : acceptor_base(cid, tsk) {}
};

/// Hash function for `unordered_sets/maps` of `command *` that is deterministic even as allocation addresses change between application runs.
struct command_hash_by_id {
	template <typename Pointer>
	constexpr size_t operator()(const Pointer instr) const {
		return std::hash<command_id>()(instr->get_id());
	}
};

using command_set = std::unordered_set<command*, command_hash_by_id>;

/// The command graph (CDAG) provides a static schedule of commands executed on individual nodes, including kernel execution and peer-to-peer data transfers via
/// push- and await-push commands. It is generated in a distributed fashion, where each cluster node only maintains the subset of commands it will execute
/// itself.
class command_graph : public graph<command> {}; // inheritance instead of type alias so we can forward-declare command_graph

} // namespace celerity::detail
