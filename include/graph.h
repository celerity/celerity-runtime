#pragma once

#include "types.h"

#include <algorithm>
#include <cassert>
#include <concepts>
#include <deque>
#include <memory>
#include <vector>


namespace celerity::detail {

template <typename Node>
concept GraphNode = requires(const Node& node) {
	{ node.get_id() < node.get_id() } -> std::convertible_to<bool>;
};

/// A `graph` maintains ownership of all nodes that have not been pruned by epoch or horizon application.
template <GraphNode Node>
class graph {
	friend struct graph_testspy;

  public:
	// Call this before pushing horizon or epoch node in order to be able to call erase_before_epoch on the same task id later.
	void begin_epoch(const task_id tid) {
		assert(m_epochs.empty() || (m_epochs.back().epoch_tid < tid && !m_epochs.back().nodes.empty()));
		m_epochs.push_back({tid, {}});
	}

	// Retain ownership of a graph node until the current epoch is erased. The node's id must be higher than any node inserted into the graph before.
	template <std::derived_from<Node> N>
	N* retain_in_current_epoch(std::unique_ptr<N> node) {
		assert(!m_epochs.empty());
		auto& nodes = m_epochs.back().nodes;
		assert(nodes.empty() || nodes.back()->get_id() < node->get_id());
		const auto ptr = node.get();
		nodes.push_back(std::move(node));
		return ptr;
	}

	// Free all graph nodes that were pushed before begin_epoch(tid) was called.
	void erase_before_epoch(const task_id tid) {
		const auto first_retained = std::find_if(m_epochs.begin(), m_epochs.end(), [=](const epoch& epoch) { return epoch.epoch_tid >= tid; });
		assert(first_retained != m_epochs.end() && first_retained->epoch_tid == tid);
		m_epochs.erase(m_epochs.begin(), first_retained);
	}

  private:
	struct epoch {
		task_id epoch_tid;
		std::vector<std::unique_ptr<Node>> nodes; // graph node pointers are stable, so it is safe to hand them to another thread
	};

	std::deque<epoch> m_epochs;
};

} // namespace celerity::detail
