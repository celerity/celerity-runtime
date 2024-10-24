#pragma once

#include "types.h"

#include <memory>
#include <vector>

namespace celerity::detail {

/// A `graph` keeps ownership of all graph nodes that have not been pruned by epoch or horizon application.
template <typename Node, typename NodeIdLess>
class epoch_graph { // TODO naming? epochized? epoch_based? epoch_partitioned?
  public:
	// Call this before pushing horizon or epoch node in order to be able to call erase_before_epoch on the same task id later.
	void begin_epoch(const task_id tid) {
		assert(m_epochs.empty() || (m_epochs.back().epoch_tid < tid && !m_epochs.back().nodes.empty()));
		m_epochs.push_back({tid, {}});
	}

	// Add a graph node to the current epoch. Its graph node id must be higher than any node inserted into the graph before.
	void append(std::unique_ptr<Node> graph_node) {
		assert(!m_epochs.empty());
		auto& nodes = m_epochs.back().nodes;
		assert(nodes.empty() || NodeIdLess{}(nodes.back().get(), graph_node.get()));
		nodes.push_back(std::move(graph_node));
	}

	// Free all graph nodes that were pushed before begin_epoch(tid) was called.
	void prune_before_epoch(const task_id tid) {
		const auto first_retained = std::partition_point(m_epochs.begin(), m_epochs.end(), [=](const graph_epoch& epoch) { return epoch.epoch_tid < tid; });
		assert(first_retained != m_epochs.end() && first_retained->epoch_tid == tid);
		m_epochs.erase(m_epochs.begin(), first_retained);
	}

	// The total number of graph nodes currently owned and not yet pruned, across all epochs.
	size_t get_live_node_count() const {
		size_t count = 0;
		for(auto& epoch : m_epochs) {
			count += epoch.nodes.size();
		}
		return count;
	}

  private:
	struct graph_epoch {
		task_id epoch_tid;
		std::vector<std::unique_ptr<Node>> nodes; // graph node pointers are stable, so it is safe to hand them to another thread
	};

	std::vector<graph_epoch> m_epochs;
};

} // namespace celerity::detail
