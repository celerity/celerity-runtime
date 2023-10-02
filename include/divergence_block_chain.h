#pragma once

#include <mutex>
#include <thread>
#include <vector>

#include "communicator.h"
#include "recorders.h"

namespace celerity::detail {
struct runtime_testspy;
}

namespace celerity::detail::divergence_checker_detail {
using task_hash = size_t;
using divergence_map = std::unordered_map<task_hash, std::vector<node_id>>;

/**
 * @brief Stores the hashes of tasks for each node.
 *
 * The data is stored densely so it can easily be exchanged through MPI collective operations.
 */
struct per_node_task_hashes {
  public:
	per_node_task_hashes(const size_t max_hash_count, const size_t num_nodes) : m_data(max_hash_count * num_nodes), m_max_hash_count(max_hash_count){};
	const task_hash& operator()(const node_id nid, const size_t i) const { return m_data.at(nid * m_max_hash_count + i); }
	task_hash* data() { return m_data.data(); }

  private:
	std::vector<task_hash> m_data;
	size_t m_max_hash_count;
};

/**
 *  @brief This class checks for divergences of tasks between nodes.
 *
 *  It is responsible for collecting the task hashes from all nodes and checking for differences -> divergence.
 *  When a divergence is found, the task record for the diverging task is printed and the program is terminated.
 *  Additionally it will also print a warning when a deadlock is suspected.
 */

class divergence_block_chain {
	friend struct divergence_block_chain_testspy;

  public:
	divergence_block_chain(task_recorder& task_recorder, std::unique_ptr<communicator> comm)
	    : m_local_nid(comm->get_local_nid()), m_num_nodes(comm->get_num_nodes()), m_per_node_hash_counts(comm->get_num_nodes()),
	      m_communicator(std::move(comm)) {
		task_recorder.add_callback([this](const task_record& task) { add_new_task(task); });
	}

	divergence_block_chain(const divergence_block_chain&) = delete;
	divergence_block_chain(divergence_block_chain&&) = delete;

	~divergence_block_chain() = default;

	divergence_block_chain& operator=(const divergence_block_chain&) = delete;
	divergence_block_chain& operator=(divergence_block_chain&&) = delete;

	bool check_for_divergence();

  private:
	node_id m_local_nid;
	size_t m_num_nodes;

	std::vector<task_hash> m_local_hashes;
	std::vector<task_record> m_task_records;
	size_t m_tasks_checked = 0;
	size_t m_hashes_added = 0;

	std::vector<int> m_per_node_hash_counts;
	std::mutex m_task_records_mutex;

	std::chrono::time_point<std::chrono::steady_clock> m_last_cleared = std::chrono::steady_clock::now();

	std::unique_ptr<communicator> m_communicator;

	void divergence_out(const divergence_map& check_map, const int task_num);

	void add_new_hashes();
	void clear(const int min_progress);
	std::pair<int, int> collect_hash_counts();
	per_node_task_hashes collect_hashes(const int min_hash_count) const;
	divergence_map create_check_map(const per_node_task_hashes& task_hashes, const int task_num) const;

	void check_for_deadlock() const;

	static void log_node_divergences(const divergence_map& check_map, const int task_num);
	static void log_task_record(const divergence_map& check_map, const task_record& task, const task_hash hash);
	void log_task_record_once(const divergence_map& check_map, const int task_num);

	void add_new_task(const task_record& task);
	task_record thread_save_get_task_record(const size_t task_num);
};

class divergence_checker {
	friend struct ::celerity::detail::runtime_testspy;

  public:
	divergence_checker(task_recorder& task_recorder, std::unique_ptr<communicator> comm, bool test_mode = false)
	    : m_block_chain(task_recorder, std::move(comm)) {
		if(!test_mode) { start(); }
	}

	divergence_checker(const divergence_checker&) = delete;
	divergence_checker(const divergence_checker&&) = delete;

	divergence_checker& operator=(const divergence_checker&) = delete;
	divergence_checker& operator=(divergence_checker&&) = delete;

	~divergence_checker() { stop(); }

  private:
	void start() {
		m_thread = std::thread(&divergence_checker::run, this);
		m_is_running = true;
	}

	void stop() {
		m_is_running = false;
		if(m_thread.joinable()) { m_thread.join(); }
	}

	void run() {
		bool is_finished = false;
		while(!is_finished || m_is_running) {
			is_finished = m_block_chain.check_for_divergence();

			std::this_thread::sleep_for(std::chrono::milliseconds(100));
		}
	}

	std::thread m_thread;
	bool m_is_running = false;
	divergence_block_chain m_block_chain;
};
}; // namespace celerity::detail::divergence_checker_detail
