#include "divergence_checker.h"

namespace celerity::detail::divergence_checker_detail {
bool divergence_block_chain::check_for_divergence() {
	add_new_hashes();

	const auto [min_hash_count, max_hash_count] = collect_hash_counts();

	if(min_hash_count == 0) {
		if(max_hash_count != 0 && m_local_nid == 0) {
			check_for_deadlock();
		} else if(max_hash_count == 0) {
			return true;
		}
		return false;
	}

	const per_node_task_hashes task_hashes = collect_hashes(min_hash_count);

	for(int j = 0; j < min_hash_count; ++j) {
		const divergence_map check_map = create_divergence_map(task_hashes, j);

		// If there is more than one hash for this task, we have a divergence!
		if(check_map.size() > 1) { reprot_divergence(check_map, j); }
	}

	clear(min_hash_count);

	return false;
}

void divergence_block_chain::reprot_divergence(const divergence_map& check_map, const int task_num) {
	if(m_local_nid == 0) { log_node_divergences(check_map, task_num + static_cast<int>(m_tasks_checked) + 1); }

	// sleep for local_nid * 100 ms such that we have a no lock synchronized output
	std::this_thread::sleep_for(std::chrono::milliseconds(m_local_nid * 100));

	log_task_record_once(check_map, task_num);

	m_communicator->barrier();

	throw std::runtime_error("Divergence in task graph detected");
}

void divergence_block_chain::add_new_hashes() {
	std::lock_guard<std::mutex> lock(m_task_records_mutex);
	for(size_t i = m_hashes_added; i < m_task_records.size(); ++i) {
		std::size_t seed = m_local_hashes.empty() ? m_last_hash : m_local_hashes.back();
		celerity::detail::utils::hash_combine(seed, std::hash<task_record>{}(m_task_records[i]));
		m_local_hashes.push_back(seed);
	}
	m_last_hash = m_local_hashes.empty() ? m_last_hash : m_local_hashes.back();
	m_hashes_added = m_task_records.size();
}

void divergence_block_chain::clear(const int min_progress) {
	m_local_hashes.erase(m_local_hashes.begin(), m_local_hashes.begin() + min_progress);
	m_tasks_checked += min_progress;

	m_last_cleared = std::chrono::steady_clock::now();
}

std::pair<int, int> divergence_block_chain::collect_hash_counts() {
	m_per_node_hash_counts[m_local_nid] = static_cast<int>(m_local_hashes.size());

	m_communicator->allgather_inplace(m_per_node_hash_counts.data(), 1);

	const auto [min, max] = std::minmax_element(m_per_node_hash_counts.cbegin(), m_per_node_hash_counts.cend());

	return {*min, *max};
}

per_node_task_hashes divergence_block_chain::collect_hashes(const int min_hash_count) const {
	per_node_task_hashes data(min_hash_count, m_num_nodes);

	m_communicator->allgather(m_local_hashes.data(), min_hash_count, data.data(), min_hash_count);

	return data;
}


divergence_map divergence_block_chain::create_divergence_map(const per_node_task_hashes& task_hashes, const int task_num) const {
	divergence_map check_map;
	for(node_id nid = 0; nid < m_num_nodes; ++nid) {
		check_map[task_hashes(nid, task_num)].push_back(nid);
	}
	return check_map;
}

void divergence_block_chain::check_for_deadlock() {
	auto diff = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - m_last_cleared);

	if(diff >= std::chrono::seconds(10) && diff - m_time_of_last_warning >= std::chrono::seconds(5)) {
		std::string warning = fmt::format("After {} seconds of waiting, node(s)", diff.count());

		std::vector<node_id> stuck_nodes;
		for(node_id nid = 0; nid < m_num_nodes; ++nid) {
			if(m_per_node_hash_counts[nid] == 0) stuck_nodes.push_back(nid);
		}
		warning += fmt::format(" {} ", fmt::join(stuck_nodes, ","));
		warning += "did not move to the next task. The runtime might be stuck.";

		CELERITY_WARN("{}", warning);
		m_time_of_last_warning = diff;
	}
}

void divergence_block_chain::log_node_divergences(const divergence_map& check_map, const int task_id) {
	std::string error = fmt::format("Divergence detected. Task Nr {} diverges on nodes:\n\n", task_id);
	for(auto& [hash, nodes] : check_map) {
		error += fmt::format("Following task-hash {:#x} resulted on {} ", hash, nodes.size() > 1 ? "nodes" : "node ");
		for(auto& node : nodes) {
			error += fmt::format("{} ", node);
		}
		error += "\n";
	}
	CELERITY_ERROR("{}", error);
}

void divergence_block_chain::log_task_record(const divergence_map& check_map, const task_record& task, const task_hash hash) {
	std::string task_record_output = fmt::format("Task record for hash {:#x}:\n\n", hash);
	task_record_output += fmt::format("id: {}, debug_name: {}, type: {}, cgid: {}\n", task.tid, task.debug_name, task.type, task.cgid);
	const auto& geometry = task.geometry;
	task_record_output += fmt::format("geometry:\n");
	task_record_output += fmt::format("\t dimensions: {}, global_size: {}, global_offset: {}, granularity: {}\n", geometry.dimensions, geometry.global_size,
	    geometry.global_offset, geometry.granularity);

	if(!task.reductions.empty()) {
		task_record_output += fmt::format("reductions: \n");
		for(const auto& red : task.reductions) {
			task_record_output += fmt::format(
			    "\t id: {}, bid: {}, buffer_name: {}, init_from_buffer: {}\n", red.rid, red.bid, red.buffer_name, red.init_from_buffer ? "true" : "false");
		}
	}

	if(!task.accesses.empty()) {
		task_record_output += fmt::format("accesses: \n");
		for(const auto& acc : task.accesses) {
			task_record_output += fmt::format("\t bid: {}, buffer_name: {}, mode: {}, req: {}\n", acc.bid, acc.buffer_name, acc.mode, acc.req);
		}
	}

	if(!task.side_effect_map.empty()) {
		task_record_output += fmt::format("side_effect_map: \n");
		for(const auto& [hoid, order] : task.side_effect_map) {
			task_record_output += fmt::format("\t hoid: {}, order: {}\n", hoid, order);
		}
	}

	if(!task.dependencies.empty()) {
		task_record_output += fmt::format("dependencies: \n");
		for(const auto& dep : task.dependencies) {
			task_record_output += fmt::format("\t node: {}, kind: {}, origin: {}\n", dep.node, dep.kind, dep.origin);
		}
	}
	CELERITY_ERROR("{}", task_record_output);
}

void divergence_block_chain::log_task_record_once(const divergence_map& check_map, const int task_num) {
	for(auto& [hash, nodes] : check_map) {
		if(nodes[0] == m_local_nid) {
			const auto task = thread_save_get_task_record(task_num + m_tasks_checked);
			log_task_record(check_map, task, hash);
		}
	}
}

void divergence_block_chain::add_new_task(const task_record& task) { //
	std::lock_guard<std::mutex> lock(m_task_records_mutex);
	// make copy of task record so that we can access it later
	m_task_records.emplace_back(task);
}

task_record divergence_block_chain::thread_save_get_task_record(const size_t task_num) {
	std::lock_guard<std::mutex> lock(m_task_records_mutex);
	return m_task_records[task_num];
}
} // namespace celerity::detail::divergence_checker_detail
