#pragma once

#include "divergence_checker.h"

using namespace celerity;
using namespace celerity::detail;
using namespace celerity::detail::divergence_checker_detail;

struct divergence_checker_detail::divergence_block_chain_testspy {
	static per_node_task_hashes pre_check(divergence_block_chain& div_test, const int max_size) {
		div_test.add_new_hashes();
		div_test.collect_hash_counts();
		return div_test.collect_hashes(max_size);
	}

	static void post_check(divergence_block_chain& div_test, const int min_size) { div_test.clear(min_size); }

	static void call_check_for_divergence_with_pre_post(std::vector<std::unique_ptr<divergence_block_chain>>& div_test) {
		std::vector<size_t> sizes;
		std::transform(div_test.begin(), div_test.end(), std::back_inserter(sizes), [](auto& div) { return div->m_task_records.size(); });
		auto [min, max] = std::minmax_element(sizes.begin(), sizes.end());

		std::vector<per_node_task_hashes> extended_lifetime_hashes;
		for(size_t i = 1; i < div_test.size(); i++) {
			extended_lifetime_hashes.push_back(divergence_block_chain_testspy::pre_check(*div_test[i], static_cast<const int>(*max)));
		}

		call_check_for_divergence(*div_test[0]);

		for(size_t i = 1; i < div_test.size(); i++) {
			divergence_block_chain_testspy::post_check(*div_test[i], static_cast<const int>(*min));
		}
	}

	static bool call_check_for_divergence(divergence_block_chain& div_test) { return div_test.check_for_divergence(); }


	static void set_last_cleared(divergence_block_chain& div_test, std::chrono::time_point<std::chrono::steady_clock> time) { div_test.m_last_cleared = time; }
};

namespace celerity::test_utils {
// Note: this is only a simulator for this specific case. In the general case, we should use something more sophisticated for tracking the allgather
// communication.
class divergence_test_communicator_provider {
  public:
	divergence_test_communicator_provider(size_t num_nodes) : m_num_nodes(num_nodes), m_inplace_data(num_nodes), m_gather_data(num_nodes) {}

	std::unique_ptr<communicator> create(node_id local_nid) {
		return std::make_unique<divergence_test_communicator>(local_nid, m_num_nodes, m_inplace_data, m_gather_data);
	}

  private:
	struct inplace_data {
		std::byte* data;
		int count;
	};

	struct gather_data {
		const std::byte* sendbuf;
		int sendcount;
		std::byte* recvbuf;
		int recvcount;
	};

	template <typename T>
	struct tracker {
		tracker(size_t num_nodes) : m_was_called(num_nodes), m_data(num_nodes) {}

		void operator()(T data, const node_id nid) {
			m_was_called[nid] = true;
			m_data[nid] = data;
		}

		bool all() const {
			return std::all_of(m_was_called.cbegin(), m_was_called.cend(), [](bool b) { return b; });
		}

		void reset() { std::fill(m_was_called.begin(), m_was_called.end(), false); }


		std::vector<bool> m_was_called;
		std::vector<T> m_data;
	};

	class divergence_test_communicator : public communicator {
	  public:
		divergence_test_communicator(node_id local_nid, size_t num_nodes, tracker<inplace_data>& inplace_data, tracker<gather_data>& gather_data)
		    : m_local_nid(local_nid), m_num_nodes(num_nodes), m_inplace_data(inplace_data), m_gather_data(gather_data) {}

	  private:
		node_id local_nid_impl() override { return m_local_nid; }
		size_t num_nodes_impl() override { return m_num_nodes; }

		void allgather_inplace_impl(std::byte* data, const int count) override {
			m_inplace_data({data, count}, m_local_nid);
			if(m_inplace_data.all()) {
				for(size_t i = 0; i < m_num_nodes; ++i) {
					for(size_t j = 0; j < m_num_nodes; ++j) {
						for(int k = 0; k < count; ++k) {
							if(j != i) { m_inplace_data.m_data[i].data[j * count + k] = m_inplace_data.m_data[j].data[j * count + k]; }
						}
					}
				}

				m_inplace_data.reset();
			}
		}

		void allgather_impl(const std::byte* sendbuf, const int sendcount, std::byte* recvbuf, const int recvcount) override {
			m_gather_data({sendbuf, sendcount, recvbuf, recvcount}, m_local_nid);
			if(m_gather_data.all()) {
				for(size_t i = 0; i < m_num_nodes; ++i) {
					for(size_t j = 0; j < m_num_nodes; ++j) {
						for(int k = 0; k < m_gather_data.m_data[i].sendcount; ++k) {
							m_gather_data.m_data[i].recvbuf[j * (m_gather_data.m_data[i].sendcount) + k] = m_gather_data.m_data[j].sendbuf[k];
						}
					}
				}

				m_gather_data.reset();
			}
		}

		void barrier_impl() override {}

		node_id m_local_nid = 0;
		size_t m_num_nodes = 1;

		tracker<inplace_data>& m_inplace_data;
		tracker<gather_data>& m_gather_data;
	};

	size_t m_num_nodes = 1;

	tracker<inplace_data> m_inplace_data{m_num_nodes};
	tracker<gather_data> m_gather_data{m_num_nodes};
};

} // namespace celerity::test_utils
