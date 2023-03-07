#pragma once

#include "buffer_manager.h"
#include "runtime.h"
#include "types.h"

#include <vector>

namespace celerity {
namespace detail {

	class abstract_buffer_reduction {
	  public:
		explicit abstract_buffer_reduction(const buffer_id output_bid) : m_output_bid(output_bid) {}
		virtual ~abstract_buffer_reduction() = default;

		void push_overlapping_data(node_id source_nid, unique_payload_ptr data) { m_overlapping_data.emplace_back(source_nid, std::move(data)); }

		virtual void reduce_to_buffer() = 0;

	  protected:
		buffer_id m_output_bid;
		std::vector<std::pair<node_id, unique_payload_ptr>> m_overlapping_data;
	};

	template <typename DataT, int Dims, typename BinaryOperation>
	class buffer_reduction final : public abstract_buffer_reduction {
	  public:
		buffer_reduction(buffer_id output_bid, BinaryOperation op, DataT identity) : abstract_buffer_reduction(output_bid), m_op(op), m_init(identity) {}

		void reduce_to_buffer() override {
			std::sort(m_overlapping_data.begin(), m_overlapping_data.end(), [](auto& lhs, auto& rhs) { return lhs.first < rhs.first; });

			DataT acc = m_init;
			for(auto& [nid, data] : m_overlapping_data) {
				acc = m_op(acc, *static_cast<const DataT*>(data.get_pointer()));
			}

			auto info = runtime::get_instance().get_buffer_manager().access_host_buffer(
			    m_output_bid, cl::sycl::access::mode::discard_write, cl::sycl::range<3>{1, 1, 1}, cl::sycl::id<3>{});
			*static_cast<DataT*>(info.ptr) = acc;
		}

	  private:
		BinaryOperation m_op;
		DataT m_init;
	};

	class reduction_manager {
	  public:
		template <typename DataT, int Dims, typename BinaryOperation>
		reduction_id create_reduction(const buffer_id bid, BinaryOperation op, DataT identity) {
			std::lock_guard lock{m_mutex};
			const auto rid = m_next_rid++;
			m_reductions.emplace(rid, std::make_unique<buffer_reduction<DataT, Dims, BinaryOperation>>(bid, op, identity));
			return rid;
		}

		bool has_reduction(reduction_id rid) const {
			std::lock_guard lock{m_mutex};
			return m_reductions.count(rid) != 0;
		}

		void push_overlapping_reduction_data(reduction_id rid, node_id source_nid, unique_payload_ptr data) {
			std::lock_guard lock{m_mutex};
			m_reductions.at(rid)->push_overlapping_data(source_nid, std::move(data));
		}

		void finish_reduction(reduction_id rid) {
			std::lock_guard lock{m_mutex};
			m_reductions.at(rid)->reduce_to_buffer();
			m_reductions.erase(rid);
		}

	  private:
		mutable std::mutex m_mutex;
		reduction_id m_next_rid = 1;
		std::unordered_map<reduction_id, std::unique_ptr<abstract_buffer_reduction>> m_reductions;
	};

} // namespace detail
} // namespace celerity
