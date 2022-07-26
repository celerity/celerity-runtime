#pragma once

#include "buffer_manager.h"
#include "runtime.h"
#include "types.h"

#include <vector>

namespace celerity {
namespace detail {

	struct reduction_info {
		buffer_id output_buffer_id = 0;
		bool initialize_from_buffer = false;
	};

	class abstract_buffer_reduction {
	  public:
		explicit abstract_buffer_reduction(buffer_id bid, bool include_current_buffer_value) : info{bid, include_current_buffer_value} {}
		virtual ~abstract_buffer_reduction() = default;

		void push_overlapping_data(node_id source_nid, unique_payload_ptr data) { overlapping_data.emplace_back(source_nid, std::move(data)); }

		virtual void reduce_to_buffer() = 0;

		reduction_info get_info() const { return info; }

	  protected:
		reduction_info info;
		std::vector<std::pair<node_id, unique_payload_ptr>> overlapping_data;
	};

	template <typename DataT, int Dims, typename BinaryOperation>
	class buffer_reduction final : public abstract_buffer_reduction {
	  public:
		buffer_reduction(buffer_id bid, BinaryOperation op, DataT identity, bool include_current_buffer_value)
		    : abstract_buffer_reduction(bid, include_current_buffer_value), m_op(op), m_init(identity) {}

		void reduce_to_buffer() override {
			std::sort(overlapping_data.begin(), overlapping_data.end(), [](auto& lhs, auto& rhs) { return lhs.first < rhs.first; });

			DataT acc = m_init;
			for(auto& [nid, data] : overlapping_data) {
				acc = m_op(acc, *static_cast<const DataT*>(data.get_pointer()));
			}

			auto host_buf = runtime::get_instance().get_buffer_manager().get_host_buffer<DataT, Dims>(
			    info.output_buffer_id, cl::sycl::access::mode::discard_write, cl::sycl::range<3>{1, 1, 1}, cl::sycl::id<3>{});
			*host_buf.buffer.get_pointer() = acc;
		}

	  private:
		BinaryOperation m_op;
		DataT m_init;
	};

	class reduction_manager {
	  public:
		template <typename DataT, int Dims, typename BinaryOperation>
		reduction_id create_reduction(const buffer_id bid, BinaryOperation op, DataT identity, bool include_current_buffer_value) {
			std::lock_guard lock{m_mutex};
			auto rid = m_next_rid++;
			m_reductions.emplace(rid, std::make_unique<buffer_reduction<DataT, Dims, BinaryOperation>>(bid, op, identity, include_current_buffer_value));
			return rid;
		}

		bool has_reduction(reduction_id rid) const {
			std::lock_guard lock{m_mutex};
			return m_reductions.count(rid) != 0;
		}

		reduction_info get_reduction(reduction_id rid) {
			std::lock_guard lock{m_mutex};
			return m_reductions.at(rid)->get_info();
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
