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

		void push_overlapping_data(node_id source_nid, raw_buffer_data data) { overlapping_data.emplace_back(source_nid, std::move(data)); }

		virtual void reduce_to_buffer() = 0;

		reduction_info get_info() const { return info; }

	  protected:
		reduction_info info;
		std::vector<std::pair<node_id, raw_buffer_data>> overlapping_data;
	};

	template <typename DataT, int Dims, typename BinaryOperation>
	class buffer_reduction final : public abstract_buffer_reduction {
	  public:
		buffer_reduction(buffer_id bid, BinaryOperation op, DataT identity, bool include_current_buffer_value)
		    : abstract_buffer_reduction(bid, include_current_buffer_value), op(op), init(identity) {}

		void reduce_to_buffer() override {
			std::sort(overlapping_data.begin(), overlapping_data.end(), [](auto& lhs, auto& rhs) { return lhs.first < rhs.first; });

			DataT acc = init;
			for(auto& [nid, data] : overlapping_data) {
				assert(data.get_range() == cl::sycl::range<3>(1, 1, 1));
				DataT other = *static_cast<const DataT*>(data.get_pointer());
				acc = op(acc, other);
			}

			raw_buffer_data raw(sizeof(DataT), cl::sycl::range<3>{1, 1, 1});
			memcpy(raw.get_pointer(), &acc, sizeof(DataT));
			runtime::get_instance().get_buffer_manager().set_buffer_data(info.output_buffer_id, {}, std::move(raw));
		}

	  private:
		BinaryOperation op;
		DataT init;
	};

	class reduction_manager {
	  public:
		template <typename DataT, int Dims, typename BinaryOperation>
		reduction_id create_reduction(const buffer_id bid, BinaryOperation op, DataT identity, bool include_current_buffer_value) {
			std::lock_guard lock{mutex};
			auto rid = next_rid++;
			reductions.emplace(rid, std::make_unique<buffer_reduction<DataT, Dims, BinaryOperation>>(bid, op, identity, include_current_buffer_value));
			return rid;
		}

		bool has_reduction(reduction_id rid) const {
			std::lock_guard lock{mutex};
			return reductions.count(rid) != 0;
		}

		reduction_info get_reduction(reduction_id rid) {
			std::lock_guard lock{mutex};
			return reductions.at(rid)->get_info();
		}

		void push_overlapping_reduction_data(reduction_id rid, node_id source_nid, raw_buffer_data data) {
			std::lock_guard lock{mutex};
			reductions.at(rid)->push_overlapping_data(source_nid, std::move(data));
		}

		void finish_reduction(reduction_id rid) {
			std::lock_guard lock{mutex};
			reductions.at(rid)->reduce_to_buffer();
			reductions.erase(rid);
		}

	  private:
		mutable std::mutex mutex;
		reduction_id next_rid = 1;
		std::unordered_map<reduction_id, std::unique_ptr<abstract_buffer_reduction>> reductions;
	};
} // namespace detail
} // namespace celerity
