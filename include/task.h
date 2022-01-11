#pragma once

#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "grid.h"
#include "intrusive_graph.h"
#include "range_mapper.h"
#include "types.h"

namespace celerity {

class handler;

namespace detail {

	enum class task_type {
		NOP,
		HOST_COMPUTE,   ///< host task with explicit global size and celerity-defined split
		DEVICE_COMPUTE, ///< device compute task
		COLLECTIVE,     ///< host task with implicit 1d global size = #ranks and fixed split
		MASTER_NODE,    ///< zero-dimensional host task
		HORIZON,        ///< task horizon
	};

	enum class execution_target {
		NONE,
		HOST,
		DEVICE,
	};

	struct command_group_storage_base {
		virtual void operator()(handler& cgh) const = 0;

		virtual ~command_group_storage_base() = default;
	};

	template <typename Functor>
	struct command_group_storage : command_group_storage_base {
		Functor fun;

		command_group_storage(Functor fun) : fun(fun) {}
		void operator()(handler& cgh) const override { fun(cgh); }
	};

	class buffer_access_map {
	  public:
		void add_access(buffer_id bid, std::unique_ptr<range_mapper_base>&& rm) { map.emplace(bid, std::move(rm)); }

		std::unordered_set<buffer_id> get_accessed_buffers() const;
		std::unordered_set<cl::sycl::access::mode> get_access_modes(buffer_id bid) const;

		/**
		 * @brief Computes the combined access-region for a given buffer, mode and subrange.
		 *
		 * @param bid
		 * @param mode
		 * @param sr The subrange to be passed to the range mappers (extended to a chunk using the global size of the task)
		 *
		 * @returns The region obtained by merging the results of all range-mappers for this buffer and mode
		 */
		GridRegion<3> get_requirements_for_access(
		    buffer_id bid, cl::sycl::access::mode mode, int kernel_dims, const subrange<3>& sr, const cl::sycl::range<3>& global_size) const;

	  private:
		std::unordered_multimap<buffer_id, std::unique_ptr<range_mapper_base>> map;
	};

	class side_effect_map : private std::unordered_map<host_object_id, experimental::side_effect_order> {
	  private:
		using map_base = std::unordered_map<host_object_id, experimental::side_effect_order>;

	  public:
		using typename map_base::const_iterator, map_base::value_type, map_base::key_type, map_base::mapped_type, map_base::const_reference,
		    map_base::const_pointer;
		using iterator = const_iterator;
		using reference = const_reference;
		using pointer = const_pointer;

		using map_base::size, map_base::count, map_base::empty, map_base::cbegin, map_base::cend, map_base::at;

		iterator begin() const { return cbegin(); }
		iterator end() const { return cend(); }
		iterator find(host_object_id key) const { return map_base::find(key); }

		void add_side_effect(host_object_id hoid, experimental::side_effect_order order);
	};

	class task : public intrusive_graph_node<task> {
	  public:
		task_type get_type() const { return type; }

		task_id get_id() const { return tid; }

		collective_group_id get_collective_group_id() const { return cgid; }

		const buffer_access_map& get_buffer_access_map() const { return access_map; }

		const side_effect_map& get_side_effect_map() const { return side_effect_map; }

		const command_group_storage_base& get_command_group() const { return *cgf; }

		int get_dimensions() const { return dimensions; }

		cl::sycl::range<3> get_global_size() const { return global_size; }

		cl::sycl::id<3> get_global_offset() const { return global_offset; }

		cl::sycl::range<3> get_granularity() const { return granularity; }

		const std::string& get_debug_name() const { return debug_name; }

		bool has_variable_split() const { return type == task_type::HOST_COMPUTE || type == task_type::DEVICE_COMPUTE; }

		execution_target get_execution_target() const {
			switch(type) {
			case task_type::NOP: return execution_target::NONE;
			case task_type::DEVICE_COMPUTE: return execution_target::DEVICE;
			case task_type::HOST_COMPUTE:
			case task_type::COLLECTIVE:
			case task_type::MASTER_NODE: return execution_target::HOST;
			case task_type::HORIZON: return execution_target::NONE;
			default: assert(!"Unhandled task type"); return execution_target::NONE;
			}
		}

		const std::vector<reduction_id>& get_reductions() const { return reductions; }

		static std::unique_ptr<task> make_nop(task_id tid) {
			return std::unique_ptr<task>(new task(tid, task_type::NOP, {}, 0, {0, 0, 0}, {}, {1, 1, 1}, nullptr, {}, {}, {}, {}));
		}

		static std::unique_ptr<task> make_host_compute(task_id tid, int dimensions, cl::sycl::range<3> global_size, cl::sycl::id<3> global_offset,
		    cl::sycl::range<3> granularity, std::unique_ptr<command_group_storage_base> cgf, buffer_access_map access_map, side_effect_map side_effect_map,
		    std::vector<reduction_id> reductions) {
			return std::unique_ptr<task>(new task(tid, task_type::HOST_COMPUTE, {}, dimensions, global_size, global_offset, granularity, std::move(cgf),
			    std::move(access_map), std::move(side_effect_map), std::move(reductions), {}));
		}

		static std::unique_ptr<task> make_device_compute(task_id tid, int dimensions, cl::sycl::range<3> global_size, cl::sycl::id<3> global_offset,
		    cl::sycl::range<3> granularity, std::unique_ptr<command_group_storage_base> cgf, buffer_access_map access_map, std::vector<reduction_id> reductions,
		    std::string debug_name) {
			return std::unique_ptr<task>(new task(tid, task_type::DEVICE_COMPUTE, {}, dimensions, global_size, global_offset, granularity, std::move(cgf),
			    std::move(access_map), {}, std::move(reductions), std::move(debug_name)));
		}

		static std::unique_ptr<task> make_collective(task_id tid, collective_group_id cgid, size_t num_collective_nodes,
		    std::unique_ptr<command_group_storage_base> cgf, buffer_access_map access_map, side_effect_map side_effect_map) {
			return std::unique_ptr<task>(new task(tid, task_type::COLLECTIVE, cgid, 1, detail::range_cast<3>(cl::sycl::range<1>{num_collective_nodes}), {},
			    {1, 1, 1}, std::move(cgf), std::move(access_map), std::move(side_effect_map), {}, {}));
		}

		static std::unique_ptr<task> make_master_node(
		    task_id tid, std::unique_ptr<command_group_storage_base> cgf, buffer_access_map access_map, side_effect_map side_effect_map) {
			return std::unique_ptr<task>(new task(
			    tid, task_type::MASTER_NODE, {}, 0, {0, 0, 0}, {}, {1, 1, 1}, std::move(cgf), std::move(access_map), std::move(side_effect_map), {}, {}));
		}

		static std::unique_ptr<task> make_horizon_task(task_id tid) {
			return std::unique_ptr<task>(new task(tid, task_type::HORIZON, {}, 0, {0, 0, 0}, {}, {1, 1, 1}, nullptr, {}, {}, {}, {}));
		}

	  private:
		task_id tid;
		task_type type;
		collective_group_id cgid;
		int dimensions;
		cl::sycl::range<3> global_size;
		cl::sycl::id<3> global_offset;
		cl::sycl::range<3> granularity;
		std::unique_ptr<command_group_storage_base> cgf;
		buffer_access_map access_map;
		detail::side_effect_map side_effect_map;
		std::vector<reduction_id> reductions;
		std::string debug_name;

		task(task_id tid, task_type type, collective_group_id cgid, int dimensions, cl::sycl::range<3> global_size, cl::sycl::id<3> global_offset,
		    cl::sycl::range<3> granularity, std::unique_ptr<command_group_storage_base> cgf, buffer_access_map access_map,
		    detail::side_effect_map side_effect_map, std::vector<reduction_id> reductions, std::string debug_name)
		    : tid(tid), type(type), cgid(cgid), dimensions(dimensions), global_size(global_size), global_offset(global_offset), granularity(granularity),
		      cgf(std::move(cgf)), access_map(std::move(access_map)), side_effect_map(std::move(side_effect_map)), reductions(std::move(reductions)),
		      debug_name(std::move(debug_name)) {
			assert(type == task_type::HOST_COMPUTE || type == task_type::DEVICE_COMPUTE || granularity.size() == 1);
			assert((type != task_type::HOST_COMPUTE && type != task_type::COLLECTIVE && type != task_type::MASTER_NODE) || side_effect_map.empty());
		}
	};

} // namespace detail
} // namespace celerity
