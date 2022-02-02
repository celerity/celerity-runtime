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
		EPOCH,
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

	enum class epoch_action {
		none,
		barrier,
		shutdown,
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

	struct task_geometry {
		int dimensions = 0;
		cl::sycl::range<3> global_size{0, 0, 0};
		cl::sycl::id<3> global_offset{};
		cl::sycl::range<3> granularity{1, 1, 1};
	};

	class task : public intrusive_graph_node<task> {
	  public:
		task_type get_type() const { return type; }

		task_id get_id() const { return tid; }

		collective_group_id get_collective_group_id() const { return cgid; }

		const buffer_access_map& get_buffer_access_map() const { return access_map; }

		const side_effect_map& get_side_effect_map() const { return side_effects; }

		const command_group_storage_base& get_command_group() const { return *cgf; }

		const task_geometry& get_geometry() const { return geometry; }

		int get_dimensions() const { return geometry.dimensions; }

		cl::sycl::range<3> get_global_size() const { return geometry.global_size; }

		cl::sycl::id<3> get_global_offset() const { return geometry.global_offset; }

		cl::sycl::range<3> get_granularity() const { return geometry.granularity; }

		const std::string& get_debug_name() const { return debug_name; }

		bool has_variable_split() const { return type == task_type::HOST_COMPUTE || type == task_type::DEVICE_COMPUTE; }

		execution_target get_execution_target() const {
			switch(type) {
			case task_type::EPOCH: return execution_target::NONE;
			case task_type::DEVICE_COMPUTE: return execution_target::DEVICE;
			case task_type::HOST_COMPUTE:
			case task_type::COLLECTIVE:
			case task_type::MASTER_NODE: return execution_target::HOST;
			case task_type::HORIZON: return execution_target::NONE;
			default: assert(!"Unhandled task type"); return execution_target::NONE;
			}
		}

		const std::vector<reduction_id>& get_reductions() const { return reductions; }

		epoch_action get_epoch_action() const { return epoch_action; }

		static std::unique_ptr<task> make_epoch(task_id tid, epoch_action action) {
			return std::unique_ptr<task>(new task(tid, task_type::EPOCH, collective_group_id{}, task_geometry{}, nullptr, {}, {}, {}, {}, action));
		}

		static std::unique_ptr<task> make_host_compute(task_id tid, task_geometry geometry, std::unique_ptr<command_group_storage_base> cgf,
		    buffer_access_map access_map, side_effect_map side_effect_map, std::vector<reduction_id> reductions) {
			return std::unique_ptr<task>(new task(tid, task_type::HOST_COMPUTE, collective_group_id{}, geometry, std::move(cgf), std::move(access_map),
			    std::move(side_effect_map), std::move(reductions), {}, {}));
		}

		static std::unique_ptr<task> make_device_compute(task_id tid, task_geometry geometry, std::unique_ptr<command_group_storage_base> cgf,
		    buffer_access_map access_map, std::vector<reduction_id> reductions, std::string debug_name) {
			return std::unique_ptr<task>(new task(tid, task_type::DEVICE_COMPUTE, collective_group_id{}, geometry, std::move(cgf), std::move(access_map), {},
			    std::move(reductions), std::move(debug_name), {}));
		}

		static std::unique_ptr<task> make_collective(task_id tid, collective_group_id cgid, size_t num_collective_nodes,
		    std::unique_ptr<command_group_storage_base> cgf, buffer_access_map access_map, side_effect_map side_effect_map) {
			const task_geometry geometry{1, detail::range_cast<3>(cl::sycl::range<1>{num_collective_nodes}), {}, {1, 1, 1}};
			return std::unique_ptr<task>(
			    new task(tid, task_type::COLLECTIVE, cgid, geometry, std::move(cgf), std::move(access_map), std::move(side_effect_map), {}, {}, {}));
		}

		static std::unique_ptr<task> make_master_node(
		    task_id tid, std::unique_ptr<command_group_storage_base> cgf, buffer_access_map access_map, side_effect_map side_effect_map) {
			return std::unique_ptr<task>(new task(tid, task_type::MASTER_NODE, collective_group_id{}, task_geometry{}, std::move(cgf), std::move(access_map),
			    std::move(side_effect_map), {}, {}, {}));
		}

		static std::unique_ptr<task> make_horizon_task(task_id tid) {
			return std::unique_ptr<task>(new task(tid, task_type::HORIZON, collective_group_id{}, task_geometry{}, nullptr, {}, {}, {}, {}, {}));
		}

	  private:
		task_id tid;
		task_type type;
		collective_group_id cgid;
		task_geometry geometry;
		std::unique_ptr<command_group_storage_base> cgf;
		buffer_access_map access_map;
		detail::side_effect_map side_effects;
		std::vector<reduction_id> reductions;
		std::string debug_name;
		detail::epoch_action epoch_action;

		task(task_id tid, task_type type, collective_group_id cgid, task_geometry geometry, std::unique_ptr<command_group_storage_base> cgf,
		    buffer_access_map access_map, detail::side_effect_map side_effects, std::vector<reduction_id> reductions, std::string debug_name,
		    detail::epoch_action epoch_action)
		    : tid(tid), type(type), cgid(cgid), geometry(geometry), cgf(std::move(cgf)), access_map(std::move(access_map)),
		      side_effects(std::move(side_effects)), reductions(std::move(reductions)), debug_name(std::move(debug_name)), epoch_action(epoch_action) {
			assert(type == task_type::HOST_COMPUTE || type == task_type::DEVICE_COMPUTE || get_granularity().size() == 1);
			assert((type != task_type::HOST_COMPUTE && type != task_type::COLLECTIVE && type != task_type::MASTER_NODE) || side_effects.empty());
		}
	};

} // namespace detail
} // namespace celerity
