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
		epoch,          ///< task epoch (graph-level serialization point)
		host_compute,   ///< host task with explicit global size and celerity-defined split
		device_compute, ///< device compute task
		collective,     ///< host task with implicit 1d global size = #ranks and fixed split
		master_node,    ///< zero-dimensional host task
		horizon,        ///< task horizon
		fence,          ///< promise-side of an async experimental::fence
	};

	enum class execution_target {
		none,
		host,
		device,
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
		void add_access(buffer_id bid, std::unique_ptr<range_mapper_base>&& rm) { m_map.emplace(bid, std::move(rm)); }

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
		    buffer_id bid, cl::sycl::access::mode mode, int kernel_dims, const subrange<3>& sr, const celerity::range<3>& global_size) const;

	  private:
		std::unordered_multimap<buffer_id, std::unique_ptr<range_mapper_base>> m_map;
	};

	using reduction_set = std::vector<reduction_info>;

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

	class fence_promise {
	  public:
		fence_promise() = default;
		fence_promise(const fence_promise&) = delete;
		fence_promise& operator=(const fence_promise&) = delete;
		virtual ~fence_promise() = default;

		virtual void fulfill() = 0;
	};

	struct task_geometry {
		int dimensions = 0;
		celerity::range<3> global_size{0, 0, 0};
		celerity::id<3> global_offset{};
		celerity::range<3> granularity{1, 1, 1};
	};

	class task : public intrusive_graph_node<task> {
	  public:
		task_type get_type() const { return m_type; }

		task_id get_id() const { return m_tid; }

		collective_group_id get_collective_group_id() const { return m_cgid; }

		const buffer_access_map& get_buffer_access_map() const { return m_access_map; }

		const side_effect_map& get_side_effect_map() const { return m_side_effects; }

		const command_group_storage_base& get_command_group() const { return *m_cgf; }

		const task_geometry& get_geometry() const { return m_geometry; }

		int get_dimensions() const { return m_geometry.dimensions; }

		celerity::range<3> get_global_size() const { return m_geometry.global_size; }

		celerity::id<3> get_global_offset() const { return m_geometry.global_offset; }

		celerity::range<3> get_granularity() const { return m_geometry.granularity; }

		const std::string& get_debug_name() const { return m_debug_name; }

		bool has_variable_split() const { return m_type == task_type::host_compute || m_type == task_type::device_compute; }

		execution_target get_execution_target() const {
			switch(m_type) {
			case task_type::epoch: return execution_target::none;
			case task_type::device_compute: return execution_target::device;
			case task_type::host_compute:
			case task_type::collective:
			case task_type::master_node: return execution_target::host;
			case task_type::horizon:
			case task_type::fence: return execution_target::none;
			default: assert(!"Unhandled task type"); return execution_target::none;
			}
		}

		const reduction_set& get_reductions() const { return m_reductions; }

		epoch_action get_epoch_action() const { return m_epoch_action; }

		fence_promise* get_fence_promise() const { return m_fence_promise.get(); }

		static std::unique_ptr<task> make_epoch(task_id tid, detail::epoch_action action) {
			return std::unique_ptr<task>(new task(tid, task_type::epoch, collective_group_id{}, task_geometry{}, nullptr, {}, {}, {}, {}, action, nullptr));
		}

		static std::unique_ptr<task> make_host_compute(task_id tid, task_geometry geometry, std::unique_ptr<command_group_storage_base> cgf,
		    buffer_access_map access_map, side_effect_map side_effect_map, reduction_set reductions) {
			return std::unique_ptr<task>(new task(tid, task_type::host_compute, collective_group_id{}, geometry, std::move(cgf), std::move(access_map),
			    std::move(side_effect_map), std::move(reductions), {}, {}, nullptr));
		}

		static std::unique_ptr<task> make_device_compute(task_id tid, task_geometry geometry, std::unique_ptr<command_group_storage_base> cgf,
		    buffer_access_map access_map, reduction_set reductions, std::string debug_name) {
			return std::unique_ptr<task>(new task(tid, task_type::device_compute, collective_group_id{}, geometry, std::move(cgf), std::move(access_map), {},
			    std::move(reductions), std::move(debug_name), {}, nullptr));
		}

		static std::unique_ptr<task> make_collective(task_id tid, collective_group_id cgid, size_t num_collective_nodes,
		    std::unique_ptr<command_group_storage_base> cgf, buffer_access_map access_map, side_effect_map side_effect_map) {
			const task_geometry geometry{1, detail::range_cast<3>(celerity::range<1>{num_collective_nodes}), {}, {1, 1, 1}};
			return std::unique_ptr<task>(
			    new task(tid, task_type::collective, cgid, geometry, std::move(cgf), std::move(access_map), std::move(side_effect_map), {}, {}, {}, nullptr));
		}

		static std::unique_ptr<task> make_master_node(
		    task_id tid, std::unique_ptr<command_group_storage_base> cgf, buffer_access_map access_map, side_effect_map side_effect_map) {
			return std::unique_ptr<task>(new task(tid, task_type::master_node, collective_group_id{}, task_geometry{}, std::move(cgf), std::move(access_map),
			    std::move(side_effect_map), {}, {}, {}, nullptr));
		}

		static std::unique_ptr<task> make_horizon(task_id tid) {
			return std::unique_ptr<task>(new task(tid, task_type::horizon, collective_group_id{}, task_geometry{}, nullptr, {}, {}, {}, {}, {}, nullptr));
		}

		static std::unique_ptr<task> make_fence(
		    task_id tid, buffer_access_map access_map, side_effect_map side_effect_map, std::unique_ptr<fence_promise> fence_promise) {
			return std::unique_ptr<task>(new task(tid, task_type::fence, collective_group_id{}, task_geometry{}, nullptr, std::move(access_map),
			    std::move(side_effect_map), {}, {}, {}, std::move(fence_promise)));
		}

	  private:
		task_id m_tid;
		task_type m_type;
		collective_group_id m_cgid;
		task_geometry m_geometry;
		std::unique_ptr<command_group_storage_base> m_cgf;
		buffer_access_map m_access_map;
		detail::side_effect_map m_side_effects;
		reduction_set m_reductions;
		std::string m_debug_name;
		detail::epoch_action m_epoch_action;
		std::unique_ptr<fence_promise> m_fence_promise;

		task(task_id tid, task_type type, collective_group_id cgid, task_geometry geometry, std::unique_ptr<command_group_storage_base> cgf,
		    buffer_access_map access_map, detail::side_effect_map side_effects, reduction_set reductions, std::string debug_name,
		    detail::epoch_action epoch_action, std::unique_ptr<fence_promise> fence_promise)
		    : m_tid(tid), m_type(type), m_cgid(cgid), m_geometry(geometry), m_cgf(std::move(cgf)), m_access_map(std::move(access_map)),
		      m_side_effects(std::move(side_effects)), m_reductions(std::move(reductions)), m_debug_name(std::move(debug_name)), m_epoch_action(epoch_action),
		      m_fence_promise(std::move(fence_promise)) {
			assert(type == task_type::host_compute || type == task_type::device_compute || get_granularity().size() == 1);
			// Only host tasks can have side effects
			assert(this->m_side_effects.empty() || type == task_type::host_compute || type == task_type::collective || type == task_type::master_node
			       || type == task_type::fence);
		}
	};

} // namespace detail
} // namespace celerity
