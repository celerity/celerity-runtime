#pragma once

#include "cgf.h"
#include "graph.h"
#include "grid.h"
#include "hint.h"
#include "intrusive_graph.h"
#include "ranges.h"
#include "reduction.h"
#include "types.h"
#include "utils.h"

#include <cassert>
#include <cstddef>
#include <memory>
#include <source_location>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <matchbox.hh>

#include "command_graph.h" // NOCOMMIT For CDAG loop template
#include "log.h"           // NOCOMMIT Just for debugging


namespace celerity {
namespace detail {

	class buffer_access_map {
	  public:
		/// Default ctor for tasks w/o buffer accesses
		buffer_access_map() = default;

		buffer_access_map(std::vector<buffer_access>&& accesses, const task_geometry& geometry);

		const std::unordered_set<buffer_id>& get_accessed_buffers() const& { return m_accessed_buffers; }

		size_t get_num_accesses() const { return m_accesses.size(); }

		std::pair<buffer_id, access_mode> get_nth_access(const size_t n) const {
			const auto& [bid, mode, _, _2] = m_accesses[n];
			return {bid, mode};
		}

		// TODO: This should probably be retunred by get_nth_access instead
		bool is_replicated(const size_t n) const { return m_accesses[n].is_replicated; }

		region<3> get_requirements_for_nth_access(const size_t n, const box<3>& execution_range) const;

		/// Returns the union of all consumer accesses made across the entire task (conceptually, the
		/// union of the set of regions obtained by calling get_consumed_region for each chunk).
		region<3> get_task_consumed_region(const buffer_id bid) const {
			if(auto it = m_task_consumed_regions.find(bid); it != m_task_consumed_regions.end()) { return it->second; }
			return {};
		}

		/// Returns the union of all producer accesses made across the entire task (conceptually, the
		/// union of the set of regions obtained by calling get_produced_region for each chunk).
		region<3> get_task_produced_region(const buffer_id bid) const {
			if(auto it = m_task_produced_regions.find(bid); it != m_task_produced_regions.end()) { return it->second; }
			return {};
		};

		/// Computes the union of all consumed regions (across multiple accesses) for a given execution range.
		region<3> compute_consumed_region(const buffer_id bid, const box<3>& execution_range) const;

		/// Computes the union of all produced regions (across multiple accesses) for a given execution range.
		region<3> compute_produced_region(const buffer_id bid, const box<3>& execution_range) const;

		/// Returns a set of bounding boxes, one for each accessed region, that must be allocated contiguously.
		box_vector<3> compute_required_contiguous_boxes(const buffer_id bid, const box<3>& execution_range) const;

	  private:
		std::vector<buffer_access> m_accesses;
		std::unordered_set<buffer_id> m_accessed_buffers; ///< Cached set of buffer ids found in m_accesses
		range<3> m_task_global_size;
		int m_task_dimensions = -1;
		std::unordered_map<buffer_id, region<3>> m_task_consumed_regions;
		std::unordered_map<buffer_id, region<3>> m_task_produced_regions;
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

		side_effect_map() = default;

		side_effect_map(const std::vector<host_object_effect>& side_effects) {
			map_base::reserve(side_effects.size());
			for(const auto& [hoid, order] : side_effects) {
				map_base::emplace(hoid, order);
			}
		}

		using map_base::size, map_base::count, map_base::empty, map_base::cbegin, map_base::cend, map_base::at;

		iterator begin() const { return cbegin(); }
		iterator end() const { return cend(); }

		bool operator==(const side_effect_map& other) const { return static_cast<const map_base&>(*this) == static_cast<const map_base&>(other); }
	};

	// TODO refactor into an inheritance hierarchy
	class task : public intrusive_graph_node<task> {
	  public:
		task_type get_type() const { return m_type; }

		task_id get_id() const { return m_tid; }

		collective_group_id get_collective_group_id() const { return m_cgid; }

		const buffer_access_map& get_buffer_access_map() const { return m_access_map; }

		const side_effect_map& get_side_effect_map() const { return m_side_effects; }

		const task_geometry& get_geometry() const { return m_geometry; }

		void set_debug_name(const std::string& debug_name) { m_debug_name = debug_name; }
		const std::string& get_debug_name() const { return m_debug_name; }

		bool has_variable_split() const {
			return (m_type == task_type::host_compute || m_type == task_type::device_compute) && std::holds_alternative<basic_task_geometry>(m_geometry);
		}

		execution_target get_execution_target() const {
			switch(m_type) {
			case task_type::epoch: return execution_target::none;
			case task_type::device_compute: return execution_target::device;
			case task_type::host_compute:
			case task_type::collective:
			case task_type::master_node: return execution_target::host;
			case task_type::horizon:
			case task_type::fence: return execution_target::none;
			default: utils::unreachable(); // LCOV_EXCL_LINE
			}
		}

		const reduction_set& get_reductions() const { return m_reductions; }

		epoch_action get_epoch_action() const { return m_epoch_action; }

		task_promise* get_task_promise() const { return m_promise.get(); }

		template <typename Launcher>
		Launcher get_launcher() const {
			return std::get<Launcher>(m_launcher);
		}

		void add_hint(std::unique_ptr<hint_base>&& h) { m_hints.emplace_back(std::move(h)); }

		// NOCOMMIT Hack
		performance_assertions perf_assertions;

		template <typename Hint>
		const Hint* get_hint() const {
			static_assert(std::is_base_of_v<hint_base, Hint>, "Hint must extend hint_base");
			for(auto& h : m_hints) {
				if(auto* ptr = dynamic_cast<Hint*>(h.get()); ptr != nullptr) { return ptr; }
			}
			return nullptr;
		}

		static std::unique_ptr<task> make_epoch(task_id tid, detail::epoch_action action, std::unique_ptr<task_promise> promise) {
			return std::unique_ptr<task>(
			    new task(tid, task_type::epoch, non_collective_group_id, basic_task_geometry{}, {}, {}, {}, {}, action, std::move(promise)));
		}

		static std::unique_ptr<task> make_host_compute(task_id tid, task_geometry geometry, host_task_launcher launcher, buffer_access_map access_map,
		    side_effect_map side_effect_map, reduction_set reductions) {
			return std::unique_ptr<task>(new task(tid, task_type::host_compute, non_collective_group_id, std::move(geometry), std::move(launcher),
			    std::move(access_map), std::move(side_effect_map), std::move(reductions), {}, nullptr));
		}

		static std::unique_ptr<task> make_device_compute(
		    task_id tid, task_geometry geometry, device_kernel_launcher launcher, buffer_access_map access_map, reduction_set reductions) {
			return std::unique_ptr<task>(new task(tid, task_type::device_compute, non_collective_group_id, std::move(geometry), std::move(launcher),
			    std::move(access_map), {}, std::move(reductions), {}, nullptr));
		}

		static std::unique_ptr<task> make_collective(task_id tid, task_geometry geometry, collective_group_id cgid, size_t num_collective_nodes,
		    host_task_launcher launcher, buffer_access_map access_map, side_effect_map side_effect_map) {
			// The geometry is required to construct the buffer_access_map, so we pass it in here even though it has to have a specific shape
			assert(get_dimensions(geometry) == 1 && get_global_size(geometry) == detail::range_cast<3>(range(num_collective_nodes))
			       && get_global_offset(geometry) == zeros);
			return std::unique_ptr<task>(
			    new task(tid, task_type::collective, cgid, geometry, std::move(launcher), std::move(access_map), std::move(side_effect_map), {}, {}, nullptr));
		}

		static std::unique_ptr<task> make_master_node(task_id tid, host_task_launcher launcher, buffer_access_map access_map, side_effect_map side_effect_map) {
			return std::unique_ptr<task>(new task(tid, task_type::master_node, non_collective_group_id, basic_task_geometry{}, std::move(launcher),
			    std::move(access_map), std::move(side_effect_map), {}, {}, nullptr));
		}

		static std::unique_ptr<task> make_horizon(task_id tid) {
			return std::unique_ptr<task>(new task(tid, task_type::horizon, non_collective_group_id, basic_task_geometry{}, {}, {}, {}, {}, {}, nullptr));
		}

		static std::unique_ptr<task> make_fence(
		    task_id tid, buffer_access_map access_map, side_effect_map side_effect_map, std::unique_ptr<task_promise> promise) {
			return std::unique_ptr<task>(new task(tid, task_type::fence, non_collective_group_id, basic_task_geometry{}, {}, std::move(access_map),
			    std::move(side_effect_map), {}, {}, std::move(promise)));
		}

	  private:
		task_id m_tid;
		task_type m_type;
		collective_group_id m_cgid;
		task_geometry m_geometry;
		command_group_launcher m_launcher;
		buffer_access_map m_access_map;
		detail::side_effect_map m_side_effects;
		reduction_set m_reductions;
		std::string m_debug_name;
		detail::epoch_action m_epoch_action;
		std::unique_ptr<task_promise> m_promise; // TODO keep user_allocation_id in struct task instead of inside task_promise
		std::vector<std::unique_ptr<hint_base>> m_hints;

		task(task_id tid, task_type type, collective_group_id cgid, task_geometry geometry, command_group_launcher launcher, buffer_access_map access_map,
		    detail::side_effect_map side_effects, reduction_set reductions, detail::epoch_action epoch_action, std::unique_ptr<task_promise> promise)
		    : m_tid(tid), m_type(type), m_cgid(cgid), m_geometry(geometry), m_launcher(std::move(launcher)), m_access_map(std::move(access_map)),
		      m_side_effects(std::move(side_effects)), m_reductions(std::move(reductions)), m_epoch_action(epoch_action), m_promise(std::move(promise)) {
			assert(type == task_type::host_compute || type == task_type::device_compute || std::get<basic_task_geometry>(m_geometry).granularity.size() == 1);
			// Only host tasks can have side effects
			assert(this->m_side_effects.empty() || type == task_type::host_compute || type == task_type::collective || type == task_type::master_node
			       || type == task_type::fence);
		}
	};

	std::unique_ptr<detail::task> make_command_group_task(const detail::task_id tid, const size_t num_collective_nodes, raw_command_group&& cg);

	[[nodiscard]] std::string print_task_debug_label(const task& tsk, bool title_case = false);

	/// Determines which overlapping regions appear between write accesses when the iteration space of `tsk` is split into `chunks`.
	std::unordered_map<buffer_id, region<3>> detect_overlapping_writes(const task& tsk, const box_vector<3>& chunks);

	/// The task graph (TDAG) represents all cluster-wide operations, such as command group submissions and fences, and their interdependencies.
	class task_graph : public graph<task> {}; // inheritance instead of type alias so we can forward declare task_graph

} // namespace detail
} // namespace celerity
