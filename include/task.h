#pragma once

#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "device_queue.h"
#include "grid.h"
#include "hint.h"
#include "host_queue.h"
#include "intrusive_graph.h"
#include "launcher.h"
#include "lifetime_extending_state.h"
#include "range_mapper.h"
#include "types.h"

namespace celerity {

class handler;

namespace detail {

	class command_launcher_storage_base {
	  public:
		command_launcher_storage_base() = default;
		command_launcher_storage_base(const command_launcher_storage_base&) = delete;
		command_launcher_storage_base(command_launcher_storage_base&&) = default;
		command_launcher_storage_base& operator=(const command_launcher_storage_base&) = delete;
		command_launcher_storage_base& operator=(command_launcher_storage_base&&) = default;
		virtual ~command_launcher_storage_base() = default;

		virtual sycl::event operator()(
		    device_queue& q, const subrange<3> execution_sr, const std::vector<void*>& reduction_ptrs, const bool is_reduction_initializer) const = 0;
		virtual std::future<host_queue::execution_info> operator()(host_queue& q, const subrange<3>& execution_sr) const = 0;
	};

	template <typename Functor>
	class command_launcher_storage : public command_launcher_storage_base {
	  public:
		command_launcher_storage(Functor&& fun) : m_fun(std::move(fun)) {}

		sycl::event operator()(
		    device_queue& q, const subrange<3> execution_sr, const std::vector<void*>& reduction_ptrs, const bool is_reduction_initializer) const override {
			return invoke<sycl::event>(q, execution_sr, reduction_ptrs, is_reduction_initializer);
		}

		std::future<host_queue::execution_info> operator()(host_queue& q, const subrange<3>& execution_sr) const override {
			return invoke<std::future<host_queue::execution_info>>(q, execution_sr);
		}

	  private:
		Functor m_fun;

		template <typename Ret, typename... Args>
		Ret invoke(Args&&... args) const {
			if constexpr(std::is_invocable_v<Functor, Args...>) {
				return m_fun(args...);
			} else {
				throw std::runtime_error("Cannot launch command function with provided arguments");
			}
		}
	};

	class buffer_access_map {
	  public:
		void add_access(buffer_id bid, std::unique_ptr<range_mapper_base>&& rm) { m_accesses.emplace_back(bid, std::move(rm)); }

		std::unordered_set<buffer_id> get_accessed_buffers() const;
		std::unordered_set<cl::sycl::access::mode> get_access_modes(buffer_id bid) const;
		size_t get_num_accesses() const { return m_accesses.size(); }
		std::pair<buffer_id, access_mode> get_nth_access(const size_t n) const {
			const auto& [bid, rm] = m_accesses[n];
			return {bid, rm->get_access_mode()};
		}

		/**
		 * @brief Computes the combined access-region for a given buffer, mode and subrange.
		 *
		 * @param bid
		 * @param mode
		 * @param sr The subrange to be passed to the range mappers (extended to a chunk using the global size of the task)
		 *
		 * @returns The region obtained by merging the results of all range-mappers for this buffer and mode
		 */
		region<3> get_mode_requirements(
		    const buffer_id bid, const access_mode mode, const int kernel_dims, const subrange<3>& sr, const range<3>& global_size) const;

		box<3> get_requirements_for_nth_access(const size_t n, const int kernel_dims, const subrange<3>& sr, const range<3>& global_size) const;

		std::vector<const range_mapper_base*> get_range_mappers(const buffer_id bid) const {
			std::vector<const range_mapper_base*> rms;
			for(const auto& [a_bid, a_rm] : m_accesses) {
				if(a_bid == bid) { rms.push_back(a_rm.get()); }
			}
			return rms;
		}

		box_vector<3> get_required_contiguous_boxes(const buffer_id bid, const int kernel_dims, const subrange<3>& sr, const range<3>& global_size) const;

	  private:
		std::vector<std::pair<buffer_id, std::unique_ptr<range_mapper_base>>> m_accesses;
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
		virtual allocation_id get_user_allocation_id() = 0;
	};

	struct task_geometry {
		int dimensions = 0;
		range<3> global_size{1, 1, 1};
		id<3> global_offset{};
		range<3> granularity{1, 1, 1};
	};

	class task : public intrusive_graph_node<task> {
	  public:
		task_type get_type() const { return m_type; }

		task_id get_id() const { return m_tid; }

		collective_group_id get_collective_group_id() const { return m_cgid; }

		const buffer_access_map& get_buffer_access_map() const { return m_access_map; }

		const side_effect_map& get_side_effect_map() const { return m_side_effects; }

		const task_geometry& get_geometry() const { return m_geometry; }

		int get_dimensions() const { return m_geometry.dimensions; }

		range<3> get_global_size() const { return m_geometry.global_size; }

		id<3> get_global_offset() const { return m_geometry.global_offset; }

		range<3> get_granularity() const { return m_geometry.granularity; }

		void set_debug_name(const std::string& debug_name) { m_debug_name = debug_name; }
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

		template <typename Launcher>
		Launcher get_launcher() const {
			return {};
		} // placeholder

		template <typename... Args>
		auto launch(Args&&... args) const {
			return (*m_launcher)(std::forward<Args>(args)...);
		}

		void extend_lifetime(std::shared_ptr<lifetime_extending_state> state) { m_attached_state.emplace_back(std::move(state)); }

		void add_hint(std::unique_ptr<hint_base>&& h) { m_hints.emplace_back(std::move(h)); }

		template <typename Hint>
		const Hint* get_hint() const {
			static_assert(std::is_base_of_v<hint_base, Hint>, "Hint must extend hint_base");
			for(auto& h : m_hints) {
				if(auto* ptr = dynamic_cast<Hint*>(h.get()); ptr != nullptr) { return ptr; }
			}
			return nullptr;
		}

		static std::unique_ptr<task> make_epoch(task_id tid, detail::epoch_action action) {
			return std::unique_ptr<task>(new task(tid, task_type::epoch, non_collective_group_id, task_geometry{}, nullptr, {}, {}, {}, action, nullptr));
		}

		static std::unique_ptr<task> make_host_compute(task_id tid, task_geometry geometry, std::unique_ptr<command_launcher_storage_base> launcher,
		    buffer_access_map access_map, side_effect_map side_effect_map, reduction_set reductions) {
			return std::unique_ptr<task>(new task(tid, task_type::host_compute, non_collective_group_id, geometry, std::move(launcher), std::move(access_map),
			    std::move(side_effect_map), std::move(reductions), {}, nullptr));
		}

		static std::unique_ptr<task> make_device_compute(task_id tid, task_geometry geometry, std::unique_ptr<command_launcher_storage_base> launcher,
		    buffer_access_map access_map, reduction_set reductions) {
			return std::unique_ptr<task>(new task(tid, task_type::device_compute, non_collective_group_id, geometry, std::move(launcher), std::move(access_map),
			    {}, std::move(reductions), {}, nullptr));
		}

		static std::unique_ptr<task> make_collective(task_id tid, collective_group_id cgid, size_t num_collective_nodes,
		    std::unique_ptr<command_launcher_storage_base> launcher, buffer_access_map access_map, side_effect_map side_effect_map) {
			const task_geometry geometry{1, detail::range_cast<3>(range(num_collective_nodes)), {}, {1, 1, 1}};
			return std::unique_ptr<task>(
			    new task(tid, task_type::collective, cgid, geometry, std::move(launcher), std::move(access_map), std::move(side_effect_map), {}, {}, nullptr));
		}

		static std::unique_ptr<task> make_master_node(
		    task_id tid, std::unique_ptr<command_launcher_storage_base> launcher, buffer_access_map access_map, side_effect_map side_effect_map) {
			return std::unique_ptr<task>(new task(tid, task_type::master_node, non_collective_group_id, task_geometry{}, std::move(launcher),
			    std::move(access_map), std::move(side_effect_map), {}, {}, nullptr));
		}

		static std::unique_ptr<task> make_horizon(task_id tid) {
			return std::unique_ptr<task>(new task(tid, task_type::horizon, non_collective_group_id, task_geometry{}, nullptr, {}, {}, {}, {}, nullptr));
		}

		static std::unique_ptr<task> make_fence(
		    task_id tid, buffer_access_map access_map, side_effect_map side_effect_map, std::unique_ptr<fence_promise> fence_promise) {
			return std::unique_ptr<task>(new task(tid, task_type::fence, non_collective_group_id, task_geometry{}, nullptr, std::move(access_map),
			    std::move(side_effect_map), {}, {}, std::move(fence_promise)));
		}

	  private:
		task_id m_tid;
		task_type m_type;
		collective_group_id m_cgid;
		task_geometry m_geometry;
		std::unique_ptr<command_launcher_storage_base> m_launcher;
		buffer_access_map m_access_map;
		detail::side_effect_map m_side_effects;
		reduction_set m_reductions;
		std::string m_debug_name;
		detail::epoch_action m_epoch_action;
		// TODO I believe that `struct task` should not store command_group_launchers, fence_promise or other state that is related to execution instead of
		// abstract DAG building. For user-initialized buffers we already notify the runtime -> executor of this state directly. Maybe also do that for these.
		std::unique_ptr<fence_promise> m_fence_promise;
		std::vector<std::shared_ptr<lifetime_extending_state>> m_attached_state;
		std::vector<std::unique_ptr<hint_base>> m_hints;

		task(task_id tid, task_type type, collective_group_id cgid, task_geometry geometry, std::unique_ptr<command_launcher_storage_base> launcher,
		    buffer_access_map access_map, detail::side_effect_map side_effects, reduction_set reductions, detail::epoch_action epoch_action,
		    std::unique_ptr<fence_promise> fence_promise)
		    : m_tid(tid), m_type(type), m_cgid(cgid), m_geometry(geometry), m_launcher(std::move(launcher)), m_access_map(std::move(access_map)),
		      m_side_effects(std::move(side_effects)), m_reductions(std::move(reductions)), m_epoch_action(epoch_action),
		      m_fence_promise(std::move(fence_promise)) {
			assert(type == task_type::host_compute || type == task_type::device_compute || get_granularity().size() == 1);
			// Only host tasks can have side effects
			assert(this->m_side_effects.empty() || type == task_type::host_compute || type == task_type::collective || type == task_type::master_node
			       || type == task_type::fence);
		}
	};

	[[nodiscard]] std::string print_task_debug_label(const task& tsk, bool title_case = false);

	/// Determines which overlapping regions appear between write accesses when the iteration space of `tsk` is split into `chunks`.
	std::unordered_map<buffer_id, region<3>> detect_overlapping_writes(const task& tsk, const box_vector<3>& chunks);

} // namespace detail
} // namespace celerity
