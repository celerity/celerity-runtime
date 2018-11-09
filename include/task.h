#pragma once

#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "grid.h"
#include "range_mapper.h"
#include "types.h"

namespace celerity {

class compute_prepass_handler;
class compute_livepass_handler;
class master_access_prepass_handler;
class master_access_livepass_handler;

enum class task_type { COMPUTE, MASTER_ACCESS };

namespace detail {

	// This is a workaround that let's us store a command group functor with auto&
	// parameter, which we require in order to be able to pass different
	// celerity::handlers for prepass and live invocations.
	template <typename PrepassHandler, typename LivepassHandler>
	struct handler_storage_base {
		virtual void operator()(PrepassHandler&) const = 0;
		virtual void operator()(LivepassHandler&) const = 0;
		virtual ~handler_storage_base() = default;
	};

	template <typename Functor, typename PrepassHandler, typename LivepassHandler>
	struct handler_storage : handler_storage_base<PrepassHandler, LivepassHandler> {
		Functor fun;

		handler_storage(Functor fun) : fun(fun) {}

		void operator()(PrepassHandler& handler) const override { fun(handler); }
		void operator()(LivepassHandler& handler) const override { fun(handler); }
	};

	using cgf_storage_base = handler_storage_base<compute_prepass_handler, compute_livepass_handler>;
	template <typename Functor>
	using cgf_storage = handler_storage<Functor, compute_prepass_handler, compute_livepass_handler>;

	using maf_storage_base = handler_storage_base<master_access_prepass_handler, master_access_livepass_handler>;
	template <typename Functor>
	using maf_storage = handler_storage<Functor, master_access_prepass_handler, master_access_livepass_handler>;

	class task {
	  public:
		task(task_id tid) : tid(tid) {}
		virtual ~task() = default;

		virtual task_type get_type() const = 0;
		virtual std::vector<buffer_id> get_accessed_buffers() const = 0;
		virtual std::unordered_set<cl::sycl::access::mode> get_access_modes(buffer_id bid) const = 0;

		task_id get_id() const { return tid; }

	  private:
		task_id tid;
	};

	class compute_task : public task {
	  public:
		compute_task(task_id tid, std::unique_ptr<cgf_storage_base>&& cgf) : task(tid), cgf(std::move(cgf)) {}

		task_type get_type() const override { return task_type::COMPUTE; }

		void set_dimensions(int dims) { dimensions = dims; }
		void set_global_size(cl::sycl::range<3> gs) { global_size = gs; }
		void set_global_offset(cl::sycl::id<3> offset) { global_offset = offset; }
		void set_debug_name(std::string name) { debug_name = name; };

		void add_range_mapper(buffer_id bid, std::unique_ptr<range_mapper_base>&& rm) { range_mappers[bid].push_back(std::move(rm)); }

		const cgf_storage_base& get_command_group() const { return *cgf; }

		int get_dimensions() const { return dimensions; }
		cl::sycl::range<3> get_global_size() const { return global_size; }
		cl::sycl::id<3> get_global_offset() const { return global_offset; }
		std::string get_debug_name() const { return debug_name; }

		std::vector<buffer_id> get_accessed_buffers() const override;
		std::unordered_set<cl::sycl::access::mode> get_access_modes(buffer_id bid) const override;

		/**
		 * @brief Computes the combined access-region for a given buffer, mode and subrange.
		 *
		 * @param bid
		 * @param mode
		 * @param sr The subrange to be passed to the range mappers (extended to a chunk using the global size of the task)
		 *
		 * @returns The region obtained by merging the results of all range-mappers for this buffer and mode
		 */
		GridRegion<3> get_requirements(buffer_id bid, cl::sycl::access::mode mode, const subrange<3>& sr) const;

	  private:
		std::unique_ptr<cgf_storage_base> cgf;
		int dimensions = 0;
		cl::sycl::range<3> global_size;
		cl::sycl::id<3> global_offset = {};
		std::string debug_name;
		std::unordered_map<buffer_id, std::vector<std::unique_ptr<range_mapper_base>>> range_mappers;
	};

	class master_access_task : public task {
	  public:
		master_access_task(task_id tid, std::unique_ptr<maf_storage_base>&& maf) : task(tid), maf(std::move(maf)) {}

		task_type get_type() const override { return task_type::MASTER_ACCESS; }

		void add_buffer_access(buffer_id bid, cl::sycl::access::mode mode, subrange<3> sr) { buffer_accesses[bid].push_back({mode, sr}); }

		const maf_storage_base& get_functor() const { return *maf; }

		std::vector<buffer_id> get_accessed_buffers() const override;
		std::unordered_set<cl::sycl::access::mode> get_access_modes(buffer_id bid) const override;
		GridRegion<3> get_requirements(buffer_id bid, cl::sycl::access::mode mode) const;

	  private:
		struct buffer_access_info {
			cl::sycl::access::mode mode;
			subrange<3> sr;
		};

		std::unique_ptr<maf_storage_base> maf;
		std::unordered_map<buffer_id, std::vector<buffer_access_info>> buffer_accesses;
	};

} // namespace detail
} // namespace celerity
