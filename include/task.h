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
class master_access_prepass_handler;
class master_access_livepass_handler;

namespace detail {

	enum class task_type { NOP, COMPUTE, MASTER_ACCESS };

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

	// TODO: It's not ideal that dependencies are only populated on the master node, but the interface exists on workers as well...
	class task : public intrusive_graph_node<task> {
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

	class nop_task : public task {
	  public:
		explicit nop_task(const task_id& tid) : task(tid) {}

		task_type get_type() const override { return task_type::NOP; }
		std::vector<buffer_id> get_accessed_buffers() const override { return {}; }
		std::unordered_set<cl::sycl::access::mode> get_access_modes(buffer_id bid) const override { return {}; }
	};

	class compute_task : public task {
	  public:
		compute_task(task_id tid, std::unique_ptr<command_group_storage_base>&& cgf) : task(tid), cgf(std::move(cgf)) {}

		task_type get_type() const override { return task_type::COMPUTE; }

		void set_dimensions(int dims) { dimensions = dims; }
		void set_global_size(cl::sycl::range<3> gs) { global_size = gs; }
		void set_global_offset(cl::sycl::id<3> offset) { global_offset = offset; }
		void set_debug_name(std::string name) { debug_name = name; };

		void add_range_mapper(buffer_id bid, std::unique_ptr<range_mapper_base>&& rm) { range_mappers[bid].push_back(std::move(rm)); }

		const command_group_storage_base& get_command_group() const { return *cgf; }

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
		std::unique_ptr<command_group_storage_base> cgf;
		int dimensions = 0;
		cl::sycl::range<3> global_size;
		cl::sycl::id<3> global_offset = {};
		std::string debug_name;
		std::unordered_map<buffer_id, std::vector<std::unique_ptr<range_mapper_base>>> range_mappers;
	};

	class master_access_task : public task {
	  public:
		master_access_task(task_id tid, std::unique_ptr<command_group_storage_base>&& maf) : task(tid), maf(std::move(maf)) {}

		task_type get_type() const override { return task_type::MASTER_ACCESS; }

		void add_buffer_access(buffer_id bid, cl::sycl::access::mode mode, subrange<3> sr) { buffer_accesses[bid].push_back({mode, sr}); }

		const command_group_storage_base& get_functor() const { return *maf; }

		std::vector<buffer_id> get_accessed_buffers() const override;
		std::unordered_set<cl::sycl::access::mode> get_access_modes(buffer_id bid) const override;
		GridRegion<3> get_requirements(buffer_id bid, cl::sycl::access::mode mode) const;

	  private:
		struct buffer_access_info {
			cl::sycl::access::mode mode;
			subrange<3> sr;
		};

		std::unique_ptr<command_group_storage_base> maf;
		std::unordered_map<buffer_id, std::vector<buffer_access_info>> buffer_accesses;
	};

} // namespace detail
} // namespace celerity
