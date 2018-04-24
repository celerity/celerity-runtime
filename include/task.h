#pragma once

#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

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

} // namespace detail

class task {
  public:
	virtual ~task() = default;
	virtual task_type get_type() const = 0;
};

class compute_task : public task {
  public:
	compute_task(std::unique_ptr<detail::cgf_storage_base>&& cgf) : task(), cgf(std::move(cgf)) {}

	task_type get_type() const override { return task_type::COMPUTE; }

	const detail::cgf_storage_base& get_command_group() const { return *cgf; }
	any_range get_global_size() const { return global_size; }
	const std::unordered_map<buffer_id, std::vector<std::unique_ptr<detail::range_mapper_base>>>& get_range_mappers() const { return range_mappers; }

	void set_global_size(any_range gs) { global_size = gs; }
	void add_range_mapper(buffer_id bid, std::unique_ptr<detail::range_mapper_base>&& rm) { range_mappers[bid].push_back(std::move(rm)); }

  private:
	std::unique_ptr<detail::cgf_storage_base> cgf;
	any_range global_size;
	std::unordered_map<buffer_id, std::vector<std::unique_ptr<detail::range_mapper_base>>> range_mappers;
};

class master_access_task : public task {
  public:
	struct buffer_access_info {
		cl::sycl::access::mode mode;
		any_range range;
		any_range offset;
	};

	master_access_task(std::unique_ptr<detail::maf_storage_base>&& maf) : task(), maf(std::move(maf)) {}

	task_type get_type() const override { return task_type::MASTER_ACCESS; }

	const std::unordered_map<buffer_id, std::vector<buffer_access_info>>& get_accesses() const { return buffer_accesses; }
	const detail::maf_storage_base& get_functor() const { return *maf; }

	void add_buffer_access(buffer_id bid, cl::sycl::access::mode mode, any_range range, any_range offset) {
		buffer_accesses[bid].push_back({mode, range, offset});
	}

  private:
	std::unique_ptr<detail::maf_storage_base> maf;
	std::unordered_map<buffer_id, std::vector<buffer_access_info>> buffer_accesses;
};

} // namespace celerity
