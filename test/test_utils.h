#pragma once

#define CELERITY_TEST
#include <celerity.h>

#include "graph_generator.h"
#include "range_mapper.h"
#include "task_manager.h"

namespace celerity {
namespace test_utils {

	class mock_buffer_factory;

	template <int Dims>
	class mock_buffer {
	  public:
		template <cl::sycl::access::mode Mode, typename Functor, typename = decltype(std::declval<Functor>()(std::declval<chunk<Dims>>()))>
		void get_access(handler& cgh, Functor rmfn) {
			using rmfn_traits = allscale::utils::lambda_traits<Functor>;
			static_assert(rmfn_traits::result_type::dims == Dims, "The returned subrange doesn't match buffer dimensions.");
			if(detail::is_prepass_handler(cgh)) {
				auto compute_cgh = dynamic_cast<detail::compute_task_handler<true>&>(cgh);
				compute_cgh.add_requirement(id, std::make_unique<detail::range_mapper<rmfn_traits::arg1_type::dims, Dims>>(rmfn, Mode, size));
			}
		}

		template <cl::sycl::access::mode Mode>
		void get_access(handler& cgh, cl::sycl::range<Dims> range, cl::sycl::id<Dims> offset = {}) {
			if(detail::is_prepass_handler(cgh)) {
				auto ma_cgh = dynamic_cast<detail::master_access_task_handler<true>&>(cgh);
				ma_cgh.add_requirement(Mode, id, detail::range_cast<3>(range), detail::id_cast<3>(offset));
			}
		}

		detail::buffer_id get_id() const { return id; }

	  private:
		friend class mock_buffer_factory;

		detail::buffer_id id;
		cl::sycl::range<Dims> size;

		mock_buffer(detail::buffer_id id, cl::sycl::range<Dims> size) : id(id), size(size) {}
	};

	class mock_buffer_factory {
	  public:
		mock_buffer_factory(detail::task_manager* tm = nullptr, detail::graph_generator* ggen = nullptr) : task_mngr(tm), ggen(ggen) {}

		template <int Dims>
		mock_buffer<Dims> create_buffer(cl::sycl::range<Dims> size, bool mark_as_host_initialized = false) {
			const detail::buffer_id bid = next_buffer_id++;
			const auto buf = mock_buffer<Dims>(bid, size);
			if(task_mngr != nullptr) { task_mngr->add_buffer(bid, detail::range_cast<3>(size), mark_as_host_initialized); }
			if(ggen != nullptr) { ggen->add_buffer(bid, detail::range_cast<3>(size)); }
			return buf;
		}

	  private:
		detail::task_manager* task_mngr;
		detail::graph_generator* ggen;
		detail::buffer_id next_buffer_id = 0;
	};

	template <typename KernelName = class test_task, typename CGF, int KernelDims = 2>
	detail::task_id add_compute_task(
	    detail::task_manager& tm, CGF cgf, cl::sycl::range<KernelDims> global_size = {1, 1}, cl::sycl::id<KernelDims> global_offset = {}) {
		tm.create_compute_task([&, gs = global_size, go = global_offset](handler& cgh) {
			cgf(cgh);
			cgh.parallel_for<KernelName>(gs, go, [](cl::sycl::id<KernelDims>) {});
		});
		return (*tm.get_task_graph()).m_vertices.size() - 1;
	}

	template <typename CGF>
	detail::task_id add_master_access_task(detail::task_manager& tm, CGF cgf) {
		tm.create_master_access_task(cgf);
		return (*tm.get_task_graph()).m_vertices.size() - 1;
	}

} // namespace test_utils
} // namespace celerity
