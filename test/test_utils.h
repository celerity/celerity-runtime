#pragma once

#define CELERITY_TEST
#include <celerity.h>

#include "graph_generator.h"
#include "task_manager.h"

namespace celerity {
namespace test_utils {

	class mock_buffer_factory;

	template <int Dims>
	class mock_buffer {
	  public:
		template <cl::sycl::access::mode Mode, typename Functor>
		void get_access(compute_prepass_handler& handler, Functor rmfn) {
			using rmfn_traits = allscale::utils::lambda_traits<Functor>;
			static_assert(rmfn_traits::result_type::dims == Dims, "The returned subrange doesn't match buffer dimensions.");
			handler.require(id, std::make_unique<detail::range_mapper<rmfn_traits::arg1_type::dims, Dims>>(rmfn, Mode, size));
		}

		template <cl::sycl::access::mode Mode, typename Functor>
		void get_access(compute_livepass_handler& handler, Functor rmfn) {}

		template <cl::sycl::access::mode Mode>
		void get_access(master_access_prepass_handler& handler, cl::sycl::range<Dims> range, cl::sycl::id<Dims> offset = {}) {
			handler.require(Mode, id, detail::range_cast<3>(range), detail::id_cast<3>(offset));
		}

		template <cl::sycl::access::mode Mode>
		void get_access(master_access_livepass_handler& handler, cl::sycl::range<Dims> range, cl::sycl::id<Dims> offset = {}) {}

		buffer_id get_id() const { return id; }

	  private:
		friend class mock_buffer_factory;

		buffer_id id;
		cl::sycl::range<Dims> size;

		mock_buffer(buffer_id id, cl::sycl::range<Dims> size) : id(id), size(size) {}
	};

	class mock_buffer_factory {
	  public:
		mock_buffer_factory(detail::task_manager* tm = nullptr, detail::graph_generator* ggen = nullptr) : task_mngr(tm), ggen(ggen) {}

		template <int Dims>
		mock_buffer<Dims> create_buffer(cl::sycl::range<Dims> size, bool mark_as_host_initialized = false) {
			const buffer_id bid = next_buffer_id++;
			const auto buf = mock_buffer<Dims>(bid, size);
			if(task_mngr != nullptr) { task_mngr->add_buffer(bid, detail::range_cast<3>(size), mark_as_host_initialized); }
			if(ggen != nullptr) { ggen->add_buffer(bid, detail::range_cast<3>(size)); }
			return buf;
		}

	  private:
		detail::task_manager* task_mngr;
		detail::graph_generator* ggen;
		buffer_id next_buffer_id = 0;
	};

	template <typename KernelName = class test_task, typename CGF, int KernelDims = 2>
	task_id add_compute_task(detail::task_manager& tm, CGF cgf, cl::sycl::range<KernelDims> global_size = {1, 1}, cl::sycl::id<KernelDims> global_offset = {}) {
		tm.create_compute_task([&, gs = global_size, go = global_offset](auto& cgh) {
			cgf(cgh);
			cgh.template parallel_for<KernelName>(gs, go, [](cl::sycl::id<KernelDims>) {});
		});
		return (*tm.get_task_graph()).m_vertices.size() - 1;
	}

	template <typename MAF>
	task_id add_master_access_task(detail::task_manager& tm, MAF maf) {
		tm.create_master_access_task(maf);
		return (*tm.get_task_graph()).m_vertices.size() - 1;
	}

} // namespace test_utils
} // namespace celerity
