#include "device_queue.h"

#include <CL/sycl.hpp>

#include "log.h"
#include "workaround.h"

namespace celerity {
namespace detail {

	void device_queue::init(const config& cfg, const device_or_selector& user_device_or_selector) {
		assert(sycl_queue == nullptr);
		const auto profiling_cfg = cfg.get_enable_device_profiling();
		device_profiling_enabled = profiling_cfg != std::nullopt && *profiling_cfg;
		if(device_profiling_enabled) { CELERITY_INFO("Device profiling enabled."); }

		const auto props = device_profiling_enabled ? cl::sycl::property_list{cl::sycl::property::queue::enable_profiling()} : cl::sycl::property_list{};
		const auto handle_exceptions = cl::sycl::async_handler{[this](cl::sycl::exception_list el) { this->handle_async_exceptions(el); }};

		auto device = std::visit(
		    [&cfg](const auto& value) { return ::celerity::detail::pick_device(cfg, value, cl::sycl::platform::get_platforms()); }, user_device_or_selector);
		sycl_queue = std::make_unique<cl::sycl::queue>(device, handle_exceptions, props);
	}

	void device_queue::handle_async_exceptions(cl::sycl::exception_list el) const {
		for(auto& e : el) {
			try {
				std::rethrow_exception(e);
			} catch(cl::sycl::exception& e) {
				CELERITY_ERROR("SYCL asynchronous exception: {}. Terminating.", e.what());
				std::terminate();
			}
		}
	}

} // namespace detail
} // namespace celerity
