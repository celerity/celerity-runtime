#include "backend/sycl_backend.h"

#include "async_event.h"
#include "backend/backend.h"
#include "cgf.h"
#include "closure_hydrator.h"
#include "dense_map.h"
#include "grid.h"
#include "named_threads.h"
#include "nd_memory.h"
#include "system_info.h"
#include "thread_queue.h"
#include "tracy.h"
#include "types.h"
#include "utils.h"
#include "workaround.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <exception>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <sycl/sycl.hpp>


namespace celerity::detail::sycl_backend_detail {

bool sycl_event::is_complete() { return m_last.get_info<sycl::info::event::command_execution_status>() == sycl::info::event_command_status::complete; }

std::optional<std::chrono::nanoseconds> sycl_event::get_native_execution_time() {
	if(!m_first.has_value()) return std::nullopt; // avoid the cost of throwing + catching a sycl exception by when profiling is disabled
	return std::chrono::nanoseconds(m_last.get_profiling_info<sycl::info::event_profiling::command_end>() //
	                                - m_first->get_profiling_info<sycl::info::event_profiling::command_start>());
}

void delayed_async_event::state::set_value(async_event event) {
	m_event = std::move(event);
	[[maybe_unused]] const bool previously_ready = m_is_ready.exchange(true, std::memory_order_release);
	assert(!previously_ready && "delayed_async_event::state::set_value() called more than once");
}

bool delayed_async_event::is_complete() {
	if(!m_state->m_is_ready.load(std::memory_order_acquire)) return false;
	return m_state->m_event.is_complete();
}

void* delayed_async_event::get_result() {
	assert(m_state->m_is_ready.load(std::memory_order_acquire));
	return m_state->m_event.get_result();
}

std::optional<std::chrono::nanoseconds> delayed_async_event::get_native_execution_time() {
	assert(m_state->m_is_ready.load(std::memory_order_acquire));
	return m_state->m_event.get_native_execution_time();
}

void flush(sycl::queue& queue) {
#if CELERITY_WORKAROUND(ACPP)
	// AdaptiveCpp does not guarantee that command groups are actually scheduled until an explicit await operation, which we cannot insert without
	// blocking the executor loop (see https://github.com/AdaptiveCpp/AdaptiveCpp/issues/599). Instead, we explicitly flush the queue to be able to continue
	// using our polling-based approach.
	queue.get_context().AdaptiveCpp_runtime()->dag().flush_async();
#else
	(void)queue;
#endif
}

// LCOV_EXCL_START
void report_errors(const sycl::exception_list& errors) {
	if(errors.size() == 0) return;

	std::vector<std::string> what;
	for(const auto& e : errors) {
		try {
			std::rethrow_exception(e);
		} catch(sycl::exception& e) { //
			what.push_back(e.what());
		} catch(std::exception& e) { //
			what.push_back(e.what());
		} catch(...) { //
			what.push_back("unknown exception");
		}
	}

	// Errors usually manifest on calls to sycl::event::get_info(), not their actual origin, and therefore will contain many duplicates
	std::sort(what.begin(), what.end());
	what.erase(std::unique(what.begin(), what.end()), what.end());

	utils::panic("asynchronous SYCL errors:\n\t{}", fmt::join(what, "\n\t"));
}
// LCOV_EXCL_STOP

} // namespace celerity::detail::sycl_backend_detail

namespace celerity::detail {

struct sycl_backend::impl {
	struct device_state {
		sycl::device sycl_device;
		sycl::context sycl_context;
		std::vector<sycl::queue> queues;
		std::optional<detail::thread_queue> submission_thread;
		std::atomic_flag active_async_error_check = false;

		device_state() = default;
		explicit device_state(const sycl::device& dev) : sycl_device(dev), sycl_context(sycl_device) {}
	};

	struct host_state {
		sycl::context sycl_context;
		thread_queue alloc_queue;
		std::vector<thread_queue> queues; // TODO naming vs alloc_queue?

		// pass devices to ensure the sycl_context receives the correct platform
		explicit host_state(const std::vector<sycl::device>& all_devices, bool enable_profiling)
		    // DPC++ requires exactly one CUDA device here, but for allocation the sycl_context mostly means "platform".
		    // - TODO assert that all devices belong to the same platform + backend here
		    // - TODO test Celerity on a (SimSYCL) system without GPUs
		    : sycl_context(all_devices.at(0)), //
		      alloc_queue(named_threads::thread_type::alloc, enable_profiling) {}
	};

	system_info system;
	dense_map<device_id, device_state> devices; // thread-safe for read access (not resized after construction)
	host_state host;
	using configuration = sycl_backend::configuration;
	configuration config;

	impl(const std::vector<sycl::device>& devices, const configuration& config)
	    : devices(devices.begin(), devices.end()), host(devices, config.profiling), config(config) //
	{
		// For now, we assume distinct memories per device. TODO some targets, (OpenMP emulated devices), might deviate from that.
		system.devices.resize(devices.size());
		system.memories.resize(2 + devices.size()); //  user + host + device memories
		system.memories[user_memory_id].copy_peers.set(user_memory_id);
		system.memories[host_memory_id].copy_peers.set(host_memory_id);
		system.memories[host_memory_id].copy_peers.set(user_memory_id);
		system.memories[user_memory_id].copy_peers.set(host_memory_id);
		for(device_id did = 0; did < devices.size(); ++did) {
			const memory_id mid = first_device_memory_id + did;
			system.devices[did].native_memory = mid;
			system.memories[mid].copy_peers.set(mid);
			system.memories[mid].copy_peers.set(host_memory_id);
			system.memories[host_memory_id].copy_peers.set(mid);
			// device-to-device copy capabilities are added in cuda_backend constructor
		}
	}

	template <typename F>
	async_event submit_alloc(F&& f) {
#if CELERITY_WORKAROUND(SIMSYCL)
		// SimSYCL is not thread safe => skip alloc_queue and complete allocations in executor thread.
		if constexpr(std::is_void_v<std::invoke_result_t<F>>) {
			return f(), make_complete_event();
		} else {
			return make_complete_event(f());
		}
#else
		return host.alloc_queue.submit(std::forward<F>(f));
#endif
	}

	thread_queue& get_host_queue(const size_t lane) {
		assert(lane <= host.queues.size());
		if(lane == host.queues.size()) { host.queues.emplace_back(named_threads::task_type_host_queue(lane), config.profiling); }
		return host.queues[lane];
	}

	sycl::queue& get_device_queue(const device_id did, const size_t lane) {
		auto& device = devices[did];
		assert(lane <= device.queues.size());
		if(lane == device.queues.size()) {
			const auto properties = config.profiling ? sycl::property_list{sycl::property::queue::enable_profiling{}, sycl::property::queue::in_order{}}
			                                         : sycl::property_list{sycl::property::queue::in_order{}};
			device.queues.emplace_back(device.sycl_device, sycl::async_handler(sycl_backend_detail::report_errors), properties);
		}
		return device.queues[lane];
	}
};

sycl_backend::sycl_backend(const std::vector<sycl::device>& devices, const configuration& config) : m_impl(new impl(devices, config)) {
	// Initialize a submission thread with hydrator for each device, if they are enabled
	if(m_impl->config.per_device_submission_threads) {
		for(device_id did = 0; did < m_impl->system.devices.size(); ++did) {
			m_impl->devices[did].submission_thread.emplace(named_threads::task_type_device_submitter(did.value), m_impl->config.profiling);
			// no need to wait for the event -> will happen before the first task is submitted
			(void)m_impl->devices[did].submission_thread->submit([did] { closure_hydrator::make_available(); });
		}
	}
}

sycl_backend::~sycl_backend() {
	// If we are using submission threads, tear down their hydrators before they are destroyed
	if(m_impl->config.per_device_submission_threads) {
		for(auto& device : m_impl->devices) {
			// no need to wait for the event -> destruction will wait for the submission thread to finish
			(void)device.submission_thread->submit([] { closure_hydrator::teardown(); });
		}
	}
}

const system_info& sycl_backend::get_system_info() const { return m_impl->system; }

void sycl_backend::init() {
	CELERITY_DETAIL_TRACY_ZONE_SCOPED("sycl::init", sycl_init);

	// Instantiate the first in-order queue on each device. At least for CUDA systems this will perform device initialization, which can take > 100 ms / device.
	for(device_id did = 0; did < m_impl->system.devices.size(); ++did) {
		(void)m_impl->get_device_queue(did, 0 /* lane */);
	}
}

void* sycl_backend::debug_alloc(const size_t size) {
	const auto ptr = sycl::malloc_host(size, m_impl->host.sycl_context);
#if CELERITY_DETAIL_ENABLE_DEBUG
	memset(ptr, static_cast<int>(sycl_backend_detail::uninitialized_memory_pattern), size);
#endif
	return ptr;
}

void sycl_backend::debug_free(void* const ptr) { sycl::free(ptr, m_impl->host.sycl_context); }

async_event sycl_backend::enqueue_host_alloc(const size_t size, const size_t alignment) {
	return m_impl->submit_alloc([this, size, alignment] {
		const auto ptr = sycl::aligned_alloc_host(alignment, size, m_impl->host.sycl_context);
#if CELERITY_DETAIL_ENABLE_DEBUG
		memset(ptr, static_cast<int>(sycl_backend_detail::uninitialized_memory_pattern), size);
#endif
		return ptr;
	});
}

async_event sycl_backend::enqueue_device_alloc(const device_id device, const size_t size, const size_t alignment) {
	return m_impl->submit_alloc([this, device, size, alignment] {
		auto& d = m_impl->devices[device];
		const auto ptr = sycl::aligned_alloc_device(alignment, size, d.sycl_device, d.sycl_context);
#if CELERITY_DETAIL_ENABLE_DEBUG
		sycl::queue(d.sycl_context, d.sycl_device, sycl::async_handler(sycl_backend_detail::report_errors), sycl::property::queue::in_order{})
		    .fill(ptr, sycl_backend_detail::uninitialized_memory_pattern, size)
		    .wait_and_throw();
#endif
		return ptr;
	});
}

async_event sycl_backend::enqueue_host_free(void* const ptr) {
	return m_impl->submit_alloc([this, ptr] { sycl::free(ptr, m_impl->host.sycl_context); });
}

async_event sycl_backend::enqueue_device_free(const device_id device, void* const ptr) {
	return m_impl->submit_alloc([this, device, ptr] { sycl::free(ptr, m_impl->devices[device].sycl_context); });
}

async_event sycl_backend::enqueue_host_task(size_t host_lane, const host_task_launcher& launcher, std::vector<closure_hydrator::accessor_info> accessor_infos,
    const range<3>& global_range, const box<3>& execution_range, const communicator* collective_comm) //
{
	auto& hydrator = closure_hydrator::get_instance();
	hydrator.arm(target::host_task, std::move(accessor_infos));
	auto launch_hydrated = hydrator.hydrate<target::host_task>(launcher);
	return m_impl->get_host_queue(host_lane).submit(
	    [=, launch_hydrated = std::move(launch_hydrated)] { launch_hydrated(global_range, execution_range, collective_comm); });
}

async_event sycl_backend::enqueue_device_kernel(const device_id device, const size_t lane, const device_kernel_launcher& launch,
    std::vector<closure_hydrator::accessor_info> accessor_infos, const box<3>& execution_range, const std::vector<void*>& reduction_ptrs) //
{
	return enqueue_device_work(device, lane, [=, this, acc_infos = std::move(accessor_infos)](sycl::queue& queue) mutable {
		CELERITY_DETAIL_TRACY_ZONE_SCOPED("sycl::submit", sycl_submit);
		auto event = queue.submit([&](sycl::handler& sycl_cgh) {
			auto& hydrator = closure_hydrator::get_instance();
			hydrator.arm(target::device, std::move(acc_infos));
			const auto launch_hydrated = hydrator.hydrate<target::device>(sycl_cgh, launch);
			launch_hydrated(sycl_cgh, execution_range, reduction_ptrs);
		});
		sycl_backend_detail::flush(queue);
		return make_async_event<sycl_backend_detail::sycl_event>(std::move(event), m_impl->config.profiling);
	});
}

async_event sycl_backend::enqueue_host_copy(size_t host_lane, const void* const source_base, void* const dest_base, const region_layout& source_layout,
    const region_layout& dest_layout, const region<3>& copy_region, const size_t elem_size) //
{
	return m_impl->get_host_queue(host_lane).submit([=] { nd_copy_host(source_base, dest_base, source_layout, dest_layout, copy_region, elem_size); });
}

void sycl_backend::check_async_errors() {
	for(size_t i = 0; i < m_impl->devices.size(); ++i) {
		auto& device = m_impl->devices[i];
		if(m_impl->config.per_device_submission_threads) {
			// Prevent multiple error checks from being enqueued at the same time
			if(!device.active_async_error_check.test_and_set()) {
				(void)device.submission_thread->submit([&]() {
					for(auto& queue : device.queues) {
						queue.throw_asynchronous();
					}
					device.active_async_error_check.clear();
				});
			}
		} else {
			for(auto& queue : device.queues) {
				queue.throw_asynchronous();
			}
		}
	}
}

system_info& sycl_backend::get_system_info() { return m_impl->system; }

async_event celerity::detail::sycl_backend::enqueue_device_work(
    const device_id device, const size_t lane, const std::function<async_event(sycl::queue&)>& work) {
	// Basic case: no per-device submission threads
	if(!m_impl->config.per_device_submission_threads) { return work(m_impl->get_device_queue(device, lane)); }

	auto& device_state = m_impl->devices[device];
	auto& submission_thread = device_state.submission_thread;
	assert(submission_thread.has_value());

	// Note: this mechanism is quite similar in principle to a std::future/promise,
	//       but implementing it with that caused a 50% (!) slowdown in system-level benchmarks
	const auto async_event_state = std::make_shared<sycl_backend_detail::delayed_async_event::state>();
	auto async_event = make_async_event<sycl_backend_detail::delayed_async_event>(async_event_state);

	(void)submission_thread->submit([this, device, lane, work, async_event_state] {
		auto event = work(m_impl->get_device_queue(device, lane));
		async_event_state->set_value(std::move(event));
	});
	return async_event;
}

bool sycl_backend::is_profiling_enabled() const { return m_impl->config.profiling; }

std::vector<sycl_backend_type> sycl_backend_enumerator::compatible_backends(const sycl::device& device) const {
	std::vector<backend_type> backends{backend_type::generic};
#if CELERITY_WORKAROUND(ACPP) && defined(SYCL_EXT_HIPSYCL_BACKEND_CUDA)
	if(device.get_backend() == sycl::backend::cuda) { backends.push_back(sycl_backend_type::cuda); }
#elif CELERITY_WORKAROUND(DPCPP)
	if(device.get_backend() == sycl::backend::ext_oneapi_cuda) { backends.push_back(sycl_backend_type::cuda); }
#endif
	assert(std::is_sorted(backends.begin(), backends.end()));
	return backends;
}

std::vector<sycl_backend_type> sycl_backend_enumerator::available_backends() const {
	std::vector<backend_type> backends{backend_type::generic};
#if CELERITY_DETAIL_BACKEND_CUDA_ENABLED
	backends.push_back(sycl_backend_type::cuda);
#endif
	assert(std::is_sorted(backends.begin(), backends.end()));
	return backends;
}

bool sycl_backend_enumerator::is_specialized(backend_type type) const {
	switch(type) {
	case backend_type::generic: return false;
	case backend_type::cuda: return true;
	default: utils::unreachable(); // LCOV_EXCL_LINE
	}
}

int sycl_backend_enumerator::get_priority(backend_type type) const {
	switch(type) {
	case backend_type::generic: return 0;
	case backend_type::cuda: return 1;
	default: utils::unreachable(); // LCOV_EXCL_LINE
	}
}

} // namespace celerity::detail

namespace celerity::detail {

std::unique_ptr<backend> make_sycl_backend(const sycl_backend_type type, const std::vector<sycl::device>& devices, const sycl_backend::configuration& config) {
	assert(std::all_of(
	    devices.begin(), devices.end(), [=](const sycl::device& d) { return utils::contains(sycl_backend_enumerator{}.compatible_backends(d), type); }));

	switch(type) {
	case sycl_backend_type::generic: //
		return std::make_unique<sycl_generic_backend>(devices, config);

	case sycl_backend_type::cuda:
#if CELERITY_DETAIL_BACKEND_CUDA_ENABLED
		return std::make_unique<sycl_cuda_backend>(devices, config);
#else
		utils::panic("CUDA backend has not been compiled");
#endif
	}
	utils::unreachable(); // LCOV_EXCL_LINE
}

} // namespace celerity::detail
