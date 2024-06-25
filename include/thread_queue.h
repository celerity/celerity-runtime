#pragma once

#include "async_event.h"
#include "double_buffered_queue.h"
#include "named_threads.h"
#include "utils.h"

#include <chrono>
#include <future>
#include <thread>
#include <type_traits>
#include <variant>

namespace celerity::detail {

/// A single-thread job queue accepting functors and returning events that conditionally forward job results.
class thread_queue {
  public:
	/// Constructs a null thread queue that cannot receive jobs.
	thread_queue() = default;

	/// Spawns a new thread queue with the given thread name. If `enable_profiling` is set to `true`, completed events from this thread queue will report a
	/// non-nullopt duration.
	explicit thread_queue(std::string thread_name, const bool enable_profiling = false) : m_impl(new impl(std::move(thread_name), enable_profiling)) {}

	// thread_queue is movable, but not copyable.
	thread_queue(const thread_queue&) = delete;
	thread_queue(thread_queue&&) = default;
	thread_queue& operator=(const thread_queue&) = delete;
	thread_queue& operator=(thread_queue&&) = default;

	/// Destruction will await all submitted and pending jobs.
	~thread_queue() {
		if(m_impl != nullptr) {
			m_impl->queue.push(job{} /* termination */);
			m_impl->thread.join();
		}
	}

	/// Submit a job to the thread queue.
	/// `fn` must take no arguments and return either `void` or a type convertible to `void *`, which will be forwarded as the result into the event.
	template <typename Fn>
	async_event submit(Fn&& fn) {
		assert(m_impl != nullptr);
		job job(std::forward<Fn>(fn));
		auto evt = make_async_event<thread_queue::event>(job.promise.get_future());
		m_impl->queue.push(std::move(job));
		return evt;
	}

  private:
	friend struct thread_queue_testspy;

	/// The object passed through std::future from queue thread to owner thread
	struct completion {
		void* result = nullptr;
		std::optional<std::chrono::nanoseconds> execution_time;
	};

	struct job {
		std::function<void*()> fn;
		std::promise<completion> promise;

		job() = default; // empty (default-constructed) fn signals termination

		/// Constructor overload for `fn` returning `void`.
		template <typename Fn, std::enable_if_t<std::is_same_v<std::invoke_result_t<Fn>, void>, int> = 0>
		job(Fn&& fn) : fn([fn = std::forward<Fn>(fn)] { return std::invoke(fn), nullptr; }) {}

		/// Constructor overload for `fn` returning `void*`.
		template <typename Fn, std::enable_if_t<std::is_invocable_r_v<void*, Fn>, int> = 0>
		job(Fn&& fn) : fn([fn = std::forward<Fn>(fn)] { return std::invoke(fn); }) {}
	};

	class event : public async_event_impl {
	  public:
		explicit event(std::future<completion> future) : m_state(std::move(future)) {}

		bool is_complete() override { return get_completed() != nullptr; }

		void* get_result() override {
			const auto completed = get_completed();
			assert(completed);
			return completed->result;
		}

		std::optional<std::chrono::nanoseconds> get_native_execution_time() override {
			const auto completed = get_completed();
			assert(completed);
			return completed->execution_time;
		}

	  private:
		// As the result from std::future can only be retrieved once and std::shared_future is not functionally necessary here, we replace the future by its
		// completion as soon as the first query succeeds.
		std::variant<std::future<completion>, completion> m_state;

		completion* get_completed() {
			if(const auto completed = std::get_if<completion>(&m_state)) return completed;
			if(auto& future = std::get<std::future<completion>>(m_state); future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
				return &m_state.emplace<completion>(future.get());
			}
			return nullptr;
		}
	};

	// pimpl'd to keep thread_queue movable
	struct impl {
		double_buffered_queue<job> queue;
		const bool enable_profiling;
		std::thread thread;

		explicit impl(std::string name, const bool enable_profiling) : enable_profiling(enable_profiling), thread(&impl::thread_main, this, std::move(name)) {}

		void execute(job& job) const {
			std::chrono::steady_clock::time_point start;
			if(enable_profiling) { start = std::chrono::steady_clock::now(); }

			completion completion;
			completion.result = job.fn();

			if(enable_profiling) {
				const auto end = std::chrono::steady_clock::now();
				completion.execution_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
			}
			job.promise.set_value(completion);
		}

		void loop() {
			for(;;) {
				queue.wait_while_empty();
				for(auto& job : queue.pop_all()) {
					if(!job.fn) return;
					execute(job);
				}
			}
		}

		void thread_main(const std::string& name) {
			set_thread_name(get_current_thread_handle(), name);

			try {
				loop();
			} catch(std::exception& e) { //
				utils::panic("exception in {}: {}", name, e.what());
			} catch(...) { //
				utils::panic("exception in {}", name);
			}
		}
	};

	std::unique_ptr<impl> m_impl;
};

} // namespace celerity::detail
