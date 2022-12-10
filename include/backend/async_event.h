#pragma once

#include <gch/small_vector.hpp>

#include "payload.h"

namespace celerity::detail::backend_detail {
class native_event_wrapper {
  public:
	virtual ~native_event_wrapper() = default;
	virtual bool is_done() const = 0;
	// virtual void wait() = 0;
};
} // namespace celerity::detail::backend_detail

namespace celerity::detail::backend {

// TODO: Naming: Future, Promise, ..?
// TODO: Should this even live here, in the backend module..?
// FIXME: We probably want this to be copyable, right..? (Currently not possible due to payload attachment hack)
class [[nodiscard]] async_event {
  public:
	async_event() = default;
	async_event(const async_event&) = delete;
	async_event(async_event&&) = default;
	async_event(std::shared_ptr<backend_detail::native_event_wrapper> native_event) { add(std::move(native_event)); }

	async_event& operator=(async_event&&) = default;

	void merge(async_event other) {
		for(size_t i = 0; i < other.m_native_events.size(); ++i) {
			m_done_cache.push_back(other.m_done_cache[i]);
			m_native_events.emplace_back(std::move(other.m_native_events[i]));
		}
		m_attached_payloads.insert(
		    m_attached_payloads.end(), std::make_move_iterator(other.m_attached_payloads.begin()), std::make_move_iterator(other.m_attached_payloads.end()));
	}

	void add(std::shared_ptr<backend_detail::native_event_wrapper> native_event) {
		m_done_cache.push_back(false);
		m_native_events.emplace_back(std::move(native_event));
	}

	bool is_done() const {
		for(size_t i = 0; i < m_native_events.size(); ++i) {
			if(!m_done_cache[i]) {
				const bool is_done = m_native_events[i]->is_done();
				if(is_done) {
					m_done_cache[i] = true;
					continue;
				}
				return false;
			}
		}
		return true;
	}

	void wait() const {
		while(!is_done()) {}
	}

	// FIXME: Workaround to extend lifetime of temporary staging copies for asynchronous transfers
	void hack_attach_payload(unique_payload_ptr ptr) { m_attached_payloads.emplace_back(std::move(ptr)); }

  private:
	mutable gch::small_vector<bool> m_done_cache;
	gch::small_vector<std::shared_ptr<backend_detail::native_event_wrapper>> m_native_events;
	gch::small_vector<unique_payload_ptr> m_attached_payloads;
};

} // namespace celerity::detail::backend
