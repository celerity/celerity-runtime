#pragma once

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
class async_event {
  public:
	async_event() = default;
	async_event(std::shared_ptr<backend_detail::native_event_wrapper> native_event) { add(std::move(native_event)); }

	void merge(async_event other) {
		for(size_t i = 0; i < other.m_native_events.size(); ++i) {
			m_done_cache.push_back(other.m_done_cache[i]);
			m_native_events.emplace_back(std::move(other.m_native_events[i]));
		}
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

  private:
	mutable std::vector<bool> m_done_cache;
	std::vector<std::shared_ptr<backend_detail::native_event_wrapper>> m_native_events;
};

} // namespace celerity::detail::backend
