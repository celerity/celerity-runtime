#pragma once

#include <numeric>
#include <vector>

#include "log.h"
#include "ranges.h"
#include "sycl_wrappers.h"
#include "types.h"

namespace celerity::detail {

// To avoid additional register pressure, we embed hydration IDs into pointers for
// accessors and reduction, with the assumption that a real pointer will never be in the
// range [0, max_hydration_id]. Embedding / extracting are currently no-ops
// and the associated helper functions only exist for documentation purposes.
// This number puts an effective limit on the number of task objects (accessors
// etc.) that can be captured into a command function.
constexpr size_t max_hydration_id = 128;

template <typename T>
using can_embed_hydration_id = std::bool_constant<sizeof(hydration_id) == sizeof(T)>;

template <typename T>
T embed_hydration_id(const hydration_id hid) {
	static_assert(can_embed_hydration_id<T>::value);
	assert(hid > 0); // Has to be greater than zero so nullptr is not considered an embedded id
	assert(hid <= max_hydration_id);
	T result;
	std::memcpy(&result, &hid, sizeof(hid));
	return result;
}

template <typename T>
hydration_id extract_hydration_id(const T value) {
	static_assert(can_embed_hydration_id<T>::value);
	hydration_id result;
	std::memcpy(&result, &value, sizeof(value));
	return result;
}

template <typename T>
bool is_embedded_hydration_id(const T value) {
	static_assert(can_embed_hydration_id<T>::value);
	const auto coid = extract_hydration_id(value);
	return coid > 0 && coid <= max_hydration_id;
}

class task_hydrator {
  public:
	struct accessor_info {
		target tgt;
		void* ptr;
		range<3> buffer_range;
		id<3> buffer_offset;
		subrange<3> accessor_sr;
	};

	struct reduction_info {
		void* ptr;
	};

	/**
	 * Enable the closure_hydrator for this thread.
	 */
	static void make_available() {
		assert(m_instance == nullptr);
		m_instance = std::unique_ptr<task_hydrator>(new task_hydrator());
	}

	static bool is_available() { return m_instance != nullptr; }

	static void teardown() { m_instance.reset(); }

	static task_hydrator& get_instance() {
		assert(m_instance != nullptr);
		return *m_instance;
	}

	// NOCOMMIT Naming... arm?!
	/**
	 * Executes the provided callback in an "armed" state, that is, task objects
	 * attempting to hydrate from within the same thread will be able to do so.
	 *
	 * Note that accessors and reductions are assumed to use separate sets of hydration ids,
	 * both within the range [1, *_infos.size()), respectively.
	 */
	template <typename Callback>
	void arm(std::vector<accessor_info> accessor_infos, std::vector<reduction_info> reduction_infos, const Callback& cb) {
		assert(accessor_infos.size() < max_hydration_id);
		assert(reduction_infos.size() < max_hydration_id);

		m_is_armed = true;
		m_accessor_infos = std::move(accessor_infos);
		m_reduction_infos = std::move(reduction_infos);
		cb();
		m_accessor_infos.clear();
		m_reduction_infos.clear();
		m_is_armed = false;
	}

	template <typename Closure>
	Closure hydrate_local_accessors(const Closure& closure, sycl::handler& cgh) {
		static_assert(std::is_copy_constructible_v<std::decay_t<Closure>>);
		m_sycl_cgh = &cgh;
		Closure hydrated{closure};
		m_sycl_cgh = nullptr;
		return hydrated;
	}

	sycl::handler& get_sycl_handler() {
		assert(m_sycl_cgh != nullptr);
		return *m_sycl_cgh;
	}

	bool has_sycl_handler() const { return m_sycl_cgh != nullptr; }

	bool can_hydrate() const { return m_is_armed; }

	accessor_info hydrate_accessor(const hydration_id hid) {
		assert(!m_accessor_infos.empty());
		assert(hid > 0);
		assert(hid <= m_accessor_infos.size());
		return m_accessor_infos[hid - 1];
	}

	reduction_info hydrate_reduction(const hydration_id hid) {
		assert(!m_reduction_infos.empty());
		assert(hid > 0);
		assert(hid <= m_reduction_infos.size());
		return m_reduction_infos[hid - 1];
	}

  private:
	inline static thread_local std::unique_ptr<task_hydrator> m_instance;
	std::vector<accessor_info> m_accessor_infos;
	std::vector<reduction_info> m_reduction_infos;
	bool m_is_armed = false;
	sycl::handler* m_sycl_cgh = nullptr;

	task_hydrator() = default;
	task_hydrator(const task_hydrator&) = delete;
	task_hydrator(task_hydrator&&) = delete;
};

}; // namespace celerity::detail