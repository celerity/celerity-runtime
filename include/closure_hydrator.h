#pragma once

#include <optional>
#include <vector>

#include "grid.h"
#include "ranges.h"
#include "sycl_wrappers.h"
#include "types.h"

namespace celerity::detail {

#if CELERITY_ACCESSOR_BOUNDARY_CHECK
struct oob_bounding_box {
	id<3> min{SIZE_MAX, SIZE_MAX, SIZE_MAX};
	id<3> max{0, 0, 0};

	box<3> into_box() const { return min[0] < max[0] && min[1] < max[1] && min[2] < max[2] ? box(min, max) : box<3>(); }
};
#endif

// To avoid additional register pressure, we embed hydration IDs into pointers for
// accessors, with the assumption that a real pointer will never be in the
// range [0, max_hydration_id]. Embedding / extracting are currently no-ops
// and the associated helper functions only exist for documentation purposes.
// This number puts an effective limit on the number of task objects (accessors
// etc.) that can be captured into a command function.
constexpr size_t max_hydration_id = 128;

template <typename T>
constexpr bool can_embed_hydration_id = std::bool_constant<sizeof(hydration_id) == sizeof(T)>::value;

template <typename T>
T embed_hydration_id(const hydration_id hid) {
	static_assert(can_embed_hydration_id<T>);
	assert(hid > 0); // Has to be greater than zero so nullptr is not considered an embedded id
	assert(hid <= max_hydration_id);
	T result;
	std::memcpy(&result, &hid, sizeof(hid));
	return result;
}

template <typename T>
hydration_id extract_hydration_id(const T value) {
	static_assert(can_embed_hydration_id<T>);
	hydration_id result;
	std::memcpy(&result, &value, sizeof(value));
	return result;
}

template <typename T>
bool is_embedded_hydration_id(const T value) {
	static_assert(can_embed_hydration_id<T>);
	const auto hid = extract_hydration_id(value);
	return hid > 0 && hid <= max_hydration_id;
}

/**
 * The closure hydrator is used to inject information into objects (currently host/device and local accessors) that have been captured into a lambda closure.
 * We abuse the copy constructor of the captured objects to modify them while the containing closure is being copied by the hydrate() function.
 * Accessors request their corresponding information by means of per-closure unique "hydration ids" that are assigned upon accessor creation.
 *
 * The hydrator is implemented as a thread-local singleton that needs to be explicitly enabled per-thread. This is because kernel command function
 * closures may be copied any number of times after having been passed to SYCL, which should not trigger the hydration mechanism.
 */
class closure_hydrator {
  public:
	struct accessor_info {
		void* ptr;
		box<3> allocated_box_in_buffer;
		box<3> accessed_box_in_buffer;

#if CELERITY_ACCESSOR_BOUNDARY_CHECK
		oob_bounding_box* out_of_bounds_indices = nullptr;
#endif
	};

	closure_hydrator(const closure_hydrator&) = delete;
	closure_hydrator(closure_hydrator&&) = delete;
	closure_hydrator& operator=(const closure_hydrator&) = delete;
	closure_hydrator& operator=(closure_hydrator&&) = delete;
	~closure_hydrator() = default;

	static void make_available() {
		assert(m_instance == nullptr);
		m_instance = std::unique_ptr<closure_hydrator>(new closure_hydrator());
	}

	static bool is_available() { return m_instance != nullptr; }

	static void teardown() { m_instance.reset(); }

	static closure_hydrator& get_instance() {
		assert(m_instance != nullptr);
		return *m_instance;
	}

	/**
	 * Puts the hydrator into the "armed" state, after which hydrate() can be called to hydrate kernel functions.
	 *
	 * accessor_infos must contain one entry for each hydration id that has been assigned to accessors in the
	 * closure that is to be hydrated, in matching order.
	 */
	void arm(const target tgt, std::vector<accessor_info> accessor_infos) {
		assert(!m_armed_for.has_value());
		assert(accessor_infos.size() < max_hydration_id);
		m_armed_for = tgt;
		m_accessor_infos = std::move(accessor_infos);
	}

	/**
	 * Hydrates the provided closure by copying it in a context where calls to get_accessor_info and get_sycl_handler are allowed.
	 */
	template <target Tgt, typename Closure, std::enable_if_t<Tgt == target::device, int> = 0>
	[[nodiscard]] auto hydrate(sycl::handler& cgh, const Closure& closure) {
		return hydrate(target::device, &cgh, closure);
	}

	/**
	 * Hydrates the provided closure by copying it in a context where calls to get_accessor_info are allowed.
	 */
	template <target Tgt, typename Closure, std::enable_if_t<Tgt == target::host_task, int> = 0>
	[[nodiscard]] auto hydrate(const Closure& closure) {
		return hydrate(target::host_task, nullptr, closure);
	}

	bool is_hydrating() const { return m_is_hydrating; }

	template <target Tgt>
	accessor_info get_accessor_info(const hydration_id hid) {
		assert(m_armed_for.has_value() && *m_armed_for == Tgt);
		assert(!m_accessor_infos.empty());
		assert(hid > 0);
		assert(hid <= m_accessor_infos.size());
		return m_accessor_infos[hid - 1];
	}

	sycl::handler& get_sycl_handler() {
		assert(m_sycl_cgh != nullptr);
		return *m_sycl_cgh;
	}

  private:
	inline static thread_local std::unique_ptr<closure_hydrator> m_instance; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
	std::vector<accessor_info> m_accessor_infos;
	std::optional<target> m_armed_for = std::nullopt;
	bool m_is_hydrating = false;
	sycl::handler* m_sycl_cgh = nullptr;

	closure_hydrator() = default;

	template <typename Closure>
	[[nodiscard]] auto hydrate(const target tgt, sycl::handler* cgh, const Closure& closure) {
		static_assert(std::is_copy_constructible_v<std::decay_t<Closure>>);
		assert(m_armed_for.has_value() && *m_armed_for == tgt);
		assert(tgt == target::host_task || cgh != nullptr);
		m_sycl_cgh = cgh;
		m_is_hydrating = true;
		Closure hydrated{closure};
		m_is_hydrating = false;
		m_sycl_cgh = nullptr;
		m_accessor_infos.clear();
		m_armed_for = std::nullopt;
		return hydrated;
	}
};

}; // namespace celerity::detail
