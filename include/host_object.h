#pragma once

#include <memory>
#include <mutex>
#include <type_traits>
#include <unordered_set>
#include <utility>

#include "runtime.h"


namespace celerity::experimental {

template <typename T>
class host_object;

} // namespace celerity::experimental

namespace celerity::detail {

class host_object_manager {
  public:
	host_object_id create_host_object() {
		const std::lock_guard lock{m_mutex};
		const auto id = m_next_id++;
		m_objects.emplace(id);
		return id;
	}

	void destroy_host_object(const host_object_id id) {
		const std::lock_guard lock{m_mutex};
		m_objects.erase(id);
	}

	// true-result only reliable if no calls to create_host_object() are pending
	bool has_active_objects() const {
		const std::lock_guard lock{m_mutex};
		return !m_objects.empty();
	}

  private:
	mutable std::mutex m_mutex;
	host_object_id m_next_id = 0;
	std::unordered_set<host_object_id> m_objects;
};

// Base for `state` structs in all host_object specializations: registers and unregisters host_objects with the host_object_manager.
struct host_object_tracker {
	detail::host_object_id id{};

	host_object_tracker() {
		if(!detail::runtime::is_initialized()) { detail::runtime::init(nullptr, nullptr); }
		id = detail::runtime::get_instance().get_host_object_manager().create_host_object();
	}

	host_object_tracker(host_object_tracker&&) = delete;
	host_object_tracker& operator=(host_object_tracker&&) = delete;

	~host_object_tracker() { detail::runtime::get_instance().get_host_object_manager().destroy_host_object(id); }
};

// see host_object deduction guides
template <typename T>
struct assert_host_object_ctor_param_is_rvalue {
	static_assert(std::is_rvalue_reference_v<T&&>,
	    "Either pass the constructor parameter as T&& or std::reference_wrapper<T>, or add explicit template arguments to host_object");
	using type = T;
};

template <typename T>
using assert_host_object_ctor_param_is_rvalue_t = typename assert_host_object_ctor_param_is_rvalue<T>::type;

template <typename T>
host_object_id get_host_object_id(const experimental::host_object<T>& ho) {
	return ho.get_id();
}

template <typename T>
typename experimental::host_object<T>::instance_type& get_host_object_instance(const experimental::host_object<T>& ho) {
	return ho.get_instance();
}


} // namespace celerity::detail

namespace celerity::experimental {

/**
 * A `host_object` wraps state that exists separately on each worker node and can be referenced in host tasks through `side_effect`s. Celerity ensures that
 * access to the object state is properly synchronized and ordered. An example usage of a host object might be a file stream that is written to from multiple
 * host tasks sequentially.
 *
 * - The generic `host_object<T>` keeps ownership of the state at any time and is the safest way to achieve side effects on the host.
 * - The `host_object<T&>` specialization attaches Celerity's tracking and synchronization mechanism to user-managed state. The user guarantees that the
 *   referenced object is not accessed in any way other than through a `side_effect` while the `host_object` is live.
 * - `host_object<void>` does not carry internal state and can be used to track access to global variables or functions like `printf()`.
 */
template <typename T>
class host_object {
	static_assert(std::is_object_v<T>); // disallow host_object<T&&> and host_object<function-type>

  public:
	using instance_type = T;

	host_object() : m_shared_state(std::make_shared<state>(std::in_place)) {}

	explicit host_object(const T& obj) : m_shared_state(std::make_shared<state>(std::in_place, obj)) {}

	explicit host_object(T&& obj) : m_shared_state(std::make_shared<state>(std::in_place, std::move(obj))) {}

	/// Constructs the object in-place with the given constructor arguments.
	template <typename... CtorParams>
	explicit host_object(const std::in_place_t, CtorParams&&... ctor_args) // requiring std::in_place avoids overriding copy and move constructors
	    : m_shared_state(std::make_shared<state>(std::in_place, std::forward<CtorParams>(ctor_args)...)) {}

  private:
	template <typename U>
	friend detail::host_object_id detail::get_host_object_id(const experimental::host_object<U>& ho);

	template <typename U>
	friend typename experimental::host_object<U>::instance_type& detail::get_host_object_instance(const experimental::host_object<U>& ho);

	struct state : detail::host_object_tracker {
		T instance;

		template <typename... CtorParams>
		explicit state(const std::in_place_t, CtorParams&&... ctor_args) : instance(std::forward<CtorParams>(ctor_args)...) {}
	};

	detail::host_object_id get_id() const { return m_shared_state->id; }
	T& get_instance() const { return m_shared_state->instance; }

	std::shared_ptr<state> m_shared_state;
};

template <typename T>
class host_object<T&> {
  public:
	using instance_type = T;

	explicit host_object(T& obj) : m_shared_state(std::make_shared<state>(obj)) {}

	explicit host_object(const std::reference_wrapper<T> ref) : m_shared_state(std::make_shared<state>(ref.get())) {}

  private:
	template <typename U>
	friend detail::host_object_id detail::get_host_object_id(const experimental::host_object<U>& ho);

	template <typename U>
	friend typename experimental::host_object<U>::instance_type& detail::get_host_object_instance(const experimental::host_object<U>& ho);

	struct state : detail::host_object_tracker {
		T& instance;

		explicit state(T& instance) : instance{instance} {}
	};

	detail::host_object_id get_id() const { return m_shared_state->id; }
	T& get_instance() const { return m_shared_state->instance; }

	std::shared_ptr<state> m_shared_state;
};

template <>
class host_object<void> {
  public:
	using instance_type = void;

	explicit host_object() : m_shared_state(std::make_shared<state>()) {}

  private:
	template <typename U>
	friend detail::host_object_id detail::get_host_object_id(const experimental::host_object<U>& ho);

	struct state : detail::host_object_tracker {};

	detail::host_object_id get_id() const { return m_shared_state->id; }

	std::shared_ptr<state> m_shared_state;
};

// The universal reference parameter T&& matches U& as well as U&& for object types U, but we don't want to implicitly invoke a copy constructor: the user
// might have intended to either create a host_object<T&> (which requires a std::reference_wrapper parameter) or move-construct the interior.
template <typename T>
explicit host_object(T&&) -> host_object<detail::assert_host_object_ctor_param_is_rvalue_t<T>>;

template <typename T>
explicit host_object(std::reference_wrapper<T>) -> host_object<T&>;

explicit host_object()->host_object<void>;

} // namespace celerity::experimental
