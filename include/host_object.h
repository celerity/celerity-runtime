#pragma once

#include <memory>
#include <type_traits>
#include <utility>

#include "runtime.h"
#include "tracy.h"

namespace celerity::experimental {

template <typename T>
class host_object;

} // namespace celerity::experimental

namespace celerity::detail {

/// Host objects that own their instance (i.e. not host_object<T&> nor host_object<void>) wrap it in a type deriving from this struct in order to pass it to
/// the executor for (virtual) destruction from within the instruction graph.
struct host_object_instance {
	host_object_instance() = default;
	host_object_instance(const host_object_instance&) = delete;
	host_object_instance(host_object_instance&&) = delete;
	host_object_instance& operator=(host_object_instance&&) = delete;
	host_object_instance& operator=(const host_object_instance&) = delete;
	virtual ~host_object_instance() = default;
};

/// A `tacker` instance is shared by all copies of any `host_object` via a `std::shared_ptr` to implement (SYCL) reference semantics.
/// It notifies the runtime of host object creation and destruction.
struct host_object_tracker {
	detail::host_object_id id{};

	explicit host_object_tracker(std::unique_ptr<host_object_instance> instance) {
		CELERITY_DETAIL_TRACY_ZONE_SCOPED("host_object::host_object", DarkSlateBlue);
		if(!detail::runtime::has_instance()) { detail::runtime::init(nullptr, nullptr); }
		id = detail::runtime::get_instance().create_host_object(std::move(instance));
	}

	host_object_tracker(const host_object_tracker&) = delete;
	host_object_tracker(host_object_tracker&&) = delete;
	host_object_tracker& operator=(host_object_tracker&&) = delete;
	host_object_tracker& operator=(const host_object_tracker&) = delete;

	~host_object_tracker() {
		CELERITY_DETAIL_TRACY_ZONE_SCOPED("~host_object::host_object", DarkCyan);
		detail::runtime::get_instance().destroy_host_object(id);
	}
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
	assert(ho.m_tracker != nullptr);
	return ho.m_tracker->id;
}

template <typename T>
typename experimental::host_object<T>::instance_type* get_host_object_instance(const experimental::host_object<T>& ho) {
	// By attaching `instance` to `tracker` instead of `host_object` directly, we guarantee that a pointer returned by get_host_object_instance (for an owning
	// host_object) can never be dangling even if the `host_object` (reference-type) has been moved from.
	assert(ho.m_tracker != nullptr);
	return ho.m_tracker->instance;
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

	host_object() : host_object(std::in_place) {}
	explicit host_object(const instance_type& obj) : host_object(std::in_place, obj) {}
	explicit host_object(instance_type&& obj) : host_object(std::in_place, std::move(obj)) {}

	/// Constructs the object in-place with the given constructor arguments.
	template <typename... CtorParams>
	explicit host_object(const std::in_place_t /* tag */, CtorParams&&... ctor_args)
	    : m_tracker(std::make_shared<tracker>(std::make_unique<instance>(std::in_place, std::forward<CtorParams>(ctor_args)...))) {}

  private:
	struct instance : detail::host_object_instance {
		instance_type value;

		template <typename... CtorParams>
		explicit instance(const std::in_place_t /* do not override copy / move ctors */, CtorParams&&... ctor_args)
		    : value(std::forward<CtorParams>(ctor_args)...) {}
	};

	struct tracker : detail::host_object_tracker {
		instance_type* instance = nullptr; // owned by host_object_instance (executor)

		explicit tracker(std::unique_ptr<struct instance> instance) : tracker(&instance->value, instance) {} // delegate to read .value before moving instance
		explicit tracker(instance_type* const ref, std::unique_ptr<struct instance>& owned) : detail::host_object_tracker(std::move(owned)), instance(ref) {}
	};

	template <typename U>
	friend detail::host_object_id detail::get_host_object_id(const experimental::host_object<U>& ho);

	template <typename U>
	friend typename experimental::host_object<U>::instance_type* detail::get_host_object_instance(const experimental::host_object<U>& ho);

	std::shared_ptr<tracker> m_tracker;
};

template <typename T>
class host_object<T&> {
  public:
	using instance_type = T;

	explicit host_object(instance_type& obj) : m_tracker(std::make_shared<tracker>(&obj)) {}
	explicit host_object(const std::reference_wrapper<instance_type> ref) : host_object(ref.get()) {}

  private:
	struct tracker : detail::host_object_tracker {
		instance_type* instance = nullptr; // owned by user
		explicit tracker(instance_type* const ref) : detail::host_object_tracker(nullptr /* no owned instance */), instance(ref) {}
	};

	template <typename U>
	friend detail::host_object_id detail::get_host_object_id(const experimental::host_object<U>& ho);

	template <typename U>
	friend typename experimental::host_object<U>::instance_type* detail::get_host_object_instance(const experimental::host_object<U>& ho);

	std::shared_ptr<tracker> m_tracker;
};

template <>
class host_object<void> {
  public:
	using instance_type = void;

	explicit host_object() : m_tracker(std::make_shared<tracker>()) {}

  private:
	struct tracker : detail::host_object_tracker {
		tracker() : detail::host_object_tracker(nullptr /* no owned instance */) {}
	};

	template <typename U>
	friend detail::host_object_id detail::get_host_object_id(const experimental::host_object<U>& ho);

	std::shared_ptr<tracker> m_tracker;
};

// The universal reference parameter T&& matches U& as well as U&& for object types U, but we don't want to implicitly invoke a copy constructor: the user
// might have intended to either create a host_object<T&> (which requires a std::reference_wrapper parameter) or move-construct the interior.
template <typename T>
explicit host_object(T&&) -> host_object<detail::assert_host_object_ctor_param_is_rvalue_t<T>>;

template <typename T>
explicit host_object(std::reference_wrapper<T>) -> host_object<T&>;

explicit host_object()->host_object<void>;

} // namespace celerity::experimental
