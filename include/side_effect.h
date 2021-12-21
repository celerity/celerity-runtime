#pragma once

#include "accessor.h"
#include "runtime.h"


namespace celerity {

template <typename T>
class host_object;

} // namespace celerity

namespace celerity::detail {

struct host_object_info {
	std::optional<task_id> last_writer;
};

class host_object_manager {
  public:
	host_object_id create_host_object() {
		const std::lock_guard lock{mutex};
		const auto id = next_id++;
		objects.emplace(id, host_object_info{});
		return id;
	}

	void destroy_host_object(const host_object_id id) {
		const std::lock_guard lock{mutex};
		objects.erase(id);
	}

	// true-result only reliable if no calls to create_host_object() are pending
	bool has_active_objects() const {
		const std::lock_guard lock{mutex};
		return !objects.empty();
	}

  private:
	mutable std::mutex mutex;
	host_object_id next_id = 0;
	std::unordered_map<host_object_id, host_object_info> objects;
};

} // namespace celerity::detail

namespace celerity {

template <typename T, access_mode Mode>
class side_effect;

template <typename T>
class host_object {
  public:
	host_object() : shared_state{std::make_shared<state>(std::in_place)} {}

	explicit host_object(const T& obj) : shared_state{std::make_shared<state>(std::in_place, obj)} {}

	explicit host_object(T&& obj) : shared_state{std::make_shared<state>(std::in_place, std::move(obj))} {}

	template <typename... CtorParams>
	explicit host_object(const std::in_place_t, CtorParams&&... ctor_args)
	    : shared_state{std::make_shared<state>(std::in_place, std::forward<CtorParams>(ctor_args)...)} {}

  private:
	template <typename, access_mode>
	friend class side_effect;

	struct state {
		T object;
		detail::host_object_id id;

		template <typename... CtorParams>
		explicit state(const std::in_place_t, CtorParams&&... ctor_args) : object{std::forward<CtorParams>(ctor_args)...} {
			if(!detail::runtime::is_initialized()) { detail::runtime::init(nullptr, nullptr); }
			id = detail::runtime::get_instance().get_host_object_manager().create_host_object();
		}

		state(state&&) = delete;
		state& operator=(state&&) = delete;

		~state() { detail::runtime::get_instance().get_host_object_manager().destroy_host_object(id); }
	};

	std::shared_ptr<state> shared_state;
};

template <typename T>
explicit host_object(T& obj) -> host_object<std::remove_const_t<T>>;

template <typename T, access_mode Mode = access_mode::read_write>
class side_effect {
	static_assert(Mode == access_mode::read || Mode == access_mode::write || Mode == access_mode::read_write,
	    "discard and atomic access modes are invalid on side_effect");

  public:
	using interior_type = std::conditional_t<Mode == access_mode::read, const T, T>;

	explicit side_effect(const host_object<T>& object, handler& cgh) : object{object} {
		if(detail::is_prepass_handler(cgh)) {
			auto& prepass_cgh = static_cast<detail::prepass_handler&>(cgh);
			prepass_cgh.add_requirement(object.shared_state->id, Mode);
		}
	}

	template <typename AccessModeTag>
	explicit side_effect(const host_object<T>& object, handler& cgh, const AccessModeTag) : side_effect{object, cgh} {}

	template <int Dims>
	interior_type& operator()(const partition<Dims>& p) const {
		return object.shared_state->object;
	}

  private:
	host_object<T> object;
};

template <typename T, typename Tag>
side_effect(const host_object<T>&, handler&, Tag) -> side_effect<T, detail::deduce_access_mode<Tag>()>;

template <typename T>
side_effect(const host_object<T>&, handler&) -> side_effect<T, access_mode::read_write>;

} // namespace celerity