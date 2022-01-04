#pragma once

#include <CL/sycl.hpp>
#include <sycl/sycl.hpp>

// TODO: Don't pollute preprocessor namespace with generic "WORKAROUND" names, prefix with "CELERITY".

#if defined(CELERITY_DPCPP)
#define WORKAROUND_DPCPP 1
#else
#define WORKAROUND_DPCPP 0
#endif

#if defined(__COMPUTECPP__)
#define WORKAROUND_COMPUTECPP 1
#define CELERITY_DETAIL_WA_VERSION_MAJOR COMPUTECPP_VERSION_MAJOR
#define CELERITY_DETAIL_WA_VERSION_MINOR COMPUTECPP_VERSION_MINOR
#define CELERITY_DETAIL_WA_VERSION_PATCH COMPUTECPP_VERSION_PATCH
#else
#define WORKAROUND_COMPUTECPP 0
#endif

#if defined(__HIPSYCL__)
#define WORKAROUND_HIPSYCL 1
#define CELERITY_DETAIL_WA_VERSION_MAJOR HIPSYCL_VERSION_MAJOR
#define CELERITY_DETAIL_WA_VERSION_MINOR HIPSYCL_VERSION_MINOR
#define CELERITY_DETAIL_WA_VERSION_PATCH HIPSYCL_VERSION_PATCH
#else
#define WORKAROUND_HIPSYCL 0
#endif

#define CELERITY_DETAIL_WA_CHECK_VERSION_1(major) (CELERITY_DETAIL_WA_VERSION_MAJOR <= major)
#define CELERITY_DETAIL_WA_CHECK_VERSION_2(major, minor)                                                                                                       \
	(CELERITY_DETAIL_WA_VERSION_MAJOR < major) || (CELERITY_DETAIL_WA_VERSION_MAJOR == major && CELERITY_DETAIL_WA_VERSION_MINOR <= minor)
#define CELERITY_DETAIL_WA_CHECK_VERSION_3(major, minor, patch)                                                                                                \
	(CELERITY_DETAIL_WA_VERSION_MAJOR < major) || (CELERITY_DETAIL_WA_VERSION_MAJOR == major && CELERITY_DETAIL_WA_VERSION_MINOR < minor)                      \
	    || (CELERITY_DETAIL_WA_VERSION_MAJOR == major && CELERITY_DETAIL_WA_VERSION_MINOR == minor && CELERITY_DETAIL_WA_VERSION_PATCH <= patch)

#define CELERITY_DETAIL_WA_GET_OVERLOAD(_1, _2, _3, NAME, ...) NAME
#define CELERITY_DETAIL_WA_MSVC_WORKAROUND(x) x // Workaround for MSVC PP expansion behavior of __VA_ARGS__
#define CELERITY_DETAIL_WA_CHECK_VERSION(...)                                                                                                                  \
	CELERITY_DETAIL_WA_MSVC_WORKAROUND(CELERITY_DETAIL_WA_GET_OVERLOAD(                                                                                        \
	    __VA_ARGS__, CELERITY_DETAIL_WA_CHECK_VERSION_3, CELERITY_DETAIL_WA_CHECK_VERSION_2, CELERITY_DETAIL_WA_CHECK_VERSION_1)(__VA_ARGS__))

#define WORKAROUND(impl, ...) (WORKAROUND_##impl == 1 && CELERITY_DETAIL_WA_CHECK_VERSION(__VA_ARGS__))


namespace celerity::detail {

// "normal" accessors return T& references to data, whereas atomic accessors return sycl::atomic<T> pointer wrappers. decltype(auto) is not enough to give
// ranged_sycl_access() a correct generic return type, so we specify it manually through the following trait.

template <typename T, sycl::access_mode Mode, sycl::access::target Target>
struct access_value_trait {
	using type = T;
	using reference = T&;
};

template <typename T, sycl::access::target Target>
struct access_value_trait<T, sycl::access_mode::read, Target> {
	using type = const T;
	using reference = const T&;
};

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations" // mode::atomic and target::local are deprecated

template <typename T, sycl::access::target Target>
struct access_value_trait<T, sycl::access_mode::atomic, Target> {
	using type = T;
	using reference = sycl::atomic<T, sycl::access::address_space::global_space>;
};

template <typename T>
struct access_value_trait<T, sycl::access_mode::atomic, sycl::access::target::local> {
	using type = T;
	using reference = sycl::atomic<T, sycl::access::address_space::local_space>;
};

#pragma GCC diagnostic pop

template <typename T, sycl::access_mode Mode, sycl::access::target Target>
using access_value_type = typename access_value_trait<T, Mode, Target>::type;

template <typename T, sycl::access_mode Mode, sycl::access::target Target>
using access_value_reference = typename access_value_trait<T, Mode, Target>::reference;

// SYCL implementations have differences in computing accessor indices involving offsets. The behavior changed between versions 1.2.1 and 2020, and there is a
// bug to iron out for ComputeCpp 2.7.0. TODO The buffer_range parameter in all these methods can be removed once we stop supporting ComputeCpp < 2.8.0.

#if WORKAROUND_HIPSYCL || WORKAROUND(COMPUTECPP, 2, 6)

template <typename T, int Dims, sycl::access_mode Mode, sycl::access::target Target, sycl::access::placeholder IsPlaceholder>
access_value_reference<T, Mode, Target> ranged_sycl_access(
    const sycl::accessor<T, Dims, Mode, Target, IsPlaceholder>& acc, const sycl::range<Dims>&, const sycl::id<Dims>& index) {
	// SYCL 1.2.1 behavior
	return acc[acc.get_offset() + index];
}

#elif WORKAROUND(COMPUTECPP, 2, 7)

template <typename T, int Dims, sycl::access_mode Mode, sycl::access::target Target, sycl::access::placeholder IsPlaceholder>
access_value_reference<T, Mode, Target> ranged_sycl_access(
    const sycl::accessor<T, Dims, Mode, Target, IsPlaceholder>& acc, const sycl::range<Dims>& buffer_range, const sycl::id<Dims>& index) {
	// The linear index computation involving an accessor offset is buggy in ComputeCpp 2.7.0, it simply sums up linear indices of offset and item.
	// We compute the linear index manually as a workaround. This is undesirable for other backends since it requires capturing the buffer range.
	size_t linear_index = 0;
	for(int d = 0; d < Dims; ++d) {
		linear_index = linear_index * buffer_range[d] + acc.get_offset()[d] + index[d];
	}
	if constexpr(Mode == sycl::access_mode::atomic) {
		return access_value_reference<T, Mode, Target>{acc.get_pointer() + linear_index};
	} else {
		const auto memory = static_cast<access_value_type<T, Mode, Target>*>(acc.get_pointer());
		return memory[linear_index];
	}
}

#else

template <typename T, int Dims, sycl::access_mode Mode, sycl::access::target Target, sycl::access::placeholder IsPlaceholder>
access_value_reference<T, Mode, Target> ranged_sycl_access(
    const sycl::accessor<T, Dims, Mode, Target, IsPlaceholder>& acc, const sycl::range<Dims>&, const sycl::id<Dims>& index) {
	// SYCL 2020 behavior
	return acc[index];
}

#endif

} // namespace celerity::detail