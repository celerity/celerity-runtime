#pragma once

#include <type_traits>

#include <CL/sycl.hpp>
#include <allscale/utils/functional_utils.h>

#include "access_modes.h"
#include "buffer.h"
#include "buffer_storage.h"
#include "ccpp_2020_compatibility_layer.h"
#include "handler.h"

namespace celerity {

template <int Dims>
class partition;

template <typename DataT, int Dims, cl::sycl::access_mode Mode, cl::sycl::target Target>
class accessor;

namespace detail {

	template <typename DataT, int Dims, cl::sycl::access_mode Mode, cl::sycl::target Target>
	class accessor_base {
	  public:
		static_assert(Dims > 0, "0-dimensional accessors NYI");
		static_assert(Dims <= 3, "accessors can only have 3 dimensions or less");
		using value_type = DataT;
		using reference = DataT&;
		using const_reference = const DataT&;
	};

	template <typename DataT, int Dims, cl::sycl::access_mode Mode, typename... Args>
	accessor<DataT, Dims, Mode, cl::sycl::target::device> make_device_accessor(Args&&...);

	template <typename DataT, int Dims, cl::sycl::access_mode Mode, typename... Args>
	accessor<DataT, Dims, Mode, cl::sycl::target::host_buffer> make_host_accessor(Args&&...);

	template <typename TagT>
	constexpr cl::sycl::access_mode deduce_access_mode();

	template <typename TagT>
	constexpr cl::sycl::access_mode deduce_access_mode_discard();

	template <typename TagT>
	constexpr cl::sycl::target deduce_access_target();

	template <typename DataT, int Dims, cl::sycl::access::mode Mode, cl::sycl::target Target, int Index>
	class accessor_subscript_proxy;
} // namespace detail

/**
 * Maps slices of the accessor backing buffer present on a host to the virtual global range of the Celerity buffer.
 */
class host_memory_layout {
  public:
	/**
	 * Layout map for a single dimension describing the offset and strides of its hyperplanes.
	 *
	 * - A zero-dimensional layout corresponds to an individual data item and is not explicitly modelled in the dimension vector.
	 * - A one-dimensional layout is an interval of one-dimensional space and is fully described by global and local offsets and a count of data items (aka
	 * 0-dimensional hyperplanes).
	 * - A two-dimensional layout is modelled as an interval of rows, which manifests as an offset (a multiple of the row width) and a stride (the row width
	 * itself). Each row (aka 1-dimensional hyperplane) is modelled by the same one-dimensional layout.
	 * - and so on for arbitrary dimensioned layouts.
	 */
	class dimension {
	  public:
		dimension() noexcept = default;

		dimension(size_t global_size, size_t global_offset, size_t local_size, size_t local_offset, size_t extent)
		    : global_size(global_size), global_offset(global_offset), local_size(local_size), local_offset(local_offset), extent(extent) {
			assert(global_offset >= local_offset);
			assert(global_size >= local_size);
		}

		size_t get_global_size() const { return global_size; }

		size_t get_local_size() const { return local_size; }

		size_t get_global_offset() const { return global_offset; }

		size_t get_local_offset() const { return local_offset; }

		size_t get_extent() const { return extent; }

	  private:
		size_t global_size{};
		size_t global_offset{};
		size_t local_size{};
		size_t local_offset{};
		size_t extent{};
	};

	// TODO: This is a temporary replacement for Boost's static_vector; entire mechanism needs more comprehensive API overhaul.
	class dimension_vector {
	  public:
		dimension_vector(size_t size) : this_size(size) {}

		dimension& operator[](size_t idx) { return values[idx]; }
		const dimension& operator[](size_t idx) const { return values[idx]; }

		size_t size() const { return this_size; }

	  private:
		/**
		 * Since contiguous dimensions can be merged when generating the memory layout, host_memory_layout is not generic over a fixed dimension count
		 * TODO: Implement this ^
		 */
		constexpr static size_t max_dimensionality = 4;
		std::array<dimension, max_dimensionality> values;
		size_t this_size;
	};

	explicit host_memory_layout(const dimension_vector& dimensions) : dimensions(dimensions) {}

	/** The layout maps per dimension, in descending dimensionality */
	const dimension_vector& get_dimensions() const { return dimensions; }

  private:
	dimension_vector dimensions;
};

/**
 * Celerity wrapper around SYCL accessors.
 *
 * @note The Celerity accessor currently does not support get_size, get_count, get_range, get_offset and get_pointer,
 * as their semantics in a distributed context are unclear.
 */
template <typename DataT, int Dims, cl::sycl::access_mode Mode>
class accessor<DataT, Dims, Mode, cl::sycl::target::device> : public detail::accessor_base<DataT, Dims, Mode, cl::sycl::target::device> {
  public:
	accessor(const accessor& other) : sycl_accessor(other.sycl_accessor) { init_from(other); }

	template <cl::sycl::target Target = cl::sycl::target::device, typename Functor>
	accessor(const buffer<DataT, Dims>& buff, handler& cgh, Functor rmfn) {
		using rmfn_traits = allscale::utils::lambda_traits<Functor>;
		static_assert(rmfn_traits::result_type::dims == Dims, "The returned subrange doesn't match buffer dimensions.");

		if(detail::is_prepass_handler(cgh)) {
			auto& prepass_cgh = dynamic_cast<detail::prepass_handler&>(cgh);
			prepass_cgh.add_requirement(
			    detail::get_buffer_id(buff), std::make_unique<detail::range_mapper<rmfn_traits::arg1_type::dims, Dims>>(rmfn, Mode, buff.get_range()));
			sycl_accessor = sycl_accessor_t();

		} else {
			if(detail::get_handler_execution_target(cgh) != detail::execution_target::DEVICE) {
				throw std::runtime_error(
				    "Calling accessor constructor with device target is only allowed in parallel_for tasks."
				    "If you want to access this buffer from within a host task, please specialize the call using one of the *_host_task tags");
			}
			auto& live_cgh = dynamic_cast<detail::live_pass_device_handler&>(cgh);
			// It's difficult to figure out which stored range mapper corresponds to this constructor call, which is why we just call the raw mapper
			// manually. This also means that we have to clamp the subrange ourselves here, which is not ideal from an encapsulation standpoint.
			const auto sr = detail::clamp_subrange_to_buffer_size(live_cgh.apply_range_mapper<Dims>(rmfn, buff.get_range()), buff.get_range());
			auto access_info = detail::runtime::get_instance().get_buffer_manager().get_device_buffer<DataT, Dims>(
			    detail::get_buffer_id(buff), Mode, detail::range_cast<3>(sr.range), detail::id_cast<3>(sr.offset));
			eventual_sycl_cgh = live_cgh.get_eventual_sycl_cgh();
			sycl_accessor = sycl_accessor_t(access_info.buffer, buff.get_range(), access_info.offset);
			backing_buffer_offset = access_info.offset;
		}
	}

	template <typename Functor, typename TagT>
	accessor(const buffer<DataT, Dims>& buff, handler& cgh, Functor rmfn, TagT tag) : accessor(buff, cgh, rmfn) {}

	template <typename Functor, typename TagT>
	accessor(const buffer<DataT, Dims>& buff, handler& cgh, Functor rmfn, TagT tag, cl::sycl::property::no_init no_init) : accessor(buff, cgh, rmfn) {}

	template <typename Functor, typename TagT>
	accessor(const buffer<DataT, Dims>& buff, handler& cgh, Functor rmfn, TagT tag, cl::sycl::property_list prop_list) {
		// in this static assert it would be more relevant to use property_list type, but if a defined type is used then it is always false and
		// always fails to compile. Hence we use a templated type so that it only produces a compile error when the ctr is called.
		static_assert(!std::is_same_v<TagT, TagT>,
		    "Currently it is not accepted to pass a property list to an accessor constructor. Please use the property cl::sycl::no_init "
		    "as a last argument in the constructor");
	}

	accessor& operator=(const accessor& other) {
		if(this != &other) {
			sycl_accessor = other.sycl_accessor;
			init_from(other);
		}
		return *this;
	}

	template <cl::sycl::access_mode M = Mode, int D = Dims>
	std::enable_if_t<detail::access::mode_traits::is_producer(M) && M != cl::sycl::access_mode::atomic && (D > 0), DataT&> operator[](
	    cl::sycl::id<Dims> index) const {
		return sycl_accessor[index - backing_buffer_offset];
	}

	template <cl::sycl::access_mode M = Mode, int D = Dims>
	std::enable_if_t<detail::access::mode_traits::is_pure_consumer(M) && (D > 0), DataT> operator[](cl::sycl::id<Dims> index) const {
		return sycl_accessor[index - backing_buffer_offset];
	}

	template <cl::sycl::access_mode M = Mode, int D = Dims>
	[[deprecated("Atomic accessors are deprecated as of SYCL 2020")]] std::enable_if_t<M == cl::sycl::access_mode::atomic && (D > 0), cl::sycl::atomic<DataT>>
	operator[](cl::sycl::id<Dims> index) const {
#pragma GCC diagnostic push
		// Ignore deprecation warnings emitted by SYCL implementations (e.g. hipSYCL)
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
		return sycl_accessor[index - backing_buffer_offset];
#pragma GCC diagnostic pop
	}

	template <int D = Dims>
	std::enable_if_t<(D > 1), detail::accessor_subscript_proxy<DataT, D, Mode, cl::sycl::target::device, 1>> operator[](const size_t d0) const {
		return {*this, d0};
	}

	friend bool operator==(const accessor& lhs, const accessor& rhs) {
		return lhs.sycl_accessor == rhs.sycl_accessor && lhs.backing_buffer_offset == rhs.backing_buffer_offset;
	}

	friend bool operator!=(const accessor& lhs, const accessor& rhs) { return !(lhs == rhs); }

  private:
#if WORKAROUND_COMPUTECPP
	using sycl_accessor_t = cl::sycl::accessor<DataT, Dims, Mode, cl::sycl::access::target::global_buffer, cl::sycl::access::placeholder::true_t>;
#else
	using sycl_accessor_t = cl::sycl::accessor<DataT, Dims, Mode, cl::sycl::target::device>;
#endif

	template <typename T, int D, cl::sycl::access_mode M, typename... Args>
	friend accessor<T, D, M, cl::sycl::target::device> detail::make_device_accessor(Args&&...);

	// see init_from
	cl::sycl::handler* const* eventual_sycl_cgh = nullptr;
	sycl_accessor_t sycl_accessor;

	cl::sycl::id<Dims> backing_buffer_offset;

	accessor(cl::sycl::handler* const* eventual_sycl_cgh, const subrange<Dims>& mapped_subrange, cl::sycl::buffer<DataT, Dims>& buffer,
	    cl::sycl::id<Dims> backing_buffer_offset)
	    : eventual_sycl_cgh(eventual_sycl_cgh),
	      // We pass a range and offset here to avoid interference from SYCL, but the offset must be relative to the *backing buffer*.
	      sycl_accessor(sycl_accessor_t(buffer, mapped_subrange.range, mapped_subrange.offset - backing_buffer_offset)),
	      backing_buffer_offset(backing_buffer_offset) {
		// SYCL 1.2.1 dictates that all kernel parameters must have standard layout.
		// However, since we are wrapping a SYCL accessor, this assertion fails for some implementations,
		// as it is currently unclear whether SYCL accessors must also have standard layout.
		// See https://github.com/KhronosGroup/SYCL-Docs/issues/94
		// static_assert(std::is_standard_layout<accessor>::value, "accessor must have standard layout");
	}

	void init_from(const accessor& other) {
		eventual_sycl_cgh = other.eventual_sycl_cgh;
		backing_buffer_offset = other.backing_buffer_offset;

		// The call to sycl::handler::require must happen inside the SYCL CGF, but since get_access within the Celerity CGF is executed before
		// the submission to SYCL, it needs to be deferred. We capture a reference to a SYCL handler pointer owned by the live pass handler that is
		// initialized once the SYCL CGF is entered. We then abuse the copy constructor that is called implicitly when the lambda is copied for the SYCL
		// kernel submission to call require().
#if !defined(__SYCL_DEVICE_ONLY__) && !defined(SYCL_DEVICE_ONLY)
		if(eventual_sycl_cgh != nullptr && *eventual_sycl_cgh != nullptr) {
			(*eventual_sycl_cgh)->require(sycl_accessor);
			eventual_sycl_cgh = nullptr; // only `require` once
		}
#endif
	}
};

// Celerity Accessor Deduction Guides
template <typename T, int D, typename Functor, typename TagT>
accessor(const buffer<T, D>& buff, handler& cgh, Functor rmfn, TagT tag)
    -> accessor<T, D, detail::deduce_access_mode<TagT>(), detail::deduce_access_target<TagT>()>;

template <typename T, int D, typename Functor, typename TagT>
accessor(const buffer<T, D>& buff, handler& cgh, Functor rmfn, TagT tag, cl::sycl::property::no_init no_init)
    -> accessor<T, D, detail::deduce_access_mode_discard<TagT>(), detail::deduce_access_target<TagT>()>;

template <typename T, int D, typename Functor, typename TagT>
accessor(const buffer<T, D>& buff, handler& cgh, Functor rmfn, TagT tag, cl::sycl::property_list prop_list)
    -> accessor<T, D, detail::deduce_access_mode_discard<TagT>(), detail::deduce_access_target<TagT>()>;

//

template <typename DataT, int Dims, cl::sycl::access_mode Mode>
class accessor<DataT, Dims, Mode, cl::sycl::target::host_buffer> : public detail::accessor_base<DataT, Dims, Mode, cl::sycl::target::host_buffer> {
  public:
	template <cl::sycl::target Target = cl::sycl::target::host_buffer, typename Functor>
	accessor(const buffer<DataT, Dims>& buff, handler& cgh, Functor rmfn) {
		static_assert(!std::is_same_v<Functor, cl::sycl::range<Dims>>,
		    "The accessor constructor overload for master-access tasks (now called 'host tasks') has "
		    "been removed with Celerity 0.2.0. Please provide a range mapper instead.");

		using rmfn_traits = allscale::utils::lambda_traits<Functor>;
		static_assert(rmfn_traits::result_type::dims == Dims, "The returned subrange doesn't match buffer dimensions.");

		if(detail::is_prepass_handler(cgh)) {
			auto& prepass_cgh = dynamic_cast<detail::prepass_handler&>(cgh);
			prepass_cgh.add_requirement(
			    detail::get_buffer_id(buff), std::make_unique<detail::range_mapper<rmfn_traits::arg1_type::dims, Dims>>(rmfn, Mode, buff.get_range()));
		} else {
			if constexpr(Target == cl::sycl::target::host_buffer) {
				if(detail::get_handler_execution_target(cgh) != detail::execution_target::HOST) {
					throw std::runtime_error(
					    "Calling accessor constructor with host_buffer target is only allowed in host tasks."
					    "If you want to access this buffer from within a parallel_for task, please specialize the call using one of the non host tags");
				}
				auto& live_cgh = dynamic_cast<detail::live_pass_host_handler&>(cgh);
				// It's difficult to figure out which stored range mapper corresponds to this constructor call, which is why we just call the raw mapper
				// manually. This also means that we have to clamp the subrange ourselves here, which is not ideal from an encapsulation standpoint.
				const auto sr = detail::clamp_subrange_to_buffer_size(live_cgh.apply_range_mapper<Dims>(rmfn, buff.get_range()), buff.get_range());
				auto access_info = detail::runtime::get_instance().get_buffer_manager().get_host_buffer<DataT, Dims>(
				    detail::get_buffer_id(buff), Mode, detail::range_cast<3>(sr.range), detail::id_cast<3>(sr.offset));

				mapped_subrange = sr;
				optional_buffer = &access_info.buffer;
				backing_buffer_offset = access_info.offset;
				virtual_buffer_range = buff.get_range();
			}
		}
	}

	template <typename Functor, typename TagT>
	accessor(const buffer<DataT, Dims>& buff, handler& cgh, Functor rmfn, TagT tag) : accessor(buff, cgh, rmfn) {}

	/**
	 * TODO: As of ComputeCpp 2.5.0 they do not support no_init prop, hence this constructor is needed along with discard deduction guide.
	 *    but once they do this should be replace for a constructor that takes a prop list as an argument.
	 */
	template <typename Functor, typename TagT>
	accessor(const buffer<DataT, Dims>& buff, handler& cgh, Functor rmfn, TagT tag, cl::sycl::property::no_init no_init) : accessor(buff, cgh, rmfn) {}

	template <typename Functor, typename TagT>
	accessor(const buffer<DataT, Dims>& buff, handler& cgh, Functor rmfn, TagT tag, cl::sycl::property_list prop_list) {
		// in this static assert it would be more relevant to use property_list type, but if a defined type is used then it is always false and
		// always fails to compile. Hence we use a templated type so that it only produces a compile error when the ctr is loaded.
		static_assert(!std::is_same_v<TagT, TagT>,
		    "Currently it is not accepted to pass a property list to an accessor constructor. Please use the property cl::sycl::no_init "
		    "as a last argument in the constructor");
	}

	template <cl::sycl::access_mode M = Mode, int D = Dims>
	std::enable_if_t<detail::access::mode_traits::is_producer(M) && (D > 0), DataT&> operator[](cl::sycl::id<Dims> index) const {
		return *(get_buffer().get_pointer() + get_linear_offset(index));
	}

	template <cl::sycl::access_mode M = Mode, int D = Dims>
	std::enable_if_t<detail::access::mode_traits::is_pure_consumer(M) && (D > 0), DataT> operator[](cl::sycl::id<Dims> index) const {
		return *(get_buffer().get_pointer() + get_linear_offset(index));
	}

	template <int D = Dims>
	std::enable_if_t<(D > 1), detail::accessor_subscript_proxy<DataT, D, Mode, cl::sycl::target::host_buffer, 1>> operator[](const size_t d0) const {
		return {*this, d0};
	}

	/**
	 * @brief Returns a pointer to the underlying buffer.
	 *
	 * Since the physical (or "backing") buffer underlying a Celerity buffer might have a different size,
	 * it is not always safe to return a pointer, as the stride might not be what is expected.
	 *
	 * However, it is always possible to get a pointer if the full buffer is being accessed.
	 *
	 * This API will be deprecated and subsequently removed in a future version.
	 *
	 * @throws Throws a std::logic_error if the buffer cannot be accessed with the expected stride.
	 */
	DataT* get_pointer() const {
		bool illegal_access = false;
		if(backing_buffer_offset != detail::id_cast<Dims>(cl::sycl::id<3>{0, 0, 0})) { illegal_access = true; }
		// We can be a bit more lenient for 1D buffers, in that the backing buffer doesn't have to have the full size.
		// (Dereferencing the pointer outside of the requested range is UB anyways).
		if(Dims > 1 && get_buffer().get_range() != virtual_buffer_range) { illegal_access = true; }
		if(illegal_access) { throw std::logic_error("Buffer cannot be accessed with expected stride"); }
		return get_buffer().get_pointer();
	}

	friend bool operator==(const accessor& lhs, const accessor& rhs) {
		return (lhs.optional_buffer == rhs.optional_buffer || (lhs.optional_buffer && rhs.optional_buffer && *lhs.optional_buffer == *rhs.optional_buffer))
		       && lhs.backing_buffer_offset == rhs.backing_buffer_offset;
	}

	friend bool operator!=(const accessor& lhs, const accessor& rhs) { return !(lhs == rhs); }

	/**
	 * Returns a pointer to the host-local backing buffer along with a mapping to the global virtual buffer.
	 *
	 * Each host keeps only part of the global (virtual) buffer locally. The layout information can be used, for example, to perform distributed I/O on the
	 * partial buffer present at each host.
	 */
	template <int KernelDims>
	std::pair<DataT*, host_memory_layout> get_host_memory(const partition<KernelDims>& part) const {
		// We already know the range mapper output for "chunk" from the constructor. The parameter is a purely semantic dependency which ensures that
		// this function is not called outside a host task.
		(void)part;

		host_memory_layout::dimension_vector dimensions(Dims);
		for(int d = 0; d < Dims; ++d) {
			dimensions[d] = {/* global_size */ virtual_buffer_range[d],
			    /* global_offset */ mapped_subrange.offset[d],
			    /* local_size */ get_buffer().get_range()[d],
			    /* local_offset */ mapped_subrange.offset[d] - backing_buffer_offset[d],
			    /* extent */ mapped_subrange.range[d]};
		}

		return {get_buffer().get_pointer(), host_memory_layout{dimensions}};
	}

  private:
	template <typename T, int D, cl::sycl::access_mode M, typename... Args>
	friend accessor<T, D, M, cl::sycl::target::host_buffer> detail::make_host_accessor(Args&&...);

	// Subange of the accessor, as set by the range mapper or requested by the user (master node host tasks only).
	// This does not necessarily correspond to the backing buffer's range.
	subrange<Dims> mapped_subrange;

	mutable detail::host_buffer<DataT, Dims>* optional_buffer = nullptr;

	// Offset of the backing buffer relative to the virtual buffer.
	cl::sycl::id<Dims> backing_buffer_offset;

	// The range of the Celerity buffer as created by the user.
	// We only need this to check whether it is safe to call get_pointer() or not.
	cl::sycl::range<Dims> virtual_buffer_range = detail::range_cast<Dims>(cl::sycl::range<3>(0, 0, 0));

	/**
	 * Constructor for pre-pass.
	 */
	accessor() = default;

	/**
	 * Constructor for live-pass.
	 */
	accessor(subrange<Dims> mapped_subrange, detail::host_buffer<DataT, Dims>& buffer, cl::sycl::id<Dims> backing_buffer_offset,
	    cl::sycl::range<Dims> virtual_buffer_range)
	    : mapped_subrange(mapped_subrange), optional_buffer(&buffer), backing_buffer_offset(backing_buffer_offset), virtual_buffer_range(virtual_buffer_range) {
	}

	detail::host_buffer<DataT, Dims>& get_buffer() const {
		assert(optional_buffer != nullptr);
		return *optional_buffer;
	}

	size_t get_linear_offset(cl::sycl::id<Dims> index) const { return detail::get_linear_index(get_buffer().get_range(), index - backing_buffer_offset); }
};

namespace detail {
	template <typename DataT, int Dims, cl::sycl::access_mode Mode, typename... Args>
	accessor<DataT, Dims, Mode, cl::sycl::target::global_buffer> make_device_accessor(Args&&... args) {
		return {std::forward<Args>(args)...};
	}

	template <typename DataT, int Dims, cl::sycl::access_mode Mode, typename... Args>
	accessor<DataT, Dims, Mode, cl::sycl::target::host_buffer> make_host_accessor(Args&&... args) {
		return {std::forward<Args>(args)...};
	}

	template <typename TagT>
	constexpr cl::sycl::access_mode deduce_access_mode() {
		if(std::is_same_v<const TagT, decltype(cl::sycl::read_only)> || //
		    std::is_same_v<const TagT, decltype(cl::sycl::read_only_host_task)>) {
			return cl::sycl::access_mode::read;
		} else if(std::is_same_v<const TagT, decltype(cl::sycl::read_write)> || //
		          std::is_same_v<const TagT, decltype(cl::sycl::read_write_host_task)>) {
			return cl::sycl::access_mode::read_write;
		} else {
			return cl::sycl::access_mode::write;
		}
	}

	template <typename TagT>
	constexpr cl::sycl::access_mode deduce_access_mode_discard() {
		if constexpr(std::is_same_v<const TagT, decltype(cl::sycl::read_only)> || //
		             std::is_same_v<const TagT, decltype(cl::sycl::read_only_host_task)>) {
			static_assert(!std::is_same_v<TagT, TagT>, "Invalid access mode + no_init");
		} else if(std::is_same_v<const TagT, decltype(cl::sycl::read_write)> || //
		          std::is_same_v<const TagT, decltype(cl::sycl::read_write_host_task)>) {
			return cl::sycl::access_mode::discard_read_write;
		} else {
			return cl::sycl::access_mode::discard_write;
		}
	}

	template <typename TagT>
	constexpr cl::sycl::target deduce_access_target() {
		if(std::is_same_v<const TagT, decltype(cl::sycl::read_only)> ||   //
		    std::is_same_v<const TagT, decltype(cl::sycl::read_write)> || //
		    std::is_same_v<const TagT, decltype(cl::sycl::write_only)>) {
			return cl::sycl::target::device;
		} else {
			return cl::sycl::target::host_buffer;
		}
	}


	template <typename DataT, cl::sycl::access::mode Mode, cl::sycl::target Target>
	class accessor_subscript_proxy<DataT, 3, Mode, Target, 2> {
		using AccessorT = celerity::accessor<DataT, 3, Mode, Target>;

	  public:
		accessor_subscript_proxy(const AccessorT& acc, const size_t d0, const size_t d1) : acc(acc), d0(d0), d1(d1) {}

		decltype(std::declval<AccessorT>()[{0, 0, 0}]) operator[](const size_t d2) const { return acc[{d0, d1, d2}]; }

	  private:
		const AccessorT& acc;
		size_t d0;
		size_t d1;
	};

	template <typename DataT, int Dims, cl::sycl::access::mode Mode, cl::sycl::target Target>
	class accessor_subscript_proxy<DataT, Dims, Mode, Target, 1> {
		template <int D>
		using AccessorT = celerity::accessor<DataT, D, Mode, Target>;

	  public:
		accessor_subscript_proxy(const AccessorT<Dims>& acc, const size_t d0) : acc(acc), d0(d0) {}

		// Note that we currently have to use SFINAE over constexpr-if + decltype(auto), as ComputeCpp 2.6.0 has
		// problems inferring the correct type in some cases (e.g. when DataT == sycl::id<>).
		template <int D = Dims>
		std::enable_if_t<D == 2, decltype(std::declval<AccessorT<2>>()[{0, 0}])> operator[](const size_t d1) const {
			return acc[{d0, d1}];
		}

		template <int D = Dims>
		std::enable_if_t<D == 3, accessor_subscript_proxy<DataT, 3, Mode, Target, 2>> operator[](const size_t d1) const {
			return {acc, d0, d1};
		}

	  private:
		const AccessorT<Dims>& acc;
		size_t d0;
	};

} // namespace detail

} // namespace celerity
