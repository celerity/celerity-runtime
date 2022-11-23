// TODO: Move parts into CPP
#pragma once

// Attempting to build truly virtualized n-dimensional buffers on top of CUDA 10.2 virtual memory facilities.

// Open questions:
// - How expensive is to create a physical allocation? Can we e.g. create a separate one for each "line" (slower index)
//   in a 2D buffer?
// - How expensive is it to map a physical allocation? If allocating is expensive but mapping is cheap, we can maintain
//   pool of allocations, grow exponentially as needed.
// - Look into CUDAs support for sparse allocations (?). What is that?
// - Later: Consider how to make interface compatible with mdspan
// - Could we have a single virtual address space across all local devices?
//     - In principle yes (see vectorAddMMAP CUDA sample), but I'm not sure how that would work for D2D memcpy between the same addresses then.
//     - For USM we may need something like this though..?
// - Look into vectorAddMMAP CUDA sample for some more pointers on requirements for using virtual memory with multiple devices

#include <cassert>
#include <chrono>
#include <cstdio>
#include <cuda.h>
#include <iostream>
#include <thread>
#include <vector>

namespace ndv {

static inline void checkDrvError(CUresult res, const char* tok, const char* file, unsigned line) {
	if(res != CUDA_SUCCESS) {
		const char* err_str = NULL;
		(void)cuGetErrorString(res, &err_str);
		std::cerr << file << ':' << line << ' ' << tok << " failed (" << (unsigned)res << "): " << err_str << std::endl;
		abort();
	}
}

#define CHECK_DRV(x) ndv::checkDrvError(x, #x, __FILE__, __LINE__);

template <int Dims>
class coordinate {
	static_assert(Dims >= 1 && Dims <= 3);

  public:
	coordinate() = default;

	template <int D = Dims, typename = std::enable_if_t<D == 1>>
	coordinate(size_t dim0) : m_values{dim0} {}

	template <int D = Dims, typename = std::enable_if_t<D == 2>>
	coordinate(size_t dim0, size_t dim1) : m_values{dim0, dim1} {}

	coordinate(size_t dim0, size_t dim1, size_t dim2) {
		m_values[0] = dim0;
		if(Dims >= 2) m_values[1] = dim1;
		if(Dims == 3) m_values[2] = dim2;
	}

	size_t& operator[](size_t idx) { return m_values[idx]; }
	const size_t& operator[](size_t idx) const { return m_values[idx]; }

	template <typename ArrayLike>
	operator ArrayLike() const {
		if constexpr(Dims == 1) {
			return {m_values[0]};
		} else if constexpr(Dims == 2) {
			return {m_values[0], m_values[1]};
		} else {
			return {m_values[0], m_values[1], m_values[2]};
		}
	}

	coordinate operator+(const coordinate& other) const {
		coordinate result;
		for(int d = 0; d < Dims; ++d) {
			result[d] = m_values[d] + other.m_values[d];
		}
		return result;
	}

	coordinate operator-(const coordinate& other) const {
		coordinate result;
		for(int d = 0; d < Dims; ++d) {
			result.m_values[d] = m_values[d] - other.m_values[d];
		}
		return result;
	}

	bool operator==(const coordinate& other) const {
		return all(other, [](const size_t a, const size_t b) { return a == b; });
	}

	bool operator!=(const coordinate& other) const { return !(*this == other); }

	bool operator<=(const coordinate& other) const {
		return all(other, [](const size_t a, const size_t b) { return a <= b; });
	}

	bool operator<(const coordinate& other) const {
		return all(other, [](const size_t a, const size_t b) { return a < b; });
	}

  private:
	// TODO: Use int64_t?
	size_t m_values[Dims] = {};

	template <typename Predicate>
	bool all(const coordinate& other, Predicate&& p) const {
		for(int d = 0; d < Dims; ++d) {
			if(!p(m_values[d], other.m_values[d])) return false;
		}
		return true;
	}
};

template <int Dims>
class point : public coordinate<Dims> {
  public:
	using coordinate<Dims>::coordinate;
};

template <int Dims>
class extent : public coordinate<Dims> {
  public:
	using coordinate<Dims>::coordinate;
	size_t size() const {
		size_t s = 1;
		for(int d = 0; d < Dims; ++d) {
			s *= (*this)[d];
		}
		return s;
	}
};

template <int Dims>
class box {
  public:
	box(point<Dims> min, point<Dims> max) : m_min(std::move(min)), m_max(std::move(max)) { assert(m_min < m_max); }
	extent<Dims> get_extent() const { return m_max - m_min; }
	size_t size() const { return get_extent().size(); }
	bool contains(const point<Dims>& pt) const {
		for(int d = 0; d < Dims; ++d) {
			if(pt[d] < m_min[d] || pt[d] >= m_max[d]) { return false; }
		}
		return true;
	}

	const point<Dims>& min() const { return m_min; }
	point<Dims>& min() { return m_min; }
	const point<Dims>& max() const { return m_max; }
	point<Dims>& max() { return m_max; }

  private:
	point<Dims> m_min;
	point<Dims> m_max;
};

// TODO: Only pass stride instead of full range?
inline size_t get_linear_id(const extent<1>&, const point<1>& p) { return p[0]; }
inline size_t get_linear_id(const extent<2>& e, const point<2>& p) { return (p[0] * e[1]) + p[1]; }
inline size_t get_linear_id(const extent<3>& e, const point<3>& p) { return (p[0] * e[1] * e[2]) + (p[1] * e[2]) + p[2]; }

inline point<1> get_point_from_linear_id(const size_t linear_id, const extent<1>&) { return linear_id; }
inline point<2> get_point_from_linear_id(const size_t linear_id, const extent<2>& e) { return {linear_id / e[1], linear_id % e[1]}; }
inline point<3> get_point_from_linear_id(const size_t linear_id, const extent<3>& e) {
	return {linear_id / (e[1] * e[2]), (linear_id % (e[1] * e[2])) / e[2], (linear_id % (e[1] * e[2])) % e[2]};
}

template <typename T, int Dims>
class accessor {
  public:
	accessor(CUdeviceptr base_ptr, box<Dims> accessed_box, extent<Dims> buffer_extent)
	    : m_base_ptr(base_ptr), m_accessed_box(accessed_box), m_buffer_extent(buffer_extent) {}

	// T& operator[](const point& pt) { return *(reinterpret_cast<T*>(m_base_ptr) + get_linear_id(m_buffer_extent, pt)); }

	// For SYCL compatibility we need to be able to write to const accessors
	T& operator[](const point<Dims>& pt) const { return *(reinterpret_cast<T*>(m_base_ptr) + get_linear_id(m_buffer_extent, pt)); }

	const box<Dims>& get_box() const { return m_accessed_box; }

	const extent<Dims>& get_buffer_extent() const { return m_buffer_extent; }

	T* get_pointer() { return reinterpret_cast<T*>(m_base_ptr); }
	const T* get_pointer() const { return reinterpret_cast<T*>(m_base_ptr); }

  private:
	CUdeviceptr m_base_ptr;
	box<Dims> m_accessed_box;
	extent<Dims> m_buffer_extent;
};

struct activate_cuda_context {
	activate_cuda_context(CUcontext ctx) : m_ctx(ctx) { CHECK_DRV(cuCtxPushCurrent(ctx)); }

	~activate_cuda_context() {
		[[maybe_unused]] CUcontext ctx;
		CHECK_DRV(cuCtxPopCurrent(&ctx));
		assert(ctx == m_ctx);
	}

  private:
	[[maybe_unused]] CUcontext m_ctx;
};

inline std::vector<CUdevice> get_peer_devices(CUdevice main_device) {
	int num_devices;
	CHECK_DRV(cuDeviceGetCount(&num_devices));

	std::vector<CUdevice> peer_devices;
	peer_devices.push_back(main_device);
	for(int i = 0; i < num_devices; ++i) {
		CUdevice dev;
		CHECK_DRV(cuDeviceGet(&dev, i));
		if(dev == main_device) { continue; }

		int can_access = 0;
		CHECK_DRV(cuDeviceCanAccessPeer(&can_access, main_device, dev));
		if(!can_access) { continue; }

		int supports_vam = 0;
		CHECK_DRV(cuDeviceGetAttribute(&supports_vam, CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED, dev));
		if(supports_vam == 0) { continue; }

		peer_devices.push_back(dev);
	}

	return peer_devices;
}

// TODO: Also need to consider alignment requirements of T?
// => I think actual alignment needs to be lcm(cuda alignment, alignof(T))
template <typename T, int Dims>
class buffer {
  public:
	buffer(CUdevice dev, extent<Dims> ext) : m_device(dev), m_extent(std::move(ext)) {
		CHECK_DRV(cuInit(0));
		CHECK_DRV(cuDevicePrimaryCtxRetain(&m_context, m_device));
		activate_cuda_context act{m_context};

		{
			int supports_vam = 0;
			CHECK_DRV(cuDeviceGetAttribute(&supports_vam, CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED, m_device));
			if(!supports_vam) {
				fprintf(stderr, "Device %d doesn't support VIRTUAL ADDRESS MANAGEMENT.\n", m_device);
				std::terminate();
			}
		}

		m_alloc_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
		m_alloc_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
		m_alloc_prop.location.id = (int)m_device;
		// TODO: There is also CU_MEM_ALLOC_GRANULARITY_RECOMMENDED ("for best performance") - use that instead?
		CHECK_DRV(cuMemGetAllocationGranularity(&m_allocation_granularity, &m_alloc_prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));

		assert(m_allocation_granularity % alignof(T) == 0);
		assert(sizeof(T) <= m_allocation_granularity && "NYI"); // TODO: Need to deal with this case as well

		m_virtual_size = get_padded_size(ext.size() * sizeof(T));
		CHECK_DRV(cuMemAddressReserve(&m_base_ptr, m_virtual_size, 0, 0, 0));
	}

	buffer(const buffer&) = delete;
	buffer(buffer&&) = default;

	~buffer() {
		activate_cuda_context act{m_context};
		for(auto& a : m_allocations) {
			CHECK_DRV(cuMemUnmap(a.ptr, a.size));
			CHECK_DRV(cuMemRelease(a.handle));
		}
		CHECK_DRV(cuMemAddressFree(m_base_ptr, m_virtual_size));
	}

	const extent<Dims>& get_extent() const { return m_extent; }

	accessor<T, Dims> access(const box<Dims>& b) {
		activate_cuda_context act{m_context};

		// !!!!! TODO: Figure this out
		// => Note that for sizes larger than 1x granularity we may have to pad the address space reservation as well!
		const size_t allocation_size = m_allocation_granularity;

		assert(b.max() <= m_extent);

		// NOCOMMIT TODO: Revisit - do we actually end up using resulting pointers from all three branches?
		const auto advance = [this, &b](const CUdeviceptr ptr, const size_t add) {
			assert(ptr >= m_base_ptr);
			// Calculate positions within virtual space
			const point<Dims> current_pt = get_point_from_linear_id((ptr - m_base_ptr) / sizeof(T), m_extent);
			const point<Dims> add_pt = get_point_from_linear_id((ptr - m_base_ptr + add) / sizeof(T), m_extent);
			assert((get_linear_id(m_extent, add_pt) - get_linear_id(m_extent, current_pt)) * sizeof(T) == add);
			point<Dims> next_pt = add_pt;
			for(int d = Dims - 1; d > 0; --d) {
				if(next_pt[d] < b.min()[d]) {
					// We're before the accessed box in d, move on to first element
					next_pt[d] = b.min()[d];
					continue;
				}
				if(next_pt[d] >= b.max()[d]) {
					// We're past the accessed box in d, move on to next element (increase d-1)
					next_pt[d] = b.min()[d];
					next_pt[d - 1] += 1;
				}
			}

			if(next_pt != add_pt) {
				// We've advanced further than requested, need to re-align pointer
				return align_ptr(false, m_base_ptr + get_linear_id(m_extent, next_pt) * sizeof(T));
			}
			// We're somewhere inside the accessed box (in dim2/dim1, we may be way past in dim0), simply add
			return ptr + add;
		};

		std::vector<physical_allocation> new_allocations;
		new_allocations.reserve(m_allocations.size());

		// Walk through existing allocations and insert new ones wherever needed
		// TODO: This could probably be optimized in two ways:
		//  - Use a series of binary searches to figure out whether requested region is already covered
		//  - Only start copying into new array once we know that we'll actually need to allocate
		const CUdeviceptr start = align_ptr(false, m_base_ptr + get_linear_id(m_extent, b.min()) * sizeof(T));
		const CUdeviceptr end = align_ptr(true, m_base_ptr + (get_linear_id(m_extent, b.max() - point<Dims>{1, 1, 1}) + 1) * sizeof(T)); // 1 element past max
		size_t i = 0;
		CUdeviceptr current = start;
		while(current < end) {
			auto next = current;
			for(; i < m_allocations.size() && m_allocations[i].ptr <= current; ++i) {
				next = advance(m_allocations[i].ptr, m_allocations[i].size);
				new_allocations.emplace_back(std::move(m_allocations[i]));
			}
			current = std::max(current, next);
			if(current >= end) { break; }
			if(i >= m_allocations.size() || m_allocations[i].ptr > current) {
				// We are missing some allocations, create them
				new_allocations.emplace_back(allocate(allocation_size, current));
				current = advance(current, allocation_size);
			}
		}

		// Move remainder of allocations
		if(i < m_allocations.size()) {
			new_allocations.insert(new_allocations.end(), std::make_move_iterator(m_allocations.begin() + i), std::make_move_iterator(m_allocations.end()));
		}

#if !defined(NDEBUG)
		// Sanity check that we didn't mess up the order
		for(size_t j = 1; j < new_allocations.size(); ++j) {
			assert(new_allocations[j].ptr > new_allocations[j - 1].ptr);
		}
#endif

		m_allocations = std::move(new_allocations);

		return accessor<T, Dims>{m_base_ptr, b, m_extent};
	}

	// TODO: Optimize for contiguous copies
	// TODO: Make non-blocking
	void copy_from(const buffer<T, Dims>& src, const box<Dims>& src_box, const box<Dims>& dst_box) {
		activate_cuda_context act{m_context}; // Only needed for synchronize

		assert(src_box.get_extent() == dst_box.get_extent());
		access(dst_box); // Allocate memory

		if(Dims == 1) {
			CHECK_DRV(cuMemcpyPeer(m_base_ptr + get_linear_id(m_extent, dst_box.min()) * sizeof(T), m_context,
			    src.m_base_ptr + get_linear_id(src.m_extent, src_box.min()) * sizeof(T), src.m_context, dst_box.size() * sizeof(T)));
			CHECK_DRV(cuCtxSynchronize());
			return;
		}

		// There is no cuMemcpy2DPeer, so we use cuMemcpy3DPeer for both 2D and 3D copies.
		const auto fastest = [](const auto& c) { return c[Dims == 3 ? 2 : 1]; };
		const auto middle = [](const auto& c) { return c[Dims == 3 ? 1 : 0]; };

		CUDA_MEMCPY3D_PEER params = {};
		params.Depth = Dims == 3 ? src_box.get_extent()[0] : 1;
		params.Height = middle(src_box.get_extent());
		params.WidthInBytes = fastest(src_box.get_extent()) * sizeof(T);

		params.dstContext = m_context;
		params.dstDevice = m_base_ptr;
		params.dstHeight = middle(m_extent);
		params.dstMemoryType = CU_MEMORYTYPE_DEVICE;
		params.dstPitch = fastest(m_extent) * sizeof(T);
		params.dstXInBytes = fastest(dst_box.min()) * sizeof(T);
		params.dstY = middle(dst_box.min());
		params.dstZ = Dims == 3 ? dst_box.min()[0] : 0;

		params.srcContext = src.m_context;
		params.srcDevice = src.m_base_ptr;
		params.srcHeight = middle(src.m_extent);
		params.srcMemoryType = CU_MEMORYTYPE_DEVICE;
		params.srcPitch = fastest(src.m_extent) * sizeof(T);
		params.srcXInBytes = fastest(src_box.min()) * sizeof(T);
		params.srcY = middle(src_box.min());
		params.srcZ = Dims == 3 ? src_box.min()[0] : 0;

		CHECK_DRV(cuMemcpy3DPeer(&params));
		CHECK_DRV(cuCtxSynchronize());
	}

	void copy_from(const T* src_ptr, const extent<Dims>& src_ext, const box<Dims>& src_box, const box<Dims>& dst_box) {
		activate_cuda_context act{m_context};

		assert(src_box.get_extent() <= src_ext);
		assert(src_box.get_extent() == dst_box.get_extent());
		access(dst_box); // Allocate memory

		if(Dims == 1) {
			CHECK_DRV(cuMemcpyHtoD(m_base_ptr + get_linear_id(m_extent, dst_box.min()) * sizeof(T),
			    reinterpret_cast<const unsigned char*>(src_ptr) + get_linear_id(src_ext, src_box.min()) * sizeof(T), dst_box.size() * sizeof(T)));
			CHECK_DRV(cuCtxSynchronize());
			return;
		}

		if(Dims == 2) {
			CUDA_MEMCPY2D params = {};
			params.Height = src_box.get_extent()[0];
			params.WidthInBytes = src_box.get_extent()[1] * sizeof(T);

			params.dstDevice = m_base_ptr;
			params.dstMemoryType = CU_MEMORYTYPE_DEVICE;
			params.dstPitch = m_extent[1] * sizeof(T);
			params.dstXInBytes = dst_box.min()[1] * sizeof(T);
			params.dstY = dst_box.min()[0];

			params.srcHost = src_ptr;
			params.srcMemoryType = CU_MEMORYTYPE_HOST;
			params.srcPitch = src_ext[1] * sizeof(T);
			params.srcXInBytes = src_box.min()[1] * sizeof(T);
			params.srcY = src_box.min()[0];

			CHECK_DRV(cuMemcpy2D(&params));
			CHECK_DRV(cuCtxSynchronize());
			return;
		}

		CUDA_MEMCPY3D params = {};
		params.Depth = src_box.get_extent()[0];
		params.Height = src_box.get_extent()[1];
		params.WidthInBytes = src_box.get_extent()[2] * sizeof(T);

		params.dstDevice = m_base_ptr;
		params.dstHeight = m_extent[1];
		params.dstMemoryType = CU_MEMORYTYPE_DEVICE;
		params.dstPitch = m_extent[2] * sizeof(T);
		params.dstXInBytes = dst_box.min()[2] * sizeof(T);
		params.dstY = dst_box.min()[1];
		params.dstZ = dst_box.min()[0];

		params.srcHost = src_ptr;
		params.srcHeight = src_ext[1];
		params.srcMemoryType = CU_MEMORYTYPE_HOST;
		params.srcPitch = src_ext[2] * sizeof(T);
		params.srcXInBytes = src_box.min()[2] * sizeof(T);
		params.srcY = src_box.min()[1];
		params.srcZ = src_box.min()[0];

		CHECK_DRV(cuMemcpy3D(&params));
		CHECK_DRV(cuCtxSynchronize());
	}

	// TODO: Can we keep this DRYer with copy_from?
	void copy_to(T* dst_ptr, const extent<Dims>& dst_ext, const box<Dims>& src_box, const box<Dims>& dst_box) const {
		activate_cuda_context act{m_context};

		assert(dst_box.get_extent() <= dst_ext);
		assert(src_box.get_extent() == dst_box.get_extent());
		// TODO: Would be nice if we could assert that the src_box is allocated

		if(Dims == 1) {
			CHECK_DRV(cuMemcpyDtoH(reinterpret_cast<unsigned char*>(dst_ptr) + get_linear_id(dst_ext, dst_box.min()) * sizeof(T),
			    m_base_ptr + get_linear_id(m_extent, src_box.min()) * sizeof(T), dst_box.size() * sizeof(T)));
			CHECK_DRV(cuCtxSynchronize());
			return;
		}

		if(Dims == 2) {
			CUDA_MEMCPY2D params = {};
			params.Height = src_box.get_extent()[0];
			params.WidthInBytes = src_box.get_extent()[1] * sizeof(T);

			params.dstHost = dst_ptr;
			params.dstMemoryType = CU_MEMORYTYPE_HOST;
			params.dstPitch = dst_ext[1] * sizeof(T);
			params.dstXInBytes = dst_box.min()[1] * sizeof(T);
			params.dstY = dst_box.min()[0];

			params.srcDevice = m_base_ptr;
			params.srcMemoryType = CU_MEMORYTYPE_DEVICE;
			params.srcPitch = m_extent[1] * sizeof(T);
			params.srcXInBytes = src_box.min()[1] * sizeof(T);
			params.srcY = src_box.min()[0];

			CHECK_DRV(cuMemcpy2D(&params));
			CHECK_DRV(cuCtxSynchronize());
			return;
		}

		CUDA_MEMCPY3D params = {};
		params.Depth = src_box.get_extent()[0];
		params.Height = src_box.get_extent()[1];
		params.WidthInBytes = src_box.get_extent()[2] * sizeof(T);

		params.dstHost = dst_ptr;
		params.dstHeight = dst_ext[1];
		params.dstMemoryType = CU_MEMORYTYPE_HOST;
		params.dstPitch = dst_ext[2] * sizeof(T);
		params.dstXInBytes = dst_box.min()[2] * sizeof(T);
		params.dstY = dst_box.min()[1];
		params.dstZ = dst_box.min()[0];

		params.srcDevice = m_base_ptr;
		params.srcHeight = m_extent[1];
		params.srcMemoryType = CU_MEMORYTYPE_DEVICE;
		params.srcPitch = m_extent[2] * sizeof(T);
		params.srcXInBytes = src_box.min()[2] * sizeof(T);
		params.srcY = src_box.min()[1];
		params.srcZ = src_box.min()[0];

		CHECK_DRV(cuMemcpy3D(&params));
		CHECK_DRV(cuCtxSynchronize());
	}

	size_t get_allocated_size() const { return m_allocated_size; }

	size_t get_allocation_granularity() const { return m_allocation_granularity; }

	CUcontext get_ctx() const { return m_context; } // NOCOMMIT

  private:
	// NOTE: In the future it might become possible to map parts of a physical allocation (APIs exist but are NYI),
	//       so there may no longer be a 1:1 correspondence between handles and pointers/size.
	struct physical_allocation {
		// NOTE: We could also get this back with cuMemRetainAllocationHandle
		// TODO: If we do not plan on remapping these we can also release them right away (the memory will only be free'd once the address is unmapped).
		CUmemGenericAllocationHandle handle;
		size_t size; // TODO: Currently redundant because we use a fixed allocation size, but may change in the future
		CUdeviceptr ptr;
	};

	CUcontext m_context;
	CUdevice m_device;
	CUmemAllocationProp m_alloc_prop = {};
	size_t m_allocation_granularity; // page size, essentially
	extent<Dims> m_extent;
	size_t m_virtual_size;
	CUdeviceptr m_base_ptr;
	// List of (distjoint) physical allocations
	// Sorted by mapped pointers, ascending
	std::vector<physical_allocation> m_allocations;
	size_t m_allocated_size = 0;

	size_t get_padded_size(const size_t size) const { return ((size + m_allocation_granularity - 1) / m_allocation_granularity) * m_allocation_granularity; }

	// Aligns pointer to first aligned address before (up=false) or after (up=true) ptr (unless it is already aligned)
	CUdeviceptr align_ptr(bool up, const CUdeviceptr ptr) const {
		return ((ptr + (up ? (m_allocation_granularity - 1) : 0)) / m_allocation_granularity) * m_allocation_granularity;
	}

	physical_allocation allocate(const size_t size, const CUdeviceptr ptr) {
		activate_cuda_context act{m_context};

		const auto padded_size = get_padded_size(size);
		CUmemGenericAllocationHandle alloc_handle;
		CHECK_DRV(cuMemCreate(&alloc_handle, padded_size, &m_alloc_prop, 0));
		// TODO: assert(is_aligned(ptr))

		// Map physical allocation into address range
		CHECK_DRV(cuMemMap(ptr, padded_size, 0, alloc_handle, 0));

		// Set memory protection flags
		// NOCOMMIT FIXME: Don't do this every time
		const auto peers = get_peer_devices(m_device); // This includes m_device
		std::vector<CUmemAccessDesc> access_descs(peers.size());
		for(size_t i = 0; i < peers.size(); ++i) {
			access_descs[i].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
			access_descs[i].location.id = peers[i];
			access_descs[i].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
		}

		CHECK_DRV(cuMemSetAccess(ptr, padded_size, access_descs.data(), access_descs.size()));

#if !defined(NDEBUG)
		// Initialize memory with known pattern for debugging
		cuMemsetD8(ptr, 0b101010, padded_size);
#endif

		// {
		// 	const point pt = {(ptr - m_base_ptr) / (m_extent[1] * sizeof(T)), ((ptr - m_base_ptr) / sizeof(T)) % m_extent[1]};
		// 	printf("Allocating %zu bytes at %zu,%zu (%p)\n", padded_size, pt[0], pt[1], (void*)ptr);
		// }

		m_allocated_size += padded_size;

		return physical_allocation({alloc_handle, padded_size, ptr});
	}
};

} // namespace ndv
