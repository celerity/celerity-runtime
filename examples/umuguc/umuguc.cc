#include <numeric>
#include <span>
#include <vector>

#include <celerity.h>
#include <fmt/ranges.h>
#include <nccl.h>
#include <thrust/async/scan.h>
#include <thrust/device_ptr.h>

// TODO: A proper task geometry API should also offer splitting utilities
#include "split.h"
#include "tilebuffer_utils.h"

#include "eigen_decomposition.hpp"

using clk = std::chrono::steady_clock;

#define USE_NCCL 1

#define ENABLE_SANITY_CHECKS 0

/**
 * How do we get the multi-pass tile buffer into Celerity?
 *
 * Notes:
 * - We need several buffers: Actual storage, point counts per slot, cumulative counts (= read/write offsets), ...
 * - These all need to be communicated (in part) between ranks. Reductions need to be performed across multiple buffers as well.
 * - We need a way of creating "images" of the tile buffer that have the same shape, but store different data (e.g. shape factors)
 * - We may want to store [cumulative] counts using a sparse data structure, potentially making transferring those more involved as well
 *
 * Open questions:
 * - How do we do compression on this data structure?
 *
 * THE PLAN - After discussion with Peter on 2024/08/29:
 * - Each rank counts the number of points per tile into a *local buffer* that is large enough to cover the
 *   subrange provided by the user (obtained from stripes).
 * - Inside the tile data structure, we compute all ranks that have overlapping writes with our written subrange
 * - We then perform p2p communication with those ranks to exchange our counts
 * - The lowest rank (or maybe all, using replicated writes) writes the sum of the per-rank counts into the *global buffer*
 *   - Update: No need to write to global buffer, each rank computes the full sum
 * - Each rank obtains the full global buffer and locally computes the prefix sum (cumulative counts)
 *   - We think doing this distributed would be a huge mess (particularly for 2D/3D buffers)
 *
 * - Then the actual sorting of points into tiles is done: Each rank gives Celerity a list of subranges that will be written by that rank
 * - For reading data from a neighborhood, we provide a list of full ranges in the global buffer
 * - This means we need the global offsets both on the host (for range mappers) and on the device (for binary search)
 *   - The detailed offsets we only need on the host
 *
 * - The only changes we need for the rough prototype are:
 *    - Support for region range mappers
 *    - Support for "SPMD range mappers"; each rank only says what it writes (+ what the global write is)
 *
 * MULTI-GPU:
 * - We need to keep track of multiple counts / offsets; one for each GPU (or chunk, if we want to support oversubscription)
 * - This makes the whole "SPMD range mappers" more ugly, because now we need to be able to specify written ranges on both a node and per-chunk level
 *   => Maybe we turn it around: Initially the data structure is asked to produce a split (if possible) for N chunks (including GPUs);
 *      this then returns a list of launches + all accessed buffers / ranges required to do so. Celerity then afterwards doesn't need to
 *      call any range mappers (EXCEPT: when doing interop with classic buffers - what does that look like?)
 *
 *
 * Notes on API design:
 * - Just do a dedicated API to create a point-tile buffer. E.g. auto tb = celerity::datatypes::tile_buffer<2>::create([=]() { ... });
 *
 *   It might hurt composability *somewhat* (you can't just use it with any normal kernel), but how realistic is that use-case anyway..?
 *   In particular because what else should the kernel do other than creating the data structure? It will be executed twice, so any expensive un- or
 *   semi-related computation shouldn't be done there anyway.
 */


/**
 * FEBRUARY 2025 REVISION
 *
 * - We cannot compute the full dense offset/count grid on each node
 * - We cannot compute the offset/count grid on the host
 *
 * - [x] Input is split along duplicated bounding boxes, each node knows its total bounding box
 * - [x] For simplicity: All devices allocate the full count buffer for the local bounding box
 *    [ ] => We EXTEND the bounding box along the 1-dimension to make the prefix sum calculation easier later on
 * - [x] Each device computes its count per tile
 *
 * - [x] Compute the sum across all local devices (using NCCL)
 *   - These counts are written into the SLICE for the local node in the 3D counts buffer (using interop task we can directly write to that slice using NCCL?)
 *
 * - [x] All nodes exchange their bounding boxes (using MPI, but we should have a utility for that)
 *   - [ ] Actually we could do two phases, a broad phase based on bounding boxes, and a narrow phase based on actual regions
 * - [x] Each node computes the set of nodes that overlap with its bounding box (or regions)
 * - [x] Each node launches a task with multiple chunks (all on device 0?) that computes the sum of the per-node counts
 *   - [x] Use atomics to avoid conflicts between chunks
 * 	   => Each node now has the global counts for its bounding box
 *   - [ ] If a node has the exact same overlapping box with two separate nodes (extremely unlikely), we have a problem (chunks must be unique for now) => error
 * - [x] Each node computes the prefix sum for its bounding box
 *	   => Parts of the bounding box that overlap between nodes will be computed multiple times (from a global perspective)
 * - [x] In a new 1D buffer, the node with the lowest rank that owns a row of the bounding box writes the COUNT per row
 *     => This can be computed as the difference between the current row and the previous
 * - [x] Each node reads from the 1D buffer up until the starting coordinate of its bounding box and computes the sum
 * - [x] Each node adds the previously computed sum to its local prefix sum
 *	 => At this point, each node now has the full prefix sum for its bounding box
 *   => NOTE: We may have to write this (again using lowest rank order) to a global buffer for determining shape factor kernel chunks later on
 *            BIG OOF: How would another node know that a particular node needs this data though?! We can't just read the full thing, that is the whole point
 *                     (or, at least half the point, we don't want to compute and store the whole thing)
 *            => I guess we have to start by reading the whole thing on all nodes and then figure something out later. Can't solve everything at once...
 *
 * - [x] Repeat previous step for computing sums in overlapping bounding boxes, but now do it only for nodes with a lower rank
 * - [x] Finally, on each device compute the sum of the rank offset and the count of all lower devices
 *
 * - THEN WE ARE READY TO WRITE POINTS. JEEZ.
 *
 * =======================
 *
 * Q: Can we do a "light" variant first, where we do the partial bounding boxes, but compute everything within host tasks / in main thread?
 *
 * TODO: SMALL THINGS LEFT TO IMPLEMENT IN CELERITY
 * - [ ] Exact allocation setting (also needed for matmul)
 * - [ ] Local indexing setting (requires exact allocation; needed for per-rank and per-device slices of 3D buffers)
 * - [ ] Initialize allocation setting (to avoid having to manually initialize per-rank and per-device slices)
 * - [ ] Data requirements (expert mapper) that receives 1D/2D boxes/regions would spare us a lot of casts
 *
 *
 * OTHER TODOS:
 * - [ ] Come up with a more consistent naming scheme for the expert mapper input vector, the expert mapper and the accessor. It's a mess currently.A
 *       => I'm leaning towards getting rid of the read_ and write_ prefixes
 *
 * - [ ] Bring back tile fill stats (min/max/avg)
 *
 * OPEN ISSUES:
 *
 * The biggest issue this current implementation has is that it is not very memory efficient.
 *  - This is because we have a lot of per-device and per-node buffers, with the latter also sometimes being duplicated across all devices.
 *  - Maybe some of these could be reused, since they typically have the same allocation shape
 *  - Also look into freeing buffers that are no longer needed
 *
 * The other big problem is that besides shape factor calculation, we spend most of the time calculating the written ranges for each device on the host.
 *  - Since this is inherently a sequential operation, I don't think we can move it to the device
 *  - A somewhat unfortunate consequence of this is that the application tends to perform better in "legacy mode", i.e., one rank per device, because
 *    in that case we divide the grid into smaller pieces for each rank.
 *  	- This means that ideally we'd have multiple "local grids" per rank, one for each device. But that makes everything A LOT more complicated.
 */

#define NCCL_CHECK(cmd)                                                                                                                                        \
	do {                                                                                                                                                       \
		ncclResult_t res = cmd;                                                                                                                                \
		if(res != ncclSuccess) {                                                                                                                               \
			printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(res));                                                            \
			exit(EXIT_FAILURE);                                                                                                                                \
		}                                                                                                                                                      \
	} while(0)

struct umuguc_point3d {
	double x, y, z;

	bool operator==(const umuguc_point3d& other) const { return x == other.x && y == other.y && z == other.z; }
	bool operator!=(const umuguc_point3d& other) const { return !(*this == other); }
};

void consume_args(const int i, const int n, int* argc, char** argv[]) {
	assert(i + n <= *argc);
	for(int j = i; j < *argc - n; ++j) {
		(*argv)[j] = (*argv)[j + n];
	}
	*argc -= n;
}

bool get_flag(const std::string_view flag, int* argc, char** argv[]) {
	for(int i = 0; i < *argc; ++i) {
		if((*argv)[i] == flag) {
			consume_args(i, 1, argc, argv);
			return true;
		}
	}
	return false;
}

template <typename T>
std::optional<T> get_arg(const std::string_view flag, int* argc, char** argv[]) {
	for(int i = 0; i < *argc; ++i) {
		if((*argv)[i] == flag) {
			if(i + 1 < *argc) {
				T result;
				std::istringstream((*argv)[i + 1]) >> result;
				consume_args(i, 2, argc, argv);
				return result;
			}
			throw std::runtime_error(fmt::format("Missing value for argument {}", flag));
		}
	}
	return std::nullopt;
}

int main(int argc, char* argv[]) {
	const auto usage = [&]() {
		fmt::print(stderr, "Usage: {} <input_file> [--write-output] [--duplicate-input <factor>] [--swap-xy] [--precise-timings]\n", argv[0]);
		exit(EXIT_FAILURE);
	};
	if(argc < 2) usage();
	const auto filename = argv[1];
	consume_args(1, 1, &argc, &argv);
	const bool write_output = get_flag("--write-output", &argc, &argv);
	const double tile_size = get_arg<double>("--tile-size", &argc, &argv).value_or(1.0);
	// TODO: Consider adding an offset parameter that controls by how many bounding boxes duplicated inputs are offset (default 1)
	//       This way we could test different levels of overlap between nodes
	const size_t duplicate_input = get_arg<size_t>("--duplicate-input", &argc, &argv).value_or(1);
	// We need inputs to be "long" in the 0th dimension for our prefix sum implementation
	// to work (somewhat) efficiently; we may have to rotate some files.
	const bool swap_xy = get_flag("--swap-xy", &argc, &argv);
	const bool precise_timings = get_flag("--precise-timings", &argc, &argv);
	if(argc != 1) usage();

	std::ifstream input(filename, std::ios::binary);
	if(!input) {
		fmt::print(stderr, "Could not open file: {}\n", filename);
		return 1;
	}
	input.seekg(0, std::ios::end);
	const size_t file_size = input.tellg();
	input.seekg(0, std::ios::beg);

	if(file_size % sizeof(umuguc_point3d) != 0) {
		fmt::print(stderr, "File size is not a multiple of point3d size\n");
		return 1;
	}

	// TODO: We currently use uint32_t for number of entries, which limits us to 4B points
	std::vector<umuguc_point3d> points(file_size / sizeof(umuguc_point3d));
	input.read(reinterpret_cast<char*>(points.data()), file_size);

	if(points.size() * duplicate_input > std::numeric_limits<uint32_t>::max()) {
		fmt::print(stderr, "Too many points for 32 bit counting. Time to upgrade!: {}\n", points.size());
		return 1;
	}

// FIXME: We need the initialize-on-allocation feature
#ifndef NDEBUG
	CELERITY_CRITICAL("NOT ALL BUFFERS ARE BEING INITIALIZED TO ZERO - DEBUG MEMORY PATTERN WILL CAUSE PROBLEMS");
#endif

	celerity::queue queue;
	auto& rt = celerity::detail::runtime::get_instance();
	const auto rank = rt.NOCOMMIT_get_local_nid();
	const auto num_ranks = rt.NOCOMMIT_get_num_nodes();
	// NOCOMMIT: We assume a uniform number of devices per node here - throw if not uniform
	const size_t num_devices = rt.NOCOMMIT_get_num_local_devices();

	if(rank == 0) {
		// TODO: How do we want to handle warmup for this benchmark? Each phase needs new allocations, but we cannot anticipate them in IDAG.
		fmt::print("NO WARMUP - TBD\n");
		fmt::print("Using tile size {:.1f}\n", tile_size);
	}

	// TODO: We should get this from metadata
	umuguc_point3d input_min = {std::numeric_limits<double>::max(), std::numeric_limits<double>::max(), std::numeric_limits<double>::max()};
	umuguc_point3d input_max = {std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest()};
	for(auto& p : points) {
		if(swap_xy) { std::swap(p.x, p.y); }

		input_min.x = std::min(input_min.x, p.x);
		input_min.y = std::min(input_min.y, p.y);
		input_min.z = std::min(input_min.z, p.z);
		input_max.x = std::max(input_max.x, p.x);
		input_max.y = std::max(input_max.y, p.y);
		input_max.z = std::max(input_max.z, p.z);
	}
	if(rank == 0) {
		fmt::print("Read {} points ({:.1f} GiB)\n", points.size(), points.size() * sizeof(umuguc_point3d) / 1024.0 / 1024.0 / 1024.0);
		fmt::print("Min: ({:.1f}, {:.1f}, {:.1f}), Max: ({:.1f}, {:.1f}, {:.1f}). Extent: ({:.1f}, {:.1f}, {:.1f})\n", input_min.x, input_min.y, input_min.z,
		    input_max.x, input_max.y, input_max.z, input_max.x - input_min.x, input_max.y - input_min.y, input_max.z - input_min.z);
	}

	const double original_extent_y = input_max.y - input_min.y;

	// If we are duplicating the input, we want to ensure that we are splitting it in such a way that we get a realistic level of overlap between GPUs / nodes.
	// Simply duplicating the input and offsetting it does not achieve this. Instead we want the split to be halfway through the original data set.
	if(duplicate_input > 1) {
		const size_t original_size = points.size();
		// Insert a copy and move it down by one bounding box
		points.insert(points.end(), points.begin(), points.end());
		for(size_t i = original_size; i < points.size(); ++i) {
			points[i].y += input_max.y - input_min.y;
		}
		// Now delete the first half of the original data set, and second half of copy
		points.erase(points.begin(), points.begin() + original_size / 2);
		points.erase(points.begin() + original_size, points.end());
		// Now recompute bounding box
		input_min.y = std::numeric_limits<double>::max();
		input_max.y = std::numeric_limits<double>::lowest();
		for(auto& p : points) {
			input_min.y = std::min(input_min.y, p.y);
			input_max.y = std::max(input_max.y, p.y);
		}
		if(rank == 0) {
			fmt::print(
			    "Min: ({:.1f}, {:.1f}, {:.1f}), Max: ({:.1f}, {:.1f}, {:.1f}). Extent: ({:.1f}, {:.1f}, {:.1f}) <-- Bounding box of duplication pattern\n",
			    input_min.x, input_min.y, input_min.z, input_max.x, input_max.y, input_max.z, input_max.x - input_min.x, input_max.y - input_min.y,
			    input_max.z - input_min.z);
		}
	}

	const umuguc_point3d global_min = input_min;
	const umuguc_point3d global_max = {input_max.x, input_max.y + original_extent_y * (duplicate_input - 1), input_max.z};
	const umuguc_point3d global_extent = {global_max.x - global_min.x, global_max.y - global_min.y, global_max.z - global_min.z};
	const celerity::range<2> global_grid_size = {
	    static_cast<uint32_t>(std::round(global_extent.y / tile_size)), static_cast<uint32_t>(std::round(global_extent.x / tile_size))};
	if(global_grid_size.size() == 0) {
		CELERITY_CRITICAL("Global grid size is {}x{} - try smaller tile size", global_grid_size[1], global_grid_size[0]);
		exit(1);
	}
	if(rank == 0) {
		// TODO: Is the XY swap really worth the confusion?
		fmt::print("Global grid size: {}x{}, effective tile size: {:.1f}x{:.1f}\n", global_grid_size[1], global_grid_size[0],
		    (global_max.x - global_min.x) / global_grid_size[1], (global_max.y - global_min.y) / global_grid_size[0]);
	}

	using celerity::subrange;
	using celerity::detail::box;
	using celerity::detail::region;
	using celerity::detail::subrange_cast;

	const size_t global_num_points = points.size() * duplicate_input;
	celerity::buffer<umuguc_point3d, 1> points_input(global_num_points);
	celerity::debug::set_buffer_name(points_input, "points input");

#if USE_NCCL
	std::vector<ncclComm_t> nccl_comms(num_devices);
	std::vector<int> device_ids;
	for(auto& dev : rt.NOCOMMIT_get_sycl_devices()) {
		device_ids.push_back(sycl::get_native<sycl::backend::cuda>(dev));
	}
	// CELERITY_INFO("Creating NCCL comms for devices {}", fmt::join(device_ids, ", "));
	NCCL_CHECK(ncclCommInitAll(nccl_comms.data(), num_devices, device_ids.data()));
#endif

	if(duplicate_input % num_ranks != 0) {
		// This is because we don't want to have to compute an exact bounding box for each rank for arbitrary splits.
		throw std::runtime_error(fmt::format("Input multiplier ({}) must be divisible by number of ranks ({})", duplicate_input, num_ranks));
	}

	auto print_delta_time = [&, previous = clk::now()](std::string_view description, std::optional<clk::time_point> against = std::nullopt) mutable {
		queue.wait(celerity::experimental::barrier);
		const auto now = clk::now();
		auto dt = now - previous;
		if(against.has_value()) { dt = now - against.value(); }
		if(rank == 0) { fmt::print("{}: {} ms\n", description, std::chrono::duration_cast<std::chrono::milliseconds>(dt).count()); }
		previous = now;
		return now;
	};

	umuguc_point3d local_min = input_min;
	umuguc_point3d local_max = input_max;
	const auto duplicates_per_rank = duplicate_input / num_ranks;
	local_min.y = input_min.y + original_extent_y * rank * duplicates_per_rank;
	local_max.y = input_max.y + original_extent_y * (rank * duplicates_per_rank + duplicates_per_rank - 1);

	// Write input to buffer, duplicate
	// NOTE: If duplicate_input is 1, this is basically host-initializing the buffer with extra steps
	{
		celerity::buffer<umuguc_point3d, 1> original_input(points.data(), points.size());

		queue.submit([&](celerity::handler& cgh) {
			celerity::accessor read_points(original_input, cgh, celerity::access::all{}, celerity::read_only);
			celerity::accessor write_duplicated_points(points_input, cgh, celerity::access::one_to_one{}, celerity::write_only, celerity::no_init);

			const size_t original_size = points.size();
			cgh.parallel_for(global_num_points, [=](celerity::id<1> id) {
				const size_t offset = id[0] / original_size;
				auto pt = read_points[id[0] % original_size];
				// The bounding box of the "duplication pattern" may be different from the original input,
				// because we split based on point count, not at the geometric center.
				// Since the pattern is created by offsetting the input by a full bounding box,
				// we have to use the original extent here for them to seamlessly fit together again.
				pt.y += original_extent_y * offset;
				write_duplicated_points[id] = pt;
			});
		});
	}

	print_delta_time("Writing [and duplicating] input to buffer");

	const umuguc_point3d local_extent = {local_max.x - local_min.x, local_max.y - local_min.y, local_max.z - local_min.z};
	// For some reason (I don't have time to investigate right now), sizing the local grid exactly to the local extent can sometimes cause
	// out-of-bounds accesses in the counting kernel. It may just be unfortunate rounding. To remedy this, we simply extend the local grid
	// size along the Y-axis in both directions by one tile. This is not a problem in practice because we only use the local grid to declare
	// data requirements, i.e., we at most incur a very small allocation and transfer overhead.
	const auto [local_grid_size, local_grid_offset] = ([&] {
		celerity::range<2> size = {
		    static_cast<uint32_t>(std::round(local_extent.y / tile_size)), static_cast<uint32_t>(std::round(local_extent.x / tile_size))};
		celerity::id<2> offset = {
		    static_cast<uint32_t>((local_min.y - global_min.y) / tile_size), static_cast<uint32_t>((local_min.x - global_min.x) / tile_size)};
		if(offset[0] + size[0] < global_grid_size[0]) size[0]++;
		if(offset[0] > 0) {
			offset[0]--;
			size[0]++;
		}
		return std::pair{size, offset};
	})(); // IIFE
	if(local_grid_size.size() == 0) {
		CELERITY_CRITICAL("Local grid size is {}x{} - try smaller tile size", local_grid_size[1], local_grid_size[0]);
		exit(1);
	}
	if(local_grid_size[1] != global_grid_size[1]) {
		// It should be possible to simply extend the grid along the X-axis, but I haven't tested this case.
		throw std::runtime_error("Local grid size does not match global grid size along X-axis.");
	}
	fmt::print("Rank {} local grid size: {}x{}, offset: {},{}, effective tile size: {:.1f}x{:.1f}\n", rank, local_grid_size[1], local_grid_size[0],
	    local_grid_offset[1], local_grid_offset[0], local_extent.x / local_grid_size[1], local_extent.y / local_grid_size[0]);

	// Sanity check: Does actual bounding box match predicted bounding box?
#if ENABLE_SANITY_CHECKS
	queue.submit([&](celerity::handler& cgh) {
		celerity::accessor read_points(points_input, cgh, celerity::access::one_to_one{}, celerity::read_only_host_task);
		cgh.host_task(points_input.get_range(), [=](celerity::partition<1> part) {
			// if(part.get_subrange().offset[0] % points.size() != 0) throw std::runtime_error("Partition offset is not a multiple of input size?");
			umuguc_point3d local_min = {std::numeric_limits<double>::max(), std::numeric_limits<double>::max(), std::numeric_limits<double>::max()};
			umuguc_point3d local_max = {std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest()};
			celerity::experimental::for_each_item(part, [&](celerity::item<1> itm) {
				auto pt = read_points[itm];
				local_min.x = std::min(local_min.x, pt.x);
				local_min.y = std::min(local_min.y, pt.y);
				local_min.z = std::min(local_min.z, pt.z);
				local_max.x = std::max(local_max.x, pt.x);
				local_max.y = std::max(local_max.y, pt.y);
				local_max.z = std::max(local_max.z, pt.z);
			});
			fmt::print("Local bounding box: Min: ({:.1f}, {:.1f}, {:.1f}), Max: ({:.1f}, {:.1f}, {:.1f}). Extent: ({:.1f}, {:.1f}, {:.1f})\n", local_min.x,
			    local_min.y, local_min.z, local_max.x, local_max.y, local_max.z, local_max.x - local_min.x, local_max.y - local_min.y,
			    local_max.z - local_min.z);

			fmt::print("predicted min y: {:.1f}, max y: {:.1f}\n", local_min.y, local_max.y);
			fmt::print("actual    min y: {:.1f}, max y: {:.1f}\n", local_min.y, local_max.y);
			fmt::print("min diff: {:.1f}, max diff: {:.1f}\n", local_min.y - local_min.y, local_max.y - local_max.y);

			if(std::abs(local_min.y - local_min.y) > 1e-6 || std::abs(local_max.y - local_max.y) > 1e-6) {
				throw std::runtime_error("Predicted bounding box does not match actual bounding box");
			}
		});
	});
#endif

	// TODO API: No need to split remote chunks into per-device chunks
	// TODO: Rename to something more descriptive
	auto chunks =
	    celerity::detail::split_1d(box_cast<3>(celerity::detail::box<1>{0, points_input.get_range()}), celerity::detail::ones, num_ranks * num_devices);
	celerity::custom_task_geometry write_tiles_geometry;
	write_tiles_geometry.global_size = range_cast<3>(points_input.get_range());
	for(size_t i = 0; i < chunks.size(); ++i) {
		write_tiles_geometry.assigned_chunks.push_back({chunks[i].get_subrange(), (i / num_devices), (i % num_devices)});
	}

	celerity::buffer<uint32_t, 3> per_device_tile_point_counts({num_ranks * num_devices, global_grid_size[0], global_grid_size[1]});
	celerity::buffer<uint32_t, 3> per_rank_tile_point_counts({num_ranks, global_grid_size[0], global_grid_size[1]});
	// Each rank constructs its own copy of the global counts within its local bounding box
	// This means incorporating counts from all other ranks that have an overlapping bounding box
	celerity::buffer<uint32_t, 3> per_rank_global_counts({num_ranks, global_grid_size[0], global_grid_size[1]});
	// Counts of all ranks that are lower than the given rank (i.e, for rank 0 it will be empty), within the local bounding box
	// Required for determining the actual write offset on this rank (lower ranks write first)
	celerity::buffer<uint32_t, 3> per_rank_lower_rank_counts({num_ranks, global_grid_size[0], global_grid_size[1]});
	// Global prefix sum (within the local bounding box)
	celerity::buffer<uint32_t, 3> per_rank_cumulative_counts({num_ranks, global_grid_size[0], global_grid_size[1]});
	// Sum of values in each row of the global bounding box. Required for computing the final global prefix sum on each rank.
	celerity::buffer<uint32_t, 1> global_row_sums({global_grid_size[0]});
	// The value that needs to be added to the local prefix sum to complete it. Computed as the sum across all rows above local bounding box.
	celerity::buffer<uint32_t, 1> per_rank_add_to_prefix_sum({num_ranks});
	// The offset each device starts writing at within each tile. Computed as the global prefix sum + the point counts of all lower ranks and devices.
	celerity::buffer<uint32_t, 3> per_device_write_offsets({num_ranks * num_devices, global_grid_size[0], global_grid_size[1]});
	// Same as per_rank_cumulative_counts, but in a single, global view. Required for exchanging counts for neighborhood access calculation.
	// TODO: Not the most elegant solution
	// => Turns out we don't need it for a 1-neighborhood (at least with all the current data sets we have)
	// BUT: We do need it for writing output at the end
	celerity::buffer<uint32_t, 2> global_cumulative_counts({global_grid_size[0], global_grid_size[1]});

	celerity::buffer<umuguc_point3d, 1> tiles_storage(global_num_points);
	celerity::buffer<sycl::double3, 1> shape_factors(global_num_points);

	celerity::debug::set_buffer_name(per_device_tile_point_counts, "per device tile point counts");
	celerity::debug::set_buffer_name(per_rank_tile_point_counts, "per rank tile point counts");
	celerity::debug::set_buffer_name(per_rank_global_counts, "per rank global counts");
	celerity::debug::set_buffer_name(per_rank_lower_rank_counts, "per rank lower rank counts");
	celerity::debug::set_buffer_name(per_rank_cumulative_counts, "per rank cumulative counts");
	celerity::debug::set_buffer_name(global_row_sums, "global row sums");
	celerity::debug::set_buffer_name(per_rank_add_to_prefix_sum, "per rank add to prefix sum");
	celerity::debug::set_buffer_name(per_device_write_offsets, "per device write offsets");
	// celerity::debug::set_buffer_name(global_cumulative_counts, "global cumulative counts");
	celerity::debug::set_buffer_name(tiles_storage, "tiles storage");
	celerity::debug::set_buffer_name(shape_factors, "shape factors");

	const auto rank_slice = [local_grid_offset, local_grid_size](const size_t rank) {
		return box<3>(subrange<3>{{rank, local_grid_offset[0], local_grid_offset[1]}, {1, local_grid_size[0], local_grid_size[1]}});
	};

	const auto device_slice = [local_grid_offset, local_grid_size, rank, num_devices](const size_t device) {
		return box<3>(subrange<3>{{rank * num_devices + device, local_grid_offset[0], local_grid_offset[1]}, {1, local_grid_size[0], local_grid_size[1]}});
	};

	queue.wait(celerity::experimental::barrier);
	const auto before_count_points = std::chrono::steady_clock::now();

	// TODO: If this becomes a pattern, build a utility around it
	std::vector<std::pair<box<3>, region<3>>> local_device_accesses;
	for(auto& chnk : write_tiles_geometry.assigned_chunks) {
		if(chnk.nid == rank) {
			local_device_accesses.push_back({chnk.box, device_slice(chnk.did.value())});
		} else {
			local_device_accesses.push_back({chnk.box, {}});
		}
	}
	// TODO: This is a *bit* flaky: We are saying that the full range is being written across all nodes. This is not true however, because we only
	//       write the local grid size on each node. This is not a problem in practice, but it does create a desync between what nodes think the
	//       state of the buffer is. In a hypothetical scenario where we access part of the buffer outside the local grid on a remote node
	//       after this, the remote node will think the data has changed elsewhere and will wait for it to be pushed. The "owning" node however
	//       knows that it didn't really write the range, and so will not generate a push (in fact it will believe someone else to be the
	//       owner). If we wanted to do this correctly, we'd have to pass the local grid size of each node.
	celerity::expert_mapper write_device_count_access{box<3>::full_range(per_device_tile_point_counts.get_range()), local_device_accesses};
	write_device_count_access.options.use_local_indexing = true;

	// TODO: Where is handler::fill... Although it's unclear how that would work in SPMD context (each rank wants to fill its own slice)
	// Easiest solution for now would probably be a "data access property" (like exact allocation, local indexing) that says what to initialize to
	// TODO: Same as below for resetting before writing points. Refactor out.
	queue.submit([&](celerity::handler& cgh) {
		celerity::custom_task_geometry<3> geo;
		std::vector<std::pair<box<3>, region<3>>> write_device_slice;
		for(size_t i = 0; i < num_devices; ++i) {
			const auto chnk_box = device_slice(i);
			geo.assigned_chunks.push_back({chnk_box, rank, i});
			write_device_slice.push_back({chnk_box, chnk_box});
		}
		celerity::accessor write_device_counts(
		    per_device_tile_point_counts, cgh, celerity::expert_mapper(write_device_slice), celerity::write_only, celerity::no_init);
		celerity::debug::set_task_name(cgh, "initialize per-device point to zero");
		cgh.assert_no_data_movement();
		cgh.parallel_for(geo, [=](celerity::id<3> id) { write_device_counts[id] = 0; });
	});

	queue.submit([&](celerity::handler& cgh) {
		celerity::accessor read_points(points_input, cgh, celerity::access::one_to_one{}, celerity::read_only);
		celerity::accessor write_counts(per_device_tile_point_counts, cgh, write_device_count_access, celerity::write_only, celerity::no_init);
		celerity::debug::set_task_name(cgh, "count points (global)");

		cgh.assert_no_data_movement(celerity::detail::data_movement_scope::inter_node);
		cgh.parallel_for(write_tiles_geometry, [=](celerity::id<1> id) {
			const auto& p = read_points[id];
			// grid_size - 1: We divide the domain such the last tile contains the maximum value
			const auto tile_x = static_cast<uint32_t>((p.x - global_min.x) / global_extent.x * (global_grid_size[1] - 1));
			const auto tile_y = static_cast<uint32_t>((p.y - global_min.y) / global_extent.y * (global_grid_size[0] - 1));
			auto device_slice = write_counts[0]; // 0 because we use local indexing
			const auto local_tile_x = tile_x - local_grid_offset[1];
			const auto local_tile_y = tile_y - local_grid_offset[0];
			sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device> ref{device_slice[local_tile_y][local_tile_x]};
			ref++;
		});
	});

	const auto before_aggregate_counts = print_delta_time("Counting points", before_count_points);

	// Sum up counts across all devices on this rank
	// This is spicy because for the first time we submit a geometry that is local to each node
	{
		celerity::custom_task_geometry sum_points_geo;
		sum_points_geo.global_size = range_cast<3>(celerity::range<1>{num_devices});
		std::vector<std::pair<box<3>, region<3>>> read_device_counts_accesses;
		std::vector<std::pair<box<3>, region<3>>> write_rank_counts_accesses;
		for(size_t i = 0; i < num_devices; ++i) {
			const auto chnk_box = box_cast<3>(box<1>(subrange<1>{i, 1}));
			sum_points_geo.assigned_chunks.push_back({chnk_box, rank, i});
			read_device_counts_accesses.push_back({chnk_box, device_slice(i)});
			if(i == 0) {
				write_rank_counts_accesses.push_back({chnk_box, rank_slice(rank)});
			} else {
				write_rank_counts_accesses.push_back({chnk_box, {}});
			}
		}

#if USE_NCCL
		queue.submit([&](celerity::handler& cgh) {
			celerity::expert_mapper read_device_counts{box<3>::full_range(per_device_tile_point_counts.get_range()), read_device_counts_accesses};
			read_device_counts.options.allocate_exactly = true;
			celerity::expert_mapper write_rank_counts{box<3>::full_range(per_rank_tile_point_counts.get_range()), write_rank_counts_accesses};
			write_rank_counts.options.allocate_exactly = true;
			celerity::accessor device_counts(per_device_tile_point_counts, cgh, read_device_counts, celerity::read_only);
			celerity::accessor rank_counts(per_rank_tile_point_counts, cgh, write_rank_counts, celerity::write_only, celerity::no_init);

			celerity::debug::set_task_name(cgh, "device count sum");
			cgh.interop_task(sum_points_geo, [=](celerity::interop_handle& ih, celerity::partition<1> part) {
				auto device_counts_ptr = ih.get_native_mem(device_counts);
				auto rank_counts_ptr = ih.get_native_mem(rank_counts);
				const size_t device_idx = part.get_subrange().offset[0];
				// CELERITY_INFO(
				//     "Hello from interop task for device {}. Device counts: {}, rank counts: {}", device_idx, (void*)device_counts_ptr,
				//     (void*)rank_counts_ptr);
				auto stream = ih.get_native_queue<sycl::backend::cuda>();
				static_assert(std::is_same_v<decltype(per_rank_tile_point_counts), celerity::buffer<uint32_t, 3>>); // Adjust NCCL type if this fails
				// TODO: Can / should we register buffer with NCCL first?
				NCCL_CHECK(ncclReduce(device_counts_ptr, rank_counts_ptr, local_grid_size.size(), ncclUint32, ncclSum, 0, nccl_comms[device_idx], stream));
			});
		});
#else
#error Aggregating device counts only implemented using NCCL for now
#endif
	}

	if(precise_timings) print_delta_time("Sum points across devices");

	// The authoritative box is the portion of the global bounding box for which this rank will write the per-row counts
	//   => The lowest rank that has a row in its local bounding box is responsible (calculated below)
	// This box and vector are again needed for neighborhood calculation later on
	box<2> authoritative_box{subrange<2>{{local_grid_offset[0], 0}, {local_grid_size[0], 1}}};
	std::vector<box<2>> bounding_box_by_rank(num_ranks);

	// Determine all ranks that have overlapping bounding boxes with this rank
	// Compute global counts within local bounding box, and sum of all counts of lower ranks within local bounding box
	{
		const box<2> my_bounding_box{subrange<2>{local_grid_offset, local_grid_size}};
		// TODO: We may want a utility in Celerity to do this (also for regions). Could be useful for synchronizing data requirements across ranks.
		MPI_Allgather(&my_bounding_box, sizeof(box<2>), MPI_BYTE, bounding_box_by_rank.data(), sizeof(box<2>), MPI_BYTE, MPI_COMM_WORLD);

		celerity::custom_task_geometry<3> global_sum_geo;
		std::vector<std::pair<box<3>, region<3>>> read_rank_counts_accesses;
		std::vector<std::pair<box<3>, region<3>>> write_global_counts_accesses;

		celerity::custom_task_geometry<3> lower_rank_sum_geo;
		std::vector<std::pair<box<3>, region<3>>> read_lower_rank_counts_accesses;
		std::vector<std::pair<box<3>, region<3>>> write_lower_rank_sum_accesses;

		for(size_t i = 0; i < num_ranks; ++i) {
			if(i == rank) continue;
			const auto intersection = celerity::detail::box_intersection(my_bounding_box, bounding_box_by_rank[i]);
			if(!intersection.empty()) {
				// CELERITY_INFO("I have an intersecting bounding box with rank {}: {}", i, intersection);

				if(i < rank) {
					const auto new_authoritative_box = region_difference(authoritative_box, bounding_box_by_rank[i]).get_boxes();
					if(new_authoritative_box.size() > 1) {
						// TODO: Can this even happen? Certainly not with duplicated input
						throw std::runtime_error("Non-contiguous authoritative region?!");
					}
					authoritative_box = new_authoritative_box.size() == 1 ? new_authoritative_box[0] : box<2>{};
				}

				// We use a 3D task geometry in this case so we can have the same 2D chunk on two different nodes (in separate slices)
				// This is not necessary strictly speaking, because we could choose anything for the remote chunk - it just works out neatly
				const auto sr = intersection.get_subrange();
				const box<3> my_box = box<3>(subrange<3>{{rank, sr.offset[0], sr.offset[1]}, {1, sr.range[0], sr.range[1]}});
				const box<3> their_box = box<3>(subrange<3>{{i, sr.offset[0], sr.offset[1]}, {1, sr.range[0], sr.range[1]}});
				global_sum_geo.assigned_chunks.push_back({my_box, rank, 0});
				global_sum_geo.assigned_chunks.push_back({their_box, i, 0});
				// CELERITY_INFO("ADDING CHUNKS: {} -> {} and {} -> {}", my_box, rank, their_box, i);
				// This also works out nicely with the 3D kernel - the slice matches the per-rank counts buffer
				read_rank_counts_accesses.push_back({my_box, their_box});
				read_rank_counts_accesses.push_back({their_box, my_box});
				// For writes we only specify the local write, this data will not be read anywhere else
				write_global_counts_accesses.push_back({my_box, my_box});
				write_global_counts_accesses.push_back({their_box, {}});

				if(i < rank) {
					lower_rank_sum_geo.assigned_chunks.push_back({my_box, rank, 0});
					read_lower_rank_counts_accesses.push_back({my_box, their_box});
					write_lower_rank_sum_accesses.push_back({my_box, my_box});
				} else {
					// TODO: We should actually be able to not specify these, the data is already there from the global sum
					lower_rank_sum_geo.assigned_chunks.push_back({their_box, i, 0});
					read_lower_rank_counts_accesses.push_back({their_box, my_box});
					write_lower_rank_sum_accesses.push_back({their_box, {}});
				}
			}
		}

		if(precise_timings) print_delta_time("Compute authoritative box");

		// CELERITY_INFO("Authoritative box for row sums for this rank: {}", authoritative_box);

		// Initialize global counts with local counts
		// TODO: Need cgh.copy (although in this case its not clear how one would specify what to copy exactly..?)
		queue.submit([&](celerity::handler& cgh) {
			celerity::custom_task_geometry<3> copy_local_counts_geo;
			copy_local_counts_geo.assigned_chunks.push_back({rank_slice(rank), rank, 0});
			celerity::accessor read_rank_counts(per_rank_tile_point_counts, cgh, celerity::access::one_to_one{}, celerity::read_only);
			celerity::accessor write_global_counts(per_rank_global_counts, cgh, celerity::access::one_to_one{}, celerity::write_only, celerity::no_init);
			cgh.assert_no_data_movement(celerity::detail::data_movement_scope::inter_node);
			celerity::debug::set_task_name(cgh, "copy local counts to global counts buffer");
			cgh.parallel_for(copy_local_counts_geo, [=](celerity::id<3> id) { write_global_counts[id] = read_rank_counts[id]; });
		});

		if(precise_timings) print_delta_time("Copy local counts to global buffer");

		// Compute the global counts within the local bounding box
		// We currently do everything on device 0 for simplicity
		queue.submit([&](celerity::handler& cgh) {
			celerity::accessor read_rank_counts(per_rank_tile_point_counts, cgh,
			    celerity::expert_mapper(box<3>::full_range(per_rank_tile_point_counts.get_range()), read_rank_counts_accesses), celerity::read_only);
			// TODO: This may lead to overlapping writes between chunks on the same device; need to tell Celerity somehow that its ok because we use atomics
			// Since we write only parts of the local buffer, we cannot declare the whole buffer as being written and have to compute the actual union instead
			celerity::accessor write_global_counts(
			    per_rank_global_counts, cgh, celerity::expert_mapper(write_global_counts_accesses), celerity::write_only, celerity::no_init);
			celerity::debug::set_task_name(cgh, "sum overlapping remote counts");
			cgh.parallel_for(global_sum_geo, [=](celerity::id<3> id) {
				const auto source_rank = read_rank_counts.experimental_get_allocation_offset()[0];
				sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device> ref{write_global_counts[id]};
				ref += read_rank_counts[source_rank][id[1]][id[2]];
			});
		});

		if(precise_timings) print_delta_time("Sum overlapping remote counts");

		// Compute sum of counts of all ranks that are lower than the current rank
		// TODO: Can we combine this with the previous task?
		// TODO: For rank 0 we don't need to do this
		queue.submit([&](celerity::handler& cgh) {
			celerity::accessor read_rank_counts(per_rank_tile_point_counts, cgh,
			    celerity::expert_mapper(box<3>::full_range(per_rank_tile_point_counts.get_range()), read_lower_rank_counts_accesses), celerity::read_only);
			celerity::accessor write_lower_rank_sum(
			    per_rank_lower_rank_counts, cgh, celerity::expert_mapper(write_lower_rank_sum_accesses), celerity::write_only, celerity::no_init);
			celerity::debug::set_task_name(cgh, "sum lower rank counts");
			cgh.assert_no_data_movement(); // We already have the data where we need it from the previous task
			cgh.parallel_for(lower_rank_sum_geo, [=](celerity::id<3> id) {
				const auto source_rank = read_rank_counts.experimental_get_allocation_offset()[0];
				sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device> ref{write_lower_rank_sum[id]};
				ref += read_rank_counts[source_rank][id[1]][id[2]];
			});
		});

		if(precise_timings) print_delta_time("Sum lower rank counts");
	}

	// Compute prefix sum
	{
		// Begin by computing the prefix sum within the local bounding box on each rank
		queue.submit([&](celerity::handler& cgh) {
			celerity::custom_task_geometry<3> geo;
			geo.assigned_chunks.push_back({rank_slice(rank), rank, 0});
			celerity::accessor read_global_counts(per_rank_global_counts, cgh, celerity::access::one_to_one{}, celerity::read_only);
			celerity::accessor write_cumulative_counts(
			    per_rank_cumulative_counts, cgh, celerity::access::one_to_one{}, celerity::write_only, celerity::no_init);
			celerity::debug::set_task_name(cgh, "prefix sum local counts");
			cgh.assert_no_data_movement();
			cgh.interop_task(geo, [=](celerity::interop_handle& ih, celerity::partition<3> part) {
				thrust::device_ptr<uint32_t> global_counts_ptr(ih.get_native_mem(read_global_counts));
				thrust::device_ptr<uint32_t> cumulative_counts_ptr(ih.get_native_mem(write_cumulative_counts));
				auto stream = ih.get_native_queue<sycl::backend::cuda>();
				thrust::async::exclusive_scan(
				    thrust::device.on(stream), global_counts_ptr, global_counts_ptr + local_grid_size.size(), cumulative_counts_ptr, 0);
			});
		});

		if(precise_timings) print_delta_time("Compute local prefix sum (thrust)");

		// Next, we compute the per-row sum of the cumulative counts for the GLOBAL bounding box
		queue.submit([&](celerity::handler& cgh) {
			celerity::custom_task_geometry<1> geo;
			std::vector<std::pair<box<3>, region<3>>> read_cumulative_counts_accesses;
			std::vector<std::pair<box<3>, region<3>>> read_global_counts_accesses;
			std::vector<std::pair<box<3>, region<3>>> write_row_sums_accesses;
			if(!authoritative_box.empty()) {
				const auto chnk_box = box_cast<3>(authoritative_box);
				geo.assigned_chunks.push_back({chnk_box, rank, 0});
				// We only need to read from the rows intersecting with the authoritative region, but it doesn't matter if we declare more
				read_cumulative_counts_accesses.push_back({chnk_box, rank_slice(rank)});
				read_global_counts_accesses.push_back({chnk_box, rank_slice(rank)});
				// We manually declare the access instead of using a one-to-one range mapper so we can declare the full write region without
				// having to also specify all remote chunks
				write_row_sums_accesses.push_back({chnk_box, chnk_box});
			}
			celerity::accessor read_cumulative_counts(
			    per_rank_cumulative_counts, cgh, celerity::expert_mapper(read_cumulative_counts_accesses), celerity::read_only);
			celerity::accessor read_global_counts(per_rank_global_counts, cgh, celerity::expert_mapper(read_global_counts_accesses), celerity::read_only);
			celerity::accessor write_row_sums(global_row_sums, cgh,
			    celerity::expert_mapper(range_cast<3>(global_row_sums.get_range()), write_row_sums_accesses), celerity::write_only, celerity::no_init);
			celerity::debug::set_task_name(cgh, "write row sums");
			cgh.assert_no_data_movement();
			cgh.parallel_for(geo, [=](celerity::id<1> id) {
				// Since the prefix sum is exclusive, we we need to add the last element of the row
				write_row_sums[id] = read_cumulative_counts[rank][id[0]][local_grid_size[1] - 1]
				                     - read_cumulative_counts[rank][id[0]][local_grid_offset[1] /* must be 0 for this to work*/]
				                     + read_global_counts[rank][id[0]][local_grid_size[1] - 1];
			});
		});

		if(precise_timings) print_delta_time("Compute row sums");

		// Compute the value that needs to be added to the local prefix sum to complete it
		// TODO: We could do this using interop task and thrust, but it's probably not worth it
		queue.submit([&](celerity::handler& cgh) {
			celerity::accessor read_row_sums(global_row_sums, cgh, celerity::access::all{}, celerity::read_only_host_task);
			celerity::accessor write_add_to_prefix_sum(
			    per_rank_add_to_prefix_sum, cgh, celerity::access::one_to_one{}, celerity::write_only_host_task, celerity::no_init);
			celerity::debug::set_task_name(cgh, "compute add to prefix sum");
			cgh.host_task(celerity::range<1>(num_ranks), [=](celerity::partition<1> part) {
				if(part.get_subrange().offset[0] != rank) throw std::runtime_error("Partition offset is not the rank?");
				uint32_t sum = 0;
				for(size_t i = 0; i < local_grid_offset[0]; ++i) {
					sum += read_row_sums[i];
				}
				write_add_to_prefix_sum[rank] = sum;
				// CELERITY_INFO("Rank {} has to add {} to its local prefix sum", rank, sum);
			});
		});

		if(precise_timings) print_delta_time("Compute per-rank add to prefix sum (host)");

		// Finally, we add the value to the local prefix sum to complete it
		queue.submit([&](celerity::handler& cgh) {
			celerity::custom_task_geometry<3> geo;
			const auto chnk_box = rank_slice(rank);
			geo.assigned_chunks.push_back({chnk_box, rank, 0});
			std::vector<std::pair<box<3>, region<3>>> write_cumulative_counts_accesses;
			write_cumulative_counts_accesses.push_back({chnk_box, chnk_box});
			std::vector<std::pair<box<3>, region<3>>> read_add_to_prefix_sum_accesses;
			read_add_to_prefix_sum_accesses.push_back({chnk_box, box<3>({rank, 0, 0}, {rank + 1, 1, 1})});
			celerity::accessor read_add_to_prefix_sum(
			    per_rank_add_to_prefix_sum, cgh, celerity::expert_mapper(read_add_to_prefix_sum_accesses), celerity::read_only);
			celerity::accessor write_cumulative_counts(
			    per_rank_cumulative_counts, cgh, celerity::expert_mapper(write_cumulative_counts_accesses), celerity::read_write);
			celerity::debug::set_task_name(cgh, "add to prefix sum");
			cgh.assert_no_data_movement(celerity::detail::data_movement_scope::inter_node);
			cgh.parallel_for(geo, [=](celerity::id<3> id) { write_cumulative_counts[id] += read_add_to_prefix_sum[rank]; });
		});
	}

	if(precise_timings) print_delta_time("Add to prefix sum");

	// Compute per-device write offset
	{
		// We begin by adding up the global prefix sum and lower rank sum on each device
		queue.submit([&](celerity::handler& cgh) {
			celerity::custom_task_geometry<3> geo;
			// We use the same list of accesses for reading lower rank sums and cumulative counts
			std::vector<std::pair<box<3>, region<3>>> read_local_bounding_box;
			std::vector<std::pair<box<3>, region<3>>> write_device_write_offsets_accesses;
			for(size_t i = 0; i < num_devices; ++i) {
				const auto chnk_box = device_slice(i);
				geo.assigned_chunks.push_back({chnk_box, rank, i});
				read_local_bounding_box.push_back({chnk_box, rank_slice(rank)});
				write_device_write_offsets_accesses.push_back({chnk_box, chnk_box});
			}
			// FIXME: We are currently getting a legitimate uninitialized read warning here, because we only write the overlapping parts of the buffer
			//        => Either initialize to zero explicitly, or implement initialization-upon-allocation option
			celerity::accessor read_lower_rank_sum(per_rank_lower_rank_counts, cgh, celerity::expert_mapper(read_local_bounding_box), celerity::read_only);
			celerity::accessor read_cumulative_counts(per_rank_cumulative_counts, cgh, celerity::expert_mapper(read_local_bounding_box), celerity::read_only);
			celerity::accessor write_device_write_offsets(
			    per_device_write_offsets, cgh, celerity::expert_mapper(write_device_write_offsets_accesses), celerity::write_only, celerity::no_init);
			celerity::debug::set_task_name(cgh, "copy lower rank sum to device write offsets");
			cgh.assert_no_data_movement(celerity::detail::data_movement_scope::inter_node);
			cgh.parallel_for(geo, [=](celerity::id<3> id) {
				write_device_write_offsets[id] = read_cumulative_counts[rank][id[1]][id[2]] + read_lower_rank_sum[rank][id[1]][id[2]];
			});
		});

		if(precise_timings) print_delta_time("Copy lower rank sum to device write offsets");

		// Now we iteratively sum up the counts of all lower devices
		for(size_t j = 0; j < num_devices - 1; ++j) {
			queue.submit([&](celerity::handler& cgh) {
				celerity::custom_task_geometry<3> geo;
				std::vector<std::pair<box<3>, region<3>>> read_lower_device_counts_accesses;
				std::vector<std::pair<box<3>, region<3>>> write_device_write_offsets_accesses;
				for(size_t i = j + 1; i < num_devices; ++i) {
					const auto chnk_box = device_slice(i);
					geo.assigned_chunks.push_back({chnk_box, rank, i});
					read_lower_device_counts_accesses.push_back({chnk_box, device_slice(j)});
					write_device_write_offsets_accesses.push_back({chnk_box, chnk_box});
				}
				celerity::accessor read_lower_device_counts(
				    per_device_tile_point_counts, cgh, celerity::expert_mapper(read_lower_device_counts_accesses), celerity::read_only);
				celerity::accessor write_device_write_offsets(
				    per_device_write_offsets, cgh, celerity::expert_mapper(write_device_write_offsets_accesses), celerity::read_write);
				celerity::debug::set_task_name(cgh, fmt::format("add device {} counts", j));
				cgh.assert_no_data_movement(celerity::detail::data_movement_scope::inter_node);
				cgh.parallel_for(
				    geo, [=](celerity::id<3> id) { write_device_write_offsets[id] += read_lower_device_counts[rank * num_devices + j][id[1]][id[2]]; });
			});
		}

		if(precise_timings) print_delta_time("Add lower device counts");
	}

	// Compute per-device written subranges in 1D buffer
	std::vector<region<1>> written_region_per_device(num_devices);
	{
		for(size_t i = 0; i < num_devices; ++i) {
			queue.submit([&](celerity::handler& cgh) {
				celerity::custom_task_geometry geo;
				geo.assigned_chunks.push_back({box<3>::full_range(celerity::detail::ones), rank, 0}); // Doesn't matter
				std::vector<std::pair<box<3>, region<3>>> read_device_slice;
				read_device_slice.push_back({geo.assigned_chunks.back().box, device_slice(i)});
				celerity::accessor device_counts(per_device_tile_point_counts, cgh, celerity::expert_mapper(read_device_slice), celerity::read_only_host_task);
				celerity::accessor device_write_offsets(
				    per_device_write_offsets, cgh, celerity::expert_mapper(read_device_slice), celerity::read_only_host_task);
				celerity::debug::set_task_name(cgh, fmt::format("compute written subranges for device {}", i));
				cgh.assert_no_data_movement(celerity::detail::data_movement_scope::inter_node);
				cgh.host_task(geo, [=, &written_region_per_device](celerity::partition<1>) {
					auto counts = device_counts[rank * num_devices + i];
					auto offsets = device_write_offsets[rank * num_devices + i];

					celerity::detail::region_builder<1> written_region_builder;
					std::optional<celerity::subrange<1>> current_sr;
					for(size_t y = local_grid_offset[0]; y < local_grid_offset[0] + local_grid_size[0]; ++y) {
						for(size_t x = local_grid_offset[1]; x < local_grid_offset[1] + local_grid_size[1]; ++x) {
							if(counts[y][x] > 0) {
								if(!current_sr.has_value()) {
									current_sr = subrange<1>{offsets[y][x], counts[y][x]};
								} else {
									if(current_sr->offset + current_sr->range == offsets[y][x]) {
										// We are the only writer between the previous and current tile
										current_sr->range += counts[y][x];
									} else {
										// Someone else wrote in between
										written_region_builder.add(box<1>(*current_sr));
										current_sr = subrange<1>{offsets[y][x], counts[y][x]};
									}
								}
							} else if(current_sr.has_value()) {
								if(current_sr->offset + current_sr->range != offsets[y][x]) {
									// Someone else wrote in between
									written_region_builder.add(box<1>(*current_sr));
									current_sr.reset();
								}
							}
						}
					}
					if(current_sr.has_value()) { written_region_builder.add(box(*current_sr)); }
					written_region_per_device[i] = std::move(written_region_builder).into_region();
				});
			});
		}
		queue.wait(); // Wait for writes in main thread

		for(size_t i = 0; i < num_devices; ++i) {
			CELERITY_INFO("Device {} writes {} disjoint subranges", i, written_region_per_device[i].get_boxes().size());
		}
	}

	print_delta_time("Aggregate counts, compute written tile regions", before_aggregate_counts);

	// Write points into tiles
	{
		// Reset per-device counts to zero
		// TODO: We're currently really inconsistent with whether the device loop is outside or inside the queue.submit() - what is better?
		queue.submit([&](celerity::handler& cgh) {
			celerity::custom_task_geometry<3> geo;
			std::vector<std::pair<box<3>, region<3>>> write_device_slice;
			for(size_t i = 0; i < num_devices; ++i) {
				const auto chnk_box = device_slice(i);
				geo.assigned_chunks.push_back({chnk_box, rank, i});
				write_device_slice.push_back({chnk_box, chnk_box});
			}
			celerity::accessor write_device_counts(
			    per_device_tile_point_counts, cgh, celerity::expert_mapper(write_device_slice), celerity::write_only, celerity::no_init);
			celerity::debug::set_task_name(cgh, "reset per-device point counts");
			cgh.assert_no_data_movement();
			cgh.parallel_for(geo, [=](celerity::id<3> id) { write_device_counts[id] = 0; });
		});

		std::vector<std::pair<box<3>, region<3>>> per_chunk_accesses;
		std::vector<std::pair<box<3>, region<3>>> device_slice_accesses;
		for(auto& chnk : write_tiles_geometry.assigned_chunks) {
			if(chnk.nid == rank) {
				per_chunk_accesses.push_back({chnk.box, region_cast<3>(written_region_per_device[chnk.did.value()])});
				device_slice_accesses.push_back({chnk.box, device_slice(chnk.did.value())});
			} else {
				per_chunk_accesses.push_back({chnk.box, {}});
				device_slice_accesses.push_back({chnk.box, {}});
			}
		}
		celerity::expert_mapper tile_accesses{range_cast<3>(tiles_storage.get_range()), per_chunk_accesses};
		celerity::expert_mapper read_write_offsets_reqs{device_slice_accesses};
		read_write_offsets_reqs.options.use_local_indexing = true;

		queue.submit([&](celerity::handler& cgh) {
			celerity::accessor read_points(points_input, cgh, celerity::access::one_to_one{}, celerity::read_only);
			celerity::accessor read_write_offsets(per_device_write_offsets, cgh, read_write_offsets_reqs, celerity::read_only);
			celerity::accessor write_counts(per_device_tile_point_counts, cgh, write_device_count_access, celerity::write_only, celerity::no_init);
			celerity::accessor write_tiles(tiles_storage, cgh, tile_accesses, celerity::write_only, celerity::no_init);

			celerity::debug::set_task_name(cgh, "write points");
			cgh.assert_no_data_movement();
			cgh.parallel_for(write_tiles_geometry, [=](celerity::id<1> id) {
				// TODO: Keep DRY with above
				const auto& p = read_points[id];
				// grid_size - 1: We divide the domain such the last tile contains the maximum value
				const auto tile_x = static_cast<uint32_t>((p.x - global_min.x) / global_extent.x * (global_grid_size[1] - 1));
				const auto tile_y = static_cast<uint32_t>((p.y - global_min.y) / global_extent.y * (global_grid_size[0] - 1));
				auto device_slice = write_counts[0]; // 0 because we use local indexing
				const auto local_tile_x = tile_x - local_grid_offset[1];
				const auto local_tile_y = tile_y - local_grid_offset[0];
				sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device> ref{device_slice[local_tile_y][local_tile_x]};
				const uint32_t offset = ref.fetch_add(uint32_t(1));
				write_tiles[read_write_offsets[0][local_tile_y][local_tile_x] + offset] = p;
			});
		});
	}

	// Compute task geometry for shape factor calculation
	celerity::custom_task_geometry<1> shape_factor_geometry;
	shape_factor_geometry.assigned_chunks.resize(num_devices);
	{
		// TODO: There may be an edge case here if the local grids do not overlap, but we don't worry about that for now.
		for(size_t i = 0; i < num_devices; ++i) {
			queue.submit([&](celerity::handler& cgh) {
				celerity::custom_task_geometry geo;
				geo.assigned_chunks.push_back({box<3>::full_range(celerity::detail::ones), rank, 0}); // Doesn't matter
				// TODO: We really need a helper function to generate these, it's the same pattern over and over
				std::vector<std::pair<box<3>, region<3>>> read_rank_slice;
				read_rank_slice.push_back({geo.assigned_chunks.back().box, rank_slice(rank)});
				celerity::expert_mapper read_cumulative_counts(read_rank_slice);
				read_cumulative_counts.options.allocate_exactly = true;
				celerity::accessor cumulative_counts(per_rank_cumulative_counts, cgh, read_cumulative_counts, celerity::read_only_host_task);
				celerity::debug::set_task_name(cgh, fmt::format("compute shape factor geometry for device {}", i));
				cgh.assert_no_data_movement(celerity::detail::data_movement_scope::inter_node);
				cgh.host_task(geo, [=, &shape_factor_geometry](celerity::partition<1> part) {
					const auto ptr = cumulative_counts.get_allocation_window(part).get_allocation();
					const std::span counts_linear(ptr, local_grid_size[0] * local_grid_size[1]);
					const size_t ideal_points_per_device = global_num_points / (num_ranks * num_devices);
					const uint32_t device_gid = rank * num_devices + i;
					const uint32_t device_offset = device_gid * ideal_points_per_device;

					// This only works under the assumption that we have a "contiguous chunk" of counts
					assert(local_grid_size[1] == global_grid_size[1]);

					const auto start_it = std::ranges::lower_bound(counts_linear, device_offset);
					const auto end_it = std::ranges::lower_bound(counts_linear, device_offset + ideal_points_per_device);
					if(start_it == counts_linear.end()) { throw std::runtime_error("Could not find start of per-device tile range?!"); }
					if(end_it == counts_linear.end() && device_gid != num_ranks * num_devices - 1) {
						throw std::runtime_error("Could not find end of per-device tile range within local bounding box - this is bad");
					}
					const auto start_offset = *start_it;
					const bool is_last_device = device_gid == num_ranks * num_devices - 1;
					const auto end_offset = (end_it != counts_linear.end() && !is_last_device) ? *end_it : global_num_points;

					// CELERITY_INFO("Device {} (global {}) on rank {} should process points {} to {}", i, device_gid, rank, start_offset, end_offset);
					shape_factor_geometry.assigned_chunks[i] = {box_cast<3>(box<1>{start_offset, end_offset}), rank, i};
				});
			});
		}
		queue.wait(); // Wait for writes in main thread
	}

	print_delta_time("Write points, compute shape factor geometry");

	// Compute the neighborhood accesses
	std::vector<std::pair<box<3>, region<3>>> shape_factor_neighborhood_accesses;
	{
		for(size_t i = 1; i < num_devices; ++i) {
			if(shape_factor_geometry.assigned_chunks[i].box.get_min()[0] != shape_factor_geometry.assigned_chunks[i - 1].box.get_max()[0]) {
				throw std::runtime_error("Per-device ranges are non-contiguous?!");
			}
		}

		const box<1> my_execution_range{
		    box_cast<1>(shape_factor_geometry.assigned_chunks.front().box).get_min(), box_cast<1>(shape_factor_geometry.assigned_chunks.back().box).get_max()};

		std::vector<box<1>> execution_range_by_rank(num_ranks);
		MPI_Allgather(&my_execution_range, sizeof(box<1>), MPI_BYTE, execution_range_by_rank.data(), sizeof(box<1>), MPI_BYTE, MPI_COMM_WORLD);

		if(std::accumulate(std::next(execution_range_by_rank.begin()), execution_range_by_rank.end(), region<1>(execution_range_by_rank[0]),
		       [](const region<1>& a, const region<1>& b) { return region_union(a, b); })
		    != box<1>::full_range(global_num_points)) {
			throw std::runtime_error(fmt::format("Per-rank ranges do not cover the full global range {}. All ranges: {}", box<1>::full_range(global_num_points),
			    fmt::join(execution_range_by_rank, ", ")));
		}

		for(size_t i = 0; i < num_ranks; ++i) {
			if(i > 0 && execution_range_by_rank[i].get_min() != execution_range_by_rank[i - 1].get_max()) {
				throw std::runtime_error(fmt::format("Per-rank ranges are non-contiguous?! All ranges: {}", fmt::join(execution_range_by_rank, ", ")));
			}
			if(i == rank) continue;

			// NOTE: This needs to be adjusted if we wanted to more than a 1-neighborhood
			if((execution_range_by_rank[i].get_min()[0] > 0 && execution_range_by_rank[i].get_min()[0] == bounding_box_by_rank[i].get_min()[0])
			    || (execution_range_by_rank[i].get_max()[0] < global_grid_size[0]
			        && execution_range_by_rank[i].get_max()[0] == bounding_box_by_rank[i].get_max()[0])) {
				CELERITY_INFO("Rank {} will need to read outside its local bounding box!", i);
				// Turns out we don't have this case for any of the data sets, probably because we are extending the local grid in both dimension by one cell.
				throw std::runtime_error("NOT YET IMPLEMENTED!");

				// Here's what the plan would have been:

				// - Each rank inspects other ranks' execution ranges. If the N-neighborhood (N=1) of another rank exceeds its own local bounding box,
				//   and we are the authoritative rank for that row, add a read access to the data requirements of the neighborhood calculation task.

				// - Copy the authoritative rows of the cumulative counts buffer into the new global cumulative counts buffer, which all ranks can read from
				//   => This is a bit ugly because it means that we potentially end up re-transferring counts for overlapping regions (which we are not
				//   authoritative for), even though we already have them locally. Not sure if there's a way around this.

				// - Compute the neighborhood access by reading from the new cumulative counts buffer, with an extended bounding box
			}
		}

		std::vector<region<1>> neighborhood_reads_per_device(num_devices);

		for(size_t i = 0; i < num_devices; ++i) {
			queue.submit([&](celerity::handler& cgh) {
				celerity::custom_task_geometry geo;
				geo.assigned_chunks.push_back({box<3>::full_range(celerity::detail::ones), rank, 0}); // Doesn't matter
				// TODO: We really need a helper function to generate these, it's the same pattern over and over
				std::vector<std::pair<box<3>, region<3>>> read_rank_slice;
				read_rank_slice.push_back({geo.assigned_chunks.back().box, rank_slice(rank)});
				celerity::expert_mapper read_cumulative_counts(read_rank_slice);
				read_cumulative_counts.options.allocate_exactly = true;
				celerity::accessor cumulative_counts(per_rank_cumulative_counts, cgh, read_cumulative_counts, celerity::read_only_host_task);
				celerity::debug::set_task_name(cgh, fmt::format("compute neighborhood reads for device {}", i));
				// NOTE: This would not be true if we had to read from the global cumulative count buffer
				cgh.assert_no_data_movement(celerity::detail::data_movement_scope::inter_node);
				// TODO: This basically a copy-paste of compute_neighborhood_reads_2d - factor out into separate function again
				cgh.host_task(geo, [=, &shape_factor_geometry, &neighborhood_reads_per_device](celerity::partition<1> part) {
					const auto ptr = cumulative_counts.get_allocation_window(part).get_allocation();
					const std::span counts_linear(ptr, local_grid_size[0] * local_grid_size[1]);

					const auto& box = shape_factor_geometry.assigned_chunks[i].box;
					const auto sr = box.get_subrange();

					// Find start and end tile for this subrange
					// TODO: We actually have this information already when computing the geometry, could retain
					// upper_bound - 1: We want to find the first tile that contains any elements, because only on those will we
					// do any computations (and thus require a neighborhood read)
					// Using lower_bound instead returns the first empty tile after the previous chunk, thus overestimating the required reads
					const auto start_it = std::ranges::upper_bound(counts_linear, sr.offset[0]) - 1;
					assert(start_it != counts_linear.end());
					const auto end_it = std::lower_bound(start_it, counts_linear.end(), sr.offset[0] + sr.range[0]);
					assert(end_it != counts_linear.end() || sr.offset[0] + sr.range[0] == global_num_points);

					const uint32_t start_tile = start_it - counts_linear.begin();
					const uint32_t end_tile = end_it != counts_linear.end() ? end_it - counts_linear.begin() : local_grid_size.size();

					// Compute inclusive start and end coordinates in 2D
					// TODO: This is all done in LOCAL coordinates - will have to be adjusted if we need to read outside local bounding box!
					const celerity::id<2> first = {start_tile / local_grid_size[1], start_tile % local_grid_size[1]};
					const celerity::id<2> last = celerity::id<2>{(end_tile - 1) / local_grid_size[1], (end_tile - 1) % local_grid_size[1]};

					celerity::detail::box_vector<1> read_boxes;
					const auto add_box = [&](const celerity::id<2>& min, const celerity::id<2>& max) {
						if(celerity::detail::all_true(min < local_grid_offset) || celerity::detail::all_true(max >= local_grid_offset + local_grid_size)) {
							// This shouldn't happen, as we've established above (reading outside local bounding box is not yet implemented)
							throw std::runtime_error("Neighborhood is outside local bounding box?");
						}
						const uint32_t start_1d = min[0] * local_grid_size[1] + min[1];
						const uint32_t end_1d = max[0] * local_grid_size[1] + max[1] + 1; // +1: Convert back to exclusive
						if(end_1d == counts_linear.size() && (rank != num_ranks - 1 || i != num_devices - 1)) {
							// Just to be really sure
							throw std::runtime_error("Neighborhood is outside local bounding box?");
						}
						read_boxes.push_back({counts_linear[start_1d], end_1d < counts_linear.size() ? counts_linear[end_1d] : global_num_points});
					};

					// Add main chunk range + left and right neighbors
					celerity::id<2> main_neighborhood_start = first;
					celerity::id<2> main_neighborhood_end = last;
					if(first[1] > 0) { main_neighborhood_start[1]--; }
					if(last[1] < local_grid_size[1] - 1) { main_neighborhood_end[1]++; }
					add_box(main_neighborhood_start, main_neighborhood_end);

					// Add top neighbors
					if(last[0] > 0) {
						auto adjusted_start = main_neighborhood_start;
						if(first[0] == 0) { adjusted_start = {1, 0}; }
						const celerity::id<2> top_neighborhood_start = {adjusted_start[0] - 1, adjusted_start[1]};
						const celerity::id<2> top_neighborhood_end = {main_neighborhood_end[0] - 1, main_neighborhood_end[1]};
						add_box(top_neighborhood_start, top_neighborhood_end);
					}

					// Add bottom neighbors
					if(first[0] < local_grid_size[0] - 1) {
						auto adjusted_end = main_neighborhood_end;
						if(last[0] == local_grid_size[0] - 1) { adjusted_end = {local_grid_size[0] - 2, local_grid_size[1] - 1}; }
						const celerity::id<2> bottom_neighborhood_start = {main_neighborhood_start[0] + 1, main_neighborhood_start[1]};
						const celerity::id<2> bottom_neighborhood_end = {adjusted_end[0] + 1, adjusted_end[1]};
						add_box(bottom_neighborhood_start, bottom_neighborhood_end);
					}

					neighborhood_reads_per_device[i] = celerity::detail::region<1>{std::move(read_boxes)};
				});
			});
		}
		queue.wait(); // Wait for writes in main thread

		region<1> local_neighborhood;
		for(auto& device_region : neighborhood_reads_per_device) {
			local_neighborhood = region_union(local_neighborhood, device_region);
		}

		const auto neighborhood_reads_by_rank = allgather_regions(local_neighborhood, num_ranks, rank);
		for(size_t i = 0; i < num_ranks; ++i) {
			if(i == rank) continue;
			if(!box_intersection(bounding_box_by_rank[i], bounding_box_by_rank[rank]).empty()) {
				// This rank may need data from us (let Celerity figure it out)
				shape_factor_geometry.assigned_chunks.push_back({box_cast<3>(execution_range_by_rank[i]), i, 0});
				shape_factor_neighborhood_accesses.push_back({box_cast<3>(execution_range_by_rank[i]), region_cast<3>(neighborhood_reads_by_rank[i])});
			}
		}

		// Don't forget to add our own data requirements
		for(size_t i = 0; i < num_devices; ++i) {
			shape_factor_neighborhood_accesses.push_back(
			    {box_cast<3>(shape_factor_geometry.assigned_chunks[i].box), region_cast<3>(neighborhood_reads_per_device[i])});
		}
	}

	print_delta_time("Compute shape factor neighborhood accesses");

	// Run shape factor calculation
	{
		// TODO: There may be cases where we'd want to split chunks that wrap around into a new row into two separate chunks, one per row.
		// This is because if the second chunk doesn't go all the way to where the previous row starts, we may end up over-allocating by a large
		// amount due to the neighborhood access. Having multiple chunks would allow for these allocations to be made separately.

		queue.submit([&](celerity::handler& cgh) {
			celerity::expert_mapper read_tile_neighborhood(shape_factor_neighborhood_accesses);
			celerity::expert_mapper write_shape_factors_reqs(range_cast<3>(shape_factors.get_range()),
			    celerity::from_range_mapper(shape_factor_geometry, shape_factors.get_range(), celerity::access::one_to_one{}));

			// TODO API: We need a shortcut for this pattern, its really common
			std::vector<std::pair<box<3>, region<3>>> rank_slice_accesses;
			for(const auto& chnk : shape_factor_geometry.assigned_chunks) {
				if(chnk.nid == rank) {
					CELERITY_INFO("Device {} on rank {} has shape factor range {}", chnk.did.value(), rank, chnk.box);
					rank_slice_accesses.push_back({chnk.box, rank_slice(rank)});
				} else {
					rank_slice_accesses.push_back({chnk.box, {}});
				}
			}

			celerity::accessor read_tiles(tiles_storage, cgh, read_tile_neighborhood, celerity::read_only);
			celerity::accessor write_shape_factors(shape_factors, cgh, write_shape_factors_reqs, celerity::write_only, celerity::no_init);
			celerity::accessor read_global_counts(per_rank_global_counts, cgh, celerity::expert_mapper(rank_slice_accesses), celerity::read_only);
			celerity::accessor read_cumulative_counts(per_rank_cumulative_counts, cgh, celerity::expert_mapper(rank_slice_accesses), celerity::read_only);
			celerity::debug::set_task_name(cgh, "compute shape factors");

			cgh.parallel_for(shape_factor_geometry, [=](celerity::id<1> id) {
				const auto outer_product = [](const umuguc_point3d& p) {
					// create the matrix by calculating p * p^T
					std::array<std::array<double, 3>, 3> matrix;
					matrix[0][0] = p.x * p.x;
					matrix[0][1] = p.x * p.y;
					matrix[0][2] = p.x * p.z;
					matrix[1][0] = p.y * p.x;
					matrix[1][1] = p.y * p.y;
					matrix[1][2] = p.y * p.z;
					matrix[2][0] = p.z * p.x;
					matrix[2][1] = p.z * p.y;
					matrix[2][2] = p.z * p.z;
					return matrix;
				};

				const auto& get_point = [=](const tilebuffer_item& itm) {
					// TODO: Wait, isn't that just the kernel's thread id?
					return read_tiles[read_cumulative_counts[rank][itm.slot[0]][itm.slot[1]] + itm.index];
				};

				const auto get_slot = [=](const celerity::id<2>& slot) {
					// Roll own span until we can compile with C++20 (Clang 14 crashes if I try)
					struct bootleg_span {
						const umuguc_point3d* data;
						size_t count;
						const umuguc_point3d* begin() const { return data; }
						const umuguc_point3d* end() const { return data + count; }
					};

					const auto slot_size = read_global_counts[rank][slot[0]][slot[1]];
					if(slot_size == 0) {
						// Empty tile
						return bootleg_span{nullptr, 0};
					}
					const auto start = read_cumulative_counts[rank][slot[0]][slot[1]];
					return bootleg_span{&read_tiles[start], slot_size};
				};

				const auto item =
				    get_current_item(id, local_grid_size, local_grid_offset, &read_cumulative_counts[rank][local_grid_offset[0]][local_grid_offset[1]]);
				const auto p = get_point(item);

				std::array<std::array<double, 3>, 3> matrix{{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}};

				const int local_search_radius = 1; // NOTE: This CANNOT simply be changed. Requires adjustment to neighborhood access calculation.
				const double radius = tile_size;   // TODO: Decouple these two parameters
				double sum_fermi = 0.0;
				for(int j = -local_search_radius; j <= local_search_radius; ++j) {
					for(int k = -local_search_radius; k <= local_search_radius; ++k) {
						const int32_t x = item.slot[1] + k;
						const int32_t y = item.slot[0] + j;
						if(x < 0 || y < 0 || uint32_t(x) >= global_grid_size[1] || uint32_t(y) >= global_grid_size[0]) continue;

						for(auto& p2 : get_slot(celerity::id<2>(y, x))) {
							const umuguc_point3d p3 = {p.x - p2.x, p.y - p2.y, p.z - p2.z};
							const auto distance_p_p2 = sqrt(p3.x * p3.x + p3.y * p3.y + p3.z * p3.z);

							if(p != p2 && distance_p_p2 <= radius + 0.01) {
								const double fermi = 1 / (std::exp((distance_p_p2 / radius) - 0.6) / 0.1 + 1);
								sum_fermi += fermi;
								const std::array<std::array<double, 3>, 3> matrix2 = outer_product(p3);

								for(int m = 0; m < 3; m++) {
									for(int n = 0; n < 3; n++) {
										matrix[m][n] += fermi * matrix2[m][n];
									}
								}
							}
						}
					}
				}

				sycl::double3 sf = {};
				const size_t write_offset = read_cumulative_counts[rank][item.slot[0]][item.slot[1]] + item.index;
				if(sum_fermi == 0.0) {
					// Early exit if there are no neighbors
					write_shape_factors[write_offset] = sf;
					return;
				}

				for(int j = 0; j < 3; j++) {
					for(int k = 0; k < 3; k++) {
						matrix[j][k] = (1 / sum_fermi) * matrix[j][k];
					}
				}

				std::array<std::array<double, 3>, 3> V;
				std::array<double, 3> d{0, 0, 0};
				eigen_decomposition<3>(matrix, V, d);

				const auto sum = d[0] + d[1] + d[2];
				sf.z() = (3 * d[0]) / sum;
				sf.y() = (2 * (d[1] - d[0])) / sum;
				sf.x() = (d[2] - d[1]) / sum;

				write_shape_factors[write_offset] = sf;
			});
		});
	}

	print_delta_time("Calculate shape factors");

	if(write_output) {
		// Same issue as for the cumulative counts if we had to compute neighborhoods outside the local bounding box:
		// If one rank needs counts outside its local bounding box, we need a globally accessible copy of the data somewhere.
		// TODO: Any way to avoid this? (For a real application we would use distributed I/O, but that is overkill here)
		celerity::buffer<uint32_t, 2> global_total_counts({global_grid_size[0], global_grid_size[1]});

		// Copy authoritative rows into global buffer
		queue.submit([&](celerity::handler& cgh) {
			celerity::custom_task_geometry<2> geo;
			std::vector<std::pair<box<3>, region<3>>> read_rank_slice_accesses;
			if(!authoritative_box.empty()) {
				const auto asr = authoritative_box.get_subrange();
				const auto full_width_box = box<3>(subrange<3>{{asr.offset[0], local_grid_offset[1], 0}, {asr.range[0], local_grid_size[1], 1}});
				geo.assigned_chunks.push_back({full_width_box, rank, 0});
				// We only need to read from the rows intersecting with the authoritative region, but it doesn't matter if we declare more
				read_rank_slice_accesses.push_back({full_width_box, rank_slice(rank)});
			}
			celerity::accessor read_rank_global_counts(per_rank_global_counts, cgh, celerity::expert_mapper(read_rank_slice_accesses), celerity::read_only);
			celerity::accessor read_rank_cumulative_counts(
			    per_rank_cumulative_counts, cgh, celerity::expert_mapper(read_rank_slice_accesses), celerity::read_only);
			celerity::accessor write_global_counts(global_total_counts, cgh,
			    celerity::expert_mapper(range_cast<3>(global_total_counts.get_range()),
			        celerity::from_range_mapper(geo, global_total_counts.get_range(), celerity::access::one_to_one{})),
			    celerity::write_only, celerity::no_init);
			celerity::accessor write_global_cumulative_counts(global_cumulative_counts, cgh,
			    celerity::expert_mapper(range_cast<3>(global_cumulative_counts.get_range()),
			        celerity::from_range_mapper(geo, global_cumulative_counts.get_range(), celerity::access::one_to_one{})),
			    celerity::write_only, celerity::no_init);
			celerity::debug::set_task_name(cgh, "copy global counts for output");
			cgh.assert_no_data_movement();
			cgh.parallel_for(geo, [=](celerity::id<2> id) {
				write_global_counts[id] = read_rank_global_counts[rank][id[0]][id[1]];
				write_global_cumulative_counts[id] = read_rank_cumulative_counts[rank][id[0]][id[1]];
			});
		});

		queue.submit([&](celerity::handler& cgh) {
			celerity::accessor read_points(tiles_storage, cgh, celerity::access::all{}, celerity::read_only_host_task);
			celerity::accessor read_shape_factors(shape_factors, cgh, celerity::access::all{}, celerity::read_only_host_task);
			celerity::accessor read_global_counts(global_total_counts, cgh, celerity::access::all{}, celerity::read_only_host_task);
			celerity::accessor read_global_cumulative_counts(global_cumulative_counts, cgh, celerity::access::all{}, celerity::read_only_host_task);

			const size_t total_point_count = points_input.get_range().size();
			cgh.host_task(celerity::on_master_node, [=]() {
				CELERITY_INFO("Writing output to disk...");

				const auto write_to_file = [&](const std::string& filename, const auto* data) {
					std::ofstream file(filename, std::ios::binary);
					for(size_t y = 0; y < global_grid_size[0]; ++y) {
						for(size_t x = 0; x < global_grid_size[1]; ++x) {
							struct header {
								int x, y, count;
							};

							header hdr = {static_cast<int>(x), static_cast<int>(y), static_cast<int>(read_global_counts[y][x])};
							file.write(reinterpret_cast<const char*>(&hdr), sizeof(header));
							file.write(reinterpret_cast<const char*>(&data[read_global_cumulative_counts[y][x]]), read_global_counts[y][x] * sizeof(*data));
						}
					}
				};

				// since sycl::double3 is padded to 4 * sizeof(double), we have to pack it first
				std::vector<umuguc_point3d> packed_shape_factors(total_point_count);
				for(size_t i = 0; i < total_point_count; ++i) {
					const auto& sf = read_shape_factors[i];
					packed_shape_factors[i].x = sf.x();
					packed_shape_factors[i].y = sf.y();
					packed_shape_factors[i].z = sf.z();
				}
				write_to_file("umuguc_points_output.bin", read_points.get_pointer());
				write_to_file("umuguc_shape_factors_output.bin", packed_shape_factors.data());
			});
		});
		queue.wait(celerity::experimental::barrier);
	}

	if(write_output) print_delta_time("Writing output");

	print_delta_time("TOTAL TIME", before_count_points);


#if USE_NCCL
	for(const auto& comm : nccl_comms) {
		NCCL_CHECK(ncclCommDestroy(comm));
	}
#endif

	return 0;
}
