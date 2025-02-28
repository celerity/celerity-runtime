#include <vector>

#include <celerity.h>
#include <fmt/ranges.h>
#include <nccl.h>
#include <thrust/async/scan.h>
#include <thrust/device_ptr.h>

// TODO: A proper task geometry API should also offer splitting utilities
#include "scratch_buffer.h"
#include "split.h"
#include "tilebuffer_utils.h"

#include "eigen_decomposition.hpp"

using clk = std::chrono::steady_clock;

#define USE_NCCL 1

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
		fmt::print(stderr, "Usage: {} <input_file> [--write-output] [--duplicate-input <factor>] [--swap-xy]\n", argv[0]);
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
	// We need inputs to be "long" in the 0th dimension for our planned (NYI) prefix sum implementation
	// to work (somewhat) efficiently; we may have to rotate some files.
	const bool swap_xy = get_flag("--swap-xy", &argc, &argv);
	if(argc != 1) usage();

	// TODO: How do we want to handle warmup for this benchmark? Each phase needs new allocations, but we cannot anticipate them in IDAG.
	fmt::print("NO WARMUP - TBD\n");
	fmt::print("Using tile size {:.1f}\n", tile_size);

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
	// std::vector<umuguc_point3d> points(10'000);
	// input.read(reinterpret_cast<char*>(points.data()), points.size() * sizeof(umuguc_point3d));

	if(points.size() * duplicate_input > std::numeric_limits<uint32_t>::max()) {
		fmt::print(stderr, "Too many points for 32 bit counting. Time to upgrade!: {}\n", points.size());
		return 1;
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
	fmt::print("Read {} points ({:.1f} GiB)\n", points.size(), points.size() * sizeof(umuguc_point3d) / 1024.0 / 1024.0 / 1024.0);
	fmt::print("Min: ({:.1f}, {:.1f}, {:.1f}), Max: ({:.1f}, {:.1f}, {:.1f}). Extent: ({:.1f}, {:.1f}, {:.1f})\n", input_min.x, input_min.y, input_min.z,
	    input_max.x, input_max.y, input_max.z, input_max.x - input_min.x, input_max.y - input_min.y, input_max.z - input_min.z);

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
		fmt::print("Min: ({:.1f}, {:.1f}, {:.1f}), Max: ({:.1f}, {:.1f}, {:.1f}). Extent: ({:.1f}, {:.1f}, {:.1f}) <-- Bounding box of duplication pattern\n",
		    input_min.x, input_min.y, input_min.z, input_max.x, input_max.y, input_max.z, input_max.x - input_min.x, input_max.y - input_min.y,
		    input_max.z - input_min.z);

		// std::filesystem::path path(filename);
		// std::ofstream output(fmt::format("duplication_pattern_{}", path.filename().c_str()), std::ios::binary);
		// output.write(reinterpret_cast<const char*>(points.data()), points.size() * sizeof(umuguc_point3d));
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
	// TODO: Is the XY swap really worth the confusion?
	fmt::print("Global grid size: {}x{}, effective tile size: {:.1f}x{:.1f}\n", global_grid_size[1], global_grid_size[0],
	    (global_max.x - global_min.x) / global_grid_size[1], (global_max.y - global_min.y) / global_grid_size[0]);

	using celerity::subrange;
	using celerity::detail::box;
	using celerity::detail::region;
	using celerity::detail::subrange_cast;

	celerity::queue queue;

	celerity::buffer<umuguc_point3d, 1> points_input(points.size() * duplicate_input);
	celerity::debug::set_buffer_name(points_input, "points input");
	celerity::buffer<umuguc_point3d, 1> tiles_storage(points.size() * duplicate_input);
	celerity::debug::set_buffer_name(tiles_storage, "tiles storage");

	auto& rt = celerity::detail::runtime::get_instance();
	const auto rank = rt.NOCOMMIT_get_local_nid();
	const auto num_ranks = rt.NOCOMMIT_get_num_nodes();
	// // NOCOMMIT: We assume a uniform number of devices per node here
	// //           => Ideally we should simply not create per-device chunks for remote nodes
	const size_t num_devices = rt.NOCOMMIT_get_num_local_devices();

#if USE_NCCL
	std::vector<ncclComm_t> nccl_comms(num_devices);
	std::vector<int> device_ids;
	// std::iota(device_ids.begin(), device_ids.end(), 0);
	for(auto& dev : rt.NOCOMMIT_get_sycl_devices()) {
		device_ids.push_back(sycl::get_native<sycl::backend::cuda>(dev));
	}
	CELERITY_INFO("Creating NCCL comms for devices {}", fmt::join(device_ids, ", "));
	NCCL_CHECK(ncclCommInitAll(nccl_comms.data(), num_devices, device_ids.data()));
#endif

	if(duplicate_input % num_ranks != 0) {
		// This is because we don't want to have to compute an exact bounding box for each rank for arbitrary splits.
		throw std::runtime_error(fmt::format("Input multiplier ({}) must be divisible by number of ranks ({})", duplicate_input, num_ranks));
	}

	// TODO: Add option to disable this for all the intermediate stages, because it requires synchronization
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

	// TODO: Can we move this to device?
	queue.submit([&](celerity::handler& cgh) {
		celerity::accessor write_points(points_input, cgh, celerity::access::one_to_one{}, celerity::write_only_host_task, celerity::no_init);
		cgh.host_task(points_input.get_range(), [=, &points, &local_min, &local_max](celerity::partition<1> part) {
			if(part.get_subrange().offset[0] % points.size() != 0) throw std::runtime_error("Partition offset is not a multiple of input size?");
			if(part.get_subrange().range[0] % points.size() != 0) throw std::runtime_error("Partition range is not a multiple of input size?");

			const auto offset = part.get_subrange().offset[0] / points.size();
			const auto count = part.get_subrange().range[0] / points.size();
			fmt::print("OFFSET IS {}, COUNT IS {}\n", offset, count);
			local_min.y = input_min.y + original_extent_y * offset;
			local_max.y = input_max.y + original_extent_y * (offset + count - 1);

			celerity::experimental::for_each_item(part, [&](celerity::item<1> itm) {
				const size_t offset = itm[0] / points.size();
				auto pt = points[itm[0] % points.size()];
				// The bounding box of the "duplication pattern" may be different from the original input,
				// because we split based on point count, not at the geometric center.
				// Since the pattern is created by offsetting the input by a full bounding box,
				// we have to use the original extent here for them to seamlessly fit together again.
				pt.y += original_extent_y * offset;
				write_points[itm] = pt;
				if(itm[0] % points.size() == 0) { fmt::print("Offsetting by {:.1f} in y\n", original_extent_y * offset); }
			});

			// std::filesystem::path path(filename);
			// std::ofstream output(fmt::format("duplicated_x{}_{}", duplicate_input, path.filename().c_str()), std::ios::binary);
			// output.write(reinterpret_cast<const char*>(write_points.get_pointer()), part.get_global_size().size() * sizeof(umuguc_point3d));
		});
	});

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
	fmt::print("Local grid size: {}x{}, offset: {},{}, effective tile size: {:.1f}x{:.1f}\n", local_grid_size[1], local_grid_size[0], local_grid_offset[1],
	    local_grid_offset[0], local_extent.x / local_grid_size[1], local_extent.y / local_grid_size[0]);

	// Sanity check: Does actual bounding box match predicted bounding box?
#if 1
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
	auto chunks =
	    celerity::detail::split_1d(box_cast<3>(celerity::detail::box<1>{0, points_input.get_range()}), celerity::detail::ones, num_ranks * num_devices);
	celerity::custom_task_geometry write_tiles_geometry;
	write_tiles_geometry.global_size = range_cast<3>(points_input.get_range());
	for(size_t i = 0; i < chunks.size(); ++i) {
		write_tiles_geometry.assigned_chunks.push_back({chunks[i].get_subrange(), (i / num_devices), (i % num_devices)});
	}

	celerity::scratch_buffer<uint32_t, 2> num_entries_scratch(write_tiles_geometry, global_grid_size);            // TODO: This should be device scope
	celerity::scratch_buffer<uint32_t, 2> num_entries_cumulative_scratch(write_tiles_geometry, global_grid_size); // TODO: This should be node scope
	// Like num_entries_cumulative_scratch, but includes per-chunk write offset within each tile (= sum of all ranks and chunks that write before it)
	celerity::scratch_buffer<uint32_t, 2> write_offsets_scratch(write_tiles_geometry, global_grid_size); // TODO: This should be device scope

	num_entries_scratch.fill(0); // The other two are initialized from host data later on

	// NEW APPROACH - Using global buffers
	// TODO: Not sure if using 3D buffers is the best idea, since we rarely exercise those code paths in graph generators. Could also just use 2D w/ offset.
	celerity::buffer<uint32_t, 3> per_device_tile_point_counts({num_ranks * num_devices, global_grid_size[0], global_grid_size[1]});
	celerity::buffer<uint32_t, 3> per_rank_tile_point_counts({num_ranks, global_grid_size[0], global_grid_size[1]});
	// Each rank constructs its own copy of the global counts within its local bounding box
	// This means incorporating counts from all other ranks that have an overlapping bounding box
	// TODO: Naming
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
	// TODO: Rename, _buffer is due to naming conflict with legacy vector below
	celerity::buffer<uint32_t, 3> per_device_write_offsets_buffer({num_ranks * num_devices, global_grid_size[0], global_grid_size[1]});

	queue.wait(celerity::experimental::barrier);
	const auto before_count_points = std::chrono::steady_clock::now();

	queue.submit([&](celerity::handler& cgh) {
		celerity::accessor read_points(points_input, cgh, celerity::access::one_to_one{}, celerity::read_only);
		// TODO API: Should this just be a normal accessor? Or simply a pointer?
		celerity::scratch_accessor write_num_entries_scratch(num_entries_scratch /* TODO write access */);
		celerity::debug::set_task_name(cgh, "count points (scratch)");

		cgh.assert_no_data_movement(celerity::detail::data_movement_scope::inter_node);
		// TODO API: How do we launch nd-range kernels with custom geometries? Required for optimized count/write
		cgh.parallel_for(write_tiles_geometry, [=](celerity::id<1> id) {
			const auto& p = read_points[id];
			// grid_size - 1: We divide the domain such the last tile contains the maximum value
			const auto tile_x = static_cast<uint32_t>((p.x - global_min.x) / global_extent.x * (global_grid_size[1] - 1));
			const auto tile_y = static_cast<uint32_t>((p.y - global_min.y) / global_extent.y * (global_grid_size[0] - 1));
			// Count points in per-chunk scratch buffer
			sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device> ref{write_num_entries_scratch[{tile_y, tile_x}]};
			ref++;
		});
	});

	// TODO: If this becomes a pattern, build a utility around it
	std::vector<std::pair<box<3>, region<3>>> local_device_accesses;
	for(auto& chnk : write_tiles_geometry.assigned_chunks) {
		if(chnk.nid == rank) {
			const celerity::id<3> offset = {chnk.nid * num_devices + chnk.did.value(), local_grid_offset[0], local_grid_offset[1]};
			const celerity::range<3> range = {1, local_grid_size[0], local_grid_size[1]};
			local_device_accesses.push_back({chnk.box, box<3>(subrange<3>{offset, range})});
		} else {
			local_device_accesses.push_back({chnk.box, {}}); // TODO: Having to do this is stupid
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

// FIXME: We need to initialize the buffer to 0. Where is handler::fill...
// Easiest solution for now would probably be a "data access property" (like exact allocation, local indexing) that says what to initialize to
#ifndef NDEBUG
	CELERITY_CRITICAL("COUNT BUFFER NOT YET INITIALIZED TO ZERO - DEBUG MEMORY PATTERN WILL CAUSE PROBLEMS");
#endif

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

	auto before_subrange_calculation = print_delta_time("Counting points", before_count_points);

	// Sum up counts across all devices on this rank
	// This is spicy because for the first time we submit a geometry that is local to each node
	{
		// TODO: This should use the geometry builder
		celerity::custom_task_geometry sum_points_geo;
		sum_points_geo.global_size = range_cast<3>(celerity::range<1>{num_devices});
		std::vector<std::pair<box<3>, region<3>>> read_device_counts_accesses;
		std::vector<std::pair<box<3>, region<3>>> write_rank_counts_accesses;
		for(size_t i = 0; i < num_devices; ++i) {
			const auto chnk_box = box_cast<3>(box<1>(subrange<1>{i, 1}));
			sum_points_geo.assigned_chunks.push_back({chnk_box, rank, i});

			const celerity::id<3> device_offset = {rank * num_devices + i, local_grid_offset[0], local_grid_offset[1]};
			const celerity::range<3> range = {1, local_grid_size[0], local_grid_size[1]};
			read_device_counts_accesses.push_back({chnk_box, box<3>(subrange<3>{device_offset, range})});

			if(i == 0) {
				const celerity::id<3> rank_offset = {rank, local_grid_offset[0], local_grid_offset[1]};
				write_rank_counts_accesses.push_back({chnk_box, box<3>(subrange<3>{rank_offset, range})});
			} else {
				write_rank_counts_accesses.push_back({chnk_box, {}});
			}
		}
		celerity::expert_mapper read_device_counts{box<3>::full_range(per_device_tile_point_counts.get_range()), read_device_counts_accesses};
		celerity::expert_mapper write_rank_counts{box<3>::full_range(per_rank_tile_point_counts.get_range()), write_rank_counts_accesses};

#if USE_NCCL
		queue.submit([&](celerity::handler& cgh) {
			// TODO: Need exact allocation hint
			celerity::accessor device_counts(per_device_tile_point_counts, cgh, read_device_counts, celerity::read_only);
			celerity::accessor rank_counts(per_rank_tile_point_counts, cgh, write_rank_counts, celerity::write_only, celerity::no_init);

			celerity::debug::set_task_name(cgh, "device count sum");
			cgh.interop_task(sum_points_geo, [=](celerity::interop_handle& ih, celerity::partition<1> part) {
				auto device_counts_ptr = ih.get_native_mem(device_counts);
				auto rank_counts_ptr = ih.get_native_mem(rank_counts);
				const size_t device_idx = part.get_subrange().offset[0];
				CELERITY_INFO(
				    "Hello from interop task for device {}. Device counts: {}, rank counts: {}", device_idx, (void*)device_counts_ptr, (void*)rank_counts_ptr);
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

	// The portion of the global bounding box for which this rank will write the per-row counts
	// The lowest rank that has a row in its local bounding box is responsible (calculated below)
	box<2> authoritative_box{subrange<2>{{local_grid_offset[0], 0}, {local_grid_size[0], 1}}};

	// Determine all ranks that have overlapping bounding boxes with this rank
	// Compute global counts within local bounding box, and sum of all counts of lower ranks within local bounding box
	{
		const box<2> my_bounding_box{subrange<2>{local_grid_offset, local_grid_size}};
		std::vector<box<2>> bounding_box_by_rank(num_ranks);
		// TODO: We may want a utility in Celerity to do this (also for regions). Could be useful for synchronizing data requirements across ranks.
		MPI_Allgather(&my_bounding_box, sizeof(box<2>), MPI_BYTE, bounding_box_by_rank.data(), sizeof(box<2>), MPI_BYTE, MPI_COMM_WORLD);

		// std::vector<box<2>> intersecting_bounding_boxes(num_ranks); // by rank
		celerity::custom_task_geometry<3> global_sum_geo;
		std::vector<std::pair<box<3>, region<3>>> read_rank_counts_accesses;
		std::vector<std::pair<box<3>, region<3>>> write_global_counts_accesses;
		region<3> write_global_counts_union_region;

		celerity::custom_task_geometry<3> lower_rank_sum_geo;
		std::vector<std::pair<box<3>, region<3>>> read_lower_rank_counts_accesses;
		std::vector<std::pair<box<3>, region<3>>> write_lower_rank_sum_accesses;
		region<3> write_lower_rank_sum_union_region;

		for(size_t i = 0; i < num_ranks; ++i) {
			if(i == rank) continue;
			const auto intersection = celerity::detail::box_intersection(my_bounding_box, bounding_box_by_rank[i]);
			if(!intersection.empty()) {
				CELERITY_INFO("I have an intersecting bounding box with rank {}: {}", i, intersection);

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
				CELERITY_INFO("ADDING CHUNKS: {} -> {} and {} -> {}", my_box, rank, their_box, i);
				// This also works out nicely with the 3D kernel - the slice matches the per-rank counts buffer
				read_rank_counts_accesses.push_back({my_box, their_box});
				read_rank_counts_accesses.push_back({their_box, my_box});
				// For writes we only specify the local write, this data will not be read anywhere else
				write_global_counts_accesses.push_back({my_box, my_box});
				write_global_counts_accesses.push_back({their_box, {}});
				write_global_counts_union_region = region_union(write_global_counts_union_region, region<3>{my_box});

				if(i < rank) {
					lower_rank_sum_geo.assigned_chunks.push_back({my_box, rank, 0});
					read_lower_rank_counts_accesses.push_back({my_box, their_box});
					write_lower_rank_sum_accesses.push_back({my_box, my_box});
					write_lower_rank_sum_union_region = region_union(write_lower_rank_sum_union_region, region<3>{my_box});
				} else {
					// TODO: We should actually be able to not specify these, the data is already there from the global sum
					lower_rank_sum_geo.assigned_chunks.push_back({their_box, i, 0});
					read_lower_rank_counts_accesses.push_back({their_box, my_box});
					write_lower_rank_sum_accesses.push_back({their_box, {}});
				}
			}
		}

		CELERITY_INFO("Authoritative box for row sums for this rank: {}", authoritative_box);

		// Initialize global counts with local counts
		// TODO: Need cgh.copy (although in this case its not clear how one would specify what to copy exactly..?)
		queue.submit([&](celerity::handler& cgh) {
			celerity::custom_task_geometry<3> copy_local_counts_geo;
			copy_local_counts_geo.assigned_chunks.push_back(
			    {box<3>(subrange<3>{{rank, local_grid_offset[0], local_grid_offset[1]}, {1, local_grid_size[0], local_grid_size[1]}}), rank, 0});
			celerity::accessor read_rank_counts(per_rank_tile_point_counts, cgh, celerity::access::one_to_one{}, celerity::read_only);
			celerity::accessor write_global_counts(per_rank_global_counts, cgh, celerity::access::one_to_one{}, celerity::write_only, celerity::no_init);
			cgh.assert_no_data_movement(celerity::detail::data_movement_scope::inter_node);
			celerity::debug::set_task_name(cgh, "copy local counts to global counts buffer");
			cgh.parallel_for(copy_local_counts_geo, [=](celerity::id<3> id) { write_global_counts[id] = read_rank_counts[id]; });
		});

		// Compute the global counts within the local bounding box
		// We currently do everything on device 0 for simplicity
		queue.submit([&](celerity::handler& cgh) {
			celerity::accessor read_rank_counts(per_rank_tile_point_counts, cgh,
			    celerity::expert_mapper(box<3>::full_range(per_rank_tile_point_counts.get_range()), read_rank_counts_accesses), celerity::read_only);
			// TODO: This may lead to overlapping writes between chunks on the same device; need to tell Celerity somehow that its ok because we use atomics
			// Since we write only parts of the local buffer, we cannot declare the whole buffer as being written and have to compute the actual union instead
			// TODO: There should be a constructor for expert_mapper that computes this automatically
			celerity::accessor write_global_counts(per_rank_global_counts, cgh,
			    celerity::expert_mapper(write_global_counts_union_region, write_global_counts_accesses), celerity::write_only, celerity::no_init);
			celerity::debug::set_task_name(cgh, "sum overlapping remote counts");
			cgh.parallel_for(global_sum_geo, [=](celerity::id<3> id) {
				const auto source_rank = read_rank_counts.experimental_get_allocation_offset()[0];
				sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device> ref{write_global_counts[id]};
				ref += read_rank_counts[source_rank][id[1]][id[2]];
			});
		});

		// Compute sum of counts of all ranks that are lower than the current rank
		// TODO: Can we combine this with the previous task?
		queue.submit([&](celerity::handler& cgh) {
			celerity::accessor read_rank_counts(per_rank_tile_point_counts, cgh,
			    celerity::expert_mapper(box<3>::full_range(per_rank_tile_point_counts.get_range()), read_lower_rank_counts_accesses), celerity::read_only);
			celerity::accessor write_lower_rank_sum(per_rank_lower_rank_counts, cgh,
			    celerity::expert_mapper(write_lower_rank_sum_union_region, write_lower_rank_sum_accesses), celerity::write_only, celerity::no_init);
			celerity::debug::set_task_name(cgh, "sum lower rank counts");
			cgh.assert_no_data_movement(); // We already have the data where we need it from the previous task
			cgh.parallel_for(lower_rank_sum_geo, [=](celerity::id<3> id) {
				const auto source_rank = read_rank_counts.experimental_get_allocation_offset()[0];
				sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device> ref{write_lower_rank_sum[id]};
				ref += read_rank_counts[source_rank][id[1]][id[2]];
			});
		});
	}

	// Compute prefix sum
	{
		// Begin by computing the prefix sum within the local bounding box on each rank
		queue.submit([&](celerity::handler& cgh) {
			celerity::custom_task_geometry<3> geo;
			geo.assigned_chunks.push_back(
			    {box<3>(subrange<3>{{rank, local_grid_offset[0], local_grid_offset[1]}, {1, local_grid_size[0], local_grid_size[1]}}), rank, 0});
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

		// Next, we compute the per-row sum of the cumulative counts for the GLOBAL bounding box
		queue.submit([&](celerity::handler& cgh) {
			celerity::custom_task_geometry<1> geo;
			std::vector<std::pair<box<3>, region<3>>> read_cumulative_counts_accesses;
			std::vector<std::pair<box<3>, region<3>>> read_global_counts_accesses;
			std::vector<std::pair<box<3>, region<3>>> write_row_sums_accesses;
			if(!authoritative_box.empty()) {
				const auto chnk_box = box_cast<3>(authoritative_box);
				geo.assigned_chunks.push_back({chnk_box, rank, 0});
				const celerity::id<3> rank_offset = {rank, local_grid_offset[0], local_grid_offset[1]};
				const celerity::range<3> range = {1, local_grid_size[0], local_grid_size[1]};
				// We only need to read from the rows intersecting with the authoritative region, but it doesn't matter if we declare more
				read_cumulative_counts_accesses.push_back({chnk_box, box<3>(subrange<3>{rank_offset, range})});
				read_global_counts_accesses.push_back({chnk_box, box<3>(subrange<3>{rank_offset, range})});
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
				CELERITY_INFO("Rank {} has to add {} to its local prefix sum", rank, sum);
			});
		});

		// Finally, we add the value to the local prefix sum to complete it
		queue.submit([&](celerity::handler& cgh) {
			celerity::custom_task_geometry<3> geo;
			const auto chnk_box = box<3>(subrange<3>{{rank, local_grid_offset[0], local_grid_offset[1]}, {1, local_grid_size[0], local_grid_size[1]}});
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

	// Compute per-device write offset
	{
		// We begin by adding up the global prefix sum and lower rank sum on each device
		queue.submit([&](celerity::handler& cgh) {
			celerity::custom_task_geometry<3> geo;
			// We use the same list of accesses for reading lower rank sums and cumulative counts
			std::vector<std::pair<box<3>, region<3>>> read_local_bounding_box;
			std::vector<std::pair<box<3>, region<3>>> write_device_write_offsets_accesses;
			for(size_t i = 0; i < num_devices; ++i) {
				const auto chnk_box =
				    box<3>(subrange<3>{{rank * num_devices + i, local_grid_offset[0], local_grid_offset[1]}, {1, local_grid_size[0], local_grid_size[1]}});
				geo.assigned_chunks.push_back({chnk_box, rank, i});
				read_local_bounding_box.push_back(
				    {chnk_box, box<3>(subrange<3>{{rank, local_grid_offset[0], local_grid_offset[1]}, {1, local_grid_size[0], local_grid_size[1]}})});
				write_device_write_offsets_accesses.push_back({chnk_box, chnk_box});
			}
			// FIXME: We are currently getting a legitimate uninitialized read warning here, because we only write the overlapping parts of the buffer
			//        => Either initialize to zero explicitly, or implement initialization-upon-allocation option
			celerity::accessor read_lower_rank_sum(per_rank_lower_rank_counts, cgh, celerity::expert_mapper(read_local_bounding_box), celerity::read_only);
			celerity::accessor read_cumulative_counts(per_rank_cumulative_counts, cgh, celerity::expert_mapper(read_local_bounding_box), celerity::read_only);
			celerity::accessor write_device_write_offsets(
			    per_device_write_offsets_buffer, cgh, celerity::expert_mapper(write_device_write_offsets_accesses), celerity::write_only, celerity::no_init);
			celerity::debug::set_task_name(cgh, "copy lower rank sum to device write offsets");
			cgh.assert_no_data_movement(celerity::detail::data_movement_scope::inter_node);
			cgh.parallel_for(geo, [=](celerity::id<3> id) {
				write_device_write_offsets[id] = read_cumulative_counts[rank][id[1]][id[2]] + read_lower_rank_sum[rank][id[1]][id[2]];
			});
		});

		// Now we iteratively sum up the counts of all lower devices
		for(size_t j = 0; j < num_devices - 1; ++j) {
			queue.submit([&](celerity::handler& cgh) {
				celerity::custom_task_geometry<3> geo;
				std::vector<std::pair<box<3>, region<3>>> read_lower_device_counts_accesses;
				std::vector<std::pair<box<3>, region<3>>> write_device_write_offsets_accesses;
				for(size_t i = j + 1; i < num_devices; ++i) {
					const auto chnk_box =
					    box<3>(subrange<3>{{rank * num_devices + i, local_grid_offset[0], local_grid_offset[1]}, {1, local_grid_size[0], local_grid_size[1]}});
					geo.assigned_chunks.push_back({chnk_box, rank, i});
					read_lower_device_counts_accesses.push_back(
					    {chnk_box, box<3>(subrange<3>{
					                   {rank * num_devices + j, local_grid_offset[0], local_grid_offset[1]}, {1, local_grid_size[0], local_grid_size[1]}})});
					write_device_write_offsets_accesses.push_back({chnk_box, chnk_box});
				}
				celerity::accessor read_lower_device_counts(per_rank_lower_rank_counts, cgh,
				    celerity::expert_mapper(box<3>::full_range(per_device_tile_point_counts.get_range()), read_lower_device_counts_accesses),
				    celerity::read_only);
				celerity::accessor write_device_write_offsets(per_device_write_offsets_buffer, cgh,
				    celerity::expert_mapper(box<3>::full_range(per_device_write_offsets_buffer.get_range()), write_device_write_offsets_accesses),
				    celerity::read_write);
				celerity::debug::set_task_name(cgh, fmt::format("add device {} counts", j));
				cgh.assert_no_data_movement(celerity::detail::data_movement_scope::inter_node);
				cgh.parallel_for(
				    geo, [=](celerity::id<3> id) { write_device_write_offsets[id] += read_lower_device_counts[rank * num_devices + j][id[1]][id[2]]; });
			});
		}
	}

	queue.wait();

	// Add up counts across ranks
	// TODO: This should just be a vector of regions
	std::vector<std::vector<celerity::subrange<1>>> written_subranges_per_device;
	std::vector<uint32_t> num_entries(global_grid_size.size());
	std::vector<uint32_t> num_entries_cumulative(global_grid_size.size());
	{
		MPI_Barrier(MPI_COMM_WORLD);

		// Get data from per-GPU scratch buffer on this node
		auto num_entries_per_device = num_entries_scratch.get_data_on_host();

		// Compute total sum across all devices
		for(const auto& data : num_entries_per_device) {
			for(size_t i = 0; i < global_grid_size.size(); ++i) {
				num_entries[i] += data[i];

#if 1
				// SANITY CHECK: All non-zero counts are within the local grid
				if(data[i] > 0) {
					const auto tile_x = i % global_grid_size[1];
					const auto tile_y = i / global_grid_size[1];
					if(tile_x < local_grid_offset[1] || tile_y < local_grid_offset[0] || tile_x >= local_grid_offset[1] + local_grid_size[1]
					    || tile_y >= local_grid_offset[0] + local_grid_size[0]) {
						CELERITY_CRITICAL("Rank {} has non-zero count for tile ({}, {}), which is outside local grid {}\n", rank, tile_y, tile_x,
						    celerity::subrange<2>{local_grid_offset, local_grid_size});
						abort();
					}
				}
#endif
			}
		}

// Sanity check: Compare the per-rank counts computed in global buffer versus what we computed here
#if 1
		{
			celerity::custom_task_geometry sanity_check_counts_geo;
			sanity_check_counts_geo.global_size = range_cast<3>(celerity::range<1>{num_ranks});
			std::vector<std::pair<box<3>, region<3>>> read_rank_counts_accesses;
			// TODO: No need to specify remote chunks
			for(size_t i = 0; i < num_ranks; ++i) {
				const auto chnk_box = box_cast<3>(box<1>(subrange<1>{i, 1}));
				sanity_check_counts_geo.assigned_chunks.push_back({chnk_box, i, 0});
				const celerity::id<3> rank_offset = {i, local_grid_offset[0], local_grid_offset[1]};
				// This is actually wrong for remote chunks (but it doesn't matter)
				const celerity::range<3> range = {1, local_grid_size[0], local_grid_size[1]};
				read_rank_counts_accesses.push_back({chnk_box, box<3>(subrange<3>{rank_offset, range})});
			}
			// TODO: We shouldn't have to specify the full read region in this case
			celerity::expert_mapper read_rank_counts{box<3>::full_range(per_rank_tile_point_counts.get_range()), read_rank_counts_accesses};

			queue.submit([&](celerity::handler& cgh) {
				celerity::accessor read_counts(per_rank_tile_point_counts, cgh, read_rank_counts, celerity::read_only_host_task);
				celerity::debug::set_task_name(cgh, "sanity check rank counts");
				cgh.assert_no_allocations(celerity::detail::allocation_scope::device);
				cgh.assert_no_data_movement(celerity::detail::data_movement_scope::inter_node);
				cgh.host_task(sanity_check_counts_geo, [=, &num_entries](celerity::partition<1> part) {
					if(part.get_subrange().offset[0] != rank) throw std::runtime_error("Partition offset is not the rank?");
					size_t correct_nonzero_tiles = 0;
					size_t error_tiles = 0;
					for(size_t y = 0; y < local_grid_size[0]; ++y) {
						for(size_t x = 0; x < local_grid_size[1]; ++x) {
							const auto global_x = local_grid_offset[1] + x;
							const auto global_y = local_grid_offset[0] + y;
							const auto idx = global_y * global_grid_size[1] + global_x;
							if(num_entries[idx] != read_counts[rank][global_y][global_x]) {
								CELERITY_CRITICAL("Rank {} has mismatch of per-rank count for tile ({}, {}) (local coordinates {}, {}): {} (legacy) vs {} "
								                  "(rank). {} tiles with non-zero "
								                  "count were correct so far.\n",
								    rank, global_y, global_x, y, x, num_entries[idx], read_counts[rank][global_y][global_x], correct_nonzero_tiles);
								if(error_tiles++ > 5) abort();
							} else if(num_entries[idx] != 0) {
								correct_nonzero_tiles++;
							}
						}
					}
				});
			});
			queue.wait();
			fmt::print("Sanity check of per-rank counts for rank {} passed\n", rank);
		}
#endif

		print_delta_time("Compute sum across devices");

		// First do an allgather to get the individual counts from all ranks
		std::vector<uint32_t> num_entries_by_rank(global_grid_size.size() * num_ranks);
		MPI_Allgather(num_entries.data(), num_entries.size(), MPI_UINT32_T, num_entries_by_rank.data(), num_entries.size(), MPI_UINT32_T, MPI_COMM_WORLD);
		// if(rank == 0) fmt::print("Entries by rank: {}\n", fmt::join(num_entries_by_rank, ","));

		// Now compute global sum
		// TODO: We could also just compute it from num_entries_by_rank - should we?
		MPI_Allreduce(MPI_IN_PLACE, num_entries.data(), num_entries.size(), MPI_UINT32_T, MPI_SUM, MPI_COMM_WORLD);

// Sanity check: Compare the per-rank global counts with what we computed here
// This is exactly the same logic we used for ther per-rank counts above
#if 1
		{
			celerity::custom_task_geometry sanity_check_global_counts_geo;
			sanity_check_global_counts_geo.global_size = range_cast<3>(celerity::range<1>{num_ranks});
			std::vector<std::pair<box<3>, region<3>>> read_rank_global_counts_accesses;
			const auto chnk_box = box_cast<3>(box<1>(subrange<1>{(size_t)rank, 1}));
			sanity_check_global_counts_geo.assigned_chunks.push_back({chnk_box, rank, 0});
			const celerity::id<3> rank_offset = {rank, local_grid_offset[0], local_grid_offset[1]};
			const celerity::range<3> range = {1, local_grid_size[0], local_grid_size[1]};
			read_rank_global_counts_accesses.push_back({chnk_box, box<3>(subrange<3>{rank_offset, range})});
			celerity::expert_mapper read_rank_global_counts{read_rank_global_counts_accesses};

			queue.submit([&](celerity::handler& cgh) {
				celerity::accessor read_counts(per_rank_global_counts, cgh, read_rank_global_counts, celerity::read_only_host_task);
				celerity::debug::set_task_name(cgh, "sanity check rank global counts");
				cgh.assert_no_allocations(celerity::detail::allocation_scope::device);
				cgh.assert_no_data_movement(celerity::detail::data_movement_scope::inter_node);
				cgh.host_task(sanity_check_global_counts_geo, [=, &num_entries](celerity::partition<1> part) {
					if(part.get_subrange().offset[0] != rank) throw std::runtime_error("Partition offset is not the rank?");
					size_t correct_nonzero_tiles = 0;
					size_t error_tiles = 0;
					for(size_t y = 0; y < local_grid_size[0]; ++y) {
						for(size_t x = 0; x < local_grid_size[1]; ++x) {
							const auto global_x = local_grid_offset[1] + x;
							const auto global_y = local_grid_offset[0] + y;
							const auto idx = global_y * global_grid_size[1] + global_x;
							if(num_entries[idx] != read_counts[rank][global_y][global_x]) {
								CELERITY_CRITICAL("Rank {} has mismatch of global count for tile ({}, {}) (local coordinates {}, {}): {} (legacy) vs {} "
								                  "(global). {} tiles with non-zero "
								                  "count were correct so far.\n",
								    rank, global_y, global_x, y, x, num_entries[idx], read_counts[rank][global_y][global_x], correct_nonzero_tiles);
								if(error_tiles++ > 5) abort();
							} else if(num_entries[idx] != 0) {
								correct_nonzero_tiles++;
							}
						}
					}
				});
			});
			queue.wait();
			fmt::print("Sanity check of per-rank GLOBAL counts for rank {} passed\n", rank);
		}
#endif

// Sanity check: Are the global row sums correct?
#if 1
		{
			queue.submit([&](celerity::handler& cgh) {
				celerity::accessor read_global_row_sums(global_row_sums, cgh, celerity::access::all{}, celerity::read_only_host_task);
				cgh.host_task(celerity::on_master_node, [=]() {
					for(size_t i = 0; i < global_grid_size[0]; ++i) {
						uint32_t expected_sum = 0;
						for(size_t j = 0; j < global_grid_size[1]; ++j) {
							expected_sum += num_entries[i * global_grid_size[1] + j];
						}
						if(read_global_row_sums[i] != expected_sum) {
							CELERITY_CRITICAL(
							    "Incorrect global row sum for row {}: {} (received) vs {} (expected)\n", i, read_global_row_sums[i], expected_sum);
							if(i > 5) abort();
						}
					}
				});
			});
			queue.wait();
			if(rank == 0) { fmt::print("Sanity check of global row sums passed\n"); }
		}
#endif

		print_delta_time("Reduce across ranks");

		if(rank == 0) {
			// Compute statistics over tile fill rates
			const auto minmax = std::minmax_element(num_entries.begin(), num_entries.end());
			const auto sum = std::accumulate(num_entries.begin(), num_entries.end(), 0);
			fmt::print("POINTS PER TILE: Min: {}, Max: {}, Avg: {}\n", *minmax.first, *minmax.second, sum / num_entries.size());
		}

		print_delta_time("Compute stats");

		// Re-initialize for write kernel
		num_entries_scratch.fill(0);

		// Compute prefix sum
		// TODO: Should we do this on device using e.g. Thrust?
		std::vector<std::vector<uint32_t>> per_device_write_offsets(num_entries_per_device.size());
		for(auto& offsets : per_device_write_offsets) {
			offsets.reserve(global_grid_size.size());
		}
		for(size_t i = 0; i < global_grid_size.size(); ++i) {
			if(i > 0) { num_entries_cumulative[i] = num_entries[i - 1] + num_entries_cumulative[i - 1]; }
			// write_offsets_for_this_rank[i] = num_entries_cumulative[i];
			uint32_t rank_offset = num_entries_cumulative[i];
			for(int r = 0; r < rank; ++r) {
				// write_offsets_for_this_rank[i] += num_entries_by_rank[r * grid_size.size() + i];
				rank_offset += num_entries_by_rank[r * global_grid_size.size() + i];
			}
			for(size_t d = 0; d < num_entries_per_device.size(); ++d) {
				per_device_write_offsets[d].push_back(rank_offset);
				rank_offset += num_entries_per_device[d][i];
			}
		}

// Sanity check: Compare the per-rank cumulative counts with what we computed here
// This is exactly the same logic we used for ther per-rank counts and per-rank global counts above
#if 1
		{
			celerity::custom_task_geometry sanity_check_cumulative_counts_geo;
			std::vector<std::pair<box<3>, region<3>>> read_rank_cumulative_counts_accesses;
			const auto chnk_box = box_cast<3>(box<1>(subrange<1>{(size_t)rank, 1}));
			sanity_check_cumulative_counts_geo.assigned_chunks.push_back({chnk_box, rank, 0});
			const celerity::id<3> rank_offset = {rank, local_grid_offset[0], local_grid_offset[1]};
			const celerity::range<3> range = {1, local_grid_size[0], local_grid_size[1]};
			read_rank_cumulative_counts_accesses.push_back({chnk_box, box<3>(subrange<3>{rank_offset, range})});
			celerity::expert_mapper read_rank_cumulative_counts{read_rank_cumulative_counts_accesses};

			queue.submit([&](celerity::handler& cgh) {
				celerity::accessor read_counts(per_rank_cumulative_counts, cgh, read_rank_cumulative_counts, celerity::read_only_host_task);
				celerity::debug::set_task_name(cgh, "sanity check rank cumulative counts");
				cgh.assert_no_allocations(celerity::detail::allocation_scope::device);
				cgh.assert_no_data_movement(celerity::detail::data_movement_scope::inter_node);
				cgh.host_task(sanity_check_cumulative_counts_geo, [=, &num_entries_cumulative](celerity::partition<1> part) {
					if(part.get_subrange().offset[0] != rank) throw std::runtime_error("Partition offset is not the rank?");
					size_t correct_nonzero_tiles = 0;
					size_t error_tiles = 0;
					for(size_t y = 0; y < local_grid_size[0]; ++y) {
						for(size_t x = 0; x < local_grid_size[1]; ++x) {
							const auto global_x = local_grid_offset[1] + x;
							const auto global_y = local_grid_offset[0] + y;
							const auto idx = global_y * global_grid_size[1] + global_x;
							if(num_entries_cumulative[idx] != read_counts[rank][global_y][global_x]) {
								CELERITY_CRITICAL("Rank {} has mismatch of cumulative count for tile ({}, {}) (local coordinates {}, {}): {} (legacy) vs {} "
								                  "(rank). {} tiles with non-zero "
								                  "count were correct so far.\n",
								    rank, global_y, global_x, y, x, num_entries_cumulative[idx], read_counts[rank][global_y][global_x], correct_nonzero_tiles);
								if(error_tiles++ > 5) abort();
							} else if(num_entries_cumulative[idx] != 0) {
								correct_nonzero_tiles++;
							}
						}
					}
				});
			});
			queue.wait();
			fmt::print("Sanity check of per-rank CUMULATIVE counts for rank {} passed\n", rank);
		}
#endif

// Sanity check: Compare the per-device write offsets with what we computed here
#if 1
		{
			celerity::custom_task_geometry sanity_check_device_offsets_geo;
			std::vector<std::pair<box<3>, region<3>>> read_device_offsets_accesses;
			const auto chnk_box = box_cast<3>(box<1>(subrange<1>{(size_t)rank, 1}));
			sanity_check_device_offsets_geo.assigned_chunks.push_back({chnk_box, rank, 0});
			const celerity::id<3> rank_offset = {rank * num_devices, local_grid_offset[0], local_grid_offset[1]};
			const celerity::range<3> range = {num_devices, local_grid_size[0], local_grid_size[1]};
			read_device_offsets_accesses.push_back({chnk_box, box<3>(subrange<3>{rank_offset, range})});
			celerity::expert_mapper read_device_offsets{read_device_offsets_accesses};

			queue.submit([&](celerity::handler& cgh) {
				celerity::accessor read_offsets(per_device_write_offsets_buffer, cgh, read_device_offsets, celerity::read_only_host_task);
				celerity::debug::set_task_name(cgh, "sanity check per device write offsets");
				cgh.assert_no_allocations(celerity::detail::allocation_scope::device);
				cgh.assert_no_data_movement(celerity::detail::data_movement_scope::inter_node);
				cgh.host_task(sanity_check_device_offsets_geo, [=, &per_device_write_offsets](celerity::partition<1> part) {
					if(part.get_subrange().offset[0] != rank) throw std::runtime_error("Partition offset is not the rank?");
					size_t correct_nonzero_tiles = 0;
					size_t error_tiles = 0;
					for(size_t i = 0; i < num_devices; ++i) {
						for(size_t y = 0; y < local_grid_size[0]; ++y) {
							for(size_t x = 0; x < local_grid_size[1]; ++x) {
								const auto global_x = local_grid_offset[1] + x;
								const auto global_y = local_grid_offset[0] + y;
								const auto idx = global_y * global_grid_size[1] + global_x;
								if(per_device_write_offsets[i][idx] != read_offsets[rank * num_devices + i][global_y][global_x]) {
									CELERITY_CRITICAL("Rank {} has mismatch of per-device write offset for device {} and tile ({}, {}) (local coordinates {}, "
									                  "{}): {} (legacy) vs {} "
									                  "(rank). {} tiles with non-zero "
									                  "count were correct so far.\n",
									    rank, i, global_y, global_x, y, x, per_device_write_offsets[i][idx],
									    read_offsets[rank * num_devices + i][global_y][global_x], correct_nonzero_tiles);
									if(error_tiles++ > 5) abort();
								} else if(per_device_write_offsets[i][idx] != 0) {
									correct_nonzero_tiles++;
								}
							}
						}
					}
				});
			});
			queue.wait();
			fmt::print("Sanity check of per-device write offsets for rank {} passed\n", rank);
		}
#endif

		// if(rank == 0) fmt::print("Cumulative: {}\n", fmt::join(num_entries_cumulative, ","));
		// fmt::print("Rank {} cumulative: {}\n", rank, fmt::join(write_offsets_for_this_rank, ","));
		print_delta_time("Compute prefix sums");

		// TODO API: Since this should be a single shared buffer between all local chunks, we should only need to provide a single value
		// TODO API: Alternatively, consider providing a "broadcast" function (what about differently sized scratch buffers though?)
		num_entries_cumulative_scratch.set_data_from_host(std::vector<std::vector<uint32_t>>(num_devices, num_entries_cumulative));
		write_offsets_scratch.set_data_from_host(per_device_write_offsets);

		print_delta_time("Copy to device");

		uint32_t* my_counts = num_entries_by_rank.data() + rank * global_grid_size.size();
		const size_t sum_of_my_counts = std::accumulate(my_counts, my_counts + global_grid_size.size(), 0);
		fmt::print("Rank {} has {} points\n", rank, sum_of_my_counts);

		print_delta_time("Compute total count");

		// Now compute the set of contiguous ranges written by this rank
		// written_subranges = compute_written_subranges(rank, num_entries, num_entries_by_rank, num_entries_cumulative);
		written_subranges_per_device =
		    compute_written_subranges_per_device(rank, num_entries, num_entries_by_rank, num_entries_cumulative, num_entries_per_device);

		print_delta_time("Compute written subranges");

		// fmt::print("Rank {} writes to: {}\n", rank, fmt::join(written_subranges, ", "));
		celerity::detail::region<1> locally_written_region;
		for(size_t d = 0; d < num_entries_per_device.size(); ++d) {
			// fmt::print("Rank {} device {} writes to: {}\n", rank, d, fmt::join(written_subranges_per_device[d], ", "));
			fmt::print("Rank {} device {} writes to {} subranges\n", rank, d, written_subranges_per_device[d].size());
			for(const auto& sr : written_subranges_per_device[d]) {
				locally_written_region = region_union(locally_written_region, celerity::detail::box<1>(sr));
			}
		}
		// fmt::print("Rank {} writes to: {}\n", rank, locally_written_region);
		print_delta_time("Convert to region");

		// TODO: Make this a debug functionality of the expert mapper interface
		//		=> This is a more expensive variant of what we already do in the graph generators, which is required b/c we don't have
		//         write information for remote nodes.
		// TODO: Receive region instead of box for all_writes
		// TODO: Also do for local devices? Or not, because that would be caught by GGENs? (do we always check that, or only for debug builds?)
		const auto verify_written_regions = [&](const celerity::detail::box<1>& all_writes) {
			const auto local_box_vector = locally_written_region.get_boxes();
			std::vector<int> num_boxes_per_rank(num_ranks);
			num_boxes_per_rank[rank] = local_box_vector.size();
			MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, num_boxes_per_rank.data(), 1, MPI_INT, MPI_COMM_WORLD);
			std::accumulate(num_boxes_per_rank.begin(), num_boxes_per_rank.end(), 0);
			size_t total_num_boxes = 0;
			std::vector<int> displs(num_ranks);
			std::vector<int> recv_counts(num_ranks);
			for(size_t i = 0; i < num_ranks; ++i) {
				displs[i] = total_num_boxes * sizeof(celerity::detail::box<1>); // displacement is in elements (which is bytes)
				total_num_boxes += num_boxes_per_rank[i];
				recv_counts[i] = num_boxes_per_rank[i] * sizeof(celerity::detail::box<1>);
			}
			celerity::detail::box_vector<1> all_boxes(total_num_boxes);
			MPI_Allgatherv(local_box_vector.data(), local_box_vector.size() * sizeof(celerity::detail::box<1>), MPI_BYTE, all_boxes.data(), recv_counts.data(),
			    displs.data(), MPI_BYTE, MPI_COMM_WORLD);

			celerity::detail::region_builder<1> builder;
			size_t offset = 0;
			for(int r = 0; r < num_ranks; ++r) {
				if(r != rank) {
					for(int i = 0; i < num_boxes_per_rank[r]; ++i) {
						builder.add(all_boxes[offset + i]);
					}
				}
				offset += num_boxes_per_rank[r];
			}
			const auto remote_writes = std::move(builder).into_region();

			celerity::detail::region<1> global_region(std::move(all_boxes));
			if(global_region != all_writes) {
				celerity::detail::utils::panic("Actual written region union {} does not match declared union written region {}", global_region, all_writes);
			}

			const auto intersection = region_intersection(remote_writes, locally_written_region);
			if(!intersection.empty()) {
				CELERITY_CRITICAL("REMOTE WRITES {}", remote_writes);
				CELERITY_CRITICAL("LOCAL WRITES {}", locally_written_region);
				celerity::detail::utils::panic("Overlapping writes detected: {} is written both locally and by remote nodes", intersection);
			}
		};
		// TODO: Do this only when explicitly enabled via CLI arg
		verify_written_regions(celerity::detail::box<1>::full_range(points_input.get_range().size()));
	}

	print_delta_time("Subrange calculation", before_subrange_calculation);

	std::vector<std::pair<box<3>, region<3>>> per_chunk_accesses;
	// std::vector<subrange<3>> my_ranges;
	// std::transform(written_subranges.begin(), written_subranges.end(), std::back_inserter(my_ranges), [](const auto& sr) { return subrange_cast<3>(sr);
	// });
	for(size_t i = 0; i < chunks.size(); ++i) {
		// per_chunk_accesses.push_back({chnk, rank == i ? my_ranges : std::vector<subrange<3>>{}});
		if(i / num_devices == rank) {
			celerity::detail::box_vector<3> device_ranges;
			for(auto& sr : written_subranges_per_device[i % num_devices]) {
				device_ranges.push_back(subrange_cast<3>(sr));
			}
			per_chunk_accesses.push_back({chunks[i], region<3>{std::move(device_ranges)}});
		} else {
			per_chunk_accesses.push_back({chunks[i], {}}); // TODO: Having to do this is stupid
		}
	}
	celerity::expert_mapper tile_accesses{box_cast<3>(box<1>::full_range(tiles_storage.get_range())), per_chunk_accesses};

	queue.submit([&](celerity::handler& cgh) {
		celerity::accessor read_points(points_input, cgh, celerity::access::one_to_one{}, celerity::read_only);
		celerity::accessor write_tiles(tiles_storage, cgh, tile_accesses, celerity::write_only, celerity::no_init);
		celerity::scratch_accessor write_num_entries_scratch(num_entries_scratch /* TODO: write access */);
		celerity::scratch_accessor read_write_offsets_scratch(write_offsets_scratch /* TODO: read access */);
		celerity::debug::set_task_name(cgh, "write points");

		cgh.parallel_for(write_tiles_geometry, [=](celerity::id<1> id) {
			// TODO: Keep DRY with above
			const auto& p = read_points[id];
			// grid_size - 1: We divide the domain such the last tile contains the maximum value
			const auto tile_x = static_cast<uint32_t>((p.x - global_min.x) / global_extent.x * (global_grid_size[1] - 1));
			const auto tile_y = static_cast<uint32_t>((p.y - global_min.y) / global_extent.y * (global_grid_size[0] - 1));
			sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device> ref{write_num_entries_scratch[{tile_y, tile_x}]};
			const uint32_t offset = ref.fetch_add(uint32_t(1));
			write_tiles[read_write_offsets_scratch[{tile_y, tile_x}] + offset] = p;
		});
	});

	print_delta_time("Writing points");

	// Copy *global* counts back to device for shape factor kernel (currently we again store local counts)
	// TODO: Should we use two different buffers for this..? It's kinda confusing (this copy missing was a bug!)
	//       => Or just std::move it to a new name?
	// TODO: This is now again the same value for all chunks, so we should either use a different scratch buffer w/ node scope, or have a broadcast function
	num_entries_scratch.set_data_from_host(std::vector<std::vector<unsigned int>>(num_devices, num_entries));

	// =============== Compute shape factors ===============
	// For this we first have to split up the tile storage into roughly equally sized parts, along tile boundaries (i.e., only distribute whole tiles)
	// We also have to compute the neighborhood offsets, which is easy because we already know the start and end offset of each chunk

	celerity::buffer<sycl::double3, 1> shape_factors(tiles_storage.get_range().size());
	celerity::debug::set_buffer_name(shape_factors, "shape factors");

	// IMPORTANT: We have to compute the neighborhood reads for all ranks on every rank!

	// FIXME: Don't compute per-device chunks for remote nodes
	/*const*/ auto shape_factor_geometry = compute_task_geometry(num_ranks * num_devices, points_input.get_range().size(), num_entries_cumulative);
	print_delta_time("Shape factor geometry computation");
	// FIXME HACK: Have to rewrite assignments b/c compute_task_geometry is not device-aware yet
	for(size_t i = 0; i < shape_factor_geometry.assigned_chunks.size(); ++i) {
		shape_factor_geometry.assigned_chunks[i].nid = i / num_devices;
		shape_factor_geometry.assigned_chunks[i].did = i % num_devices;
	}

	/////////// EXTREME HACK //////////////
	// TODO API: Figure this out: How do we move scratch buffers between tasks with different geometries?
	// => ACTUALLY: In this case it could simply be "node scope" buffers, so that would work fine...
	celerity::scratch_buffer<uint32_t, 2> num_entries_scratch_2(shape_factor_geometry, global_grid_size);
	celerity::scratch_buffer<uint32_t, 2> num_entries_cumulative_scratch_2(shape_factor_geometry, global_grid_size);
	num_entries_scratch_2.set_data_from_host(num_entries_scratch.get_data_on_host());
	num_entries_cumulative_scratch_2.set_data_from_host(num_entries_cumulative_scratch.get_data_on_host());
	/////////// EXTREME HACK //////////////

	const auto per_chunk_neighborhood_reads =
	    compute_neighborhood_reads_2d(shape_factor_geometry, global_grid_size, points_input.get_range().size(), num_entries, num_entries_cumulative);
	print_delta_time("Neighborhood reads computation");

	if(shape_factor_geometry.assigned_chunks.size() != (size_t)num_ranks * num_devices) {
		CELERITY_CRITICAL("Failed to create enough chunks for all ranks");
		return 1;
	}

	if(rank == 0) {
		std::vector<box<3>> chunks;
		std::transform(shape_factor_geometry.assigned_chunks.begin(), shape_factor_geometry.assigned_chunks.end(), std::back_inserter(chunks),
		    [](const auto& cg) { return cg.box; });
		fmt::print("Shape factor kernel chunks: {}\n", fmt::join(chunks, ", "));
		std::vector<subrange<3>> neighborhoods;
		std::transform(
		    per_chunk_neighborhood_reads.begin(), per_chunk_neighborhood_reads.end(), std::back_inserter(neighborhoods), [](const auto& n) { return n[0]; });
		fmt::print("Accessed subranges: {}\n", fmt::join(neighborhoods, ", "));
	}

	std::vector<std::pair<box<3>, region<3>>> shape_factors_per_chunk_accesses;
	for(size_t i = 0; i < shape_factor_geometry.assigned_chunks.size(); ++i) {
		const auto& [box, _, _2] = shape_factor_geometry.assigned_chunks[i];
		// FIXME: compute_neighborhood_reads_2d should just return a region
		celerity::detail::box_vector<3> read_boxes;
		for(const auto& sr : per_chunk_neighborhood_reads[i]) {
			read_boxes.push_back(sr);
		}
		shape_factors_per_chunk_accesses.push_back({box, region<3>{std::move(read_boxes)}});
	}

	celerity::expert_mapper read_tile_neighborhood(
	    celerity::detail::subrange_cast<3>(celerity::subrange<1>{{}, points_input.get_range().size()}), shape_factors_per_chunk_accesses);

	queue.submit([&](celerity::handler& cgh) {
		celerity::accessor read_tiles(tiles_storage, cgh, read_tile_neighborhood, celerity::read_only);
		celerity::accessor write_shape_factors(shape_factors, cgh, celerity::access::one_to_one{}, celerity::write_only, celerity::no_init);
		celerity::scratch_accessor read_num_entries_scratch(num_entries_scratch_2);
		celerity::scratch_accessor read_num_entries_cumulative_scratch(num_entries_cumulative_scratch_2);
		celerity::debug::set_task_name(cgh, "compute shape factors");

		cgh.parallel_for(shape_factor_geometry, [=, total_size = points_input.get_range().size()](celerity::id<1> id) {
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

			const auto& access_point = [=](const tilebuffer_item& itm) {
				// TODO: Wait, isn't that just the kernel's thread id?
				// const auto linear_idx = celerity::detail::get_linear_index(grid_size, itm.slot);
				return read_tiles[read_num_entries_cumulative_scratch[itm.slot] + itm.index];
			};

			const auto get_slot = [=](const celerity::id<2>& slot) {
				// Roll own span until we can compile with C++20 (Clang 14 crashes if I try)
				struct bootleg_span {
					const umuguc_point3d* data;
					size_t count;
					const umuguc_point3d* begin() const { return data; }
					const umuguc_point3d* end() const { return data + count; }
				};

				if(read_num_entries_scratch[slot] == 0) {
					// Empty tile
					return bootleg_span{nullptr, 0};
				}
				const auto start = read_num_entries_cumulative_scratch[slot];
				return bootleg_span{&read_tiles[start], read_num_entries_scratch[slot]};
			};

			const auto item = get_current_item(id, total_size, global_grid_size, &read_num_entries_cumulative_scratch[celerity::id<2>{0, 0}]);
			if(!item.within_bounds) { return; }

			const auto p = access_point(item);

			std::array<std::array<double, 3>, 3> matrix{{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}};

			const int local_search_radius = 1; // NOTE: We also assume this for the neighborhood range calculation!
			const double radius = tile_size;   // TODO: Decouple these two parameters. Needs adjustment to compute_neighborhood_reads_2d
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

			sycl::double3 sf;
			const size_t write_offset = read_num_entries_cumulative_scratch[item.slot] + item.index;
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

	print_delta_time("Shape factor computation");

	if(write_output) {
		queue.submit([&](celerity::handler& cgh) {
			celerity::accessor read_points(tiles_storage, cgh, celerity::access::all{}, celerity::read_only_host_task);
			celerity::accessor read_shape_factors(shape_factors, cgh, celerity::access::all{}, celerity::read_only_host_task);

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

							header hdr = {static_cast<int>(x), static_cast<int>(y), static_cast<int>(num_entries[y * global_grid_size[1] + x])};
							file.write(reinterpret_cast<const char*>(&hdr), sizeof(header));
							file.write(reinterpret_cast<const char*>(&data[num_entries_cumulative[y * global_grid_size[1] + x]]),
							    num_entries[y * global_grid_size[1] + x] * sizeof(*data));
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
