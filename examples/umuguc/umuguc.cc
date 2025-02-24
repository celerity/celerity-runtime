#include <vector>

#include <celerity.h>
#include <fmt/ranges.h>

// TODO: A proper task geometry API should also offer splitting utilities
#include "scratch_buffer.h"
#include "split.h"
#include "tilebuffer_utils.h"

#include "eigen_decomposition.hpp"

using clk = std::chrono::steady_clock;

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
		fmt::print(stderr, "Usage: {} <input_file> [--write-output]\n", argv[0]);
		exit(EXIT_FAILURE);
	};
	if(argc < 2) usage();
	const auto filename = argv[1];
	consume_args(1, 1, &argc, &argv);
	const bool write_output = get_flag("--write-output", &argc, &argv);
	const double tile_size = get_arg<double>("--tile-size", &argc, &argv).value_or(1.0);
	if(argc != 1) usage();

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

	if(points.size() > std::numeric_limits<uint32_t>::max()) {
		fmt::print(stderr, "Too many points for 32 bit counting. Time to upgrade!: {}\n", points.size());
		return 1;
	}

	umuguc_point3d min = {std::numeric_limits<double>::max(), std::numeric_limits<double>::max(), std::numeric_limits<double>::max()};
	umuguc_point3d max = {std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest()};
	for(const auto& p : points) {
		min.x = std::min(min.x, p.x);
		min.y = std::min(min.y, p.y);
		min.z = std::min(min.z, p.z);
		max.x = std::max(max.x, p.x);
		max.y = std::max(max.y, p.y);
		max.z = std::max(max.z, p.z);
	}
	fmt::print("Read {} points ({:.1f} GiB)\n", points.size(), points.size() * sizeof(umuguc_point3d) / 1024.0 / 1024.0 / 1024.0);
	fmt::print("Min: ({:.1f}, {:.1f}, {:.1f}), Max: ({:.1f}, {:.1f}, {:.1f}). Extent: ({:.1f}, {:.1f}, {:.1f})\n", min.x, min.y, min.z, max.x, max.y, max.z,
	    max.x - min.x, max.y - min.y, max.z - min.z);

	const celerity::range<2> grid_size = {static_cast<uint32_t>((max.y - min.y) / tile_size), static_cast<uint32_t>((max.x - min.x) / tile_size)};
	if(grid_size.size() == 0) {
		CELERITY_CRITICAL("Grid size is {}x{} - try smaller tile size", grid_size[1], grid_size[0]);
		exit(1);
	}
	fmt::print(
	    "Using buffer size: {}x{}, tile size: {:.1f}x{:.1f}\n", grid_size[1], grid_size[0], (max.x - min.x) / grid_size[1], (max.y - min.y) / grid_size[0]);

	celerity::queue queue;

	celerity::buffer<umuguc_point3d, 1> points_input(points.data(), points.size());
	celerity::debug::set_buffer_name(points_input, "points input");
	celerity::buffer<umuguc_point3d, 1> tiles_storage(points.size());
	celerity::debug::set_buffer_name(tiles_storage, "tiles storage");

	auto& rt = celerity::detail::runtime::get_instance();
	const auto rank = rt.NOCOMMIT_get_local_nid();
	const auto num_ranks = rt.NOCOMMIT_get_num_nodes();
	// // NOCOMMIT: We assume a uniform number of devices per node here
	// //           => Ideally we should simply not create per-device chunks for remote nodes
	const size_t num_devices = rt.NOCOMMIT_get_num_local_devices();

	auto print_delta_time = [&, previous = clk::now()](std::string_view description, std::optional<clk::time_point> against = std::nullopt) mutable {
		queue.wait(celerity::experimental::barrier);
		const auto now = clk::now();
		auto dt = now - previous;
		if(against.has_value()) { dt = now - against.value(); }
		if(rank == 0) { fmt::print("{}: {} ms\n", description, std::chrono::duration_cast<std::chrono::milliseconds>(dt).count()); }
		previous = now;
	};

	// TODO API: No need to split remote chunks into per-device chunks
	auto chunks =
	    celerity::detail::split_1d(box_cast<3>(celerity::detail::box<1>{0, points_input.get_range()}), celerity::detail::ones, num_ranks * num_devices);
	celerity::custom_task_geometry write_tiles_geometry;
	write_tiles_geometry.global_size = range_cast<3>(points_input.get_range());
	for(size_t i = 0; i < chunks.size(); ++i) {
		write_tiles_geometry.assigned_chunks.push_back({chunks[i].get_subrange(), (i / num_devices), (i % num_devices)});
	}

	celerity::scratch_buffer<uint32_t, 2> num_entries_scratch(write_tiles_geometry, grid_size);
	celerity::scratch_buffer<uint32_t, 2> num_entries_cumulative_scratch(write_tiles_geometry, grid_size); // TODO: This should be node scope
	// Like num_entries_cumulative_scratch, but includes per-chunk write offset within each tile (= sum of all ranks and chunks that write before it)
	celerity::scratch_buffer<uint32_t, 2> write_offsets_scratch(write_tiles_geometry, grid_size);
	num_entries_scratch.fill(0); // The other two are initialized from host data later on

	// WARMUP. TODO: How do we want to handle this in general for this benchmark? Each phase needs new allocations, but we cannot anticipate them in IDAG.
	queue.submit([&](celerity::handler& cgh) {
		celerity::accessor read_points(points_input, cgh, celerity::access::one_to_one{}, celerity::read_only);
		// TODO API: Should this just be a normal accessor? Or simply a pointer?
		celerity::scratch_accessor write_num_entries_scratch(num_entries_scratch /* TODO write access */);
		celerity::debug::set_task_name(cgh, "count points");

		// TODO API: How do we launch nd-range kernels with custom geometries? Required for optimized count/write
		cgh.parallel_for(write_tiles_geometry, [=](celerity::id<1> id) {
			const auto& p = read_points[id];
			// grid_size - 1: We divide the domain such the last tile contains the maximum value
			const auto tile_x = static_cast<uint32_t>((p.x - min.x) / (max.x - min.x) * (grid_size[1] - 1));
			const auto tile_y = static_cast<uint32_t>((p.y - min.y) / (max.y - min.y) * (grid_size[0] - 1));
			// Count points in per-chunk scratch buffer
			sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device> ref{write_num_entries_scratch[{tile_y, tile_x}]};
			ref++;
		});
	});

	queue.wait(celerity::experimental::barrier);
	const auto before_count_points = std::chrono::steady_clock::now();
	num_entries_scratch.fill(0); // The other two are initialized from host data later on

	queue.submit([&](celerity::handler& cgh) {
		celerity::accessor read_points(points_input, cgh, celerity::access::one_to_one{}, celerity::read_only);
		// TODO API: Should this just be a normal accessor? Or simply a pointer?
		celerity::scratch_accessor write_num_entries_scratch(num_entries_scratch /* TODO write access */);
		celerity::debug::set_task_name(cgh, "count points");

		// TODO API: How do we launch nd-range kernels with custom geometries? Required for optimized count/write
		cgh.parallel_for(write_tiles_geometry, [=](celerity::id<1> id) {
			const auto& p = read_points[id];
			// grid_size - 1: We divide the domain such the last tile contains the maximum value
			const auto tile_x = static_cast<uint32_t>((p.x - min.x) / (max.x - min.x) * (grid_size[1] - 1));
			const auto tile_y = static_cast<uint32_t>((p.y - min.y) / (max.y - min.y) * (grid_size[0] - 1));
			// Count points in per-chunk scratch buffer
			sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device> ref{write_num_entries_scratch[{tile_y, tile_x}]};
			ref++;
		});
	});

	print_delta_time("Counting points", before_count_points);

	// Add up counts across ranks
	// std::vector<celerity::subrange<1>> written_subranges;
	std::vector<std::vector<celerity::subrange<1>>> written_subranges_per_device;
	std::vector<uint32_t> num_entries(grid_size.size());
	std::vector<uint32_t> num_entries_cumulative(grid_size.size());
	{
		MPI_Barrier(MPI_COMM_WORLD);

		// Get data from per-GPU scratch buffer on this node
		auto num_entries_per_device = num_entries_scratch.get_data_on_host();
		// Compute total sum across all nodes
		for(const auto& data : num_entries_per_device) {
			for(size_t i = 0; i < grid_size.size(); ++i) {
				num_entries[i] += data[i];
			}
		}

		// First do an allgather to get the individual counts from all ranks
		std::vector<uint32_t> num_entries_by_rank(grid_size.size() * num_ranks);
		MPI_Allgather(num_entries.data(), num_entries.size(), MPI_UINT32_T, num_entries_by_rank.data(), num_entries.size(), MPI_UINT32_T, MPI_COMM_WORLD);
		// if(rank == 0) fmt::print("Entries by rank: {}\n", fmt::join(num_entries_by_rank, ","));

		// Now compute global sum
		// TODO: We could also just compute it from num_entries_by_rank - should we?
		MPI_Allreduce(MPI_IN_PLACE, num_entries.data(), num_entries.size(), MPI_UINT32_T, MPI_SUM, MPI_COMM_WORLD);
		// for(size_t i = 0; i < grid_size[0]; ++i) {
		// 	for(size_t j = 0; j < grid_size[1]; ++j) {
		// 		fmt::print("Rank {} {},{}: {}\n", rank, j, i, num_entries[i * grid_size[0] + j]);
		// 	}
		// }

		if(rank == 0) {
			// Compute statistics over tile fill rates
			const auto minmax = std::minmax_element(num_entries.begin(), num_entries.end());
			const auto sum = std::accumulate(num_entries.begin(), num_entries.end(), 0);
			fmt::print("POINTS PER TILE: Min: {}, Max: {}, Avg: {}\n", *minmax.first, *minmax.second, sum / num_entries.size());
		}

		// Re-initialize for write kernel
		num_entries_scratch.fill(0);

		// Compute prefix sum
		// TODO: Should we do this on device using e.g. Thrust?
		std::vector<std::vector<uint32_t>> per_device_write_offsets(num_entries_per_device.size());
		for(auto& offsets : per_device_write_offsets) {
			offsets.reserve(grid_size.size());
		}
		for(size_t i = 0; i < grid_size.size(); ++i) {
			if(i > 0) { num_entries_cumulative[i] = num_entries[i - 1] + num_entries_cumulative[i - 1]; }
			// write_offsets_for_this_rank[i] = num_entries_cumulative[i];
			uint32_t rank_offset = num_entries_cumulative[i];
			for(int r = 0; r < rank; ++r) {
				// write_offsets_for_this_rank[i] += num_entries_by_rank[r * grid_size.size() + i];
				rank_offset += num_entries_by_rank[r * grid_size.size() + i];
			}
			for(size_t d = 0; d < num_entries_per_device.size(); ++d) {
				per_device_write_offsets[d].push_back(rank_offset);
				rank_offset += num_entries_per_device[d][i];
			}
		}
		// if(rank == 0) fmt::print("Cumulative: {}\n", fmt::join(num_entries_cumulative, ","));
		// fmt::print("Rank {} cumulative: {}\n", rank, fmt::join(write_offsets_for_this_rank, ","));

		// TODO API: Since this should be a single shared buffer between all local chunks, we should only need to provide a single value
		// TODO API: Alternatively, consider providing a "broadcast" function (what about differently sized scratch buffers though?)
		num_entries_cumulative_scratch.set_data_from_host(std::vector<std::vector<uint32_t>>(num_devices, num_entries_cumulative));
		write_offsets_scratch.set_data_from_host(per_device_write_offsets);

		uint32_t* my_counts = num_entries_by_rank.data() + rank * grid_size.size();
		const size_t sum_of_my_counts = std::accumulate(my_counts, my_counts + grid_size.size(), 0);
		fmt::print("Rank {} has {} points\n", rank, sum_of_my_counts);

		// Now compute the set of contiguous ranges written by this rank
		// written_subranges = compute_written_subranges(rank, num_entries, num_entries_by_rank, num_entries_cumulative);
		written_subranges_per_device =
		    compute_written_subranges_per_device(rank, num_entries, num_entries_by_rank, num_entries_cumulative, num_entries_per_device);

		// fmt::print("Rank {} writes to: {}\n", rank, fmt::join(written_subranges, ", "));
		celerity::detail::region<1> locally_written_region;
		for(size_t d = 0; d < num_entries_per_device.size(); ++d) {
			// fmt::print("Rank {} device {} writes to: {}\n", rank, d, fmt::join(written_subranges_per_device[d], ", "));
			for(const auto& sr : written_subranges_per_device[d]) {
				locally_written_region = region_union(locally_written_region, celerity::detail::box<1>(sr));
			}
		}
		// fmt::print("Rank {} writes to: {}\n", rank, locally_written_region);

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
		verify_written_regions(celerity::detail::box<1>::full_range(points.size()));
	}

	print_delta_time("Subrange calculation");

	using celerity::subrange;
	using celerity::detail::box;
	using celerity::detail::region;
	using celerity::detail::subrange_cast;

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
			per_chunk_accesses.push_back({chunks[i], {}}); // TODO: Do we even need to do this? (Ideally not!)
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
			const auto tile_x = static_cast<uint32_t>((p.x - min.x) / (max.x - min.x) * (grid_size[1] - 1));
			const auto tile_y = static_cast<uint32_t>((p.y - min.y) / (max.y - min.y) * (grid_size[0] - 1));
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
	/*const*/ auto shape_factor_geometry = compute_task_geometry(num_ranks * num_devices, points.size(), num_entries_cumulative);
	print_delta_time("Shape factor geometry computation");
	// FIXME HACK: Have to rewrite assignments b/c compute_task_geometry is not device-aware yet
	for(size_t i = 0; i < shape_factor_geometry.assigned_chunks.size(); ++i) {
		shape_factor_geometry.assigned_chunks[i].nid = i / num_devices;
		shape_factor_geometry.assigned_chunks[i].did = i % num_devices;
	}

	/////////// EXTREME HACK //////////////
	// TODO API: Figure this out: How do we move scratch buffers between tasks with different geometries?
	// => ACTUALLY: In this case it could simply be "node scope" buffers, so that would work fine...
	celerity::scratch_buffer<uint32_t, 2> num_entries_scratch_2(shape_factor_geometry, grid_size);
	celerity::scratch_buffer<uint32_t, 2> num_entries_cumulative_scratch_2(shape_factor_geometry, grid_size);
	num_entries_scratch_2.set_data_from_host(num_entries_scratch.get_data_on_host());
	num_entries_cumulative_scratch_2.set_data_from_host(num_entries_cumulative_scratch.get_data_on_host());
	/////////// EXTREME HACK //////////////

	const auto per_chunk_neighborhood_reads =
	    compute_neighborhood_reads_2d(shape_factor_geometry, grid_size, points.size(), num_entries, num_entries_cumulative);
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
	    celerity::detail::subrange_cast<3>(celerity::subrange<1>{{}, points.size()}), shape_factors_per_chunk_accesses);

	queue.submit([&](celerity::handler& cgh) {
		celerity::accessor read_tiles(tiles_storage, cgh, read_tile_neighborhood, celerity::read_only);
		celerity::accessor write_shape_factors(shape_factors, cgh, celerity::access::one_to_one{}, celerity::write_only, celerity::no_init);
		celerity::scratch_accessor read_num_entries_scratch(num_entries_scratch_2);
		celerity::scratch_accessor read_num_entries_cumulative_scratch(num_entries_cumulative_scratch_2);
		celerity::debug::set_task_name(cgh, "compute shape factors");

		cgh.parallel_for(shape_factor_geometry, [=, total_size = points.size()](celerity::id<1> id) {
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

			const auto item = get_current_item(id, total_size, grid_size, &read_num_entries_cumulative_scratch[celerity::id<2>{0, 0}]);
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
					if(x < 0 || y < 0 || uint32_t(x) >= grid_size[1] || uint32_t(y) >= grid_size[0]) continue;

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

			cgh.host_task(celerity::on_master_node, [=]() {
				CELERITY_INFO("Writing output to disk...");

				const auto write_to_file = [&](const std::string& filename, const auto* data) {
					std::ofstream file(filename, std::ios::binary);
					for(size_t y = 0; y < grid_size[0]; ++y) {
						for(size_t x = 0; x < grid_size[1]; ++x) {
							struct header {
								int x, y, count;
							};

							header hdr = {static_cast<int>(x), static_cast<int>(y), static_cast<int>(num_entries[y * grid_size[1] + x])};
							file.write(reinterpret_cast<const char*>(&hdr), sizeof(header));
							file.write(reinterpret_cast<const char*>(&data[num_entries_cumulative[y * grid_size[1] + x]]),
							    num_entries[y * grid_size[1] + x] * sizeof(*data));
						}
					}
				};

				// since sycl::double3 is padded to 4 * sizeof(double), we have to pack it first
				std::vector<umuguc_point3d> packed_shape_factors(points.size());
				for(size_t i = 0; i < points.size(); ++i) {
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

	return 0;
}
