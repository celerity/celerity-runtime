#include "fmt/ranges.h"
#include <cstdio>

#include <celerity.h>
#include <geometry_builder.h>

#if !defined(NDEBUG) || CELERITY_SYCL_IS_SIMSYCL
const size_t default_mat_size = 128;
#else
const size_t default_mat_size = 1024;
#endif

template <typename T>
void set_identity(celerity::queue queue, celerity::buffer<T, 2> mat, bool reverse) {
	queue.submit([&](celerity::handler& cgh) {
		celerity::accessor dw{mat, cgh, celerity::access::one_to_one{}, celerity::write_only, celerity::no_init};
		const auto range = mat.get_range();

		celerity::debug::set_task_name(cgh, "set identity");
		celerity::experimental::hint(cgh, celerity::experimental::hints::split_2d{});
		cgh.parallel_for<class set_identity_kernel>(range, [=](celerity::item<2> item) {
			if(!reverse) {
				dw[item] = item[0] == item[1];
			} else {
				dw[item] = item[0] == (range[1] - item[1] - 1);
			}
		});
	});
}

// Fill matrix with something that is not all zeroes but still easy to verify.
template <typename T>
void fill_with_range(celerity::queue queue, celerity::buffer<T, 2> mat, const int min, const int max) {
	queue.submit([&](celerity::handler& cgh) {
		celerity::accessor dw{mat, cgh, celerity::access::one_to_one{}, celerity::write_only, celerity::no_init};
		const auto range = mat.get_range();

		celerity::debug::set_task_name(cgh, "fill with range");
		celerity::experimental::hint(cgh, celerity::experimental::hints::split_2d{});
		cgh.parallel_for<class set_identity_kernel>(range, [=](celerity::item<2> item) { dw[item] = item.get_linear_id() % (max - min) + min; });
	});
}

template <typename T>
void set_zero(celerity::queue queue, celerity::buffer<T, 2> mat) {
	queue.submit([&](celerity::handler& cgh) {
		celerity::accessor dw{mat, cgh, celerity::access::one_to_one{}, celerity::write_only, celerity::no_init};
		celerity::debug::set_task_name(cgh, "set zero");
		celerity::experimental::hint(cgh, celerity::experimental::hints::split_2d{});
		cgh.parallel_for(mat.get_range(), [=](celerity::item<2> item) { dw[item] = T{0}; });
	});
}


template <typename T>
void multiply(celerity::queue queue, celerity::buffer<T, 2> mat_a, celerity::buffer<T, 2> mat_b, celerity::buffer<T, 2> mat_c) {
	queue.submit([&](celerity::handler& cgh) {
#define USE_REGISTER_TILING 0

#if USE_REGISTER_TILING
		// FIXME: Multi-node / device NYI
		celerity::accessor a{mat_a, cgh, celerity::access::all(), celerity::read_only};
		celerity::accessor b{mat_b, cgh, celerity::access::all(), celerity::read_only};
		celerity::accessor c{mat_c, cgh, celerity::access::all{}, celerity::write_only, celerity::no_init};
#else
		celerity::accessor a{mat_a, cgh, celerity::access::slice<2>(1), celerity::read_only};
		celerity::accessor b{mat_b, cgh, celerity::access::slice<2>(0), celerity::read_only};
		celerity::accessor c{mat_c, cgh, celerity::access::one_to_one{}, celerity::write_only, celerity::no_init};
#endif

#if USE_REGISTER_TILING
		constexpr int TILE_SIZE = 16;
		constexpr int THREAD_TILE_SIZE = 4;
		const int group_size = TILE_SIZE;
		celerity::local_accessor<T, 2> scratch_a{{group_size * THREAD_TILE_SIZE, group_size * THREAD_TILE_SIZE}, cgh};
		celerity::local_accessor<T, 2> scratch_b{{group_size * THREAD_TILE_SIZE, group_size * THREAD_TILE_SIZE}, cgh};
#else
		const size_t group_size = 16;
		celerity::local_accessor<T, 2> scratch_a{{group_size, group_size}, cgh};
		celerity::local_accessor<T, 2> scratch_b{{group_size, group_size}, cgh};
#endif

		celerity::debug::set_task_name(cgh, "matrix multiplication");
		const size_t mat_size = mat_c.get_range()[0];
#if USE_REGISTER_TILING
		cgh.parallel_for(celerity::nd_range<2>{mat_c.get_range() / THREAD_TILE_SIZE, {group_size, group_size}}, [=](celerity::nd_item<2> item) {
			T sums[THREAD_TILE_SIZE * THREAD_TILE_SIZE] = {0};
			const auto lid = item.get_local_id();
			const auto grp = item.get_group();
			for(size_t K = 0; K < mat_size; K += TILE_SIZE * THREAD_TILE_SIZE) {
				for(int i = 0; i < THREAD_TILE_SIZE; ++i) {
					for(int j = 0; j < THREAD_TILE_SIZE; ++j) {
						scratch_a[lid[0] * THREAD_TILE_SIZE + i][lid[1] * THREAD_TILE_SIZE + j] =
						    a[{(grp[0] * group_size * THREAD_TILE_SIZE + lid[0] * THREAD_TILE_SIZE + i), K + lid[1] * THREAD_TILE_SIZE + j}];
						scratch_b[lid[0] * THREAD_TILE_SIZE + i][lid[1] * THREAD_TILE_SIZE + j] =
						    b[{(K + lid[0] * THREAD_TILE_SIZE + i), (grp[1] * group_size * THREAD_TILE_SIZE + lid[1] * THREAD_TILE_SIZE + j)}];
					}
				}
				celerity::group_barrier(item.get_group());
				// TODO: Verify that this is correct with two non-identity matrices!!
				for(size_t k = 0; k < TILE_SIZE * THREAD_TILE_SIZE; ++k) {
					for(int i = 0; i < THREAD_TILE_SIZE; ++i) {
						for(int j = 0; j < THREAD_TILE_SIZE; ++j) {
							sums[i * THREAD_TILE_SIZE + j] += scratch_a[lid[0] * THREAD_TILE_SIZE + i][k] * scratch_b[k][lid[1] * THREAD_TILE_SIZE + j];
						}
					}
				}
				celerity::group_barrier(item.get_group());
			}

			for(int i = 0; i < THREAD_TILE_SIZE; ++i) {
				for(int j = 0; j < THREAD_TILE_SIZE; ++j) {
					c[{(grp[0] * group_size * THREAD_TILE_SIZE + lid[0] * THREAD_TILE_SIZE + i),
					    +(grp[1] * group_size * THREAD_TILE_SIZE + lid[1] * THREAD_TILE_SIZE + j)}] = sums[i * THREAD_TILE_SIZE + j];
				}
			}
		});
#else
		cgh.parallel_for(celerity::nd_range<2>{mat_c.get_range(), {group_size, group_size}}, [=](celerity::nd_item<2> item) {
			T sum{};
			const auto lid = item.get_local_id();
			for(size_t j = 0; j < mat_size; j += group_size) {
				scratch_a[lid] = a[item.get_group(0) * group_size + lid[0]][j + lid[1]];
				scratch_b[lid] = b[j + lid[0]][item.get_group(1) * group_size + lid[1]];
				celerity::group_barrier(item.get_group());

				for(size_t k = 0; k < group_size; ++k) {
					const auto a_ik = scratch_a[lid[0]][k];
					const auto b_kj = scratch_b[k][lid[1]];
					sum += a_ik * b_kj;
				}
				celerity::group_barrier(item.get_group());
			}
			c[item.get_global_id()] = sum;
		});
#endif
	});
}

// TODO: Naming
template <int Dims>
class grid_data_requirements {
  public:
	grid_data_requirements(celerity::grid_geometry<Dims> geo) : m_geo(std::move(geo)) { m_requirements.resize(m_geo.get_grid().get_cells().size()); }

	// TODO: Maintain somewhat imprecise terminology from range mappers? Or call "identity"?
	void add_one_to_one(celerity::cartesian_grid<Dims> grid) {
		if(grid.get_grid_size() != m_geo.get_grid().get_grid_size()) { throw std::runtime_error("Grid size mismatch"); }
		for(size_t i = 0; i < grid.get_cells().size(); ++i) {
			const auto& cell = grid.get_cells()[i];
			m_requirements[i] = region_union(m_requirements[i], box_cast<3>(cell.box));
		}
	}

	// TODO: Add region overload
	void add(const celerity::id<2>& cell, celerity::detail::box<Dims> box) {
		const auto linear_idx = cell[0] * m_geo.get_grid().get_grid_size()[1] + cell[1];
		if(linear_idx >= m_requirements.size()) { throw std::runtime_error("Cell index out of bounds"); }
		m_requirements[linear_idx] = region_union(m_requirements[linear_idx], box_cast<3>(box));
	}

	operator celerity::expert_mapper() const {
		// TODO: We should have an option for the user to provide this, so we don't have to compute it
		// 			=> Or in the cast of partial materialization, we CANNOT compute it
		celerity::detail::region<3> union_access;
		std::vector<std::pair<celerity::detail::box<3>, celerity::detail::region<3>>> per_chunk_accesses;
		const auto& cells = m_geo.get_grid().get_cells();
		for(size_t i = 0; i < m_requirements.size(); ++i) {
			const auto& req = m_requirements[i];
			union_access = region_union(union_access, req);
			per_chunk_accesses.push_back({box_cast<3>(cells[i].box), req});
		}
		return celerity::expert_mapper{union_access, per_chunk_accesses};
	}

  private:
	celerity::grid_geometry<Dims> m_geo;
	std::vector<celerity::detail::region<3>> m_requirements;
};

template <typename T>
void multiply_blocked(celerity::queue queue, celerity::buffer<T, 2> mat_a, celerity::buffer<T, 2> mat_b, celerity::buffer<T, 2> mat_c) {
	const size_t group_size = 16;

	celerity::cartesian_grid<2> matrix_partition(celerity::detail::box<2>::full_range(mat_c.get_range()));
	// TODO: Obviously we need a better API to split across all nodes
	matrix_partition.split(celerity::detail::runtime::get_instance().NOCOMMIT_get_num_nodes(), {group_size, group_size});

	// TODO: Do we even need a builder in this case?
	// => HERE we could create a 1D geometry from a 2D partition. The key is that we maintain the coordinates!!
	// celerity::geometry_builder<2> gb2{output_partition};
	// TODO: Option to do partial materialization here
	// gb2.assign(); // TODO: ? Policy - round robin, grouped round robin (?), ...

	// THEN: Get all (locally materialized) chunks AND THEIR COORDINATES (how? only works if geometry is based on partition)

	celerity::grid_geometry geo(matrix_partition, celerity::range<2>{group_size, group_size});

	// The key thing we want to achieve is that we can iterate over all chunks and easily select their required data based on coordinates

	grid_data_requirements writes{geo};
	writes.add_one_to_one(matrix_partition);

	////////////////////

	// TODO API: Should this maybe be celerity::geometry::make_empty<2>(...)?
	// celerity::geometry_builder<2> gb{mat_c.get_range()};

	// gb.split_2d_but_recursive_and_only_for_local_chunks();
	// auto geo = gb.make();
	// auto geo = gb.make_nd({group_size, group_size, 1});

	// We want to apply the same chunking to the data as we did to the kernel. Is that a common thing or a coincidence in this case?
	//	- Maybe there is an underlying concept here, something like a "partition" that works on arbitrary index spaces?
	//	- Although the name "partition" implies that there is no overlap.
	//	- In a sense we would be doing data partitioning in that case
	//		=> How would that work for non-square matrices?
	// We then want to cycle through these blocks as we submit tasks

	// TODO: Here we could choose to only materialize those chunks that are along the main axes of the local chunk. Because the others don't need any of our
	// data, right?
	// BUT: How do we do this in such a way that the data requirements are still correct? Currently we assume all chunks to exist

	// LETS DO THIS MANUALLY FOR NOW
	const size_t num_nodes = celerity::detail::runtime::get_instance().NOCOMMIT_get_num_nodes();
	const size_t num_devices = celerity::detail::runtime::get_instance().NOCOMMIT_get_num_local_devices();
	if(num_devices != 1) throw std::runtime_error("multi-device NYI");

	const auto mat_size = mat_c.get_range();
	if(mat_size[0] != mat_size[1]) throw std::runtime_error("only square matrices supported");
	if(std::sqrt(num_nodes) != std::floor(std::sqrt(num_nodes))) throw std::runtime_error("number of nodes must be a square number");
	if(mat_size[0] % size_t(std::sqrt(num_nodes)) != 0) throw std::runtime_error("matrix size must be divisible by square root of number of nodes");

	const size_t block_size = mat_size[0] / std::sqrt(num_nodes);

	for(size_t K = 0; K < mat_size[0]; K += block_size) {
		// celerity::detail::region<3> union_access_a = celerity::detail::box<3>::full_range(range_cast<3>(mat_a.get_range()));
		// celerity::detail::region<3> union_access_b = celerity::detail::box<3>::full_range(range_cast<3>(mat_b.get_range()));
		// std::vector<std::pair<celerity::detail::box<3>, celerity::detail::region<3>>> per_chunk_accesses_a;
		// std::vector<std::pair<celerity::detail::box<3>, celerity::detail::region<3>>> per_chunk_accesses_b;

		// TODO: Does it make sense to do oversubscription here? I.e., there is no reason to have the same number of blocks as there are nodes, right?
		//	=> We could always just create N^2 blocks for N nodes..? Would make things easier, and we'd only have to deal with square block matrices
		// for(auto& ac : geo.assigned_chunks) {
		// 	auto sr_a = ac.box.get_subrange();
		// 	auto sr_b = ac.box.get_subrange();
		// 	sr_a.offset[1] = K;
		// 	sr_b.offset[0] = K;

		// 	per_chunk_accesses_a.push_back(std::pair{ac.box, celerity::detail::box{sr_a}});
		// 	per_chunk_accesses_b.push_back(std::pair{ac.box, celerity::detail::box{sr_b}});
		// }

		// TODO API: We have to do this somehow inside expert_mapper, but how? We don't have the buffer size available. Do it BAM?
		// for(auto& [_, region] : per_chunk_accesses_a) {
		// 	if(!celerity::detail::box<2>::full_range(mat_a.get_range()).covers(celerity::detail::box_cast<2>(celerity::detail::bounding_box(region)))) {
		// 		throw std::runtime_error(fmt::format("Access {} is out of bounds for matrix A", region));
		// 	}
		// }
		// for(auto& [_, region] : per_chunk_accesses_b) {
		// 	if(!celerity::detail::box<2>::full_range(mat_b.get_range()).covers(celerity::detail::box_cast<2>(celerity::detail::bounding_box(region)))) {
		// 		throw std::runtime_error(fmt::format("Access {} is out of bounds for matrix B", region));
		// 	}
		// }

		// celerity::expert_mapper data_reqs_a(union_access_a, per_chunk_accesses_a);
		// celerity::expert_mapper data_reqs_b(union_access_b, per_chunk_accesses_b);

		grid_data_requirements<2> data_reqs_a{geo};
		grid_data_requirements<2> data_reqs_b{geo};

		for(auto& cell : geo.get_grid().get_cells()) {
			const auto idx = K / block_size;
			data_reqs_a.add(cell.pos, matrix_partition.get_cell({cell.pos[0], idx}));
			data_reqs_b.add(cell.pos, matrix_partition.get_cell({idx, cell.pos[1]}));
			// Easy optimization: Instead of starting at 0/0 for each block, we start at with the data that already exists on that node
			// => THIS REQUIRES THAT INITIALIZATION IS ALSO BLOCKED ALREADY!
			// => Note that this also results in bitwise different results!! (floating point addition not commutative)

			// HOLD UP: This either also requires adjustment in the kernel, or local indexing

			// const auto grid_size = matrix_partition.get_grid_size();
			// data_reqs_a.add(cell.pos, matrix_partition.get_cell({cell.pos[0], (cell.pos[1] + idx) % grid_size[1]}));
			// data_reqs_b.add(cell.pos, matrix_partition.get_cell({(cell.pos[0] + idx) % grid_size[0], cell.pos[1]}));
		}

		queue.submit([&](celerity::handler& cgh) {
			celerity::accessor a{mat_a, cgh, data_reqs_a, celerity::read_only};
			celerity::accessor b{mat_b, cgh, data_reqs_b, celerity::read_only};
			celerity::accessor c{mat_c, cgh, celerity::access::one_to_one{}, celerity::read_write};

			celerity::local_accessor<T, 2> scratch_a{{group_size, group_size}, cgh};
			celerity::local_accessor<T, 2> scratch_b{{group_size, group_size}, cgh};

			celerity::debug::set_task_name(cgh, "matmul blocked");
			// cgh.parallel_for(geo, [=](celerity::item<2> item) {
			cgh.parallel_for(geo.operator celerity::nd_custom_task_geometry<2>(), [=](celerity::nd_item<2> item) {
				// T sum{};
				// for(size_t k = 0; k < block_size; ++k) {
				// 	sum += a[{item[0], K + k}] * b[{K + k, item[1]}];
				// }
				// c[item] += sum;

				T sum{};
				const auto gid = item.get_global_id();
				const auto lid = item.get_local_id();
				for(size_t j = 0; j < block_size; j += group_size) {
					scratch_a[lid] = a[gid[0]][K + j + lid[1]];
					scratch_b[lid] = b[K + j + lid[0]][gid[1]];
					celerity::group_barrier(item.get_group());
					for(size_t k = 0; k < group_size; ++k) {
						const auto a_ik = scratch_a[lid[0]][k];
						const auto b_kj = scratch_b[k][lid[1]];
						sum += a_ik * b_kj;
					}
					celerity::group_barrier(item.get_group());
				}
				c[gid] += sum;
			});
		});
	}
}

template <typename T>
void verify(
    celerity::queue& queue, celerity::buffer<T, 2> mat_c, const int fill_min, const int fill_max, celerity::experimental::host_object<bool> passed_obj) {
	queue.submit([&](celerity::handler& cgh) {
		celerity::accessor c{mat_c, cgh, celerity::access::one_to_one{}, celerity::read_only_host_task};
		celerity::experimental::side_effect passed{passed_obj, cgh};

		celerity::debug::set_task_name(cgh, "verification");
		const auto mat_size = mat_c.get_range();
		cgh.host_task(mat_c.get_range(), [=](celerity::partition<2> part) {
			*passed = true;
			const auto& sr = part.get_subrange();
			for(size_t i = sr.offset[0]; i < sr.offset[0] + sr.range[0]; ++i) {
				for(size_t j = sr.offset[1]; j < sr.offset[1] + sr.range[1]; ++j) {
					const float received = c[i][j];
					// The original matrix was initialized based on its linear id. We expect the result to be flipped horizontally,
					// so we have to first compute the original global index.
					const size_t original_global_index = i * mat_size[1] + mat_size[1] - j - 1;
					const float expected = (original_global_index % (fill_max - fill_min)) + fill_min;
					if(expected != received) {
						CELERITY_ERROR("Verification failed for element {},{}: {} (received) != {} (expected)", i, j, received, expected);
						*passed = false;
						return;
					}
				}
			}
			if(*passed) { CELERITY_INFO("Verification passed for {}", part.get_subrange()); }
		});
	});
}

/**
 * This is actually a somewhat interesting challenge:
 * - We can do blocked matmul on inter-node level, but normal matmul (per GPU) on intra-node level
 * - More fancy would be to do blocked matmul per GPU as well, but this would require launching multiple kernels within a single task
 *   - Technically possible, but unclear what the dependency situation is like
 * - For real world we would need the "discard/ephemeral" hint for buffer memory
 * - MAJOR issue: Sub-tasks would all be concurrent, but we DON'T want to execute them all at the same time!!
 *   - NEVERMIND: Since each task read-writes a specific block, they are all serialized!!
 *     BUT: In general, do we need a way of limiting concurrency in such scenarios?
 * - MAYBE do we actually prefer "local" coordinates when accessing blocks? I.e., each block starts at 0 and goes to N/num_blocks
 *   - Same for the kernel itself, actually
 *
 * - ANOTHER issue: Assuming that each node owns the same chunk in A and B, we could actually run a first multiplication on each
 *                  without requiring any data transfers, if we DON'T start at i/j = 0 for each
 *		=> BUT: This will result in bitwise different results, b/c floating point addition is not commutative
 *		=> SAME applies to per-GPU chunk "commutative dependencies" - if we end up doing that
 *
 * - THE GIFT THAT KEEPS ON GIVING: For register tiling we now also need to do thread coarsening!! Task and buffer geometry are no longer a 1:1 match
 *   => Here we would likely want something like a builder::scale(0.25) ?
 */


// NEXT STEPS:

// - [x] Switch from subranges / chunks to boxes (also in UMUGUC / FVM)
// - [x] Implement nd-range variant of custom geometry (needs to be separate type for nd_item!)
//   - [x] Implement nd-range kernel
// - Look into multi-device support
// - Figure out high-level API
// - Medium term: We probably want to get rid of the BAM altogether, and move towards a system where
//   tasks contain geometries that have associated data requirements.

// Disadvantages of having expert_mapper match on chunks:
// - Does not allow for identical chunks. DO WE NEED THOSE?
// - Lookup is O(n)

// SHOULD device chunks even be part of this whole thing? Or should they be a separate stage?


// TO RECAP:
// - Partition data, allow for "block indexing" (e.g. block {2, 1})
// - Thread coarsening
// - Ephemeral hint
// - Prescribe stride (either any or dense)
//		- Requires to split allocations
//		- Allow access to pointer (how does typing work for this?)
// - Maybe: Local indexing
//		- BUT: How useful is it really, when you have e.g. a buffer that is not evenly divisible by number of blocks?
//             You need to be able to tell whether threads are exceeding the domain or not
//			=> Maybe pass in global coordinates as extra object? Does not using a kernel parameter create register pressure?
// - Multi-device support TBD?!
// - 1D kernel on 2D data: Highly optimized GEMM

int main(int argc, char* argv[]) {
	const size_t mat_size = argc > 1 ? std::stoul(argv[1]) : default_mat_size;
	const bool use_blocked = argc > 2 ? std::stoi(argv[2]) : 0;

	fmt::print("Matrix size is {}x{}, doing {} multiplication\n", mat_size, mat_size, use_blocked ? "BLOCKED" : "normal");

	celerity::queue queue;

	const auto range = celerity::range<2>(mat_size, mat_size);
	celerity::buffer<float, 2> mat_a_buf(range);
	celerity::buffer<float, 2> mat_b_buf(range);
	celerity::buffer<float, 2> mat_c_buf(range);

	celerity::debug::set_buffer_name(mat_a_buf, "mat_a");
	celerity::debug::set_buffer_name(mat_b_buf, "mat_b");
	celerity::debug::set_buffer_name(mat_c_buf, "mat_c");

	// TODO: We should create geometry here and pass it into all functions

	const auto setup = [&](const int fill_min, const int fill_max) {
		fill_with_range(queue, mat_a_buf, fill_min, fill_max);
		set_identity(queue, mat_b_buf, true);
		set_zero(queue, mat_c_buf);
	};

	const auto run = [&] {
		if(use_blocked) {
			multiply_blocked(queue, mat_a_buf, mat_b_buf, mat_c_buf);
		} else {
			multiply(queue, mat_a_buf, mat_b_buf, mat_c_buf);
		}
	};

	puts("With warmup");
	setup(0, 7);
	run();

	setup(8, 13);

	queue.wait(celerity::experimental::barrier);
	const auto before = std::chrono::steady_clock::now();
	run();
	queue.wait(celerity::experimental::barrier);
	const auto after = std::chrono::steady_clock::now();

	const double gflops = 2.0 * mat_size * mat_size * mat_size / 1e9;
	const double seconds = std::chrono::duration_cast<std::chrono::duration<double>>(after - before).count();
	fmt::print("Multiplication took {}ms, {:.1f} GFLOPS/s\n", std::chrono::duration_cast<std::chrono::milliseconds>(after - before).count(), gflops / seconds);

	// each node verifies part of the result, so we pass per-node verification results through a host object
	celerity::experimental::host_object<bool> passed_obj(false);
	verify(queue, mat_c_buf, 8, 13, passed_obj);

	// The value of `passed` can differ between hosts if only part of the verification failed.
	const bool passed = queue.fence(passed_obj).get();
	return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
