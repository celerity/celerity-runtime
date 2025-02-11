#include "fmt/ranges.h"
#include <cstdio>

#include <celerity.h>
#include <geometry_builder.h>

#if !defined(NDEBUG) || CELERITY_SYCL_IS_SIMSYCL
const size_t default_mat_size = 128;
#else
const size_t default_mat_size = 1024;
#endif

#define USE_REGISTER_TILING 1

constexpr int group_size = 16;

// The choice of distributed block represents an important tradeoff / tuning parameter:
// - Larger blocks are more efficient to execute (less launch overhead to execution time ratio)
// 		- Also fewer overall tasks => less launch overhead
// - Smaller blocks means more fine-grained data dependencies
// 		- First tasks can start earlier
// TODO: What is a good size?
const size_t distributed_block_size = 2048;
static_assert(distributed_block_size % group_size == 0);

#if USE_REGISTER_TILING
constexpr int register_tile_size = 4;
constexpr int local_memory_tile_size = group_size * register_tile_size;
#endif

template <typename T>
void set_identity(celerity::queue queue, celerity::buffer<T, 2> mat, bool reverse, const std::optional<celerity::custom_task_geometry<2>>& geo) {
	queue.submit([&](celerity::handler& cgh) {
		celerity::accessor dw{mat, cgh, celerity::access::one_to_one{}, celerity::write_only, celerity::no_init};
		const auto range = mat.get_range();

		celerity::debug::set_task_name(cgh, "set identity");
		celerity::experimental::hint(cgh, celerity::experimental::hints::split_2d{});
		// NOCOMMIT DRY
		if(geo.has_value()) {
			cgh.parallel_for(*geo, [=](celerity::item<2> item) {
				if(!reverse) {
					dw[item] = item[0] == item[1];
				} else {
					dw[item] = item[0] == (range[1] - item[1] - 1);
				}
			});
		} else {
			cgh.parallel_for(range, [=](celerity::item<2> item) {
				if(!reverse) {
					dw[item] = item[0] == item[1];
				} else {
					dw[item] = item[0] == (range[1] - item[1] - 1);
				}
			});
		}
	});
}

// Fill matrix with something that is not all zeroes but still easy to verify.
template <typename T>
void fill_with_range(
    celerity::queue queue, celerity::buffer<T, 2> mat, const int min, const int max, const std::optional<celerity::custom_task_geometry<2>>& geo) {
	queue.submit([&](celerity::handler& cgh) {
		celerity::accessor dw{mat, cgh, celerity::access::one_to_one{}, celerity::write_only, celerity::no_init};
		const auto range = mat.get_range();

		celerity::debug::set_task_name(cgh, "fill with range");
		celerity::experimental::hint(cgh, celerity::experimental::hints::split_2d{});
		// NOCOMMIT DRY
		if(geo.has_value()) {
			cgh.parallel_for(*geo, [=](celerity::item<2> item) { dw[item] = item.get_linear_id() % (max - min) + min; });
		} else {
			cgh.parallel_for(range, [=](celerity::item<2> item) { dw[item] = item.get_linear_id() % (max - min) + min; });
		}
	});
}

template <typename T>
void set_zero(celerity::queue queue, celerity::buffer<T, 2> mat, const std::optional<celerity::custom_task_geometry<2>>& geo) {
	queue.submit([&](celerity::handler& cgh) {
		celerity::accessor dw{mat, cgh, celerity::access::one_to_one{}, celerity::write_only, celerity::no_init};
		celerity::debug::set_task_name(cgh, "set zero");
		celerity::experimental::hint(cgh, celerity::experimental::hints::split_2d{});
		// NOCOMMIT DRY
		if(geo.has_value()) {
			cgh.parallel_for(*geo, [=](celerity::item<2> item) { dw[item] = T{0}; });
		} else {
			cgh.parallel_for(mat.get_range(), [=](celerity::item<2> item) { dw[item] = T{0}; });
		}
	});
}

template <typename T>
void multiply(celerity::queue queue, celerity::buffer<T, 2> mat_a, celerity::buffer<T, 2> mat_b, celerity::buffer<T, 2> mat_c) {
	queue.submit([&](celerity::handler& cgh) {
#if USE_REGISTER_TILING
		const auto coarsened_slice = [](const int dim) {
			return [dim](celerity::chunk<2> chnk, const celerity::range<2>& buffer_size) {
				celerity::chunk<2> original_chunk = chnk;
				original_chunk.offset *= register_tile_size;
				original_chunk.range *= register_tile_size;
				return celerity::access::slice<2>(dim)(original_chunk, buffer_size);
			};
		};

		const auto coarsened_o2o = [](celerity::chunk<2> chnk, const celerity::range<2>& buffer_size) {
			celerity::subrange<2> sr = chnk;
			sr.offset *= register_tile_size;
			sr.range *= register_tile_size;
			return sr;
		};

		celerity::accessor a{mat_a, cgh, coarsened_slice(1), celerity::read_only};
		celerity::accessor b{mat_b, cgh, coarsened_slice(0), celerity::read_only};
		celerity::accessor c{mat_c, cgh, coarsened_o2o, celerity::write_only, celerity::no_init};
#else
		celerity::accessor a{mat_a, cgh, celerity::access::slice<2>(1), celerity::read_only};
		celerity::accessor b{mat_b, cgh, celerity::access::slice<2>(0), celerity::read_only};
		celerity::accessor c{mat_c, cgh, celerity::access::one_to_one{}, celerity::write_only, celerity::no_init};
#endif

#if USE_REGISTER_TILING
		celerity::local_accessor<T, 2> scratch_a{{local_memory_tile_size, local_memory_tile_size}, cgh};
		celerity::local_accessor<T, 2> scratch_b{{local_memory_tile_size, local_memory_tile_size}, cgh};
#else
		celerity::local_accessor<T, 2> scratch_a{{group_size, group_size}, cgh};
		celerity::local_accessor<T, 2> scratch_b{{group_size, group_size}, cgh};
#endif

		const size_t K = mat_a.get_range()[1];
		celerity::debug::set_task_name(cgh, "matrix multiplication");
#if USE_REGISTER_TILING
		cgh.parallel_for(celerity::nd_range<2>{mat_c.get_range() / register_tile_size, {group_size, group_size}}, [=](celerity::nd_item<2> item) {
			T sums[register_tile_size * register_tile_size] = {0};
			const auto lid = item.get_local_id();
			const auto grp = item.get_group();
			for(size_t k0 = 0; k0 < K; k0 += local_memory_tile_size) {
				for(int i = 0; i < register_tile_size; ++i) {
					for(int j = 0; j < register_tile_size; ++j) {
						scratch_a[i * group_size + lid[0]][j * group_size + lid[1]] =
						    a[{grp[0] * group_size * register_tile_size + i * group_size + lid[0], k0 + j * group_size + lid[1]}];
						scratch_b[i * group_size + lid[0]][j * group_size + lid[1]] =
						    b[{k0 + i * group_size + lid[0], grp[1] * group_size * register_tile_size + j * group_size + lid[1]}];
					}
				}
				celerity::group_barrier(item.get_group());
				for(size_t k1 = 0; k1 < local_memory_tile_size; ++k1) {
					for(int i = 0; i < register_tile_size; ++i) {
						for(int j = 0; j < register_tile_size; ++j) {
							sums[i * register_tile_size + j] += scratch_a[lid[0] * register_tile_size + i][k1] * scratch_b[k1][lid[1] * register_tile_size + j];
						}
					}
				}
				celerity::group_barrier(item.get_group());
			}

			for(int i = 0; i < register_tile_size; ++i) {
				for(int j = 0; j < register_tile_size; ++j) {
					c[{grp[0] * group_size * register_tile_size + lid[0] * register_tile_size + i,
					    grp[1] * group_size * register_tile_size + lid[1] * register_tile_size + j}] = sums[i * register_tile_size + j];
				}
			}
		});
#else
		celerity::experimental::hint(cgh, celerity::experimental::hints::split_2d{});
		cgh.parallel_for(celerity::nd_range<2>{mat_c.get_range(), {group_size, group_size}}, [=](celerity::nd_item<2> item) {
			T sum{};
			const auto lid = item.get_local_id();
			for(size_t j = 0; j < K; j += group_size) { // TODO: Akward j iterating to K
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


// We need to partition A, B and C into blocks of some size, chosen by user (e.g. 1024x1024).
// This partitioning is NOT related to node assignment.
// However, it likely makes sense to assign neighboring blocks to the same node.
// It would therefore be elegant if we first partition C into N chunks of arbitrary size, for N nodes.
// We then subdivide each of those chunks into the desired block size.
// We then launch the kernel over the nested blocks.
// HOWEVER: To compute the required input data for each block, we need it to use the same coordinate system as the partitionings of blocks A and B.
// If the chunks are nested, how would that work?
//
// Some options:
// - Partition C into blocks at top level, then assign multiple blocks somehow
// - Have a way of getting the "global" coordinates of a block (how would that work though? If we want to be able to do different grids, e.g. 1D split inside 2D
// grid, that doesn't make sense)
// - ...?

template <typename T>
void multiply_blocked(celerity::queue queue, celerity::buffer<T, 2> mat_a, celerity::buffer<T, 2> mat_b, celerity::buffer<T, 2> mat_c, bool is_warmup) {
	celerity::cartesian_grid<2> c_partition(celerity::detail::box<2>::full_range(mat_c.get_range()));
	// TODO: Obviously we need a better API to split across all nodes
	// matrix_partition.split(celerity::detail::runtime::get_instance().NOCOMMIT_get_num_nodes(), {group_size, group_size});
	if(mat_c.get_range() % distributed_block_size != celerity::detail::zeros) {
		throw std::runtime_error(
		    fmt::format("Matrix C with dimensions {} is not divisible by distributed block size {}", mat_c.get_range(), distributed_block_size));
	}
	c_partition.split(mat_c.get_range() / distributed_block_size, {distributed_block_size, distributed_block_size});

	// We now want to split A and B the same way as C
	celerity::cartesian_grid<2> a_partition(celerity::detail::box<2>::full_range(mat_a.get_range()));
	celerity::cartesian_grid<2> b_partition(celerity::detail::box<2>::full_range(mat_b.get_range()));
	a_partition.split(mat_a.get_range() / distributed_block_size, {distributed_block_size, distributed_block_size});
	b_partition.split(mat_b.get_range() / distributed_block_size, {distributed_block_size, distributed_block_size});

	// TODO: Do we even need a builder in this case?
	// => HERE we could create a 1D geometry from a 2D partition. The key is that we maintain the coordinates!!
	// celerity::geometry_builder<2> gb2{output_partition};
	// TODO: Option to do partial materialization here
	// gb2.assign(); // TODO: ? Policy - round robin, grouped round robin (?), ...
	// Look into block-cyclic distributions like ScaLAPACK: https://www.netlib.org/utk/papers/factor/node3.html
	// 	=> Important point: Some algorithms work on successively smaller portions of the matrix, e.g. Gaussian elimination.
	//  => Here a block-cyclic distribution results in better load balancing

	// THEN: Get all (locally materialized) chunks AND THEIR COORDINATES (how? only works if geometry is based on partition)

	// TODO: What about matrices that aren't evenly divisible by block size?

#if USE_REGISTER_TILING
	if(mat_c.get_range() % register_tile_size != celerity::detail::zeros) {
		throw std::runtime_error(fmt::format("Matrix C with dimensions {} is not divisible by register tile size {}", mat_c.get_range(), register_tile_size));
	}
	// Thread coarsening
	// TODO: Have API for that in builder?
	celerity::cartesian_grid<2> c_coarse(celerity::detail::box<2>::full_range(mat_c.get_range() / register_tile_size));
	c_coarse.split(c_partition.get_grid_size(), {distributed_block_size / register_tile_size, distributed_block_size / register_tile_size});
	celerity::grid_geometry geo(c_coarse, celerity::range<2>{group_size, group_size});
#else
	celerity::grid_geometry geo(c_partition, celerity::range<2>{group_size, group_size});
#endif

	// NOCOMMIT: This assumes the same number of devices on each node - OTHERWISE UB!!
	geo.assign(celerity::detail::runtime::get_instance().NOCOMMIT_get_num_nodes(), celerity::detail::runtime::get_instance().NOCOMMIT_get_num_local_devices());

	// The key thing we want to achieve is that we can iterate over all chunks and easily select their required data based on coordinates

	// Or we could just use one_to_one range mapper, as before
	// grid_data_requirements writes{geo};
	// writes.add_one_to_one(matrix_partition);

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
	// const size_t num_nodes = celerity::detail::runtime::get_instance().NOCOMMIT_get_num_nodes();
	// const size_t num_devices = celerity::detail::runtime::get_instance().NOCOMMIT_get_num_local_devices();
	// if(num_devices != 1) throw std::runtime_error("multi-device NYI");

	// const auto mat_size = mat_c.get_range();
	// if(mat_size[0] != mat_size[1]) throw std::runtime_error("only square matrices supported");
	// if(std::sqrt(num_nodes) != std::floor(std::sqrt(num_nodes))) throw std::runtime_error("number of nodes must be a square number");
	// if(mat_size[0] % size_t(std::sqrt(num_nodes)) != 0) throw std::runtime_error("matrix size must be divisible by square root of number of nodes");

	// const size_t block_size = mat_size[0] / std::sqrt(num_nodes);

	const size_t K = mat_a.get_range()[1];

	const size_t superblock_size = 4;
	const auto num_superblocks = mat_c.get_range() / distributed_block_size / superblock_size;
	CELERITY_CRITICAL("NUM SUPERBLOCKS: {}", num_superblocks);

	// TODO XY order - does it matter?
	for(size_t Y = 0; Y < num_superblocks[0]; ++Y) {
		for(size_t X = 0; X < num_superblocks[1]; ++X) {
			for(size_t k0 = 0; k0 < K; k0 += distributed_block_size) {
				grid_data_requirements<2> data_reqs_a{geo};
				grid_data_requirements<2> data_reqs_b{geo};
				grid_data_requirements<2> data_reqs_c{geo};

				auto custom_geo = geo.operator celerity::nd_custom_task_geometry<2>();
				std::erase_if(custom_geo.assigned_chunks, [&](auto& achnk) {
					for(auto& cell : geo.get_grid().get_cells()) {
						if(box_cast<3>(cell.box) == achnk.box) {
							if(cell.pos[0] / superblock_size != X || cell.pos[1] / superblock_size != Y) {
								// fmt::print("Skipping cell {} b/c not in superblock {}/{}\n", cell.pos, X, Y);
								return true;
							}
							return false;
						}
					}
					return true; // Should not be reached
				});

				for(auto& cell : geo.get_grid().get_cells()) {
					// NOCOMMIT TODO Why is this required?
					if(std::ranges::none_of(custom_geo.assigned_chunks, [&](auto& achnk) { return box_cast<3>(cell.box) == achnk.box; })) { continue; }

					const auto idx = k0 / distributed_block_size;
					data_reqs_a.add(cell.pos, a_partition.get_cell({cell.pos[0], idx}));
					data_reqs_b.add(cell.pos, b_partition.get_cell({idx, cell.pos[1]}));
					// This is just a one to one mapping in case we aren't doing thread coarsening
					data_reqs_c.add(cell.pos, c_partition.get_cell(cell.pos));
				}

				queue.submit([&](celerity::handler& cgh) {
					// NOCOMMIT TODO: We have to check whether data requirements are in bounds. But where? BAM doesn't have buffer dimensions.
					celerity::accessor a{mat_a, cgh, data_reqs_a, celerity::read_only};
					celerity::accessor b{mat_b, cgh, data_reqs_b, celerity::read_only};
					celerity::accessor c{mat_c, cgh, data_reqs_c, celerity::read_write};

					celerity::debug::set_task_name(cgh, "matmul blocked");

					if(!is_warmup) { cgh.assert_no_allocations(); }

#if USE_REGISTER_TILING
					celerity::local_accessor<T, 2> scratch_a{{local_memory_tile_size, local_memory_tile_size}, cgh};
					celerity::local_accessor<T, 2> scratch_b{{local_memory_tile_size, local_memory_tile_size}, cgh};

					// TODO: Can we keep DRY with basic variant? (Needs local indexing probably)
					// This kernel performs worse than basic variant on RTX 3090 because of the addition of k0 (could be solved with local indexing), and
					// weirdly, that distributed_block_size is known statically (unrolling gone wrong?).
					// Workaround for latter ¯\_(ツ)_/¯
					auto dynamic_dist_block_sz = distributed_block_size;
					cgh.parallel_for(custom_geo, [=](celerity::nd_item<2> item) {
						T sums[register_tile_size * register_tile_size] = {0};
						const auto lid = item.get_local_id();
						const auto grp = item.get_group();
						for(size_t k1 = 0; k1 < dynamic_dist_block_sz; k1 += local_memory_tile_size) {
							for(int i = 0; i < register_tile_size; ++i) {
								for(int j = 0; j < register_tile_size; ++j) {
									scratch_a[i * group_size + lid[0]][j * group_size + lid[1]] =
									    a[{grp[0] * group_size * register_tile_size + i * group_size + lid[0], k0 + k1 + j * group_size + lid[1]}];
									scratch_b[i * group_size + lid[0]][j * group_size + lid[1]] =
									    b[{k0 + k1 + i * group_size + lid[0], grp[1] * group_size * register_tile_size + j * group_size + lid[1]}];
								}
							}
							celerity::group_barrier(item.get_group());
							for(size_t k2 = 0; k2 < local_memory_tile_size; ++k2) {
								for(int i = 0; i < register_tile_size; ++i) {
									for(int j = 0; j < register_tile_size; ++j) {
										sums[i * register_tile_size + j] +=
										    scratch_a[lid[0] * register_tile_size + i][k2] * scratch_b[k2][lid[1] * register_tile_size + j];
									}
								}
							}
							celerity::group_barrier(item.get_group());
						}

						for(int i = 0; i < register_tile_size; ++i) {
							for(int j = 0; j < register_tile_size; ++j) {
								c[{grp[0] * group_size * register_tile_size + lid[0] * register_tile_size + i,
								    grp[1] * group_size * register_tile_size + lid[1] * register_tile_size + j}] += sums[i * register_tile_size + j];
							}
						}
					});

#else
					celerity::local_accessor<T, 2> scratch_a{{group_size, group_size}, cgh};
					celerity::local_accessor<T, 2> scratch_b{{group_size, group_size}, cgh};

					cgh.parallel_for(geo.operator celerity::nd_custom_task_geometry<2>(), [=](celerity::nd_item<2> item) {
						T sum{};
						const auto gid = item.get_global_id();
						const auto lid = item.get_local_id();
						for(size_t j = 0; j < distributed_block_size; j += group_size) {
							scratch_a[lid] = a[gid[0]][k0 + j + lid[1]];
							scratch_b[lid] = b[k0 + j + lid[0]][gid[1]];
							celerity::group_barrier(item.get_group());
							for(size_t k1 = 0; k1 < group_size; ++k1) {
								const auto a_ik = scratch_a[lid[0]][k1];
								const auto b_kj = scratch_b[k1][lid[1]];
								sum += a_ik * b_kj;
							}
							celerity::group_barrier(item.get_group());
						}
						c[gid] += sum;

						// Naive matmul
						// T sum{};
						// for(size_t k = 0; k < block_size; ++k) {
						// 	sum += a[{item[0], K + k}] * b[{K + k, item[1]}];
						// }
						// c[item] += sum;
					});
#endif
				});

				// SUBTASKS:
				// - Would be a huge undertaking
				// 		- Would require storing multiple task launchers within a single task, for example
				// - Unclear how accessors would even work (have two levels of accessors, inner level takes outer level as parameter?!)
				//		- also with respect to CGF diagnostics
				// - When would subtask lambda be evaluated?! It would have to be early, otherwise we introduce a lot of latency
				// 		- Most likely during TDAG generation, but that would mean that we need to do chunking in TDAG already
				// - The big advantage: We could do the same type of chunking as on outer level and have precise control over order of executions

				// ALSO:
				// - Creating device chunks on the top level is awkward. It's kind of an implementation detail from the perspective of distributed scheduling.
				// 		- Of course we could submit different tasks for each device on the top level, but that is even more stupid.
				// - If we want to do blocked matmul on device level within the top-level geometry, we need overlapping chunks, which messes with the whole grid
				// setup.
				// - Is there some kind of middleground?

				// YES: New plan after talking to peter: If we want to do blocked device matmul, we simply have to submit multiple tasks, one for each step.
				// The justification is this: A geometry may have overlapping chunks, but writes must be exclusive (unless replicated). If there is a conflict,
				// submit multiple tasks. This way we can control the order of execution for each device, and do the same optimization as on the top level.
				//
				// What we need: Nested cartesian grid, where each cell can contain subcells, that we can iterate over. Only local chunks contain subcells.
				// => OR: Have API to tell whether a chunk is a local chunk. Only if it is, descend lower.
				//
				// Ad ephemeral storage: This wouldn't work with multiple tasks (because it would've been a property of the accessor).
				// BUT: We can still do something like buffer:discard_non_owned() hint.
				// AND: Importantly, Peter (and now I) believe that storage of slices will not be a problem even for weak scaling.

				// => The only downside of this, compared to subtasks, is that we have to assume the same number of devices on each node.
				//		In a sense, we kind of "leak" the device-level algorithm to the distributed level.
				//		Different number of devices (or different splits) CAN be supported, but it requires to segregate the distributed chunks into groups as
				// well.
				// 		I.e., submit a different number of tasks for those nodes that have two devices vs those that have four. Ultimately all nodes need to
				// submit
				//      all tasks of course, but the global chunks need to be set up in such a way that only the correct set of nodes does something.

				// Should we support arbitrarily nested grids?
				// => Maybe: This way we could easily express oversubscription, first do a normal 2D split, and then recursively split again.
				// Then launch task over level 1 grid (assuming 0 = coarsest), and express data dependencies in terms of that level as well.
			}
		}
	}
}

// TODO: Use 2D geometry as well
template <typename T>
void verify(celerity::queue& queue, celerity::buffer<T, 2> mat_c, const size_t K, const int fill_min, const int fill_max,
    celerity::experimental::host_object<bool> passed_obj) {
	const size_t M = mat_c.get_range()[1];
	queue.submit([&](celerity::handler& cgh) {
		celerity::accessor c{mat_c, cgh, celerity::access::one_to_one{}, celerity::read_only_host_task};
		celerity::experimental::side_effect passed{passed_obj, cgh};

		celerity::debug::set_task_name(cgh, "verification");
		cgh.host_task(mat_c.get_range(), [=](celerity::partition<2> part) {
			*passed = true;
			const auto& sr = part.get_subrange();
			for(size_t i = sr.offset[0]; i < sr.offset[0] + sr.range[0]; ++i) {
				for(size_t j = sr.offset[1]; j < sr.offset[1] + sr.range[1]; ++j) {
					const float received = c[i][j];
					float expected = 0;

					// If B is identity matrix
					// if(j < K) {
					// 	const size_t original_global_index = i * K + j;
					// 	expected = (original_global_index % (fill_max - fill_min)) + fill_min;
					// }

					// If B is reverse identity matrix
					if(M < K || (M >= K && j >= M - K)) {
						const size_t original_global_index = i * K + M - j - 1;
						expected = (original_global_index % (fill_max - fill_min)) + fill_min;
					}

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
	const bool use_blocked = argc > 1 ? std::stoi(argv[1]) : 0;
	const size_t N = argc > 2 ? std::stoul(argv[2]) : default_mat_size;
	if(argc > 3 && argc != 5) {
		fprintf(stderr, "Usage: %s [<use blocked>] [<N>] [<K> <M>]\n", argv[0]);
		exit(1);
	}
	const size_t K = argc == 5 ? std::stoul(argv[3]) : N;
	const size_t M = argc == 5 ? std::stoul(argv[4]) : N;

	fmt::print("Multiplying {}x{} matrix times {}x{} to produce {}x{}, using {} multiplication\n", N, K, K, M, N, M, use_blocked ? "BLOCKED" : "normal");

	celerity::queue queue;

	celerity::buffer<float, 2> mat_a_buf({N, K});
	celerity::buffer<float, 2> mat_b_buf({K, M});
	celerity::buffer<float, 2> mat_c_buf({N, M});

	celerity::debug::set_buffer_name(mat_a_buf, "mat_a");
	celerity::debug::set_buffer_name(mat_b_buf, "mat_b");
	celerity::debug::set_buffer_name(mat_c_buf, "mat_c");

	celerity::cartesian_grid<2> a_partition(celerity::detail::box<2>::full_range(mat_a_buf.get_range()));
	celerity::cartesian_grid<2> b_partition(celerity::detail::box<2>::full_range(mat_b_buf.get_range()));
	celerity::cartesian_grid<2> c_partition(celerity::detail::box<2>::full_range(mat_c_buf.get_range()));
	c_partition.split(mat_c_buf.get_range() / distributed_block_size, {distributed_block_size, distributed_block_size});
	a_partition.split(mat_a_buf.get_range() / distributed_block_size, {distributed_block_size, distributed_block_size});
	b_partition.split(mat_b_buf.get_range() / distributed_block_size, {distributed_block_size, distributed_block_size});

	celerity::grid_geometry a_geo(a_partition, celerity::range<2>{group_size, group_size});
	a_geo.assign(
	    celerity::detail::runtime::get_instance().NOCOMMIT_get_num_nodes(), celerity::detail::runtime::get_instance().NOCOMMIT_get_num_local_devices());
	celerity::grid_geometry b_geo(b_partition, celerity::range<2>{group_size, group_size});
	b_geo.assign(
	    celerity::detail::runtime::get_instance().NOCOMMIT_get_num_nodes(), celerity::detail::runtime::get_instance().NOCOMMIT_get_num_local_devices());
	celerity::grid_geometry c_geo(c_partition, celerity::range<2>{group_size, group_size});
	c_geo.assign(
	    celerity::detail::runtime::get_instance().NOCOMMIT_get_num_nodes(), celerity::detail::runtime::get_instance().NOCOMMIT_get_num_local_devices());

	std::optional<celerity::custom_task_geometry<2>> opt_a_geo;
	std::optional<celerity::custom_task_geometry<2>> opt_b_geo;
	std::optional<celerity::custom_task_geometry<2>> opt_c_geo;

	if(use_blocked) {
		opt_a_geo = a_geo;
		opt_b_geo = b_geo;
		opt_c_geo = c_geo;
	}

	const auto setup = [&](const int fill_min, const int fill_max) {
		fill_with_range(queue, mat_a_buf, fill_min, fill_max, opt_a_geo);
		set_identity(queue, mat_b_buf, true, opt_b_geo);
		set_zero(queue, mat_c_buf, opt_c_geo);
	};

	const auto run = [&](bool is_warmup) {
		if(use_blocked) {
			multiply_blocked(queue, mat_a_buf, mat_b_buf, mat_c_buf, is_warmup);
		} else {
			multiply(queue, mat_a_buf, mat_b_buf, mat_c_buf);
		}
	};

	[[maybe_unused]] const auto print_mat = [&queue](auto mat) {
		if(mat.get_range().size() > 256) {
			fmt::print("Matrix too large to print\n");
			return;
		}
		queue.submit([&](celerity::handler& cgh) {
			celerity::accessor acc{mat, cgh, celerity::access::all{}, celerity::read_only_host_task};
			const auto size = mat.get_range();
			cgh.host_task(celerity::on_master_node, [=] {
				for(size_t i = 0; i < size[0]; ++i) {
					for(size_t j = 0; j < size[1]; ++j) {
						fmt::print("{:2} ", acc[{i, j}]);
					}
					fmt::print("\n");
				}
			});
		});
		queue.wait();
	};

	puts("With warmup");
	setup(0, 7);
	run(true);

	setup(8, 13);

	queue.wait(celerity::experimental::barrier);
	const auto before = std::chrono::steady_clock::now();
	run(false);
	queue.wait(celerity::experimental::barrier);
	const auto after = std::chrono::steady_clock::now();

	const double gflops = 2.0 * N * K * M / 1e9;
	const double seconds = std::chrono::duration_cast<std::chrono::duration<double>>(after - before).count();
	fmt::print("Multiplication took {}ms, {:.1f} GFLOPS/s\n", std::chrono::duration_cast<std::chrono::milliseconds>(after - before).count(), gflops / seconds);

	// each node verifies part of the result, so we pass per-node verification results through a host object
	celerity::experimental::host_object<bool> passed_obj(false);
	verify(queue, mat_c_buf, K, 8, 13, passed_obj);

	// The value of `passed` can differ between hosts if only part of the verification failed.
	const bool passed = queue.fence(passed_obj).get();
	return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
