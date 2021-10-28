#include "transformers/naive_split.h"

#include <cassert>
#include <vector>

#include "command.h"
#include "ranges.h"
#include "task.h"

namespace celerity {
namespace detail {

	// We simply split in the first dimension for now
	std::vector<chunk<3>> split_equal(const chunk<3>& full_chunk, const cl::sycl::range<3>& granularity, size_t num_chunks, int dims) {
#ifndef NDEBUG
		assert(num_chunks > 0);
		for(int d = 0; d < dims; ++d) {
			assert(granularity[d] > 0);
			assert(full_chunk.range[d] % granularity[d] == 0);
		}
#endif

		// If global range is not divisible by (num_chunks * granularity), assign ceil(quotient) to the first few chunks and floor(quotient) to the remaining
		const auto small_chunk_size_dim0 = full_chunk.range[0] / (num_chunks * granularity[0]) * granularity[0];
		const auto large_chunk_size_dim0 = small_chunk_size_dim0 + granularity[0];
		const auto num_large_chunks = (full_chunk.range[0] - small_chunk_size_dim0 * num_chunks) / granularity[0];
		assert(num_large_chunks * large_chunk_size_dim0 + (num_chunks - num_large_chunks) * small_chunk_size_dim0 == full_chunk.range[0]);

		std::vector<chunk<3>> result(num_chunks, {full_chunk.offset, full_chunk.range, full_chunk.global_size});
		for(auto i = 0u; i < num_large_chunks; ++i) {
			result[i].range[0] = large_chunk_size_dim0;
			result[i].offset[0] += i * large_chunk_size_dim0;
		}
		for(auto i = num_large_chunks; i < num_chunks; ++i) {
			result[i].range[0] = small_chunk_size_dim0;
			result[i].offset[0] += num_large_chunks * large_chunk_size_dim0 + (i - num_large_chunks) * small_chunk_size_dim0;
		}

#ifndef NDEBUG
		size_t total_range_dim0 = 0;
		for(size_t i = 0; i < result.size(); ++i) {
			total_range_dim0 += result[i].range[0];
			if(i == 0) {
				assert(result[i].offset[0] == full_chunk.offset[0]);
			} else {
				assert(result[i].offset[0] == result[i - 1].offset[0] + result[i - 1].range[0]);
			}
		}
		assert(total_range_dim0 == full_chunk.range[0]);
#endif

		return result;
	}

	naive_split_transformer::naive_split_transformer(size_t num_chunks, size_t num_workers) : num_chunks(num_chunks), num_workers(num_workers) {
		assert(num_chunks >= num_workers);
	}

	void naive_split_transformer::transform_task(const task* tsk, command_graph& cdag) {
		if(!tsk->has_variable_split()) return;

		// Assign each chunk to a node
		std::vector<node_id> nodes(num_chunks);
		// We assign chunks next to each other to the same worker (if there is more chunks than workers), as this is likely to produce less
		// transfers between tasks than a round-robin assignment (for typical stencil codes).
		const auto chunks_per_node = num_workers > 0 ? num_chunks / num_workers : num_chunks;
		for(auto i = 0u; i < num_chunks; ++i) {
			nodes[i] = (i / chunks_per_node) % num_workers;
		}

		auto& task_commands = cdag.task_commands(tsk->get_id());
		assert(task_commands.size() == 1);

		const auto original = static_cast<task_command*>(task_commands[0]);

		// TODO: For now we can only handle newly created tasks (i.e. no existing dependencies/dependents)
		assert(std::distance(original->get_dependencies().begin(), original->get_dependencies().end()) == 0);
		assert(std::distance(original->get_dependents().begin(), original->get_dependents().end()) == 0);

		chunk<3> full_chunk{tsk->get_global_offset(), tsk->get_global_size(), tsk->get_global_size()};
		auto chunks = split_equal(full_chunk, tsk->get_granularity(), num_chunks, tsk->get_dimensions());
		for(size_t i = 0; i < chunks.size(); ++i) {
			cdag.create<task_command>(nodes[i], tsk->get_id(), subrange{chunks[i]});
		}

		// Remove original
		cdag.erase(original);
	}

} // namespace detail
} // namespace celerity
