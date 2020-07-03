#include "transformers/naive_split.h"

#include <cassert>
#include <vector>

#include "command.h"
#include "ranges.h"
#include "task.h"

namespace celerity {
namespace detail {

	// We simply split in the first dimension for now
	std::vector<chunk<3>> split_equal(const chunk<3>& full_chunk, const cl::sycl::range<3>& local_size, size_t num_chunks, int dims) {
		assert(num_chunks > 0);
		for(int d = 0; d < dims; ++d) {
			assert(local_size[d] > 0);
			assert(full_chunk.range[d] % local_size[d] == 0);
		}

		chunk<3> chnk;
		chnk.global_size = full_chunk.global_size;
		chnk.offset = full_chunk.offset;
		chnk.range = full_chunk.range;

		auto ideal_chunk_size_dim0 = full_chunk.range[0] / num_chunks;
		auto floor_chunk_size_dim0 = ideal_chunk_size_dim0 / local_size[0] * local_size[0];
		assert(full_chunk.range[0] >= num_chunks * floor_chunk_size_dim0);
		chnk.range[0] = floor_chunk_size_dim0;

		std::vector<chunk<3>> result;
		for(auto i = 0u; i < num_chunks; ++i) {
			result.push_back(chnk);
			chnk.offset[0] += floor_chunk_size_dim0;
		}
		result.back().range[0] += full_chunk.range[0] - num_chunks * floor_chunk_size_dim0;
		return result;
	}

	naive_split_transformer::naive_split_transformer(size_t num_chunks, size_t num_workers) : num_chunks(num_chunks), num_workers(num_workers) {
		assert(num_chunks >= num_workers);
	}

	void naive_split_transformer::transform_task(const std::shared_ptr<const task>& tsk, command_graph& cdag) {
		if(tsk->get_type() != task_type::COMPUTE) return;
		const auto ctsk = dynamic_cast<const compute_task*>(tsk.get());

		// Assign each chunk to a node
		std::vector<node_id> nodes(num_chunks);
		// We assign chunks next to each other to the same worker (if there is more chunks than workers), as this is likely to produce less
		// transfers between tasks than a round-robin assignment (for typical stencil codes).
		const auto chunks_per_node = num_workers > 0 ? num_chunks / num_workers : num_chunks;
		for(auto i = 0u; i < num_chunks; ++i) {
			nodes[i] = (i / chunks_per_node) % num_workers;
		}

		auto& task_commands = cdag.task_commands(ctsk->get_id());
		assert(task_commands.size() == 1);
		assert(isa<compute_command>(task_commands[0]));

		const auto original = static_cast<compute_command*>(task_commands[0]);

		// TODO: For now we can only handle newly created computes (i.e. no existing dependencies/dependents)
		assert(std::distance(original->get_dependencies().begin(), original->get_dependencies().end()) == 0);
		assert(std::distance(original->get_dependents().begin(), original->get_dependents().end()) == 0);

		chunk<3> full_chunk{ctsk->get_global_offset(), ctsk->get_global_size(), ctsk->get_global_size()};
		auto local_size = ctsk->get_local_size();
		auto chunks = split_equal(full_chunk, local_size, num_chunks, ctsk->get_dimensions());
		for(size_t i = 0; i < chunks.size(); ++i) {
			cdag.create<compute_command>(nodes[i], ctsk->get_id(), chunks[i]);
		}

		// Remove original
		cdag.erase(original);
	}

} // namespace detail
} // namespace celerity
