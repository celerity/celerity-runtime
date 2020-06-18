#include "transformers/naive_split.h"

#include <cassert>
#include <vector>

#include "command.h"
#include "ranges.h"
#include "task.h"

namespace celerity {
namespace detail {

	std::vector<chunk<3>> split_equal(const chunk<1>& full_chunk, size_t num_chunks) {
		assert(num_chunks > 0);
		chunk<1> chnk;
		chnk.global_size = full_chunk.global_size;
		chnk.offset = full_chunk.offset;
		chnk.range = cl::sycl::range<1>(full_chunk.range.size() / num_chunks);

		std::vector<chunk<3>> result;
		for(auto i = 0u; i < num_chunks; ++i) {
			result.push_back(chnk);
			chnk.offset = chnk.offset + chnk.range;
			if(i == num_chunks - 1) { result[i].range[0] += full_chunk.range.size() % num_chunks; }
		}
		return result;
	}

	// We simply split by row for now
	// TODO: There's other ways to split in 2D as well.
	std::vector<chunk<3>> split_equal(const chunk<2>& full_chunk, size_t num_chunks) {
		const auto rows =
		    split_equal(chunk<1>{cl::sycl::id<1>(full_chunk.offset[0]), cl::sycl::range<1>(full_chunk.range[0]), cl::sycl::range<1>(full_chunk.global_size[0])},
		        num_chunks);
		std::vector<chunk<3>> result;
		for(auto& row : rows) {
			result.push_back(
			    chunk<2>{cl::sycl::id<2>(row.offset[0], full_chunk.offset[1]), cl::sycl::range<2>(row.range[0], full_chunk.range[1]), full_chunk.global_size});
		}
		return result;
	}

	// We simply split by planes for now
	std::vector<chunk<3>> split_equal(const chunk<3>& full_chunk, size_t num_chunks) {
		assert(num_chunks > 0);

		const auto dim0_size = full_chunk.global_size[0];
		const auto dim1_size = full_chunk.global_size[1];
		const auto dim2_size = full_chunk.global_size[2];

		chunk<3> chnk;
		chnk.global_size = full_chunk.global_size;
		chnk.offset = full_chunk.offset;
		chnk.range = cl::sycl::range<3>(dim0_size / num_chunks, dim1_size, dim2_size);

		std::vector<chunk<3>> result;
		for(auto i = 0u; i < num_chunks; ++i) {
			result.push_back(chnk);
			chnk.offset[0] = chnk.offset[0] + chnk.range[0];
			if(i == num_chunks - 1) result[i].range[0] += dim0_size % num_chunks;
		}
		return result;
	}

	naive_split_transformer::naive_split_transformer(size_t num_chunks, size_t num_workers) : num_chunks(num_chunks), num_workers(num_workers) {
		assert(num_chunks >= num_workers);
	}

	void naive_split_transformer::transform_task(const std::shared_ptr<const task>& tsk, command_graph& cdag) {
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
		std::vector<chunk<3>> chunks;
		switch(tsk->get_dimensions()) {
		case 1: {
			chunks = split_equal(chunk<1>(full_chunk), num_chunks);
		} break;
		case 2: {
			chunks = split_equal(chunk<2>(full_chunk), num_chunks);
		} break;
		case 3: {
			chunks = split_equal(chunk<3>(full_chunk), num_chunks);
		} break;
		default: assert(false);
		}

		for(size_t i = 0; i < chunks.size(); ++i) {
			cdag.create<task_command>(nodes[i], tsk->get_id(), chunks[i]);
		}

		// Remove original
		cdag.erase(original);
	}

} // namespace detail
} // namespace celerity
