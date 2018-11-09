#include "transformers/naive_split.h"

#include <cassert>
#include <numeric>
#include <vector>

#include "command.h"
#include "graph_builder.h"
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

	std::vector<chunk<3>> split_equal(const chunk<3>& full_chunk, size_t num_chunks) { throw std::runtime_error("3D split_equal NYI"); }

	naive_split_transformer::naive_split_transformer(size_t num_workers) : num_workers(num_workers) {}

	void naive_split_transformer::transform_task(const std::shared_ptr<const task>& tsk, scoped_graph_builder& gb) {
		if(tsk->get_type() != task_type::COMPUTE) return;
		const auto ctsk = dynamic_cast<const compute_task*>(tsk.get());
		if(num_workers == 1) return;

		std::vector<node_id> nodes(num_workers);
		std::iota(nodes.begin(), nodes.end(), 1);

		auto computes = gb.get_commands(command::COMPUTE);
		for(auto& cid : computes) {
			auto& cmd_data = gb.get_command_data(cid);
			const subrange<3> sr = cmd_data.data.compute.subrange;

			std::vector<chunk<3>> chunks;
			switch(ctsk->get_dimensions()) {
			case 1: {
				const chunk<1> full_chunk(cl::sycl::id<1>(sr.offset), cl::sycl::range<1>(sr.range), cl::sycl::range<1>(ctsk->get_global_size()));
				chunks = split_equal(full_chunk, num_workers);
			} break;
			case 2: {
				const chunk<2> full_chunk(cl::sycl::id<2>(sr.offset), cl::sycl::range<2>(sr.range), cl::sycl::range<2>(ctsk->get_global_size()));
				chunks = split_equal(full_chunk, num_workers);
			} break;
			case 3: {
				const chunk<3> full_chunk(cl::sycl::id<3>(sr.offset), cl::sycl::range<3>(sr.range), cl::sycl::range<3>(ctsk->get_global_size()));
				chunks = split_equal(full_chunk, num_workers);
			} break;
			default: assert(false);
			}

			gb.split_command(cid, chunks, nodes);
		}

		gb.commit();
	}

} // namespace detail
} // namespace celerity
