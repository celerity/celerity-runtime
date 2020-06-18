#pragma once

#include "graph_transformer.h"

namespace celerity {
namespace detail {

	class naive_split_transformer : public graph_transformer {
	  public:
		/**
		 * The naive split transformer splits every DEVICE_COMPUTE command within a given task into a certain number of chunks.
		 *
		 * @arg num_chunks The number of chunks each DEVICE_COMPUTE command should be split into.
		 * @arg num_workers The number of workers that the resulting chunks should be assigned to.
		 */
		naive_split_transformer(size_t num_chunks, size_t num_workers);

		void transform_task(const std::shared_ptr<const task>& tsk, command_graph& cdag) override;

	  private:
		const size_t num_chunks;
		const size_t num_workers;
	};

} // namespace detail
} // namespace celerity
