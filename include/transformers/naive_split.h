#pragma once

#include "graph_transformer.h"

namespace celerity {
namespace detail {

	class naive_split_transformer : public graph_transformer {
	  public:
		explicit naive_split_transformer(size_t num_workers);

		void transform_task(const std::shared_ptr<const task>& tsk, scoped_graph_builder& gb) override;

	  private:
		size_t num_workers;
	};

} // namespace detail
} // namespace celerity
