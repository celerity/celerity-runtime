#pragma once

#include <memory>

#include "types.h"

namespace celerity {

class task;

namespace detail {

	class scoped_graph_builder;

	class graph_transformer {
	  public:
		virtual void transform_task(const std::shared_ptr<const task>& tsk, scoped_graph_builder& gb) = 0;

		virtual ~graph_transformer() = default;
	};

} // namespace detail
} // namespace celerity
