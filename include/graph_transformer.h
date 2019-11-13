#pragma once

#include <memory>

#include "command_graph.h"

namespace celerity {
namespace detail {

	class task;

	class graph_transformer {
	  public:
		virtual void transform_task(const std::shared_ptr<const task>& tsk, command_graph& cdag) = 0;

		virtual ~graph_transformer() = default;
	};

} // namespace detail
} // namespace celerity
