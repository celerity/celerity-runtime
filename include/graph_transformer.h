#pragma once

#include <memory>

namespace celerity {
namespace detail {

	class scoped_graph_builder;
	class task;

	class graph_transformer {
	  public:
		virtual void transform_task(const std::shared_ptr<const task>& tsk, scoped_graph_builder& gb) = 0;

		virtual ~graph_transformer() = default;
	};

} // namespace detail
} // namespace celerity
