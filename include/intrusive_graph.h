#pragma once

#include <list>
#include <type_traits>
#include <optional>

#include <boost/range.hpp>

namespace celerity {
namespace detail {

	template <typename T>
	class intrusive_graph_node {
	  public:
		struct dependency {
			T* node;
			bool is_anti;
		};

		using dependent = dependency;

	  public:
		intrusive_graph_node() { static_assert(std::is_base_of<intrusive_graph_node<T>, T>::value, "T must be child class (CRTP)"); }

		virtual ~intrusive_graph_node() {
			for(auto& dep : dependents) {
				auto dep_it_end = static_cast<intrusive_graph_node*>(dep.node)->dependencies.end();
				auto this_it =
				    std::find_if(static_cast<intrusive_graph_node*>(dep.node)->dependencies.begin(), dep_it_end, [&](auto d) { return d.node == this; });
				assert(this_it != dep_it_end);
				static_cast<intrusive_graph_node*>(dep.node)->dependencies.erase(this_it);
			}

			for(auto& dep : dependencies) {
				auto dep_it_end = static_cast<intrusive_graph_node*>(dep.node)->dependents.end();
				auto this_it =
				    std::find_if(static_cast<intrusive_graph_node*>(dep.node)->dependents.begin(), dep_it_end, [&](auto d) { return d.node == this; });
				assert(this_it != dep_it_end);
				static_cast<intrusive_graph_node*>(dep.node)->dependents.erase(this_it);
			}
		}

		void add_dependency(dependency dep) {
			// Check for (direct) cycles
			assert(!has_dependent(dep.node));

			auto it = maybe_get_dep(dependencies, dep.node);
			if(it != std::nullopt) {
				// Already exists, potentially upgrade to full dependency
				if((*it)->is_anti && !dep.is_anti) {
					(*it)->is_anti = false;
					// In this case we also have to upgrade corresponding dependent within dependency
					auto this_it = maybe_get_dep(dep.node->dependents, static_cast<T*>(this));
					assert(this_it != std::nullopt);
					assert((*this_it)->is_anti);
					(*this_it)->is_anti = false;
				}
				return;
			}

			dependencies.emplace_back(dep);
			dep.node->dependents.emplace_back(dependent{static_cast<T*>(this), dep.is_anti});

			pseudo_critical_path_length = std::max(pseudo_critical_path_length, static_cast<intrusive_graph_node*>(dep.node)->pseudo_critical_path_length + 1);
		}

		void remove_dependency(T* node) {
			auto it = maybe_get_dep(dependencies, node);
			if(it != std::nullopt) {
				{
					auto& dep_dependents = static_cast<intrusive_graph_node*>((*it)->node)->dependents;
					auto this_it = maybe_get_dep(dep_dependents, static_cast<T*>(this));
					assert(this_it != std::nullopt);
					dep_dependents.erase(*this_it);
				}
				dependencies.erase(*it);
			}
		}

		bool has_dependency(T* node, std::optional<bool> is_anti = std::nullopt) {
			auto result = maybe_get_dep(dependencies, node);
			if(result == std::nullopt) return false;
			return is_anti != std::nullopt ? (*result)->is_anti == is_anti : true;
		}

		bool has_dependent(T* node, std::optional<bool> is_anti = std::nullopt) {
			auto result = maybe_get_dep(dependents, node);
			if(result == std::nullopt) return false;
			return is_anti != std::nullopt ? (*result)->is_anti == is_anti : true;
		}

		auto get_dependencies() const { return boost::make_iterator_range(dependencies.cbegin(), dependencies.cend()); }
		auto get_dependents() const { return boost::make_iterator_range(dependents.cbegin(), dependents.cend()); }

		unsigned get_pseudo_critical_path_length() const { return pseudo_critical_path_length; }

	  private:
		// TODO grep "list<" and think about each (here probably boost::small_vector)
		std::list<dependency> dependencies;
		std::list<dependent> dependents;

		// This only (potentially) grows when adding dependencies,
		// it never shrinks and does not take into account later changes further up in the dependency chain
		// (that is all that is needed for celerity use).
		unsigned pseudo_critical_path_length = 0;

		template <typename Dep>
		std::optional<typename std::list<Dep>::iterator> maybe_get_dep(std::list<Dep>& deps, T* node) {
			auto it = std::find_if(deps.begin(), deps.end(), [&](auto d) { return d.node == node; });
			if(it == deps.end()) return std::nullopt;
			return it;
		}
	};

} // namespace detail
} // namespace celerity
