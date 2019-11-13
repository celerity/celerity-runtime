#pragma once

#include <list>
#include <type_traits>

#include <boost/optional.hpp>
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
			if(it != boost::none) {
				// Already exists, potentially upgrade to full dependency
				if((*it)->is_anti && !dep.is_anti) {
					(*it)->is_anti = false;
					// In this case we also have to upgrade corresponding dependent within dependency
					auto this_it = maybe_get_dep(dep.node->dependents, static_cast<T*>(this));
					assert(this_it != boost::none);
					assert((*this_it)->is_anti);
					(*this_it)->is_anti = false;
				}
				return;
			}

			dependencies.emplace_back(dep);
			dep.node->dependents.emplace_back(dependent{static_cast<T*>(this), dep.is_anti});
		}

		void remove_dependency(T* node) {
			auto it = maybe_get_dep(dependencies, node);
			if(it != boost::none) {
				{
					auto& dep_dependents = static_cast<intrusive_graph_node*>((*it)->node)->dependents;
					auto this_it = maybe_get_dep(dep_dependents, static_cast<T*>(this));
					assert(this_it != boost::none);
					dep_dependents.erase(*this_it);
				}
				dependencies.erase(*it);
			}
		}

		bool has_dependency(T* node, boost::optional<bool> is_anti = boost::none) {
			auto result = maybe_get_dep(dependencies, node);
			if(result == boost::none) return false;
			return is_anti != boost::none ? (*result)->is_anti == is_anti : true;
		}

		bool has_dependent(T* node, boost::optional<bool> is_anti = boost::none) {
			auto result = maybe_get_dep(dependents, node);
			if(result == boost::none) return false;
			return is_anti != boost::none ? (*result)->is_anti == is_anti : true;
		}

		auto get_dependencies() const { return boost::make_iterator_range(dependencies.cbegin(), dependencies.cend()); }
		auto get_dependents() const { return boost::make_iterator_range(dependents.cbegin(), dependents.cend()); }

	  private:
		std::list<dependency> dependencies;
		std::list<dependent> dependents;

		template <typename Dep>
		boost::optional<typename std::list<Dep>::iterator> maybe_get_dep(std::list<Dep>& deps, T* node) {
			auto it = std::find_if(deps.begin(), deps.end(), [&](auto d) { return d.node == node; });
			if(it == deps.end()) return boost::none;
			return it;
		}
	};

} // namespace detail
} // namespace celerity
