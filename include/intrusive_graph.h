#pragma once

#include <algorithm>
#include <cassert>
#include <list>
#include <optional>
#include <type_traits>

#include <gch/small_vector.hpp>

namespace celerity {
namespace detail {

	enum class dependency_kind {
		anti_dep = 0, // Data anti-dependency, can be resolved by duplicating buffers
		true_dep = 1, // True data flow or temporal dependency
	};

	enum class dependency_origin {
		dataflow,                       // buffer access dependencies generate task and command dependencies
		side_effect_order,              // sequential-order side effects or side-effects behind epochs generate true dependencies
		collective_group_serialization, // all nodes must execute kernels within the same collective group in the same order
		execution_front,                // horizons and epochs are temporally ordered after all preceding tasks or commands on the same node
		last_epoch,                     // nodes without other true-dependencies require an edge to the last epoch for temporal ordering
	};

	enum class conflict_origin {
		collective_group,
		side_effect_order,
	};

	// TODO: Move to utility header..?
	template <typename Iterator>
	class iterable_range {
	  public:
		iterable_range(Iterator first, Iterator last) : m_first(first), m_last(last) {}

		Iterator begin() const { return m_first; }
		Iterator end() const { return m_last; }
		friend Iterator begin(const iterable_range& ir) { return ir.m_first; }
		friend Iterator end(const iterable_range& ir) { return ir.m_last; }

		auto& front() const { return *m_first; }
		bool empty() const { return m_first == m_last; }

	  private:
		Iterator m_first;
		Iterator m_last;
	};

	template <typename T>
	class intrusive_graph_node {
	  public:
		struct dependency {
			T* node;
			dependency_kind kind;
			dependency_origin origin; // context information for graph printing
		};

		using dependent = dependency;

		struct conflict {
			T* node;
			conflict_origin origin;
		};

	  public:
		intrusive_graph_node() { static_assert(std::is_base_of<intrusive_graph_node<T>, T>::value, "T must be child class (CRTP)"); }

	  protected:
		~intrusive_graph_node() { // protected: Statically disallow destruction through base pointer, since dtor is not polymorphic
			for(auto& dep : m_dependents) {
				auto dep_it_end = static_cast<intrusive_graph_node*>(dep.node)->m_dependencies.end();
				auto this_it =
				    std::find_if(static_cast<intrusive_graph_node*>(dep.node)->m_dependencies.begin(), dep_it_end, [&](auto d) { return d.node == this; });
				assert(this_it != dep_it_end);
				static_cast<intrusive_graph_node*>(dep.node)->m_dependencies.erase(this_it);
			}

			for(auto& dep : m_dependencies) {
				auto dep_it_end = static_cast<intrusive_graph_node*>(dep.node)->m_dependents.end();
				auto this_it =
				    std::find_if(static_cast<intrusive_graph_node*>(dep.node)->m_dependents.begin(), dep_it_end, [&](auto d) { return d.node == this; });
				assert(this_it != dep_it_end);
				static_cast<intrusive_graph_node*>(dep.node)->m_dependents.erase(this_it);
			}

			for(auto& dep : m_conflicts) {
				auto cf_it_end = static_cast<intrusive_graph_node*>(dep.node)->m_conflicts.end();
				auto this_it =
				    std::find_if(static_cast<intrusive_graph_node*>(dep.node)->m_conflicts.begin(), cf_it_end, [&](auto cf) { return cf.node == this; });
				assert(this_it != cf_it_end);
				static_cast<intrusive_graph_node*>(dep.node)->m_conflicts.erase(this_it);
			}
		}

	  public:
		void add_dependency(dependency dep) {
			// Check for (direct) cycles
			assert(!has_dependent(dep.node));

			if(const auto it = maybe_get_dep(m_dependencies, dep.node)) {
				// We assume that for dependency kinds A and B, max(A, B) is strong enough to satisfy both.
				static_assert(dependency_kind::anti_dep < dependency_kind::true_dep);

				// Already exists, potentially upgrade to full dependency
				if((*it)->kind < dep.kind) {
					(*it)->kind = dep.kind;
					(*it)->origin = dep.origin; // This unfortunately loses origin information from the lesser dependency

					// In this case we also have to upgrade corresponding dependent within dependency
					auto this_it = maybe_get_dep(dep.node->m_dependents, static_cast<T*>(this));
					assert(this_it != std::nullopt);
					assert((*this_it)->kind != dep.kind);
					(*this_it)->kind = dep.kind;
					(*this_it)->origin = dep.origin;
				}
				return;
			}

			m_dependencies.emplace_back(dep);
			dep.node->m_dependents.emplace_back(dependent{static_cast<T*>(this), dep.kind, dep.origin});

			m_pseudo_critical_path_length = std::max(m_pseudo_critical_path_length, dep.node->m_pseudo_critical_path_length + 1);

			// True dependencies serialize execution, so they subsume conflicts
			if(dep.kind == dependency_kind::true_dep) { remove_conflict(dep.node); }
		}

		void remove_dependency(T* node) {
			auto it = maybe_get_dep(m_dependencies, node);
			if(it != std::nullopt) {
				{
					auto& dep_dependents = static_cast<intrusive_graph_node*>((*it)->node)->m_dependents;
					auto this_it = maybe_get_dep(dep_dependents, static_cast<T*>(this));
					assert(this_it != std::nullopt);
					dep_dependents.erase(*this_it);
				}
				m_dependencies.erase(*it);
			}
		}

		bool has_dependency(T* node, std::optional<dependency_kind> kind = std::nullopt) {
			auto result = maybe_get_dep(m_dependencies, node);
			if(result == std::nullopt) return false;
			return kind != std::nullopt ? (*result)->kind == kind : true;
		}

		bool has_dependent(T* node, std::optional<dependency_kind> kind = std::nullopt) {
			auto result = maybe_get_dep(m_dependents, node);
			if(result == std::nullopt) return false;
			return kind != std::nullopt ? (*result)->kind == kind : true;
		}

		auto get_dependencies() const { return iterable_range{m_dependencies.cbegin(), m_dependencies.cend()}; }
		auto get_dependents() const { return iterable_range{m_dependents.cbegin(), m_dependents.cend()}; }

		int get_pseudo_critical_path_length() const { return m_pseudo_critical_path_length; }

		void add_conflict(conflict cf) {
			assert(cf.node != this);
			if(!has_conflict(cf.node) && !has_dependency(cf.node, dependency_kind::true_dep) && !has_dependent(cf.node, dependency_kind::true_dep)) {
				assert(!cf.node->has_conflict(static_cast<T*>(this)));
				m_conflicts.push_back(cf);
				cf.node->m_conflicts.push_back(conflict{static_cast<T*>(this), cf.origin});
			}
		}

		void remove_conflict(T* const node) {
			constexpr auto remove_conflict_direction = [](intrusive_graph_node* const from, intrusive_graph_node* const to) {
				const auto remove_begin = std::remove_if(from->m_conflicts.begin(), from->m_conflicts.end(), [=](const conflict& cf) { return cf.node == to; });
				from->m_conflicts.erase(remove_begin, from->m_conflicts.end());
			};
			remove_conflict_direction(this, node);
			remove_conflict_direction(node, this);
		}

		bool has_conflict(const T* const node) const {
			return std::find_if(m_conflicts.begin(), m_conflicts.end(), [&](const conflict& cf) { return cf.node == node; }) != m_conflicts.end();
		}

		auto get_conflicts() const { return iterable_range{m_conflicts.begin(), m_conflicts.end()}; }

	  private:
		gch::small_vector<dependency> m_dependencies;
		gch::small_vector<dependent> m_dependents;
		gch::small_vector<conflict> m_conflicts;

		// This only (potentially) grows when adding dependencies,
		// it never shrinks and does not take into account later changes further up in the dependency chain
		// (that is all that is needed for celerity use).
		int m_pseudo_critical_path_length = 0;

		template <typename Dep>
		std::optional<typename gch::small_vector<Dep>::iterator> maybe_get_dep(gch::small_vector<Dep>& deps, T* node) {
			auto it = std::find_if(deps.begin(), deps.end(), [&](auto d) { return d.node == node; });
			if(it == deps.end()) return std::nullopt;
			return it;
		}
	};

} // namespace detail
} // namespace celerity
