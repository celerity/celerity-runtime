#pragma once

#include <algorithm>
#include <cassert>
#include <optional>
#include <type_traits>

#include <gch/small_vector.hpp>

namespace celerity {
namespace detail {

	enum class dependency_kind {
		anti_dep = 0, // Data anti-dependency, can be resolved by duplicating buffers
		true_dep = 1, // True data flow or temporal dependency
	};

	// TODO this is purely debug info (as is dependency_kind, as it turns out). Move into dependency records and drop `struct dependency`.
	enum class dependency_origin {
		dataflow,                       // buffer access dependencies generate task and command dependencies
		collective_group_serialization, // all nodes must execute kernels within the same collective group in the same order
		execution_front,                // horizons and epochs are temporally ordered after all preceding tasks or commands on the same node
		last_epoch,                     // nodes without other true-dependencies require an edge to the last epoch for temporal ordering
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
		size_t size() const { return std::distance(m_first, m_last); }

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

	  public:
		intrusive_graph_node() { static_assert(std::is_base_of_v<intrusive_graph_node<T>, T>, "T must be child class (CRTP)"); }

	  protected:
		~intrusive_graph_node() { // protected: Statically disallow destruction through base pointer, since dtor is not polymorphic
			for(auto& dep : m_dependents) {
				auto this_it = find_by_node(dep.node->m_dependencies, static_cast<T*>(this));
				assert(this_it != dep.node->m_dependencies.end());
				dep.node->m_dependencies.erase(this_it);
			}

			for(auto& dep : m_dependencies) {
				auto this_it = find_by_node(dep.node->m_dependents, static_cast<T*>(this));
				assert(this_it != dep.node->m_dependents.end());
				dep.node->m_dependents.erase(this_it);
			}
		}

	  public:
		void add_dependency(dependency dep) {
			// Check for (direct) cycles
			assert(!has_dependent(dep.node));

			if(const auto it = find_by_node(m_dependencies, dep.node); it != m_dependencies.end()) {
				// We assume that for dependency kinds A and B, max(A, B) is strong enough to satisfy both.
				static_assert(dependency_kind::anti_dep < dependency_kind::true_dep);

				// Already exists, potentially upgrade to full dependency
				if(it->kind < dep.kind) {
					it->kind = dep.kind;
					it->origin = dep.origin; // This unfortunately loses origin information from the lesser dependency

					// In this case we also have to upgrade corresponding dependent within dependency
					auto this_it = find_by_node(dep.node->m_dependents, static_cast<T*>(this));
					assert(this_it != dep.node->m_dependents.end());
					assert(this_it->kind != dep.kind);
					this_it->kind = dep.kind;
					this_it->origin = dep.origin;
				}
				return;
			}

			m_dependencies.emplace_back(dep);
			dep.node->m_dependents.emplace_back(dependent{static_cast<T*>(this), dep.kind, dep.origin});

			m_pseudo_critical_path_length =
			    std::max(m_pseudo_critical_path_length, static_cast<intrusive_graph_node*>(dep.node)->m_pseudo_critical_path_length + 1);
		}

		void remove_dependency(const T* const node) {
			if(const auto it = find_by_node(m_dependencies, node); it != m_dependencies.end()) {
				auto& dep_dependents = static_cast<intrusive_graph_node*>(it->node)->m_dependents;
				auto this_it = find_by_node(dep_dependents, static_cast<T*>(this));
				assert(this_it != dep_dependents.end());
				dep_dependents.erase(this_it);
				m_dependencies.erase(it);
			}
		}

		bool has_dependency(const T* const node, const std::optional<dependency_kind> kind = std::nullopt) const {
			if(const auto it = find_by_node(m_dependencies, node); it != m_dependencies.end()) { return kind != std::nullopt ? it->kind == kind : true; }
			return false;
		}

		bool has_dependent(const T* const node, const std::optional<dependency_kind> kind = std::nullopt) const {
			if(const auto it = find_by_node(m_dependents, node); it != m_dependents.end()) { return kind != std::nullopt ? it->kind == kind : true; }
			return false;
		}

		auto get_dependencies() const { return iterable_range{m_dependencies.cbegin(), m_dependencies.cend()}; }
		auto get_dependents() const { return iterable_range{m_dependents.cbegin(), m_dependents.cend()}; }

		int get_pseudo_critical_path_length() const { return m_pseudo_critical_path_length; }

	  private:
		gch::small_vector<dependency> m_dependencies;

		// TODO This variable can be modified even after a DAG node is final, which easily leads to data races when a DAG is created by one thread and read /
		// processed by another. Remove this member altogether after refactoring TDAG / CDAG generation to work without it.
		gch::small_vector<dependent> m_dependents;

		// This only (potentially) grows when adding dependencies,
		// it never shrinks and does not take into account later changes further up in the dependency chain
		// (that is all that is needed for celerity use).
		int m_pseudo_critical_path_length = 0;

		template <typename Range>
		static auto find_by_node(Range& rng, const T* const node) {
			using std::begin, std::end;
			return std::find_if(begin(rng), end(rng), [&](auto d) { return d.node == node; });
		}
	};

} // namespace detail
} // namespace celerity
