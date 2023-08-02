#pragma once

#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <queue>
#include <type_traits>
#include <variant>
#include <vector>

#include <spdlog/fmt/fmt.h>

#include "grid.h"
#include "utils.h"

// Some toggles that affect performance (but also change the behavior!)
// TODO: Consider making these template arguments instead (inside some config object), and add these:
// - Preferred merge dimension
// - In-place updates
// - Localized updates
// - Min/max children
// - Linear vs quadratic split
#define CELERITY_DETAIL_REGION_MAP_MERGE_ON_UPDATE 1
#define CELERITY_DETAIL_REGION_MAP_CLAMP_RESULTS_TO_REQUEST_BOUNDARY 1
#define CELERITY_DETAIL_REGION_MAP_MERGE_RESULTS 1
#define CELERITY_DETAIL_REGION_MAP_USE_QUADRATIC_ASSIGNMENT_ON_INSERT 1

namespace celerity::detail {

struct region_map_testspy;

namespace region_map_detail {

	// TODO PERF: Do some experiments with these
	constexpr size_t max_children = 8;
	constexpr size_t min_children = 2;

	template <int D, int Dims>
	bool is_lo_inside(const box<Dims>& a, const box<Dims>& b) {
		static_assert(D < Dims);
		const auto a_min = a.get_min();
		if(a_min[D] <= b.get_min()[D]) return false;
		if(a_min[D] >= b.get_max()[D]) return false;
		return true;
	}

	template <int D, int Dims>
	bool is_hi_inside(const box<Dims>& a, const box<Dims>& b) {
		static_assert(D < Dims);
		const auto a_max = a.get_max();
		if(a_max[D] <= b.get_min()[D]) return false;
		if(a_max[D] >= b.get_max()[D]) return false;
		return true;
	}

	template <int Dims>
	box<Dims> compute_bounding_box(const box<Dims>& a, const box<Dims>& b) {
		const auto min_a = a.get_min();
		const auto min_b = b.get_min();
		const auto max_a = a.get_max();
		const auto max_b = b.get_max();
		auto new_min = min_a;
		auto new_max = max_a;
		for(size_t d = 0; d < Dims; ++d) {
			new_min[d] = std::min(min_a[d], min_b[d]);
			new_max[d] = std::max(max_a[d], max_b[d]);
		}
		return {new_min, new_max};
	}

	template <int Dims>
	bool do_overlap(const box<Dims>& a, const box<Dims>& b) {
		return !box_intersection(a, b).empty();
	}

	template <int Dims>
	bool is_inside(const box<Dims>& box, const id<Dims>& point) {
		auto box_min = box.get_min();
		auto box_max = box.get_max();
		bool inside = true;
		for(size_t d = 0; d < Dims; ++d) {
			// Min is inclusive and max exclusive
			inside &= box_min[d] <= point[d] && box_max[d] > point[d];
		}
		return inside;
	}

	/**
	 * Check that the region map's tree structure is in a good state:
	 * - Root bounding box is equal to extent
	 * - Bounding box hierarchy is correct
	 * - Depth is set correctly on all nodes
	 * - No node is overfull
	 */
	template <typename RegionMap>
	void sanity_check_region_map(const RegionMap& rm) {
#if !defined(NDEBUG)
		assert(rm.m_root->get_bounding_box() == rm.m_extent);
		rm.m_root->sanity_check_bounding_boxes();

		size_t max_depth = 0;
		std::queue<std::pair<box<RegionMap::dimensions>, const typename RegionMap::types::inner_node_type*>> node_queue;
		node_queue.push(std::make_pair(rm.m_root->get_bounding_box(), rm.m_root.get()));

		while(!node_queue.empty()) {
			const auto [node_bbox, node] = node_queue.front();
			node_queue.pop();

			assert(node->get_depth() >= max_depth); // We're traversing breadth-first
			max_depth = std::max(max_depth, node->get_depth());
			if(node->contains_leaves()) { assert(node->get_depth() == max_depth); }

			assert(node->m_children.size() == node->m_child_boxes.size());
			// TODO: This can actually fail for non-root nodes as well, as we currently do not try to balance assignments.
			// assert(node->m_children.size() >= MIN_CHILDREN || node == m_root.get());
			assert(node->m_children.size() <= max_children);

			for(size_t i = 0; i < node->m_children.size(); ++i) {
				if(!node->contains_leaves()) {
					const auto& child_node = node->get_child_node(i);
					assert(child_node.get_depth() == node->get_depth() + 1);
					node_queue.push(std::make_pair(node->m_child_boxes[i], &child_node));
				}
			}
		}
#endif
	}

	template <typename ValueType, int Dims>
	class inner_node;

	/**
	 * Convenience types shared by inner_node and region_map_impl.
	 */
	template <typename ValueType, int Dims>
	class region_map_types {
	  public:
		static_assert(Dims <= 3);

		using inner_node_type = inner_node<ValueType, Dims>;
		using unique_inner_node_ptr = std::unique_ptr<inner_node_type>;
		using inner_node_child_type = std::variant<unique_inner_node_ptr, ValueType>;
		using entry = std::pair<box<Dims>, ValueType>;

		struct insert_node_action {
			box<Dims> box;
			ValueType value;
			bool processed_locally = false;
		};

		struct erase_node_action {
			box<Dims> box;
			bool processed_locally = false;
		};

		using update_action = std::variant<insert_node_action, erase_node_action>;
		using orphan = std::pair<box<Dims>, inner_node_child_type>;

		struct insert_result {
			unique_inner_node_ptr spilled_node;
			// This should always be the same as spilled_node->get_bounding_box (TODO: assert?)
			box<Dims> spilled_box;
		};
	};

	template <typename ValueType, int Dims>
	class inner_node {
		friend struct celerity::detail::region_map_testspy;

		using types = region_map_types<ValueType, Dims>;

	  public:
		inner_node(bool contains_leaves, size_t depth) : m_depth(depth), m_contains_leaves(contains_leaves) {
			// During splits we temporarily need to store one additional child
			m_children.reserve(max_children + 1);
		}

		~inner_node() = default;

		inner_node(const inner_node&) = delete;
		inner_node(inner_node&&) noexcept = default;
		inner_node& operator=(const inner_node&) = delete;
		inner_node& operator=(inner_node&&) noexcept = default;

		/**
		 * Whether this node contains leaves, i.e. ValueType entries, or more inner_nodes.
		 */
		bool contains_leaves() const { return m_contains_leaves; }

		size_t num_children() const { return m_children.size(); }

		size_t get_depth() const { return m_depth; }

		/**
		 * Recursively sets depth on this node and all of its children.
		 */
		void set_depth(const size_t depth) {
			this->m_depth = depth;
			if(!m_contains_leaves) {
				for(size_t i = 0; i < m_children.size(); ++i) {
					get_child_node(i).set_depth(depth + 1);
				}
			}
		}

		/**
		 * Either updates the value of the given box directly, or prepares the subtree for insertion of said entry by creating
		 * a hole of the appropriate size.
		 *
		 * Inserting a new value usually means splitting up existing boxes within the tree (for example placing a smaller
		 * rectangle inside a larger one results in 5 total rectangles). This function calculates the set of actions
		 * required to perform such a split, which are then dispatched from the root.
		 *
		 * There are two special cases that can be handled more efficiently:
		 * - If the number of new boxes created due to a split is determined to fit inside this subtree without overflowing
		 *   it, a localized update is performed, and no actions need to be dispatched from the root.
		 * - If the box to be updated matches an existing box inside this subtree exactly, an in-place update is performed
		 *   and no further actions are required.
		 *
		 * @param actions The list of erase and insert actions required to create a hole for the new entry.
		 * @returns True if a localized update operation was performed that may require a bounding box recomputation.
		 */
		bool update_box(const box<Dims>& box, const ValueType& value, std::vector<typename types::update_action>& actions) {
			if(!m_contains_leaves) {
				bool any_child_did_local_update = false;
				for(size_t i = 0; i < m_child_boxes.size(); ++i) {
					if(do_overlap<Dims>(m_child_boxes[i], box)) {
						const auto did_local_update = get_child_node(i).update_box(box, value, actions);
						if(did_local_update) {
							m_child_boxes[i] = get_child_node(i).get_bounding_box();
							any_child_did_local_update = true;
						}
					}
				}
				return any_child_did_local_update;
			}

			size_t erase_action_count = 0;
			const auto previous_action_count = actions.size();

			for(size_t i = 0; i < m_child_boxes.size(); ++i) {
				const auto& child_box = m_child_boxes[i];
				if(!do_overlap<Dims>(child_box, box)) continue;

				if(box == child_box) {
					// Exact overlap. Simply update box in-place.
					get_child_value(i) = value;
					break;
				}

				actions.push_back(typename types::erase_node_action{child_box});
				erase_action_count++;

				// Full overlap, no need to split anything.
				// TODO PERF: We may already determine this in do_overlap (return either "partial" or "full" enum?)
				if(box.covers(child_box)) { continue; }

				// Partial overlap. Check in each dimension which sides of the box intersect with the current box, creating new boxes along the way.
				// TODO PERF: A split may not even be necessary, if the value remains the same. Is this something worth optimizing for?
				detail::box<Dims> remainder = child_box;

				const auto& child_value = get_child_value(i);

				const auto split_along = [&](const auto dim) {
					if(is_lo_inside<dim.value, Dims>(box, child_box)) {
						auto new_box_max = remainder.get_max();
						new_box_max[dim.value] = box.get_min()[dim.value];
						const auto new_box = detail::box(remainder.get_min(), new_box_max);

						auto new_remainder_min = remainder.get_min();
						new_remainder_min[dim.value] = box.get_min()[dim.value];
						remainder = detail::box(new_remainder_min, remainder.get_max());

						actions.push_back(typename types::insert_node_action{new_box, child_value});
					}
					if(is_hi_inside<dim.value, Dims>(box, child_box)) {
						auto new_box_min = remainder.get_min();
						new_box_min[dim.value] = box.get_max()[dim.value];
						const auto new_box = detail::box(new_box_min, remainder.get_max());

						auto new_remainder_max = remainder.get_max();
						new_remainder_max[dim.value] = box.get_max()[dim.value];
						remainder = detail::box(remainder.get_min(), new_remainder_max);

						actions.push_back(typename types::insert_node_action{new_box, child_value});
					}
				};

				// TODO PERF: We might want to switch the order of these checks to maximize the length of boxes along the fastest changing dimension.
				if constexpr(Dims >= 3) { split_along(std::integral_constant<int, 2>()); }
				if constexpr(Dims >= 2) { split_along(std::integral_constant<int, 1>()); }
				if constexpr(Dims >= 1) { split_along(std::integral_constant<int, 0>()); }
			}

			// If we didn't do anything or an in-place update, stop here.
			if(actions.size() == previous_action_count) return false;

			// Otherwise check whether we can perform all actions locally.
			const size_t insert_action_count = actions.size() - previous_action_count - erase_action_count;
			const size_t new_num_children = m_children.size() + insert_action_count - erase_action_count;

			// We can only process actions locally if we have enough space for all children.
			const bool wont_overflow = new_num_children <= max_children;
			// However, we also must ensure that we don't end up with too few children, as we otherwise degrade tree health over time.
			const bool wont_underflow = new_num_children >= min_children;

			if(wont_overflow && wont_underflow) {
				// First, process all erases.
				for(size_t i = previous_action_count; i < actions.size(); ++i) {
					if(auto* const erase_action = std::get_if<typename types::erase_node_action>(&actions[i])) {
						erase_action->processed_locally = true;
						std::vector<typename types::orphan> orphans(0);
						[[maybe_unused]] const auto did_erase = erase(erase_action->box, orphans);
						assert(did_erase);
						assert(orphans.empty()); // This should never happen as we are in a leaf node.
					}
				}

				// Now, process the inserts.
				for(size_t i = previous_action_count; i < actions.size(); ++i) {
					if(auto* const insert_action = std::get_if<typename types::insert_node_action>(&actions[i])) {
						insert_action->processed_locally = true;
						assert(m_children.size() < max_children);
						insert_child_value(insert_action->box, insert_action->value);
					}
				}

				assert(m_children.size() == new_num_children);
				return true;
			}

			return false;
		}

		template <typename Functor>
		void apply_to_values(const Functor& f, std::vector<typename types::entry>& updated_nodes) {
			if(!m_contains_leaves) {
				for(size_t i = 0; i < m_children.size(); ++i) {
					get_child_node(i).apply_to_values(f, updated_nodes);
				}
				return;
			}

			for(size_t i = 0; i < m_children.size(); ++i) {
				ValueType& child_value = get_child_value(i);
				auto new_value = f(child_value);
				if(new_value != child_value) {
					child_value = std::move(new_value);
					updated_nodes.push_back(std::make_pair(m_child_boxes[i], child_value));
				}
			}
		}

		/**
		 * Inserts a the provided entry into the tree. It is assumed that the box fits
		 * into a currently existing hole and does not overlap with any other box in
		 * the subtree.
		 *
		 * @returns If the insertion caused a node to be split, the spilled node is returned.
		 *
		 * TODO: Structurally very similar to insert_subtree - can we DRY up?
		 */
		std::optional<typename types::insert_result> insert(const box<Dims>& box, const ValueType& value) {
			if(!m_contains_leaves) {
				// Value belongs deeper into the tree. Find child that best fits it.
				// TODO PERF: Resolve ties in area increase according to [Guttman 1984]
				size_t best_i = std::numeric_limits<size_t>::max();
				size_t smallest_area_delta = std::numeric_limits<size_t>::max();
				for(size_t i = 0; i < m_child_boxes.size(); ++i) {
					const auto area_delta = compute_bounding_box(m_child_boxes[i], box).get_area() - m_child_boxes[i].get_area();
					if(area_delta < smallest_area_delta) {
						smallest_area_delta = area_delta;
						best_i = i;
					}
				}
				assert(best_i < m_children.size());
				assert(smallest_area_delta < std::numeric_limits<size_t>::max());

				auto ret = get_child_node(best_i).insert(box, value);

				// Bounding box of child might have changed.
				// TODO PERF: I think we can skip this if area_delta == 0 and child was not split
				m_child_boxes[best_i] = get_child_node(best_i).get_bounding_box();

				sanity_check_bounding_boxes();

				if(ret.has_value()) {
					assert(ret->spilled_node->m_depth == m_depth + 1);
					return insert_subtree(ret->spilled_box, std::move(ret->spilled_node));
				}

				return std::nullopt;
			}

			// Try inserting value directly, or...
			if(m_children.size() < max_children) {
				insert_child_value(box, value);
				return std::nullopt;
			}

			// ...split if we are full (include new value in split decision).
			insert_child_value(box, value);

			auto [seed1, seed2] = pick_split_seeds();

			auto node1 = std::make_unique<inner_node>(true, m_depth);
			node1->insert_child_value(m_child_boxes[seed1], std::move(std::get<ValueType>(m_children[seed1])));
			auto node2 = std::make_unique<inner_node>(true, m_depth);
			node2->insert_child_value(m_child_boxes[seed2], std::move(std::get<ValueType>(m_children[seed2])));

// TODO PERF: Quadratic does seem to perform better at least for some scenarios (e.g. walking full tree).
#if !CELERITY_DETAIL_REGION_MAP_USE_QUADRATIC_ASSIGNMENT_ON_INSERT
			auto bbox1 = m_child_boxes[seed1];
			auto bbox2 = m_child_boxes[seed2];
			for(size_t i = 0; i < m_child_boxes.size(); ++i) {
				if(i == seed1 || i == seed2) continue;

				const auto new_bbox1 = compute_bounding_box(bbox1, m_child_boxes[i]);
				const auto new_bbox2 = compute_bounding_box(bbox2, m_child_boxes[i]);

				// Assign value to node that results in smaller area increase.
				if((new_bbox1.area() - bbox1.area()) < (new_bbox2.area() - bbox2.area())) {
					node1->insert_child_value(m_child_boxes[i], std::move(std::get<ValueType>(m_children[i])));
					bbox1 = new_bbox1;
				} else {
					node2->insert_child_value(m_child_boxes[i], std::move(std::get<ValueType>(m_children[i])));
					bbox2 = new_bbox2;
				}
			}
#else
			// Greedily assign all values to groups, O(N^2)
			auto bbox1 = m_child_boxes[seed1];
			auto bbox2 = m_child_boxes[seed2];
			auto area1 = bbox1.get_area();
			auto area2 = bbox2.get_area();
			std::vector<bool> assigned(m_children.size(), false);
			assigned[seed1] = true;
			assigned[seed2] = true;
			size_t num_assigned = 2;
			while(num_assigned < m_children.size()) {
				size_t smallest_area_delta = std::numeric_limits<size_t>::max();
				size_t smallest_i = std::numeric_limits<size_t>::max();
				detail::box<Dims> smallest_bbox;
				size_t smallest_area = 0;
				size_t target_node = 0;

				for(size_t i = 0; i < m_children.size(); ++i) {
					if(assigned[i]) continue;

					const auto new_bbox1 = compute_bounding_box(m_child_boxes[i], bbox1);
					const auto new_bbox2 = compute_bounding_box(m_child_boxes[i], bbox2);
					const auto new_area1 = new_bbox1.get_area();
					const auto new_area2 = new_bbox2.get_area();

					const auto ad1 = (new_area1 - area1);
					const auto ad2 = (new_area2 - area2);

					if(ad1 < smallest_area_delta) {
						smallest_area_delta = ad1;
						smallest_i = i;
						smallest_bbox = new_bbox1;
						smallest_area = new_area1;
						target_node = 1;
					}
					if(ad2 < smallest_area_delta) {
						smallest_area_delta = ad2;
						smallest_i = i;
						smallest_bbox = new_bbox2;
						smallest_area = new_area2;
						target_node = 2;
					}
				}

				assert(target_node != 0);
				if(target_node == 1) {
					node1->insert_child_value(m_child_boxes[smallest_i], std::move(std::get<ValueType>(m_children[smallest_i])));
					bbox1 = smallest_bbox;
					area1 = smallest_area;
				} else {
					node2->insert_child_value(m_child_boxes[smallest_i], std::move(std::get<ValueType>(m_children[smallest_i])));
					bbox2 = smallest_bbox;
					area2 = smallest_area;
				}

				assigned[smallest_i] = true;
				num_assigned++;
			}
#endif

			assert(!node1->m_children.empty());
			assert(!node2->m_children.empty());
			assert(node1->m_children.size() <= max_children);
			assert(node2->m_children.size() <= max_children);
			// TODO: This is currently not guaranteed; we may want to balance insertions if area increase is a tie.
			// assert(!node1.is_underfull());
			// assert(!node2.is_underfull());

			typename types::insert_result result{std::move(node2), bbox2};

			// Replace this with node 1, return node 2
			*this = std::move(*node1);
			sanity_check_bounding_boxes();
			return result;
		}


		/**
		 * Inserts the given subtree as a child into this subtree, either directly
		 * or further down (depending on its depth).
		 *
		 * @returns If the insertion caused a node to be split, the spilled node is returned.
		 *
		 * TODO: Structurally very similar to insert - can we DRY up?
		 */
		std::optional<typename types::insert_result> insert_subtree(const box<Dims>& box, std::unique_ptr<inner_node<ValueType, Dims>>&& subtree) {
			assert(!m_contains_leaves);
			assert(subtree->m_depth > m_depth);

			// Check if subtree should be inserted as child of this node.
			if(subtree->m_depth > m_depth + 1) {
				// Subtree belongs deeper into the tree. Find child that best fits it.
				// TODO PERF: Resolve ties in area increase according to [Guttman 1984]
				size_t best_i = std::numeric_limits<size_t>::max();
				size_t smallest_area_delta = std::numeric_limits<size_t>::max();
				for(size_t i = 0; i < m_child_boxes.size(); ++i) {
					const auto area_delta = compute_bounding_box(m_child_boxes[i], box).get_area() - m_child_boxes[i].get_area();
					if(area_delta < smallest_area_delta) {
						smallest_area_delta = area_delta;
						best_i = i;
					}
				}
				assert(best_i < m_children.size());
				assert(smallest_area_delta < std::numeric_limits<size_t>::max());

				auto ret = get_child_node(best_i).insert_subtree(box, std::move(subtree));

				// Bounding box of child might have changed.
				// TODO PERF: I think we can skip this if area_delta == 0 and child was not split
				m_child_boxes[best_i] = get_child_node(best_i).get_bounding_box();

				sanity_check_bounding_boxes();

				if(ret.has_value()) {
					assert(ret->spilled_node->m_depth == m_depth + 1);
					return insert_subtree(ret->spilled_box, std::move(ret->spilled_node));
				}

				return std::nullopt;
			}

			// Try inserting value directly, or...
			if(m_children.size() < max_children) {
				insert_child_node(box, std::move(subtree));
				return std::nullopt;
			}

			// ...split if we are full (include new value in split decision).
			insert_child_node(box, std::move(subtree));

			auto [seed1, seed2] = pick_split_seeds();

			auto node1 = std::make_unique<inner_node>(false, m_depth);
			node1->insert_child_node(m_child_boxes[seed1], std::move(std::get<std::unique_ptr<inner_node>>(m_children[seed1])));
			auto node2 = std::make_unique<inner_node>(false, m_depth);
			node2->insert_child_node(m_child_boxes[seed2], std::move(std::get<std::unique_ptr<inner_node>>(m_children[seed2])));

			auto bbox1 = m_child_boxes[seed1];
			auto bbox2 = m_child_boxes[seed2];
			for(size_t i = 0; i < m_children.size(); ++i) {
				if(i == seed1 || i == seed2) continue;

				assert(get_child_node(i).m_depth == m_depth + 1); // Sanity check
				const auto new_bbox1 = compute_bounding_box(bbox1, m_child_boxes[i]);
				const auto new_bbox2 = compute_bounding_box(bbox2, m_child_boxes[i]);

				// Assign value to node that results in smaller area increase.
				if((new_bbox1.get_area() - bbox1.get_area()) < (new_bbox2.get_area() - bbox2.get_area())) {
					node1->insert_child_node(m_child_boxes[i], std::move(std::get<typename types::unique_inner_node_ptr>(m_children[i])));
					bbox1 = new_bbox1;
				} else {
					node2->insert_child_node(m_child_boxes[i], std::move(std::get<typename types::unique_inner_node_ptr>(m_children[i])));
					bbox2 = new_bbox2;
				}
			}

			assert(!node1->m_children.empty());
			assert(!node2->m_children.empty());
			assert(node1->m_children.size() <= max_children);
			assert(node2->m_children.size() <= max_children);
			// TODO: This is currently not guaranteed; we may want to balance insertions if area increase is a tie.
			// assert(!node1.is_underfull());
			// assert(!node2.is_underfull());

			typename types::insert_result result{std::move(node2), bbox2};

			// Replace this with node 1, return node 2
			*this = std::move(*node1);
			sanity_check_bounding_boxes();
			return result;
		}

		/**
		 * Erases a box if it is contained in the subtree.
		 *
		 * @param orphans A list of entries or subtrees that were orphaned due to dissolving a node.
		 * @returns True if the box was erased in this subtree.
		 */
		bool erase(const box<Dims>& box, std::vector<typename types::orphan>& orphans) {
			bool did_erase = false;

			if(!m_contains_leaves) {
				for(size_t i = 0; i < m_children.size(); ++i) {
					if(do_overlap<Dims>(box, m_child_boxes[i])) {
						auto& child = get_child_node(i);
						if(child.erase(box, orphans)) {
							did_erase = true;

							if(child.is_underfull()) {
								if(!child.m_children.empty()) {
									for(size_t j = 0; j < child.m_children.size(); ++j) {
										orphans.push_back(std::make_pair(child.m_child_boxes[j], std::move(child.m_children[j])));
									}
								}
								erase_child(i);
							} else {
								m_child_boxes[i] = get_child_node(i).get_bounding_box();
							}
						}
					}

					if(did_erase) break;
				}
			} else {
				for(size_t i = 0; i < m_children.size(); ++i) {
					if(m_child_boxes[i] == box) {
						did_erase = true;
						erase_child(i);
						break;
					}
				}
			}

			sanity_check_bounding_boxes();
			return did_erase;
		}

		/**
		 * Recursively finds all entries that intersect with box.
		 */
		void query(const box<Dims>& box, std::vector<typename types::entry>& intersecting) const {
			if(!m_contains_leaves) {
				for(size_t i = 0; i < m_children.size(); ++i) {
					if(do_overlap<Dims>(m_child_boxes[i], box)) { get_child_node(i).query(box, intersecting); }
				}
				return;
			}
			for(size_t i = 0; i < m_children.size(); ++i) {
				if(do_overlap<Dims>(m_child_boxes[i], box)) { intersecting.push_back(std::make_pair(m_child_boxes[i], get_child_value(i))); }
			}
		}

		/**
		 * Returns the entry containing a given point, if such an entry exists.
		 */
		std::optional<typename types::entry> point_query(const id<Dims>& point) const {
			for(size_t i = 0; i < m_children.size(); ++i) {
				if(is_inside<Dims>(m_child_boxes[i], point)) {
					if(!m_contains_leaves) {
						const auto result = get_child_node(i).point_query(point);
						if(result.has_value()) return result;
					} else {
						return std::make_pair(m_child_boxes[i], get_child_value(i));
					}
				}
			}
			return std::nullopt;
		}

		typename types::unique_inner_node_ptr eject_only_child() {
			assert(!m_contains_leaves);
			assert(m_children.size() == 1);
			auto child = std::move(std::get<typename types::unique_inner_node_ptr>(m_children[0]));
			m_children.clear();
			m_child_boxes.clear();
			return std::move(child);
		}

		// NOTE: Not O(1)!
		box<Dims> get_bounding_box() const {
			assert(!m_child_boxes.empty());
			box<Dims> bbox = m_child_boxes[0];
			for(size_t i = 1; i < m_child_boxes.size(); ++i) {
				bbox = compute_bounding_box(bbox, m_child_boxes[i]);
			}
			return bbox;
		}

		void insert_child_node(const box<Dims>& box, std::unique_ptr<inner_node>&& node) {
			assert(m_children.size() < max_children + 1); // During splits we temporarily go one above the max
			m_child_boxes.push_back(box);
			m_children.emplace_back(std::move(node));
		}

		template <typename Callback>
		void for_each(const Callback& cb) const {
			for(size_t i = 0; i < m_children.size(); ++i) {
				if(!m_contains_leaves) {
					get_child_node(i).for_each(cb);
				} else {
					cb(m_child_boxes[i], get_child_value(i));
				}
			}
		}

		auto format_to(fmt::format_context::iterator out, const size_t level) const {
			const auto padding = std::string(2 * level, ' ');
			auto bounding_box = get_bounding_box();
			if(!m_contains_leaves) {
				out = fmt::format_to(out, "{}inner node with bbox {} and {} children:\n", padding, bounding_box, m_children.size());
				for(size_t i = 0; i < m_children.size(); ++i) {
					out = get_child_node(i).format_to(out, level + 1);
				}
			} else {
				out = fmt::format_to(out, "{}leaf node with bbox {} and {} values:\n", padding, bounding_box, m_children.size());
				const auto value_padding = std::string(2 * (level + 1), ' ');
				for(size_t i = 0; i < m_children.size(); ++i) {
					auto& v = get_child_value(i);
					out = fmt::format_to(out, "{}{} : ", value_padding, m_child_boxes[i]);
					if constexpr(fmt::is_formattable<ValueType>::value) {
						out = fmt::format_to(out, "{}\n", v);
					} else {
						out = fmt::format_to(out, "(value not printable)\n");
					}
				}
			}
			return out;
		}

	  private:
		template <typename RegionMap>
		friend void sanity_check_region_map(const RegionMap& rm);

		size_t m_depth;

		bool m_contains_leaves;
		// TODO PERF: Consider storing these in small vectors
		std::vector<box<Dims>> m_child_boxes;
		std::vector<typename types::inner_node_child_type> m_children;

		inner_node& get_child_node(size_t index) { return *std::get<typename types::unique_inner_node_ptr>(m_children[index]); }
		const inner_node& get_child_node(size_t index) const { return *std::get<typename types::unique_inner_node_ptr>(m_children[index]); }

		ValueType& get_child_value(size_t index) { return std::get<ValueType>(m_children[index]); }
		const ValueType& get_child_value(size_t index) const { return std::get<ValueType>(m_children[index]); }

		void insert_child_value(const box<Dims>& box, const ValueType& value) {
			assert(m_children.size() < max_children + 1); // During splits we temporarily go one above the max
#if !defined(NDEBUG)
			for(auto& b : m_child_boxes) {
				// New box must not overlap with any other
				assert(box_intersection(b, box).empty());
			}
#endif
			m_child_boxes.push_back(box);
			m_children.emplace_back(value);
		}

		void erase_child(const size_t index) {
			m_child_boxes.erase(m_child_boxes.begin() + index);
			m_children.erase(m_children.begin() + index);
		}

		// TODO PERF: We're doing a quadratic split for now - revisit.
		std::pair<size_t, size_t> pick_split_seeds() {
			size_t worst_area = 0;
			size_t worst_i = std::numeric_limits<size_t>::max();
			size_t worst_j = std::numeric_limits<size_t>::max();
			for(size_t i = 0; i < m_child_boxes.size(); ++i) {
				for(size_t j = i + 1; j < m_child_boxes.size(); ++j) {
					const auto area = compute_bounding_box(m_child_boxes[i], m_child_boxes[j]).get_area();
					if(area > worst_area) {
						worst_area = area;
						worst_i = i;
						worst_j = j;
					}
				}
			}
			assert(worst_i < m_children.size());
			assert(worst_j < m_children.size());
			return std::make_pair(worst_i, worst_j);
		}

		bool is_underfull() const { return m_children.size() < min_children; }

		box<Dims> sanity_check_bounding_boxes() const {
#if !defined(NDEBUG)
			// After an erase this node might not have any children. Return empty box in that case. TODO this breaks for Dims == 0 (where area is always 1)!
			if(m_child_boxes.empty()) { return box_cast<Dims>(box<3>({0, 0, 0}, {0, 0, 0})); }

			box<Dims> result = m_child_boxes[0];
			for(size_t i = 1; i < m_child_boxes.size(); ++i) {
				const box<Dims> child_box = m_contains_leaves ? m_child_boxes[i] : get_child_node(i).sanity_check_bounding_boxes();
				assert(m_child_boxes[i] == child_box);
				result = compute_bounding_box(result, child_box);
			}
			return result;
#endif
			return {};
		}
	};

	inline void assert_dimensionality(const box<3>& box, const int dims) {
#if !defined(NDEBUG)
		assert(box.get_min_dimensions() <= dims);
#endif
	}

	inline void assert_dimensionality(const region<3>& reg, const int dims) {
#if !defined(NDEBUG)
		for(const auto& box : reg.get_boxes()) {
			assert_dimensionality(box, dims);
		}
#endif
	}

	/**
	 * The region map is implemented as a customized R-Tree [Guttman 1984]. In order to maintain
	 * performance over time, entries with compatible boxes and equal values will be merged.
	 *
	 * The implementation logic is split between this class, which acts as a wrapper around the root node,
	 * as well as inner_node, which implements the recursive tree operations. This class is responsible for
	 * dispatching the recursive calls as well as handling the merging of entries and reinsertion of orphaned
	 * entries/nodes after update operations. It is also responsible for merging the final list of query results.
	 *
	 * TODO PERF: Try to minimize the number of value copies we do during intermediate steps (e.g. when merging)
	 * TODO PERF: Look into bulk-loading algorithms for updating multiple boxes at once
	 */
	template <typename ValueType, int Dims>
	class region_map_impl {
		friend struct celerity::detail::region_map_testspy;
		using types = region_map_types<ValueType, Dims>;

	  public:
		using value_type = ValueType;
		static constexpr size_t dimensions = Dims;

		region_map_impl(const range<Dims>& extent, ValueType default_value = ValueType{})
		    : m_extent(subrange<Dims>({}, extent)), m_root(std::make_unique<typename types::inner_node_type>(true, 0)) {
			m_root->insert(this->m_extent, default_value);
		}

		~region_map_impl() = default;

		region_map_impl(const region_map_impl&) = delete;
		region_map_impl(region_map_impl&&) noexcept = default;
		region_map_impl& operator=(const region_map_impl&) = delete;
		region_map_impl& operator=(region_map_impl&&) noexcept = default;

		/**
		 * Updates the value for the provided box within the tree.
		 *
		 * This operation consists of roughly three steps:
		 *   1) Prepare the tree by creating a "hole" for the new box. This usually
		 *      means splitting existing boxes within the tree. The required set
		 *      of operations is propagated back up the tree.
		 *      In some situations an in-place or localized update can be performed,
		 *      in this case step 2 is skipped (see inner_node::update_box).
		 *   2) Perform all erase and insert operations calculated in step 1.
		 *   3) Attempt to merge the box as well as any other newly created boxes
		 *      with their surrounding entries.
		 */
		void update_box(const box<Dims>& box, const ValueType& value) {
			assert(m_root != nullptr && "Moved from?");

			const auto clamped_box = box_intersection(m_extent, box);

			// This can happen e.g. for empty buffers, or if the box is
			// completely outside the region map's extent for some reason.
			if(box.empty() > 0) return;

			m_update_actions.clear();
			m_root->update_box(clamped_box, value, m_update_actions);

			m_merge_candidates.clear();

			// If there are any actions it means there was no in-place update.
			if(!m_update_actions.empty()) {
				// In this case we have to insert the new box.
				m_update_actions.push_back(typename types::insert_node_action{clamped_box, value});
			} else {
				// Otherwise just check whether the in-place updated box can be merged.
				m_merge_candidates.push_back(std::make_pair(clamped_box, value));
			}

#if !defined(NDEBUG)
			// Sanity check: Erased and inserted boxes must cover the same space
			region<Dims> erased;
			region<Dims> inserted;
			for(const auto& a : m_update_actions) {
				utils::match(
				    a,
				    [&](const typename types::erase_node_action& erase_action) {
					    assert(region_intersection(erased, erase_action.box).empty());
					    erased = region_union(erased, erase_action.box);
				    },
				    [&](const typename types::insert_node_action& insert_action) {
					    assert(region_intersection(inserted, insert_action.box).empty());
					    inserted = region_union(inserted, insert_action.box);
				    });
			}
			assert(erased == inserted);
#endif

			for(const auto& a : m_update_actions) {
				utils::match(
				    a,
				    [&](const typename types::erase_node_action& erase_action) {
					    if(!erase_action.processed_locally) { erase(erase_action.box); }
				    },
				    [&](const typename types::insert_node_action& insert_action) {
					    if(!insert_action.processed_locally) { insert(insert_action.box, insert_action.value); }
					    // Even if the action was processed locally already, we still have to try and merge the new box.
					    m_merge_candidates.push_back(std::make_pair(insert_action.box, insert_action.value));
				    });
			}

			sanity_check_region_map(*this);

#if CELERITY_DETAIL_REGION_MAP_MERGE_ON_UPDATE
			try_merge(std::move(m_merge_candidates));
#endif

			sanity_check_region_map(*this);
		}

		/**
		 * Applies the provided functor to all values and attempts to merge all entries that had their values changed.
		 */
		template <typename Functor>
		void apply_to_values(const Functor& f) {
			assert(m_root != nullptr && "Moved from?");

			static_assert(std::is_same_v<std::invoke_result_t<Functor, ValueType>, ValueType>, "Functor must return value of same type");

			m_updated_nodes.clear();
			m_root->apply_to_values(f, m_updated_nodes);

#if CELERITY_DETAIL_REGION_MAP_MERGE_ON_UPDATE
			// Now attempt to merge boxes that had their value modified by the functor.
			try_merge(std::move(m_updated_nodes));
#endif

			sanity_check_region_map(*this);
		}

		/**
		 * Finds all entries intersecting with request, clamps them to the extent and merges them.
		 *
		 * TODO PERF: In most cases we are unlikely to store the returned values, and the copy is unnecessary. Return const reference instead?
		 */
		std::vector<typename types::entry> get_region_values(const box<Dims>& request) const {
			assert(m_root != nullptr && "Moved from?");

			m_query_results_raw.clear();
			m_root->query(request, m_query_results_raw);

#if !CELERITY_DETAIL_REGION_MAP_CLAMP_RESULTS_TO_REQUEST_BOUNDARY && !CELERITY_DETAIL_REGION_MAP_MERGE_RESULTS
			return m_query_results_raw;
#endif

#if CELERITY_DETAIL_REGION_MAP_CLAMP_RESULTS_TO_REQUEST_BOUNDARY
			// Clamp to query request box
			m_query_results_clamped.clear();
			for(auto& [b, v] : m_query_results_raw) {
				const auto r_min = request.get_min();
				const auto r_max = request.get_max();
				const auto v_min = b.get_min();
				const auto v_max = b.get_max();
				auto clamped_min = v_min;
				auto clamped_max = v_max;

				for(size_t d = 0; d < Dims; ++d) {
					clamped_min[d] = std::max(v_min[d], r_min[d]);
					clamped_max[d] = std::min(v_max[d], r_max[d]);
				}
				m_query_results_clamped.push_back(std::make_pair(box<Dims>{clamped_min, clamped_max}, v));
			}
#else
			std::swap(m_query_results_raw, m_query_results_clamped);
#endif

#ifdef NDEBUG
			// In 1D everything that can be merged will be merged on update.
			// (Nevertheless, assert this in debug builds).
			if(Dims == 1) return m_query_results_clamped;
#endif

#if !CELERITY_DETAIL_REGION_MAP_MERGE_RESULTS
			return m_query_results_clamped;
#endif

			// Do a greedy quadratic merge
			// TODO PERF: Can we come up with a more efficient solution here? Maybe some sort of line-sweeping algorithm?
			bool did_merge = true;
			std::vector<bool> is_merged(m_query_results_clamped.size(), false);
			while(did_merge) {
				did_merge = false;
				for(size_t i = 0; i < m_query_results_clamped.size(); ++i) {
					if(is_merged[i]) continue;
					for(size_t j = i + 1; j < m_query_results_clamped.size(); ++j) {
						if(is_merged[j]) continue;
						if(m_query_results_clamped[i].second != m_query_results_clamped[j].second) continue;
						if(can_merge(m_query_results_clamped[i].first, m_query_results_clamped[j].first)) {
							assert(Dims > 1 || !CELERITY_DETAIL_REGION_MAP_MERGE_ON_UPDATE); // 1D should already have merged on update.
							// TODO PERF: Computing the bbox from scratch isn't ideal, as we really only need to adjust one dimension.
							m_query_results_clamped[i].first = compute_bounding_box(m_query_results_clamped[i].first, m_query_results_clamped[j].first);
							is_merged[j] = true;
							did_merge = true;
						}
					}
				}
			}
			std::vector<typename types::entry> results_merged;
			for(size_t i = 0; i < m_query_results_clamped.size(); ++i) {
				if(!is_merged[i]) results_merged.emplace_back(std::move(m_query_results_clamped[i]));
			}

			return results_merged;
		}

		auto format_to(fmt::format_context::iterator out) const {
			out = fmt::format_to(out, "Region Map\n");
			return m_root->format_to(out, 0);
		}

		range<Dims> get_extent() const { return m_extent.get_range(); }

	  private:
		template <typename RegionMap>
		friend void sanity_check_region_map(const RegionMap& rm);

		// The extent specifies the boundaries for the region map to which all entries are clamped,
		// and which initially contains the default value. Currently always starts at [0,0,0].
		box<Dims> m_extent;

		std::unique_ptr<typename types::inner_node_type> m_root;

		// These vectors are frequently used during updates, queries etc.
		// We keep them here as to not have to allocate them from scratch every time,
		// even though they are not persisting any class state.
		std::vector<typename types::update_action> m_update_actions;
		std::vector<typename types::entry> m_merge_candidates;
		std::vector<typename types::entry> m_updated_nodes;
		std::vector<typename types::orphan> m_erase_orphans;
		mutable std::vector<typename types::entry> m_query_results_raw;
		mutable std::vector<typename types::entry> m_query_results_clamped;

		/**
		 * Inserts a new entry into the tree.
		 * Precondition: The insert location must be empty.
		 */
		void insert(const box<Dims>& box, const ValueType& value) {
			auto ret = m_root->insert(box, value);
			if(ret.has_value()) { reroot(std::move(*ret)); }
		}

		/**
		 * Inserts a subtree (either from a dissolved parent or after a split) into the tree.
		 */
		void insert_subtree(const box<Dims>& box, typename types::unique_inner_node_ptr&& subtree) {
			auto ret = m_root->insert_subtree(box, std::move(subtree));
			if(ret.has_value()) { reroot(std::move(*ret)); }
		}

		/**
		 * Creates a new root node that is parent to the current root node and new_sibling,
		 * increasing the tree's height by 1.
		 */
		void reroot(typename types::insert_result new_sibling) {
			auto new_root = std::make_unique<typename types::inner_node_type>(false, 0);
			const auto old_root_bbox = m_root->get_bounding_box();
			new_root->insert_child_node(old_root_bbox, std::move(m_root));
			new_root->insert_child_node(new_sibling.spilled_box, std::move(new_sibling.spilled_node));
			m_root = std::move(new_root);
			m_root->set_depth(0);
		}

		/**
		 * Erases a box from the tree. If the parent box becomes underfull it is dissolved and its children
		 * are reinserted.
		 */
		void erase(const box<Dims>& box) {
			m_erase_orphans.clear();
			[[maybe_unused]] const auto did_erase = m_root->erase(box, m_erase_orphans);
			assert(did_erase);

			for(auto& o : m_erase_orphans) {
				utils::match(
				    o.second,                                                                                    //
				    [&](ValueType& v) { insert(o.first, v); },                                                   //
				    [&](typename types::unique_inner_node_ptr& in) { insert_subtree(o.first, std::move(in)); }); //
			}

			if(!m_root->contains_leaves() && m_root->num_children() == 1) {
				// Decrease tree height by 1 level.
				m_root = std::move(m_root->eject_only_child());
				m_root->set_depth(0);
			}
		}

		/**
		 * Calculates whether two boxes can be merged. In order to be mergeable, the two boxes
		 * have to touch in one dimension and match exactly in all remaining dimensions.
		 */
		bool can_merge(const box<Dims>& box_a, const box<Dims>& box_b) const {
			bool adjacent = false;
			for(size_t d = 0; d < Dims; ++d) {
				if(box_a.get_min()[d] != box_b.get_min()[d] || box_a.get_max()[d] != box_b.get_max()[d]) {
					// Dimension does not match exactly, but could still be adjacent.
					// If we already are adjacent in another dimension, we cannot merge.
					if(!adjacent && (box_a.get_max()[d] == box_b.get_min()[d] || box_b.get_max()[d] == box_a.get_min()[d])) {
						adjacent = true;
					} else {
						return false;
					}
				}
			}

			assert(adjacent);
			return true;
		}

		/**
		 * Try to merge a list of candidate entries with their neighbors within the tree.
		 */
		void try_merge(std::vector<typename types::entry>&& merge_candidates) {
#if !defined(NDEBUG)
			// Sanity check: Merge candidates do not overlap
			region<Dims> candidate_union;
			for(auto& [box, value] : merge_candidates) {
				assert(region_intersection(candidate_union, box).empty());
				candidate_union = region_union(candidate_union, box);
			}
#endif

			// For each candidate, probe around it in every direction to check whether there is a box with same value.
			//   If yes, check if it can be merged.
			//     If yes, erase the two boxes, insert the new one and add it as a merge candidate.
			// Repeat until no more merges are possible.
			bool did_merge = true;
			std::vector<bool> merged(merge_candidates.size(), false);
			while(did_merge) {
				did_merge = false;

				for(size_t i = 0; i < merge_candidates.size(); ++i) {
					if(merged[i]) continue;
					auto& [box, value] = merge_candidates[i];

					// TODO PERF: Order of dimensions can affect merge results
					for(size_t d = 0; d < Dims; ++d) {
						const auto min = box.get_min();
						const auto max = box.get_max();
						std::optional<detail::box<Dims>> other_box;
						if(min[d] > 0) {
							auto probe = min;
							probe[d] -= 1;
							const auto neighbor = m_root->point_query(probe);
							assert(neighbor != std::nullopt);
							if(neighbor->second == value && can_merge(box, neighbor->first)) { other_box = neighbor->first; }
						}
						if(!other_box.has_value() && max[d] < m_extent.get_max()[d]) {
							auto probe = min;
							// Point queries are exclusive on the "max" coordinate of a box, so there is no need to increment this by 1.
							// In fact, we would miss boxes that are exactly 1 unit wide in this dimension if we incremented.
							probe[d] = max[d];
							const auto neighbor = m_root->point_query(probe);
							assert(neighbor != std::nullopt);
							if(neighbor->second == value && can_merge(box, neighbor->first)) { other_box = neighbor->first; }
						}

						if(other_box.has_value()) {
							// First figure out whether this box is also in our candidates list, and if so, remove it.
							for(size_t j = 0; j < merge_candidates.size(); ++j) {
								if(j == i) continue;
								if(merge_candidates[j].first == other_box) {
									assert(merged[j] == false);
									merged[j] = true;
									break;
								}
							}

							// Now erase the two boxes, insert the merged one and mark it as a new candidate.
							erase(box);
							erase(*other_box);
							const auto new_box = compute_bounding_box(box, *other_box);
							insert(new_box, value);

							// Overwrite merge candidate with new box for next round
							merge_candidates[i].first = new_box;

							did_merge = true;
							break; // No need to check other dimensions, move on to next candidate box.
						}
					}
				}
			}

			// Cache candidate allocation (yes, it's ugly)
			m_merge_candidates = std::move(merge_candidates);
		}

		/**
		 * Invokes the provided callback for every entry (box/value pair) within the region map,
		 * for debugging / testing / instrumentation.
		 */
		template <typename Callback>
		void for_each(const Callback& cb) const {
			return m_root->for_each(cb);
		}
	};

	// Specialization for 0-dimensional buffers (= a single value of type ValueType).
	// NOTE: AllScale boxes don't support 0 dimensions. We use 1 for now.
	template <typename ValueType>
	class region_map_impl<ValueType, 0> {
	  public:
		region_map_impl(const range<0>& /* extent */, ValueType default_value) : m_value(default_value) {}

		void update_box(const box<1>& /* box */, const ValueType& value) { m_value = value; }

		std::vector<std::pair<box<1>, ValueType>> get_region_values(const box<1>& /* request */) const { return {{box<1>{0, 1}, m_value}}; }

		template <typename Functor>
		void apply_to_values(const Functor& f) {
			m_value = f(m_value);
		}

	  private:
		ValueType m_value;
	};

} // namespace region_map_detail

/**
 * The region_map is a spatial data structure for storing values within an n-dimensional extent.
 * Each point within the extent can hold a single value of type ValueType, and all points are initially
 * set to a provided default value.
 */
template <typename ValueType>
class region_map {
	friend struct region_map_testspy;

  public:
	/**
	 * @param extent The extent of the region map defines the set of points for which it can hold values.
	 *               All update operations and query results are clamped to this extent.
	 */
	region_map(range<3> extent, int dims, ValueType default_value = ValueType{}) : m_dims(dims) {
		using namespace region_map_detail;
		assert_dimensionality(box<3>(subrange<3>{id<3>{}, extent}), dims);
		switch(m_dims) {
		case 0: m_region_map.template emplace<region_map_impl<ValueType, 0>>(range_cast<0>(extent), default_value); break;
		case 1: m_region_map.template emplace<region_map_impl<ValueType, 1>>(range_cast<1>(extent), default_value); break;
		case 2: m_region_map.template emplace<region_map_impl<ValueType, 2>>(range_cast<2>(extent), default_value); break;
		case 3: m_region_map.template emplace<region_map_impl<ValueType, 3>>(range_cast<3>(extent), default_value); break;
		default: assert(false);
		}
	}

	/**
	 * Sets a new value for the provided region within the region map.
	 */
	void update_region(const region<3>& region, const ValueType& value) {
		region_map_detail::assert_dimensionality(region, m_dims);
		for(const auto& box : region.get_boxes()) {
			update_box(box, value);
		}
	}

	/**
	 * Sets a new value for the provided box within the region map.
	 */
	void update_box(const box<3>& box, const ValueType& value) {
		using namespace region_map_detail;
		switch(m_dims) {
		case 0: get_map<0>().update_box(box_cast<1>(box), value); break;
		case 1: get_map<1>().update_box(box_cast<1>(box), value); break;
		case 2: get_map<2>().update_box(box_cast<2>(box), value); break;
		case 3: get_map<3>().update_box(box_cast<3>(box), value); break;
		default: assert(false);
		}
	}

	/**
	 * Returns all entries in the region map that intersect with the request region.
	 *
	 * @returns A list of boxes clamped to the request region, and their associated values.
	 */
	std::vector<std::pair<box<3>, ValueType>> get_region_values(const region<3>& request) const {
		region_map_detail::assert_dimensionality(request, m_dims);
		std::vector<std::pair<box<3>, ValueType>> results;
		for(const auto& box : request.get_boxes()) {
			const auto r = get_region_values(box);
			results.insert(results.begin(), r.cbegin(), r.cend());
		}
		return results;
	}

	/**
	 * Returns all entries in the region map that intersect with the request box.
	 *
	 * @returns A list of boxes clamped to the request box, and their associated values.
	 */
	std::vector<std::pair<box<3>, ValueType>> get_region_values(const box<3>& request) const {
		using namespace region_map_detail;
		std::vector<std::pair<box<3>, ValueType>> results;
		switch(m_dims) {
		// TODO: AllScale box doesn't support 0 dimensions, fall back to 1
		case 0: {
			const auto results1 = get_map<0>().get_region_values(box_cast<1>(request));
			results.reserve(results1.size());
			std::transform(
			    results1.cbegin(), results1.cend(), std::back_inserter(results), [](const auto& p) { return std::make_pair(box_cast<3>(p.first), p.second); });
		} break;
		case 1: {
			const auto results1 = get_map<1>().get_region_values(box_cast<1>(request));
			results.reserve(results1.size());
			std::transform(
			    results1.cbegin(), results1.cend(), std::back_inserter(results), [](const auto& p) { return std::make_pair(box_cast<3>(p.first), p.second); });
		} break;
		case 2: {
			const auto results2 = get_map<2>().get_region_values(box_cast<2>(request));
			results.reserve(results2.size());
			std::transform(
			    results2.cbegin(), results2.cend(), std::back_inserter(results), [](const auto& p) { return std::make_pair(box_cast<3>(p.first), p.second); });
		} break;
		case 3: {
			results = get_map<3>().get_region_values(box_cast<3>(request));
		} break;
		default: assert(false);
		}
		return results;
	}

	/**
	 * Applies a function f to every value within the region map and stores the result in its place.
	 */
	template <typename Functor>
	void apply_to_values(const Functor& f) {
		static_assert(std::is_invocable_r_v<ValueType, Functor, const ValueType&>, "Functor must receive and return a value of type ValueType");

		switch(m_dims) {
		case 0: get_map<0>().apply_to_values(f); break;
		case 1: get_map<1>().apply_to_values(f); break;
		case 2: get_map<2>().apply_to_values(f); break;
		case 3: get_map<3>().apply_to_values(f); break;
		default: assert(false);
		}
	}

	auto format_to(fmt::format_context::iterator out) const {
		switch(m_dims) {
		case 1: return get_map<1>().format_to(out);
		case 2: return get_map<2>().format_to(out);
		case 3: return get_map<3>().format_to(out);
		}
	}

  private:
	int m_dims;
	std::variant<std::monostate, region_map_detail::region_map_impl<ValueType, 0>, region_map_detail::region_map_impl<ValueType, 1>,
	    region_map_detail::region_map_impl<ValueType, 2>, region_map_detail::region_map_impl<ValueType, 3>>
	    m_region_map;

	template <int Dims>
	region_map_detail::region_map_impl<ValueType, Dims>& get_map() {
		static_assert(Dims >= 0 && Dims <= 3);
		return std::get<Dims + 1>(m_region_map);
	}

	template <int Dims>
	const region_map_detail::region_map_impl<ValueType, Dims>& get_map() const {
		static_assert(Dims >= 0 && Dims <= 3);
		return std::get<Dims + 1>(m_region_map);
	}
};

} // namespace celerity::detail
