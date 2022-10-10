#pragma once

#include <chrono>
#include <numeric>
#include <set>
#include <type_traits>
#include <vector>

// NOCOMMIT
#include <limits>
#include <memory>
#include <optional>
#include <queue>
#include <unordered_set> // NOCOMMIT Just for get_num_regions...
#include <variant>

// #include <boost/container/static_vector.hpp>

#include <CL/sycl.hpp>

#include "grid.h"

// NOCOMMIT
#define FMT_HEADER_ONLY
#include <spdlog/fmt/fmt.h>

namespace celerity {
namespace detail {

	// NOCOMMIT ...this cost me an hour (copied only cool_region_map.h w/o changes to allscale API)
	static_assert(std::is_same_v<decltype(std::declval<GridBox<3>>().get_min()), GridPoint<3>&>, "Patch allscale: Need reference");

	// ------------------------------------------------------------------------------------------------------------------
	// ------------------------------------------------------------------------------------------------------------------
	// ------------------------------------------ NEW IMPLEMENTATION ----------------------------------------------------
	// ------------------------------------------------------------------------------------------------------------------
	// ------------------------------------------------------------------------------------------------------------------
	//
	// Assumptions:
	//  - ValueType is small / cheap to copy
	//  - Updates / queries are localized and don't vary much over time
	//
	// General considerations:
	// - Should algorithms be implemented recursively or iteratively?
	// - Nomenclature : "area" vs "volume" vs ?
	// - Don't pass around vectors from lower levels upwards - have a ref to root which can store "pending operations" or something
	// - Require equality comparable? Otherwise at least we cannot do merging (and should disable it during compile time)...
	// - merging algorithms etc degenerate when there are lots of different (at worst all different) values. in each inner node, keep a list of value hashes
	// (and their counts?) that are contained in subtree? This way we can throw away entire branches early, or even decide at the root that a value is unique
	// and there is no use in merging..?
	//
	// Argument for rolling own box/region classes: Since we are not really concerned about size, and know that we need at most D=3,
	// we can always store 3 values. This would make casting dimensions free (since we need D=3 in graph generator) and we could
	// even do it in bulk (i.e., reinterpret an array).
	//
	// ==> Document all of this well (start by porting some of the documentation from the python impl to here).
	//
	//
	// TODO: This something we should consider: When faced with an ambiguous split/merge decision - always favor largest extent in fastest dimension
	//
	//
	// does it ever make sense to dissolve children in root? probably not, because they will just be re-inserted!!
	// CAN ONLY SKIP BBOX RECOMPUTE IF AREA DELTA == 0 **AND** child didnt spllit!!
	//
	// TODO:
	// 	- Implement perfect overlap optimization - this will make a HUGE difference for e.g. simple_value_update
	//	- Optimize erase/inserts during update_box - If a node can hold all new boxes, this can be a totally localized operation,
	//	  no need to go up again with actions!! => furthermore, bbox doesn't even change
	//	- Try enabling vectorization by using a std::array instead of boost::stable_vector for bounding boxes.
	//	  Store empty (?) boxes for unfilled child slots - so that they never get selected for anything.
	//


	// NOCOMMIT Wrap implementation details in "impl" namespace or something?

	// NOCOMMIT Look into reducing value copies by passing around pointers or using move ctors

	// NOCOMMIT Figure this out! (Also make it a template parameter? Or a runtime argument?)
	constexpr size_t MAX_CHILDREN = 8;
	constexpr size_t MAX_VALUES = MAX_CHILDREN;
	constexpr size_t MIN_CHILDREN = 2;

	template <typename ValueType, int Dims>
	struct insert_node_action {
		GridBox<Dims> box;
		// NOCOMMIT TODO: Consider not copying this for every action (use shared_ptr?!)
		ValueType value;
		bool processed_locally = false;
	};

	template <int Dims>
	struct erase_node_action {
		GridBox<Dims> box;
		bool processed_locally = false;
	};

	// NOCOMMIT Rename to update_actions or something?
	// NOTE: An empty set of values indicates that an in-place update took place! (not ideal from a clarity standpoint...)
	template <typename ValueType, int Dims>
	using value_node_actions = std::vector<std::variant<insert_node_action<ValueType, Dims>, erase_node_action<Dims>>>;

	template <typename ValueType, size_t Dims, typename MetricsSink>
	class inner_node;

	template <typename ValueType, size_t Dims, typename MetricsSink>
	using inner_node_child_type = std::variant<std::unique_ptr<inner_node<ValueType, Dims, MetricsSink>>, std::unique_ptr<ValueType>>;

	// NOCOMMIT Make these scalar types instead of vectors?
	template <typename ValueType, int Dims, typename MetricsSink>
	using erase_reinserts = std::vector<std::pair<GridBox<Dims>, inner_node_child_type<ValueType, Dims, MetricsSink>>>;

	template <typename ValueType, int Dims>
	using query_results = std::vector<std::pair<GridBox<Dims>, const ValueType*>>;

	template <int D, int Dims>
	bool is_lo_inside(const GridBox<Dims>& a, const GridBox<Dims>& b) {
		static_assert(D < Dims);
		const auto a_min = a.get_min();
		if(a_min[D] <= b.get_min()[D]) return false;
		if(a_min[D] >= b.get_max()[D]) return false;
		return true;
	}

	template <int D, int Dims>
	bool is_hi_inside(const GridBox<Dims>& a, const GridBox<Dims>& b) {
		static_assert(D < Dims);
		const auto a_max = a.get_max();
		if(a_max[D] <= b.get_min()[D]) return false;
		if(a_max[D] >= b.get_max()[D]) return false;
		return true;
	}

	// template <int Dims>
	template <size_t Dims>
	GridBox<Dims> compute_bounding_box(const GridBox<Dims>& a, const GridBox<Dims>& b) {
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

	template <size_t Dims>
	bool do_overlap(const GridBox<Dims>& a, const GridBox<Dims>& b) {
		// NOCOMMIT Useless function (if we keep gridbox)
		return a.intersectsWith(b);
	}


	template <size_t Dims>
	bool is_inside(const GridBox<Dims>& box, const GridPoint<Dims>& point) {
		auto box_min = box.get_min();
		auto box_max = box.get_max();
		bool inside = true;
		for(size_t d = 0; d < Dims; ++d) {
			// To avoid ambiguities we consider the min inclusive and max exclusive
			// NOCOMMIT TODO: Can this cause problems? Are there cases where we don't find a neighbor due to this?
			inside &= box_min[d] <= point[d] && box_max[d] > point[d];
		}
		return inside;
	}

	template <typename ValueType, size_t Dims, typename MetricsSink>
	class cool_region_map;

	// NOCOMMIT naming (stats? introspection?)
	class metrics_sink {
	  public:
		void count_lookup() { lookups++; }
		void count_update() { updates++; }
		void count_in_place_update() { in_place_updates++; }
		void count_localized_update() { localized_updates++; }
		void count_insert() { inserts++; }
		void count_erase() { erases++; }
		void count_dissolve() { dissolves++; }

		void set_max_depth(size_t max_depth) { this->max_depth = max_depth; }
		void set_area_per_depth(std::vector<size_t> area_per_depth) { this->area_per_depth = area_per_depth; }
		void set_fills_per_depth(std::vector<std::vector<size_t>> fills_per_depth) { this->fills_per_depth = fills_per_depth; }
		void set_value_count(size_t value_count) { this->value_count = value_count; }

		// TODO: Stuff we may want to log:
		// - Max level
		// - Total average intersecting children per lookup
		// - Average intersecting children per lookup per level

		// private:
		size_t lookups = 0;
		size_t updates = 0;
		size_t in_place_updates = 0;
		size_t localized_updates = 0;
		size_t inserts = 0;
		size_t erases = 0;
		size_t dissolves = 0;

		size_t max_depth;
		std::vector<size_t> area_per_depth;
		std::vector<std::vector<size_t>> fills_per_depth;
		size_t value_count = 0;
	};

	class metrics_null_sink {
	  public:
		void count_lookup() {}
		void count_update() {}
		void count_in_place_update() {}
		void count_localized_update() {}
		void count_insert() {}
		void count_erase() {}
		void count_dissolve() {}

		void set_max_depth(size_t) {}
		void set_area_per_depth(std::vector<size_t>) {}
		void set_fills_per_depth(std::vector<std::vector<size_t>>) {}
		void set_value_count(size_t value_count) {}
	};

	// NOCOMMIT TODO: Move to utility header or something?
	template <typename LoopFn, size_t... Is>
	constexpr void for_constexpr(std::index_sequence<Is...>, LoopFn&& fn) {
		(fn(Is), ...);
	}

	// NOCOMMIT FIXME: This really isn't a pure inner node, it can also be a leaf node.
	template <typename ValueType, size_t Dims, typename MetricsSink>
	class inner_node {
		friend struct cool_region_map_testspy;
		// NOCOMMIT TODO: Do we even need this as a friend..? Just expose root interface as public
		friend class cool_region_map<ValueType, Dims, MetricsSink>;

	  public:
		// NOCOMMIT TODO: Do we have to pass the bbox here..?
		inner_node(bool is_leaf, size_t depth, MetricsSink* metrics) : metrics(metrics), depth(depth), is_leaf(is_leaf) {
			// During splits we temporarily need to store one additional child
			children.reserve(MAX_CHILDREN + 1);
		}

		// NOCOMMIT TODO Rule of 5
		/*
		inner_node& operator=(const inner_node& other) {
		    root = other.root;
		    bounding_box = other.bounding_box;
		    depth = other.depth;
		    values = other.values;
		    // IMPORTANT: other might be one of this->children, so we have to make sure we do this last!
		    // NOCOMMIT TODO: Is this still UB?
		    // children = other.children;
		    auto foo = other.children;
		    children = foo;
		    return *this;
		}
		*/

		inner_node& operator=(const inner_node& other) = delete;
		inner_node& operator=(inner_node&&) = default;
#if 0
		inner_node& operator=(inner_node&& other) {
			root = std::move(other.root);
			bounding_box = std::move(other.bounding_box);
			depth = other.depth;
			values = std::move(other.values);
			// IMPORTANT: other might be one of this->children, so we have to make sure we do this last!
			// NOCOMMIT TODO: Is this still UB?
			// children = other.children;
			auto foo = std::move(other.children);
			children = std::move(foo);
			return *this;
		}
#endif

		// NOCOMMIT TODO: Naming - doesn't really update anything (but also look into doing in-place updates)
		// NOCOMMIT TODO: Do we even need "value" here? (Maybe for exact overlap)
		/**
		 * @returns True if a local update operation was performed that may require a bounding box recomputation.
		 */
		bool update_box(const GridBox<Dims>& box, const ValueType& value, value_node_actions<ValueType, Dims>& update_actions) {
			if(!is_leaf) {
				bool any_child_did_local_update = false;
				for(size_t i = 0; i < child_boxes.size(); ++i) {
					if(do_overlap<Dims>(child_boxes[i], box)) {
						const auto did_local_update = get_child_node(i).update_box(box, value, update_actions);
						if(did_local_update) {
							recompute_child_bounding_box(i);
							any_child_did_local_update = true;
						}
					}
				}
				return any_child_did_local_update;
			}

			size_t erase_action_count = 0;
			const auto previous_action_count = update_actions.size();

			for(size_t i = 0; i < child_boxes.size(); ++i) {
				const auto& child_box = child_boxes[i];
				if(!do_overlap<Dims>(child_box, box)) continue;

				if(box == child_box) {
					// Exact overlap. Simply update box in-place.
					get_child_value(i) = value;
					metrics->count_in_place_update();
					break;
				}

				update_actions.push_back(erase_node_action<Dims>{child_box});
				erase_action_count++;

				// Full overlap, no need to split anything.
				// NOCOMMIT FIXME: We may already determine this in do_overlap (maybe return some kind of overlap_info that can also be used for left/right
				// checks below?)
				if(box.covers(child_box)) { continue; }

				// Partial overlap. Check in each dimension which sides of the box intersect with the current box, creating new boxes along the way.
				GridBox<Dims> remainder = child_box;

				// NOCOMMIT TODO: Test whether this is actually the correct approach: Start with fastest changing dimension to maximize box extent in that dim
				// NOCOMMIT TODO: CAN WE DRY THIS UP?

				// NOCOMMIT TODO Can we not create value nodes on heap here..?

				const auto& child_value = get_child_value(i);

				if constexpr(Dims == 3) {
					if(is_lo_inside<2, Dims>(box, child_box)) {
						auto new_box = remainder;
						new_box.get_max()[2] = box.get_min()[2];
						remainder.get_min()[2] = box.get_min()[2];
						update_actions.push_back(insert_node_action<ValueType, Dims>{new_box, child_value});
					}
					if(is_hi_inside<2, Dims>(box, child_box)) {
						auto new_box = remainder;
						new_box.get_min()[2] = box.get_max()[2];
						remainder.get_max()[2] = box.get_max()[2];
						update_actions.push_back(insert_node_action<ValueType, Dims>{new_box, child_value});
					}
				}

				if constexpr(Dims >= 2) {
					if(is_lo_inside<1, Dims>(box, child_box)) {
						auto new_box = remainder;
						new_box.get_max()[1] = box.get_min()[1];
						remainder.get_min()[1] = box.get_min()[1];
						update_actions.push_back(insert_node_action<ValueType, Dims>{new_box, child_value});
					}
					if(is_hi_inside<1, Dims>(box, child_box)) {
						auto new_box = remainder;
						new_box.get_min()[1] = box.get_max()[1];
						remainder.get_max()[1] = box.get_max()[1];
						update_actions.push_back(insert_node_action<ValueType, Dims>{new_box, child_value});
					}
				}

				if constexpr(Dims >= 1 /* always true */) {
					if(is_lo_inside<0, Dims>(box, child_box)) {
						auto new_box = remainder;
						new_box.get_max()[0] = box.get_min()[0];
						remainder.get_min()[0] = box.get_min()[0];
						update_actions.push_back(insert_node_action<ValueType, Dims>{new_box, child_value});
					}

					if(is_hi_inside<0, Dims>(box, child_box)) {
						auto new_box = remainder;
						new_box.get_min()[0] = box.get_max()[0];
						remainder.get_max()[0] = box.get_max()[0];
						update_actions.push_back(insert_node_action<ValueType, Dims>{new_box, child_value});
					}
				}
			}

			// If we didn't do anything, stop here.
			if(update_actions.size() == previous_action_count) return false;

			// Otherwise check whether we can perform all actions locally.
			const size_t insert_action_count = update_actions.size() - previous_action_count - erase_action_count;

			// We can only process actions locally if we have enough space for all children.
			const bool wont_overflow = children.size() + insert_action_count <= MAX_CHILDREN;
			// However, we also must ensure that we don't end up with too few children, as we otherwise degrade tree health over time.
			const bool wont_underflow = children.size() - erase_action_count + insert_action_count >= MIN_CHILDREN;

			if(wont_overflow && wont_underflow) {
				// First, process all erases.
				for(size_t i = previous_action_count; i < update_actions.size(); ++i) {
					if(std::holds_alternative<erase_node_action<Dims>>(update_actions[i])) {
						auto& erase_action = std::get<erase_node_action<Dims>>(update_actions[i]);
						erase_action.processed_locally = true;
						erase_reinserts<ValueType, Dims, MetricsSink> to_reinsert;
						[[maybe_unused]] const auto ret = erase(erase_action.box, to_reinsert);
						assert(ret);
						assert(to_reinsert.empty()); // This should never happen as we are in a leaf node.
					}
				}

				// Now, process the inserts.
				for(size_t i = previous_action_count; i < update_actions.size(); ++i) {
					if(std::holds_alternative<insert_node_action<ValueType, Dims>>(update_actions[i])) {
						auto& insert_action = std::get<insert_node_action<ValueType, Dims>>(update_actions[i]);
						insert_action.processed_locally = true;
						assert(children.size() < MAX_CHILDREN);
						insert_child_value(insert_action.box, insert_action.value);
					}
				}

				metrics->count_localized_update();
				return true;
			}

			return false;
		}

		template <typename Functor>
		void apply_to_values(Functor&& f, query_results<ValueType, Dims>& updated_nodes) {
			if(!is_leaf) {
				for(size_t i = 0; i < children.size(); ++i) {
					get_child_node(i).apply_to_values(std::forward<Functor>(f), updated_nodes);
				}
				return;
			}

			for(size_t i = 0; i < children.size(); ++i) {
				ValueType& child_value = get_child_value(i);
				const auto new_value = std::forward<Functor>(f)(child_value);
				if(new_value != child_value) {
					child_value = std::move(new_value);
					updated_nodes.push_back(std::make_pair(child_boxes[i], &child_value));
				}
			}
		}

		// NOCOMMIT Move elsewhere
		struct insert_result {
			std::unique_ptr<inner_node<ValueType, Dims, MetricsSink>> spilled_node;
			// NOCOMMIT Is there really any use in having this here..? Can't we compute it on the fly where we need it?
			// (i.e, do we pass it around more than once, which would warrant caching it here)
			GridBox<Dims> spilled_box;
		};

		// NOCOMMIT MAKE PRIVATE
		std::optional<insert_result> insert(const GridBox<Dims>& box, const ValueType& value) {
			if(!is_leaf) {
				// NOCOMMIT TODO: Should we ignore children that are full..?
				size_t best_i = std::numeric_limits<size_t>::max();
				size_t smallest_area_delta = std::numeric_limits<size_t>::max();
				for(size_t i = 0; i < child_boxes.size(); ++i) {
					const auto area_delta = compute_bounding_box(child_boxes[i], box).area() - child_boxes[i].area();
					if(area_delta < smallest_area_delta) {
						smallest_area_delta = area_delta;
						best_i = i;
					}
				}

				auto ret = get_child_node(best_i).insert(box, value);

				// Bounding box of child might have changed.
				// NOCOMMIT TODO Can't we just skip this iff smallest_area_delta == 0?
				recompute_child_bounding_box(best_i);

				if(ret.has_value()) { return insert_subtree(ret->spilled_box, std::move(ret->spilled_node)); }
				return std::nullopt;
			}

			// Try inserting value directly, or...
			if(children.size() < MAX_VALUES) {
				insert_child_value(box, value);
				return std::nullopt;
			}

			// ...split if we are full.
			// (Include new value in split decision).
			insert_child_value(box, value);

			auto [seed1, seed2] = get_split_seeds();

			// NOCOMMIT I think we can DRY this up with child split above
			auto node1 = std::make_unique<inner_node>(true, depth, metrics);
			node1->insert_child_value(child_boxes[seed1], std::move(std::get<std::unique_ptr<ValueType>>(children[seed1])));
			auto node2 = std::make_unique<inner_node>(true, depth, metrics);
			node2->insert_child_value(child_boxes[seed2], std::move(std::get<std::unique_ptr<ValueType>>(children[seed2])));

#if 1
			auto bbox1 = child_boxes[seed1];
			auto bbox2 = child_boxes[seed2];
			for(size_t i = 0; i < child_boxes.size(); ++i) {
				if(i == seed1 || i == seed2) continue;

				const auto new_bbox1 = compute_bounding_box(bbox1, child_boxes[i]);
				const auto new_bbox2 = compute_bounding_box(bbox2, child_boxes[i]);

				// Assign value to node that results in smaller area increase.
				if((new_bbox1.area() - bbox1.area()) < (new_bbox2.area() - bbox2.area())) {
					node1->insert_child_value(child_boxes[i], std::move(std::get<std::unique_ptr<ValueType>>(children[i])));
					bbox1 = new_bbox1;
				} else {
					node2->insert_child_value(child_boxes[i], std::move(std::get<std::unique_ptr<ValueType>>(children[i])));
					bbox2 = new_bbox2;
				}
			}
#else
			// Greedily assign all values O(N^2) - unfortunately does exactly the same as above (likely because values are ordered!)

			auto area1 = node1.get_bounding_box().area();
			auto area2 = node2.get_bounding_box().area();
			std::vector<bool> assigned(values.size(), false);
			size_t num_assigned = 0;
			while(num_assigned < values.size()) {
				size_t smallest_area_delta = std::numeric_limits<size_t>::max();
				size_t smallest_i = std::numeric_limits<size_t>::max();
				int target_node = 0;

				for(size_t i = 0; i < values.size(); ++i) {
					if(assigned[i]) continue;

					auto& v = values[i];

					const auto new_bbox1 = compute_bounding_box(v.get_bounding_box(), node1.get_bounding_box());
					const auto new_bbox2 = compute_bounding_box(v.get_bounding_box(), node2.get_bounding_box());

					const auto ad1 = (new_bbox1.area() - area1);
					const auto ad2 = (new_bbox2.area() - area2);

					if(ad1 < smallest_area_delta) {
						smallest_area_delta = ad1;
						smallest_i = i;
						target_node = 1;
					}
					if(ad2 < smallest_area_delta) {
						smallest_area_delta = ad2;
						smallest_i = i;
						target_node = 2;
					}
				}

				assert(target_node != 0);
				if(target_node == 1) {
					node1.values.push_back(values[smallest_i]);
					const auto new_bbox = compute_bounding_box(values[smallest_i].get_bounding_box(), node1.get_bounding_box());
					node1.bounding_box = new_bbox;
					area1 = new_bbox.area();
				} else {
					node2.values.push_back(values[smallest_i]);
					const auto new_bbox = compute_bounding_box(values[smallest_i].get_bounding_box(), node2.get_bounding_box());
					node2.bounding_box = new_bbox;
					area2 = new_bbox.area();
				}

				assigned[smallest_i] = true;
				num_assigned++;
			}
#endif

			assert(!node1->children.empty());
			assert(!node2->children.empty());
			assert(node1->children.size() <= MAX_VALUES);
			assert(node2->children.size() <= MAX_VALUES);
			// assert(!node1.is_underfull());
			// assert(!node2.is_underfull());

			insert_result result{std::move(node2), bbox2};

			// Replace this with node 1, return node 2
			// NOCOMMIT Find cleaner way
			*this = std::move(*node1);

			sanity_check();
			return result;
		}

		std::optional<insert_result> insert_subtree(const GridBox<Dims>& box, std::unique_ptr<inner_node<ValueType, Dims, MetricsSink>>&& subtree) {
			assert(!is_leaf);
			assert(subtree->depth > depth);

			// Check if subtree should be inserted as child of this node.
			if(subtree->depth == depth + 1) {
				if(children.size() < MAX_CHILDREN) {
					insert_child_node(box, std::move(subtree));
					return std::nullopt;
				}

				// We are full, need to split.
				// (Include new child in split decision).
				insert_child_node(box, std::move(subtree));

				auto [seed1, seed2] = get_split_seeds();

				auto node1 = std::make_unique<inner_node>(false, depth, metrics);
				node1->insert_child_node(child_boxes[seed1], std::move(std::get<std::unique_ptr<inner_node>>(children[seed1])));
				auto node2 = std::make_unique<inner_node>(false, depth, metrics);
				node2->insert_child_node(child_boxes[seed2], std::move(std::get<std::unique_ptr<inner_node>>(children[seed2])));

#if 0 // This is a bit faster for the wave sim example split for some reason
				for(auto& c : children) {
					assert(c.depth == depth + 1);                 // NOCOMMIT Sanity check debugging
					assert(c.values.size() != c.children.size()); // NOCOMMIT SANITY CHECK
					const auto bbox1 = compute_bounding_box(c->get_bounding_box(), node1->get_bounding_box());
					const auto bbox2 = compute_bounding_box(c->get_bounding_box(), node2->get_bounding_box());

					if(bbox1.area() < bbox2.area()) {
						node1->children.push_back(std::move(c));
						node1->bounding_box = bbox1;
					} else {
						node2->children.push_back(c);
						node2->bounding_box = bbox2;
					}
				}
#else
				auto bbox1 = child_boxes[seed1];
				auto bbox2 = child_boxes[seed2];
				for(size_t i = 0; i < children.size(); ++i) {
					if(i == seed1 || i == seed2) continue;

					assert(get_child_node(i).depth == depth + 1); // Sanity check
					const auto new_bbox1 = compute_bounding_box(bbox1, child_boxes[i]);
					const auto new_bbox2 = compute_bounding_box(bbox2, child_boxes[i]);

					// Assign value to node that results in smaller area increase.
					if((new_bbox1.area() - bbox1.area()) < (new_bbox2.area() - bbox2.area())) {
						// NOCOMMIT Instead return unique_ptr from get_child_node?
						node1->insert_child_node(child_boxes[i], std::move(std::get<std::unique_ptr<inner_node>>(children[i])));
						bbox1 = new_bbox1;
					} else {
						node2->insert_child_node(child_boxes[i], std::move(std::get<std::unique_ptr<inner_node>>(children[i])));
						bbox2 = new_bbox2;
					}
				}
#endif

				assert(!node1->children.empty());
				assert(!node2->children.empty());
				// NOCOMMIT TODO Should we try to get some sort of even-ish split here? (In some cases it seems to do N-1/1)
				assert(node1->children.size() <= MAX_CHILDREN);
				assert(node2->children.size() <= MAX_CHILDREN);
				// assert(!node1.is_underfull());
				// assert(!node2.is_underfull()); // NOCOMMIT Can we ensure this?

				insert_result result{std::move(node2), bbox2};

				// Replace this with node 1, return node 2
				// NOCOMMIT Find cleaner way
				*this = std::move(*node1);
				return result;
			}

			// Subtree belongs deeper into the tree. Find child that best fits it.
			// NOCOMMIT DRY THIS UP

			// NOCOMMIT Move this up to be consistent with other methods (first branch usually goes down in tree)

			size_t best_i = std::numeric_limits<size_t>::max();
			size_t smallest_area_delta = std::numeric_limits<size_t>::max();
			for(size_t i = 0; i < child_boxes.size(); ++i) {
				const auto area_delta = compute_bounding_box(child_boxes[i], box).area() - child_boxes[i].area();
				if(area_delta < smallest_area_delta) {
					smallest_area_delta = area_delta;
					best_i = i;
				}
			}

			auto ret = get_child_node(best_i).insert_subtree(box, std::move(subtree));
			// Bounding box of child might have changed.
			// NOCOMMIT Same as above, if delta == 0 => no recompute?
			recompute_child_bounding_box(best_i);

			sanity_check();

			if(ret.has_value()) {
				assert(ret->spilled_node->depth == depth + 1);
				return insert_subtree(ret->spilled_box, std::move(ret->spilled_node));
			}

			return std::nullopt;
		}

		// NOCOMMIT MAKE PRIVATE
		// Also returning optional to signal whether something was erased is yuck
		bool erase(const GridBox<Dims>& box, erase_reinserts<ValueType, Dims, MetricsSink>& to_reinsert) {
			bool did_erase = false;

			if(!is_leaf) {
				for(size_t i = 0; i < children.size(); ++i) {
					if(do_overlap<Dims>(box, child_boxes[i])) {
						auto& child = get_child_node(i);
						if(child.erase(box, to_reinsert)) {
							did_erase = true;

							if(child.is_underfull()) {
								metrics->count_dissolve();
								if(!child.children.empty()) {
									for(size_t j = 0; j < child.children.size(); ++j) {
										to_reinsert.push_back(std::make_pair(child.child_boxes[j], std::move(child.children[j])));
									}
								}
								erase_child(i);
							} else {
								recompute_child_bounding_box(i);
							}
						}
					}

					if(did_erase) break;
				}
			} else {
				for(size_t i = 0; i < children.size(); ++i) {
					if(child_boxes[i] == box) {
						did_erase = true;
						erase_child(i);
						break;
					}
				}
			}

			sanity_check();
			return did_erase;
		}

		// NOCOMMIT MAKE PRIVATE
		void query(const GridBox<Dims>& box, query_results<ValueType, Dims>& intersecting) const {
			for(size_t i = 0; i < children.size(); ++i) {
				if(do_overlap<Dims>(child_boxes[i], box)) {
					if(!is_leaf) {
						get_child_node(i).query(box, intersecting);
					} else {
						// NOCOMMIT TODO: Return pointer to bbox?
						intersecting.push_back(std::make_pair(child_boxes[i], &get_child_value(i)));
					}
				}
			}
		}

		// NOCOMMIT MAKE PRIVATE
		std::optional<std::pair<GridBox<Dims>, const ValueType*>> point_query(const GridPoint<Dims>& point) const {
			for(size_t i = 0; i < children.size(); ++i) {
				if(is_inside<Dims>(child_boxes[i], point)) {
					if(!is_leaf) {
						const auto result = get_child_node(i).point_query(point);
						if(result.has_value()) return result;
					} else {
						return std::make_pair(child_boxes[i], &get_child_value(i));
					}
				}
			}
			return std::nullopt;
		}

	  private:
		MetricsSink* metrics;
		size_t depth; // NOCOMMIT TODO Or "level"?

		bool is_leaf;
		// boost::container::static_vector<GridBox<Dims>, MAX_CHILDREN + 1> child_boxes;
		std::vector<GridBox<Dims>> child_boxes;
		// NOCOMMIT TODO: Measure variant overhead (as opposed to union)
		// NOCOMMIT TODO: Consider storing ValueType directly for smallish values? (otherwise overhead in inner nodes too large)
		// using child_type = std::variant<std::unique_ptr<inner_node>, std::unique_ptr<ValueType>>;
		// boost::container::static_vector<child_type, MAX_CHILDREN + 1> children;
		std::vector<inner_node_child_type<ValueType, Dims, MetricsSink>> children;

		inner_node& get_child_node(size_t index) { return *std::get<std::unique_ptr<inner_node>>(children[index]); }
		const inner_node& get_child_node(size_t index) const { return *std::get<std::unique_ptr<inner_node>>(children[index]); }

		ValueType& get_child_value(size_t index) { return *std::get<std::unique_ptr<ValueType>>(children[index]); }
		const ValueType& get_child_value(size_t index) const { return *std::get<std::unique_ptr<ValueType>>(children[index]); }

		// NOCOMMIT TODO: Can we pass value as pointer here..?
		void insert_child_value(const GridBox<Dims>& box, const ValueType& value) {
			assert(children.size() < MAX_CHILDREN + 1); // During splits we temporarily go one above the max
#if !defined(NDEBUG)
			for(auto& b : child_boxes) {
				// New box must not overlap with any other
				assert(GridRegion<Dims>::intersect(b, box).empty());
			}
#endif
			child_boxes.push_back(box);
			children.emplace_back(std::make_unique<ValueType>(value));
		}

		// NOCOMMIT DRY THIS UP
		void insert_child_value(const GridBox<Dims>& box, std::unique_ptr<ValueType>&& value) {
			assert(children.size() < MAX_CHILDREN + 1); // During splits we temporarily go one above the max
#if !defined(NDEBUG)
			for(auto& b : child_boxes) {
				// New box must not overlap with any other
				assert(GridRegion<Dims>::intersect(b, box).empty());
			}
#endif
			child_boxes.push_back(box);
			children.emplace_back(std::move(value));
		}

		// NOCOMMIT DRY THIS UP
		void insert_child_node(const GridBox<Dims>& box, std::unique_ptr<inner_node>&& node) {
			assert(children.size() < MAX_CHILDREN + 1); // During splits we temporarily go one above the max
			child_boxes.push_back(box);
			children.emplace_back(std::move(node));
		}

		void erase_child(size_t index) {
			child_boxes.erase(child_boxes.begin() + index);
			children.erase(children.begin() + index);
		}

		// NOCOMMIT Rename (OR: Don't - store reference to cool_region_map directly? Then we don't have to update all the time)
		void set_depth(size_t depth) {
			this->depth = depth;
			for(auto& c : children) {
				if(std::holds_alternative<std::unique_ptr<inner_node>>(c)) { std::get<std::unique_ptr<inner_node>>(c)->set_depth(depth + 1); }
			}
		}

		// NOCOMMIT We're doing a quadratic split for now. Revisit.
		std::pair<size_t, size_t> get_split_seeds() {
			size_t worst_area = 0;
			size_t worst_i = std::numeric_limits<size_t>::max();
			size_t worst_j = std::numeric_limits<size_t>::max();
			for(size_t i = 0; i < child_boxes.size(); ++i) {
				for(size_t j = i + 1; j < child_boxes.size(); ++j) {
					const auto area = compute_bounding_box(child_boxes[i], child_boxes[j]).area();
					if(area > worst_area) {
						worst_area = area;
						worst_i = i;
						worst_j = j;
					}
				}
			}
			return std::make_pair(worst_i, worst_j);
		}

		bool is_underfull() const {
			if(children.size() < MIN_CHILDREN) return true;
			return false;
		}

		void recompute_child_bounding_box(size_t index) {
			assert(!is_leaf); // NOCOMMIT Is this true? If so, rename to recompute_child_node_bounding_box

			auto& child_node = get_child_node(index);
			auto bounding_box = child_node.child_boxes[0];
			for(size_t i = 1; i < child_node.child_boxes.size(); ++i) {
				bounding_box = compute_bounding_box(bounding_box, child_node.child_boxes[i]);
			}
			child_boxes[index] = bounding_box;
		}

		// NOCOMMIT REMOVE THIS AGAIN? Or unify somehow with toplevel sanity check
		void sanity_check() const {
#if !defined(NDEBUG)
			if(is_leaf) return;
			// NOCOMMIT Do this recursively?
			for(size_t i = 0; i < child_boxes.size(); ++i) {
				auto& child_node = get_child_node(i);
				assert(!child_node.child_boxes.empty());
				GridBox<Dims> child_bbox = child_node.child_boxes[0];
				for(size_t j = 1; j < child_node.child_boxes.size(); ++j) {
					child_bbox = compute_bounding_box(child_bbox, child_node.child_boxes[j]);
				}
				assert(child_bbox == child_boxes[i]);
			}
#endif
		}

		// NOTE: Not O(1)!
		GridBox<Dims> get_bounding_box() const {
			assert(!child_boxes.empty());
			GridBox<Dims> bbox = child_boxes[0];
			for(size_t i = 1; i < child_boxes.size(); ++i) {
				bbox = compute_bounding_box(bbox, child_boxes[i]);
			}
			return bbox;
		}

		void print(std::ostream& os, size_t level = 0) const {
			const auto padding = std::string(2 * level, ' ');
			auto bounding_box = get_bounding_box();
			if(!is_leaf) {
				os << padding << "inner node with bbox " << bounding_box << " and " << children.size() << " children:\n";
				for(size_t i = 0; i < children.size(); ++i) {
					get_child_node(i).print(os, level + 1);
				}
			} else {
				os << padding << "leaf node with bbox " << bounding_box << " and " << children.size() << " values:\n";
				const auto value_padding = std::string(2 * (level + 1), ' ');
				for(size_t i = 0; i < children.size(); ++i) {
					auto& v = get_child_value(i);
					os << value_padding << child_boxes[i] << " : ";
					// NOCOMMIT FIXME: Make sure arbitrary value type is printable
					if constexpr(std::is_integral_v<ValueType>) {
						os << v;
					} else {
						for(auto& x : v) {
							os << x << " ";
						}
					}
					os << "\n";
				}
			}
		}

		void collect_values(std::unordered_set<ValueType>& values) const {
			if(!is_leaf) {
				for(size_t i = 0; i < children.size(); ++i) {
					get_child_node(i).collect_values(values);
				}
			} else {
				for(size_t i = 0; i < children.size(); ++i) {
					values.insert(get_child_value(i));
				}
			}
		}
	};

	template <typename ValueType, size_t Dims, typename MetricsSink = metrics_null_sink>
	class cool_region_map {
		friend struct region_map_testspy; // NOCOMMIT
		friend struct cool_region_map_testspy;
		// static_assert(Dims == 2, "1/3D NYI"); // NOCOMMIT

	  public:
		cool_region_map(const cl::sycl::range<Dims>& extent, ValueType default_value = ValueType{})
		    : extent(subrange_to_grid_box(subrange<Dims>{id_cast<Dims>(cl::sycl::id<3>{0, 0, 0}), extent})),
		      root(std::make_unique<inner_node<ValueType, Dims, MetricsSink>>(true, 0, &metrics)) {
			root->insert(this->extent, default_value);
		}

		// NOCOMMIT Rule of 5. Also consider making this copyable.
		// my_cool_region_map_wrapper(const my_cool_region_map_wrapper&) = delete;

		// NOCOMMIT TODO: Should this take a region or just a set of subranges? (Really, it's the same!)
		// TODO: Can we optimize bulk updates somehow..?
		void update_region(const GridRegion<Dims>& region, const ValueType& value) {
			// NOCOMMIT TODO Assert region within extent
			region.scanByBoxes([&](const GridBox<Dims>& box) { update_box(box, value); });
		}

		void update_box(const GridBox<Dims>& box, const ValueType& value) {
			// TODO: This can now happen as we added support for empty buffers.
			// Early exit opportunity?
			// assert(box.area() > 0);
			metrics.count_update();

			value_node_actions<ValueType, Dims> actions;
			// NOCOMMIT Streamline this
			root->update_box(box, value, actions);
			// auto& actions = current_node_actions;


			// NOCOMMIT TODO: Why should try and copy less values... These might be expensive to copy (e.g. vector or set)
			std::vector<std::optional<std::pair<GridBox<Dims>, ValueType>>> merge_candidates;

			// If there are any actions it means there was no in-place update.
			if(!actions.empty()) {
				// In this case we have to insert the new box.
				actions.push_back(insert_node_action<ValueType, Dims>{box, value});
			} else {
				// Otherwise just check whether the in-place updated box can be merged.
				merge_candidates.push_back(std::make_pair(box, value));
			}

#ifndef NDEBUG
			// Sanity check: Erased and inserted boxes must cover the same space
			GridRegion<Dims> erased, inserted;
			for(const auto& a : actions) {
				if(std::holds_alternative<erase_node_action<Dims>>(a)) {
					auto& erase_action = std::get<erase_node_action<Dims>>(a);
					assert(GridRegion<Dims>::intersect(erased, erase_action.box).empty());
					erased = GridRegion<Dims>::merge(erased, erase_action.box);
				} else if(std::holds_alternative<insert_node_action<ValueType, Dims>>(a)) {
					auto& insert_action = std::get<insert_node_action<ValueType, Dims>>(a);
					assert(GridRegion<Dims>::intersect(inserted, insert_action.box).empty());
					inserted = GridRegion<Dims>::merge(inserted, insert_action.box);
				}
			}
			assert(erased == inserted);
#endif

			for(const auto& a : actions) {
				if(std::holds_alternative<erase_node_action<Dims>>(a)) {
					auto& erase_action = std::get<erase_node_action<Dims>>(a);
					if(!erase_action.processed_locally) { erase(erase_action.box); }
				} else if(std::holds_alternative<insert_node_action<ValueType, Dims>>(a)) {
					auto& insert_action = std::get<insert_node_action<ValueType, Dims>>(a);
					if(!insert_action.processed_locally) { insert(insert_action.box, insert_action.value); }
					// Even if the action was processed locally already, we still have to try and merge the new box.
					merge_candidates.push_back(std::make_pair(insert_action.box, insert_action.value));
				}
			}

			// NOCOMMIT WAY TOO LATE TO DO THIS!! NEEDS TO HAPPEN ABOVE SOMEHOW
			actions.clear();

			sanity_check();

			try_merging(merge_candidates);

			sanity_check();
		}

		// NOCOMMIT: Benchmark this as well!
		template <typename Functor>
		void apply_to_values(Functor&& f) {
			static_assert(std::is_same_v<std::invoke_result_t<Functor, ValueType>, ValueType>, "Functor must return value of same type");

			query_results<ValueType, Dims> updated_nodes;
			root->apply_to_values(std::forward<Functor>(f), updated_nodes);

			// Current query result holds all boxes that had their value modified by the functor.
			// Now attempt to merge these boxes.
			std::vector<std::optional<std::pair<GridBox<Dims>, ValueType>>> merge_candidates;
			merge_candidates.reserve(updated_nodes.size());
			std::transform(
			    updated_nodes.cbegin(), updated_nodes.cend(), std::back_inserter(merge_candidates), [](const std::pair<GridBox<Dims>, const ValueType*>& p) {
				    // NOCOMMIT Oof this loop is stupid, such a minor type change - can we avoid?
				    return std::make_pair(p.first, *p.second);
			    });
			try_merging(merge_candidates);
		}

		// NOCOMMIT TODO: Return subranges instead of boxes?
		// NOCOMMIT TODO: Return pointers on values instead?
		// std::vector<std::pair<GridBox<Dims>, ValueType>> get_region_values(GridRegion<Dims> request) const {
		std::vector<std::pair<GridBox<Dims>, ValueType>> get_region_values(const GridBox<Dims>& request) const {
			metrics.count_lookup();

			// NOCOMMIT Streamline this
			query_results<ValueType, Dims> intersecting;
			root->query(request, intersecting);

			// Clamp to query request box
			std::vector<std::pair<GridBox<Dims>, ValueType>> results;
			for(auto& [b, v] : intersecting) {
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
				// NOCOMMIT Also return pointers to values here..?
				results.push_back(std::make_pair(GridBox<Dims>{clamped_min, clamped_max}, *v));
			}

// NOCOMMIT MOVE BELOW INTO SEPARATE FN?
#ifdef NDEBUG
			// In 1D everything that can be merged will be merged on update.
			// (Nevertheless, assert this in debug builds).
			if(Dims == 1) return results;
#endif

			// Do a greedy quadratic merge NOCOMMIT COme up with more efficient solution
			bool did_merge = true;
			// TODO: Should we delete right away or is it cheaper to just remember the indices we should skip?!
			std::vector<bool> is_merged(results.size(), false);
			while(did_merge) {
				did_merge = false;
				for(size_t i = 0; i < results.size(); ++i) {
					if(is_merged[i]) continue;
					for(size_t j = i + 1; j < results.size(); ++j) {
						if(is_merged[j]) continue;
						// NOCOMMIT NOTE: The overhead of passing constructed value_nodes into can_merge is INSANE (+100% in rsim_pattern)
						// ==> Short circuit here for now.
						if(results[i].second != results[j].second) continue;
						if(can_merge(results[i].first, results[j].first)) {
							assert(Dims > 1); // 1D should already have merged on update.
							// FIXME: Computing the bbox from scratch isn't ideal, as we really only need to adjust one dimension.
							results[i].first = compute_bounding_box(results[i].first, results[j].first);
							is_merged[j] = true;
							did_merge = true;
						}
					}
				}
			}

			std::vector<std::pair<GridBox<Dims>, ValueType>> results_merged;
			results_merged.reserve(results.size() / 2);
			for(size_t i = 0; i < results.size(); ++i) {
				if(!is_merged[i]) results_merged.push_back(results[i]);
			}

			return results_merged;
		}

		// NOCOMMIT TODO: Should this be decoupled from the run-time metrics? No harm in always enabling this?
		const MetricsSink& get_metrics() const {
			size_t max_depth = 0;
			std::vector<size_t> area_per_depth(1);
			std::vector<std::vector<size_t>> fills_per_depth(1);
			size_t value_area = 0;
			size_t value_count = 0;

			// Store nodes to visit and their depth
			std::queue<std::pair<GridBox<Dims>, const inner_node<ValueType, Dims, MetricsSink>*>> node_queue;
			node_queue.push(std::make_pair(root->get_bounding_box(), root.get()));

			while(!node_queue.empty()) {
				const auto node_bbox = node_queue.front().first;
				const auto& node = *node_queue.front().second;
				const size_t depth = node.depth;
				node_queue.pop();

				if(depth > max_depth) {
					max_depth = depth;
					area_per_depth.resize(max_depth + 1);
					fills_per_depth.resize(max_depth + 1);
				}

				area_per_depth[depth] += node_bbox.area();

				if(!node.is_leaf) {
					assert(depth <= max_depth);
					fills_per_depth[depth].push_back(node.children.size());
					for(size_t i = 0; i < node.children.size(); ++i) {
						node_queue.push(std::make_pair(node.child_boxes[i], &node.get_child_node(i)));
					}
				} else {
					assert(depth == max_depth);
					fills_per_depth[depth].push_back(node.children.size());
					value_count += node.children.size();
					for(size_t i = 0; i < node.children.size(); ++i) {
						value_area += node.child_boxes[i].area();
					}
				}
			}

			assert(value_area == extent.area());

			metrics.set_max_depth(max_depth);
			metrics.set_area_per_depth(area_per_depth);
			metrics.set_fills_per_depth(fills_per_depth);
			metrics.set_value_count(value_count);

			return metrics;
		}

		friend std::ostream& operator<<(std::ostream& os, const cool_region_map& crm) {
			crm.print(os);
			return os;
		}

	  private:
		// NOCOMMIT TODO: Alternative have a bool that decides whether to collect metrics, and inner nodes contain ref to parent and pass metrics through to
		// root that way?
		mutable MetricsSink metrics;
		GridBox<Dims> extent;

		std::unique_ptr<inner_node<ValueType, Dims, MetricsSink>> root;

		void insert(const GridBox<Dims>& box, const ValueType& value) {
			metrics.count_insert();

			auto ret = root->insert(box, value);
			if(ret.has_value()) {
				auto new_root = std::make_unique<inner_node<ValueType, Dims, MetricsSink>>(false, 0, &metrics);

				// NOCOMMIT OOF CAN WE DRY UP? ==> Change "recompue_child_bounding_box" to run on child after all..?
				GridBox<Dims> old_root_bbox = root->child_boxes[0];
				for(size_t i = 1; i < root->child_boxes.size(); ++i) {
					old_root_bbox = compute_bounding_box(old_root_bbox, root->child_boxes[i]);
				}

				new_root->insert_child_node(old_root_bbox, std::move(root));
				new_root->insert_child_node(ret->spilled_box, std::move(ret->spilled_node));
				root = std::move(new_root);
				root->set_depth(0);
			}
		}

		void insert_subtree(const GridBox<Dims>& box, std::unique_ptr<inner_node<ValueType, Dims, MetricsSink>>&& subtree) {
			// NOCOMMIT TODO Count metric?

			auto ret = root->insert_subtree(box, std::move(subtree));
			// NOCOMMIT DRY up
			if(ret.has_value()) {
				auto new_root = std::make_unique<inner_node<ValueType, Dims, MetricsSink>>(false, 0, &metrics);

				// NOCOMMIT OOF CAN WE DRY UP? ==> Change "recompue_child_bounding_box" to run on child after all..?
				GridBox<Dims> old_root_bbox = root->child_boxes[0];
				for(size_t i = 1; i < root->child_boxes.size(); ++i) {
					old_root_bbox = compute_bounding_box(old_root_bbox, root->child_boxes[i]);
				}

				new_root->insert_child_node(old_root_bbox, std::move(root));
				new_root->insert_child_node(ret->spilled_box, std::move(ret->spilled_node));
				root = std::move(new_root);
				root->set_depth(0);
			}
		}

		void erase(const GridBox<Dims>& box) {
			metrics.count_erase();

			// NOCOMMIT Streamline this
			erase_reinserts<ValueType, Dims, MetricsSink> to_reinsert;
			root->erase(box, to_reinsert);
			// auto& to_reinsert = current_erase_reinserts;

			for(auto& tr : to_reinsert) {
				if(std::holds_alternative<std::unique_ptr<ValueType>>(tr.second)) {
					auto& v = std::get<std::unique_ptr<ValueType>>(tr.second);
					// NOCOMMIT Yuck, we already have a pointer at this point
					insert(tr.first, *v);
				} else if(std::holds_alternative<std::unique_ptr<inner_node<ValueType, Dims, MetricsSink>>>(tr.second)) {
					auto& in = std::get<std::unique_ptr<inner_node<ValueType, Dims, MetricsSink>>>(tr.second);
					insert_subtree(tr.first, std::move(in));
				} else {
					assert(false);
				}
			}

			// NOCOMMIT WAY TOO LATE TO DO THIS!! NEEDS TO HAPPEN ABOVE SOMEHOW
			to_reinsert.clear();

			if(!root->is_leaf && root->children.size() == 1) {
				// Decrease tree height by 1 level.
				auto new_root = std::move(std::get<std::unique_ptr<inner_node<ValueType, Dims, MetricsSink>>>(root->children[0]));
				root = std::move(new_root);
				root->set_depth(0);
			}
		}

		// NOCOMMIT Can we also use this for result merges?
		bool can_merge(const GridBox<Dims>& bbox_a, const GridBox<Dims>& bbox_b) const {
			// In order to be mergeable, the two bounding boxes have to be adjacent in one dimension and match exactly in all remaining dimensions.
			bool adjacent = false;
			for(size_t d = 0; d < Dims; ++d) {
				if(bbox_a.get_min()[d] != bbox_b.get_min()[d] || bbox_a.get_max()[d] != bbox_b.get_max()[d]) {
					// Dimension does not match exactly, but could still be adjacent.
					// If we already are adjacent in another dimension, we cannot merge.
					if(!adjacent && (bbox_a.get_max()[d] == bbox_b.get_min()[d] || bbox_b.get_max()[d] == bbox_a.get_min()[d])) {
						adjacent = true;
					} else {
						return false;
					}
				}
			}

			assert(adjacent);
			return true;
		}

		// NOCOMMIT TODO 2022: This is stupid, why does it receive a mutable vector when we never use the result?
		//                     Also we are invalidating the iterator by inserting in loop. Use two vectors and swap instead. Get rid of optional.
		void try_merging(std::vector<std::optional<std::pair<GridBox<Dims>, ValueType>>>& merge_candidates) {
			for(size_t i = 0; i < merge_candidates.size(); ++i) {
				// Probe into every direction, check whether there is a box with same value.
				// If yes, check if it can be merged.
				// If yes, erase the two boxes, insert the new one and add it as a merge candidate.
				// => Additionally make sure that the merged box is not another existing merge candidate (for now just go through queue and erase if necessary)

				// If a candidate was merged with an earlier one, it is set to std::nullopt.
				if(!merge_candidates[i].has_value()) continue;

				auto& [box, value] = *merge_candidates[i];

				for(size_t d = 0; d < Dims; ++d) {
					const auto min = box.get_min();
					const auto max = box.get_max();
					std::optional<GridBox<Dims>> other_box;
					if(min[d] > 0) {
						auto probe = min;
						probe[d] -= 1;
						const auto neighbor = root->point_query(probe);
						assert(neighbor != std::nullopt);
						if(*(neighbor->second) == value && can_merge(box, neighbor->first)) { other_box = neighbor->first; }
					}
					if(!other_box.has_value() && max[d] < extent.get_max()[d]) {
						auto probe = min;
						// Point queries are exclusive on the "max" coordinate of a box, so there is no need to increment this by 1.
						// In fact, we would miss boxes that are exactly 1 unit wide in this dimension if we incremented.
						probe[d] = max[d];
						const auto neighbor = root->point_query(probe);
						assert(neighbor != std::nullopt);
						if(*(neighbor->second) == value && can_merge(box, neighbor->first)) { other_box = neighbor->first; }
					}

					if(other_box.has_value()) {
						// First figure out whether this box is also in our candidates list, and if so, remove it.
						for(size_t j = i + 1; j < merge_candidates.size(); ++j) {
							if(merge_candidates[j].has_value() && merge_candidates[j]->first == other_box) {
								merge_candidates[j] = std::nullopt;
								break;
							}
						}

						// Now erase the two boxes, insert the merged one and mark it as a new candidate.
						erase(box);
						erase(*other_box);
						const auto new_box = compute_bounding_box(box, *other_box);
						insert(new_box, value);
						// NOCOMMIT TODO: Uuh - that's not safe (we are currently iterating over this vector)
						merge_candidates.push_back(std::make_pair(new_box, value));

						break; // No need to check other dimensions, move on to next candidate box.
					}
				}
			}
		}

	  private:
		// NOCOMMIT
		void sanity_check() const {
#ifndef NDEBUG
			assert(root->get_bounding_box() == extent);

			size_t max_depth = 0;
			std::queue<std::pair<GridBox<Dims>, const inner_node<ValueType, Dims, MetricsSink>*>> node_queue;
			node_queue.push(std::make_pair(root->get_bounding_box(), root.get()));

			while(!node_queue.empty()) {
				const auto node_bbox = node_queue.front().first;
				const auto& node = *node_queue.front().second;
				node_queue.pop();

				if(node.depth > max_depth) { max_depth = node.depth; }

				assert(node.children.size() == node.child_boxes.size());

				if(!node.is_leaf) {
					assert(node.depth <= max_depth);
					GridBox<Dims> bounding_box = node.child_boxes[0];
					for(size_t i = 0; i < node.children.size(); ++i) {
						auto& child_node = node.get_child_node(i);
						assert(child_node.depth == node.depth + 1);
						node_queue.push(std::make_pair(node.child_boxes[i], &child_node));
						bounding_box = compute_bounding_box(bounding_box, node.child_boxes[i]);
					}
					assert(bounding_box == node_bbox);
				} else {
					if(!node.children.empty() /* can only happen at depth == 0 */) {
						assert(node.depth == max_depth);
						GridBox<Dims> bounding_box = node.child_boxes[0];
						for(size_t i = 0; i < node.children.size(); ++i) {
							bounding_box = compute_bounding_box(bounding_box, node.child_boxes[i]);
						}
						assert(bounding_box == node_bbox);
					}
				}
			}
#endif
		}

		void print(std::ostream& os) const {
			os << "COOL REGION MAP\n";
			root->print(os, 0);
		}

		// Computes the number of *distinct regions*, i.e., entries with different values (NOT necessarily the number of boxes).
		// NB: Debug utility, NOT cheap
		size_t get_num_regions() const {
			std::unordered_set<ValueType> unique_values;
			root->collect_values(unique_values);
			return unique_values.size();
		}
	};

	template <size_t DimsOut, size_t DimsIn>
	GridBox<DimsOut> box_cast(const GridBox<DimsIn>& other) {
		GridPoint<DimsOut> min;
		GridPoint<DimsOut> max;
		for(size_t o = 0; o < DimsOut; ++o) {
			min[o] = o < DimsIn ? other.get_min()[o] : 0;
			max[o] = o < DimsIn ? other.get_max()[o] : 1;
		}
		return GridBox<DimsOut>(min, max);
	}

	// NOCOMMIT HACK _2: Same function exists in legacy region map
	inline void assert_dimensionality_2(const GridBox<3>& box, const int dims) {
		[[maybe_unused]] const auto& min = box.get_min();
		[[maybe_unused]] const auto& max = box.get_max();
		if(dims < 3) {
			assert(min[2] == 0);
			assert(max[2] == 1);
		}
		if(dims == 1) {
			assert(min[1] == 0);
			assert(max[1] == 1);
		}
	}

	inline void assert_dimensionality_2(const GridRegion<3>& reg, const int dims) {
		reg.scanByBoxes([&](const GridBox<3>& box) { assert_dimensionality_2(box, dims); });
	}

	// NOCOMMIT TODO Do we want this?
	template <typename ValueType>
	class my_cool_region_map_wrapper {
		friend struct region_map_testspy; // NOCOMMIT
	  public:
		// NOCOMMIT TODO: Pass dims into here
		// NOCOMMIT FIXME: Take box instead of range here
		my_cool_region_map_wrapper(cl::sycl::range<3> extent, int dims, ValueType default_value = ValueType{}) : dims(dims) {
			assert_dimensionality_2(subrange_to_grid_box(subrange<3>{id<3>{}, extent}), dims);
			switch(dims) {
			case 1: region_map.template emplace<cool_region_map<ValueType, 1>>(range_cast<1>(extent), default_value); break;
			case 2: region_map.template emplace<cool_region_map<ValueType, 2>>(range_cast<2>(extent), default_value); break;
			case 3: region_map.template emplace<cool_region_map<ValueType, 3>>(range_cast<3>(extent), default_value); break;
			default: assert(false);
			}
		}

		void update_region(const GridRegion<3>& region, const ValueType& value) {
			assert_dimensionality_2(region, dims);
			region.scanByBoxes([&](const GridBox<3>& box) { update_box(box, value); });
		}

		void update_box(const GridBox<3>& box, const ValueType& value) {
			switch(dims) {
			case 1: std::get<1>(region_map).update_box(box_cast<1>(box), value); break;
			case 2: std::get<2>(region_map).update_box(box_cast<2>(box), value); break;
			case 3: std::get<3>(region_map).update_box(box_cast<3>(box), value); break;
			default: assert(false);
			}
		}

		// NOCOMMIT TODO: Implement on cool_region_map itself?
		// ==> Is there any benefit to not doing this on a per-box basis? I.e. merging opportunities? Possibly.
		//     (I think basically it depends on how the region happens to be stored in terms of boxes!)
		std::vector<std::pair<GridBox<3>, ValueType>> get_region_values(const GridRegion<3>& request) const {
			assert_dimensionality_2(request, dims);
			std::vector<std::pair<GridBox<3>, ValueType>> results;
			request.scanByBoxes([&](const GridBox<3>& box) {
				const auto r = get_region_values(box);
				results.insert(results.begin(), r.cbegin(), r.cend());
			});
			return results;
		}

		// NOCOMMIT TODO Oof - that's not cheap for large result sets...
		std::vector<std::pair<GridBox<3>, ValueType>> get_region_values(const GridBox<3>& request) const {
			std::vector<std::pair<GridBox<3>, ValueType>> results;
			switch(dims) {
			case 1: {
				const auto results1 = std::get<1>(region_map).get_region_values(box_cast<1>(request));
				results.reserve(results1.size());
				std::transform(results1.cbegin(), results1.cend(), std::back_inserter(results),
				    [](const auto& p) { return std::make_pair(box_cast<3>(p.first), p.second); });
			} break;
			case 2: {
				const auto results2 = std::get<2>(region_map).get_region_values(box_cast<2>(request));
				results.reserve(results2.size());
				std::transform(results2.cbegin(), results2.cend(), std::back_inserter(results),
				    [](const auto& p) { return std::make_pair(box_cast<3>(p.first), p.second); });
			} break;
			case 3: {
				results = std::get<3>(region_map).get_region_values(box_cast<3>(request));
			} break;
			default: assert(false);
			}
			return results;
		}

		template <typename Functor>
		void apply_to_values(Functor&& f) {
			switch(dims) {
			case 1: std::get<1>(region_map).apply_to_values(std::forward<Functor>(f)); break;
			case 2: std::get<2>(region_map).apply_to_values(std::forward<Functor>(f)); break;
			case 3: std::get<3>(region_map).apply_to_values(std::forward<Functor>(f)); break;
			default: assert(false);
			}
		}

	  private:
		int dims;
		std::variant<std::monostate, cool_region_map<ValueType, 1>, cool_region_map<ValueType, 2>, cool_region_map<ValueType, 3>> region_map;
	};


} // namespace detail
} // namespace celerity
