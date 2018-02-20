#pragma once

#include <algorithm>
#include <map>
#include <memory>
#include <set>

#include "allscale/api/core/data.h"
#include "allscale/utils/assert.h"

#include "allscale/utils/printer/set.h"

namespace allscale {
namespace api {
namespace user {
namespace data {

	/**
	 * This header file defines an example data item covering a generic map of key-value pairs.
	 * The corresponding elements are:
	 * 		- a range type which corresponds to a set of keys
	 * 		- a fragment type capable of storing a share of the represented data
	 * 		- a facade type to be offered to the user as an interface
	 */


	// ---------------------------------------------------------------------------------
	//								 Declarations
	// ---------------------------------------------------------------------------------



	template<typename Element>
	class SetRegion;

	template<typename Key, typename Value>
	class Map;

	template<typename Key, typename Value>
	class MapFragment;



	// ---------------------------------------------------------------------------------
	//								  Definitions
	// ---------------------------------------------------------------------------------

	/**
	 * The implementation of a set-region enumerating the covered elements explicitly.
	 *
	 * @tparam Element the type of element to describe an element within the set; the type
	 * 		has to be serializable
	 */
	template<typename Element>
	class SetRegion {

		/**
		 * The elements covered by this region, explicitly enumerated.
		 */
		std::set<Element> elements;

	public:

		/**
		 * Adds a new element to this region.
		 */
		void add(const Element& e) {
			elements.insert(e);
		}

		/**
		 * Add multiple elements at once.
		 */
		template<typename ... Rest>
		void add(const Element& e, const Rest& ... rest) {
			add(e); add(rest...);
		}

		/**
		 * Terminal case for adding multiple elements.
		 */
		void add() { /* nothing */ }

		/**
		 * Obtains a list of all covered elements.
		 */
		const std::set<Element>& getElements() const {
			return elements;
		}

		// -- requirements imposed by the region concept --

		/**
		 * Determines whether this region is empty.
		 */
		bool empty() const {
			return elements.empty();
		}

		/**
		 * A comparison operator comparing regions on equality.
		 */
		bool operator==(const SetRegion& other) const {
			return elements == other.elements;
		}

		/**
		 * A comparison operator comparing regions for inequality.
		 */
		bool operator!=(const SetRegion& other) const {
			return !(*this == other);
		}

		/**
		 * An operator to merge two set regions.
		 */
		static SetRegion merge(const SetRegion& a, const SetRegion& b) {
			SetRegion res;
			std::set_union(a.elements.begin(),a.elements.end(),b.elements.begin(),b.elements.end(),std::inserter(res.elements, res.elements.begin()));
			return res;
		}

		/**
		 * An operator to intersect two set regions.
		 */
		static SetRegion intersect(const SetRegion& a, const SetRegion& b) {
			SetRegion res;
			std::set_intersection(a.elements.begin(), a.elements.end(), b.elements.begin(), b.elements.end(), std::inserter(res.elements, res.elements.begin()));
			return res;
		}

		/**
		 * An operator to compute the set-difference of two set regions.
		 */
		static SetRegion difference(const SetRegion& a, const SetRegion& b) {
			SetRegion res;
			std::set_difference(a.elements.begin(), a.elements.end(), b.elements.begin(), b.elements.end(), std::inserter(res.elements, res.elements.begin()));
			return res;
		}

		static SetRegion span(const SetRegion&, const SetRegion&) {
			std::cout << "Unsupported operation: cannot computed span on set regions!";
			exit(1);
		}

		/**
		 * An operator to load an instance of this range from the given archive.
		 */
		static SetRegion load(utils::ArchiveReader&) {
			assert_not_implemented();
			return SetRegion();
		}

		/**
		 * An operator to store an instance of this range into the given archive.
		 */
		void store(utils::ArchiveWriter&) const {
			assert_not_implemented();
			// nothing so far
		}

		/**
		 * Enables printing the elements of this set region.
		 */
		friend std::ostream& operator<<(std::ostream& out, const SetRegion& region) {
			return out << region.elements;
		}
	};

	/**
	 * An implementation of a fragment of a map-like data item. Each fragment
	 * stores a sub-section of the key-value pairs to be maintained by the overall map.
	 *
	 * @tparam Key the key type of the map to be stored
	 * @tparam Value the value type of the data to be associated to the key
	 */
	template<typename Key, typename Value>
	class MapFragment {

		/**
		 * The region this fragment is covering.
		 */
		SetRegion<Key> region;

		/**
		 * The data stored in this fragment.
		 */
		std::map<Key,Value> data;

		// enables the facade to access internal data of this class.
		friend class Map<Key,Value>;

	public:

		using shared_data_type = core::no_shared_data;
		using facade_type = Map<Key,Value>;
		using region_type = SetRegion<Key>;

		/**
		 * Create a new fragment covering the given region.
		 */
		MapFragment(const region_type& region)
			: MapFragment(core::no_shared_data(),region) {}

		/**
		 * Create a new fragment covering the given region.
		 */
		MapFragment(const core::no_shared_data&, const region_type& region) : region(region) {
			for(const auto& key : region.getElements()) {
				data[key]; // initialize content by accessing elements
			}
		}

		/**
		 * Obtains a facade to this fragment to be forwarded by the data manager to the user code
		 * for interacting with this fragment.
		 */
		Map<Key,Value> mask() {
			return Map<Key,Value>(*this);
		}

		/**
		 * Obtains the range of data covered by this fragment.
		 */
		const region_type& getCoveredRegion() const {
			return region;
		}

		/**
		 * Resizes this fragment to provide enough space to store values for the given key-set.
		 */
		void resize(const region_type& keys) {

			// update the covered region
			region = keys;

			// build up new data storage
			std::map<Key,Value> newData;
			for(const auto& key : keys.getElements()) {
				auto pos = data.find(key);
				newData[key] = (pos != data.end()) ? pos->second : Value();
			}

			// swap data containers
			data.swap(newData);
		}

		/**
		 * Merges all the data from the given fragment into this fragment.
		 */
		void insert(const MapFragment& other, const region_type& fraction) {
			assert_true(core::isSubRegion(fraction,region))
					<< "Cannot insert non-sub-set region into this fragment.";
			assert_true(core::isSubRegion(fraction,other.region))
					<< "Cannot load non-sub-set region from other fragment.";
			// move in data
			for(const auto& cur : fraction.getElements()) {
				auto pos = other.data.find(cur);
				assert_true(pos != other.data.end());
				data[cur] = pos->second;
			}
		}

		void extract(utils::ArchiveWriter&, const region_type&) const {
			assert_not_implemented();
		}

		void insert(utils::ArchiveReader&) {
			assert_not_implemented();
		}

	};


	/**
	 * The map facade forming the actual data item to be managed by the
	 * runtime system.
	 *
	 * @tparam Key a key type, needs to be serializable
	 * @tparam Value a value type, needs to be serializable as well
	 */
	template<typename Key, typename Value>
	class Map : public core::data_item<MapFragment<Key,Value>> {

		/**
		 * A pointer to an underlying fragment owned if used in an unmanaged state.
		 */
		std::unique_ptr<MapFragment<Key,Value>> owned;

		/**
		 * A reference to the fragment instance operating on, referencing the owned fragment or an externally managed one.
		 */
		MapFragment<Key,Value>& base;

		/**
		 * Enables fragments to use the private constructor below.
		 */
		friend class MapFragment<Key,Value>;

		/**
		 * The constructor to be utilized by the fragment to create a facade for an existing fragment.
		 */
		Map(MapFragment<Key,Value>& base) : base(base) {}

	public:

		/**
		 * Creates a new map covering the given region.
		 */
		Map(const SetRegion<Key>& keys) : owned(std::make_unique<MapFragment<Key,Value>>(keys)), base(*owned) {}

		/**
		 * Provides read/write access to one of the values stored within this map.
		 */
		Value& operator[](const Key& key) {
			auto pos = base.data.find(key);
			assert_true(pos != base.data.end()) << "Access to invalid key: " << key << " - covered region: " << base.region;
			return pos->second;
		}

		/**
		 * Provides read access to one of the values stored within this map.
		 */
		const Value& operator[](const Key& key) const {
			auto pos = base.data.find(key);
			assert_true(pos != base.data.end()) << "Access to invalid key: " << key << " - covered region: " << base.region;
			return pos->second;
		}

	};

} // end namespace data
} // end namespace user
} // end namespace api
} // end namespace allscale
