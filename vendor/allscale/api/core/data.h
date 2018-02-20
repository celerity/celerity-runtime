#pragma once

#include <type_traits>

#include "allscale/utils/concepts.h"
#include "allscale/utils/serializer.h"

namespace allscale {
namespace api {
namespace core {

	namespace sema {

		// c++ versions of the data item element access helper functions, facilitating compiler analysis
		template<typename DataItem, typename T>
		T& _data_item_element_access(DataItem&, const typename DataItem::region_type&, T& ref) {
			return ref;
		}

		template<typename DataItem, typename T>
		const T& _data_item_element_access(const DataItem&, const typename DataItem::region_type&, const T& ref) {
			return ref;
		}

		/**
		 * A user-defined read requirement on a region of a data item.
		 */
		template<typename DataItem>
		void needs_read_access(const DataItem& item, const typename DataItem::region_type& region) {
			int a = 0; a = _data_item_element_access(item,region,a);
		};

		/**
		 * A user-defined write requirement on a region of a data item.
		 */
		template<typename DataItem>
		void needs_write_access(const DataItem& item, const typename DataItem::region_type& region) {
			int a = 0; _data_item_element_access(item,region,a) = 0;
		};

		/**
		 * Instruct compiler to ignore dependencies in the enclosing scope.
		 */
		inline void no_more_dependencies() {};

	}

	// a macro to wrap up data_item_element_access calls,
	// eliminating the overhead of creating a region instance on every access
	// the ternary operation enforces type checks even on reference compilations
	#ifndef ALLSCALECC
		#define data_item_element_access(DataItem,Region,Res) \
			((false) ? allscale::api::core::sema::_data_item_element_access(DataItem,Region,Res) : Res)
	#else
		#define data_item_element_access(DataItem,Region,Res) allscale::api::core::sema::_data_item_element_access(DataItem,Region,Res)
	#endif

	// ---------------------------------------------------------------------------------
	//									  Regions
	// ---------------------------------------------------------------------------------


	template<typename R, typename _ = void>
	struct is_region : public std::false_type {};

	template<typename R>
	struct is_region<R,typename std::enable_if<

			// regions have to be values (constructible, assignable, comparable)
			utils::is_value<R>::value &&

			// regions have to be serializable
			utils::is_serializable<R>::value &&

			// there has to be an emptiness check
			std::is_same<decltype((bool (R::*)(void) const)(&R::empty)), bool (R::*)(void) const>::value &&

			// there has to be an union operation
			std::is_same<decltype((R (*)(const R&, const R&))(&R::merge)), R(*)(const R&, const R&)>::value &&

			// there has to be an intersection operation
			std::is_same<decltype((R(*)(const R&, const R&))(&R::intersect)), R(*)(const R&, const R&)>::value &&

			// there has to be a set difference operation
			std::is_same<decltype((R(*)(const R&, const R&))(&R::difference)), R(*)(const R&, const R&)>::value &&

			// there has to be a span operator, computing the hull of two regions
			std::is_same<decltype((R(*)(const R&, const R&))(&R::span)), R(*)(const R&, const R&)>::value,

		void>::type> : public std::true_type {};




	// ---------------------------------------------------------------------------------
	//									 Fragments
	// ---------------------------------------------------------------------------------



	template<typename F, typename _ = void>
	struct is_fragment : public std::false_type {};

	template<typename F>
	struct is_fragment<F, typename std::enable_if<

		// fragment needs to expose a region type
		is_region<typename F::region_type>::value &&

		// fragments need to be constructible for a given region
		std::is_same<decltype(F(std::declval<const typename F::shared_data_type&>(), std::declval<const typename F::region_type&>())), F>::value &&

		// fragments need to be destructible
		std::is_destructible<F>::value &&

		// the region covered by the fragment has to be obtainable
		std::is_same<decltype((void (F::*)(utils::Archive&) const)(&F::getCoveredRegion)), void (F::*)(utils::Archive&) const>::value &&

		// there has to be a resize operator
		std::is_same<decltype((void (F::*)(const typename F::region_type&))(&F::resize)), void (F::*)(const typename F::region_type&)>::value &&

		// there is an insert operator importing data from an existing fragment
		std::is_same<decltype((void (F::*)(const F&, const typename F::region_type&))(&F::insert)), void (F::*)(const F&, const typename F::region_type&)>::value &&

		// there is a extract operator extracting a region of data from the present fragment
		std::is_same<decltype((void (F::*)(utils::ArchiveWriter&, const typename F::region_type&) const)(&F::extract)), void (F::*)(utils::ArchiveWriter&, const typename F::region_type&) const>::value &&

		// there is a insert operator, importing previously extracted data into this fragment
		std::is_same<decltype((void (F::*)(utils::ArchiveReader&))(&F::insert)), void (F::*)(utils::ArchiveReader&)>::value &&

		// can be concerted into a facade
		std::is_same<decltype((typename F::facade_type (F::*)(void))(&F::mask)), typename F::facade_type(F::*)(void)>::value,

		void>::type> : public std::true_type{};





	// ---------------------------------------------------------------------------------
	//									SharedData
	// ---------------------------------------------------------------------------------


	template<typename R, typename _ = void>
	struct is_shared_data : public std::false_type {};

	template<typename R>
	struct is_shared_data<R,typename std::enable_if<

			// regions have to be values (constructible, assignable, comparable)
			std::is_destructible<R>::value &&

			// regions have to be serializable
			utils::is_serializable<R>::value,

		void>::type> : public std::true_type {};


	// ---------------------------------------------------------------------------------
	//									  Facade
	// ---------------------------------------------------------------------------------


	template<typename F, typename _ = void>
	struct is_facade : public std::false_type {};

	template<typename F>
	struct is_facade<F,typename std::enable_if<

			// facade must not be copy-constructible
			!std::is_copy_constructible<F>::value &&

			// nor copy-assignable
			!std::is_copy_assignable<F>::value &&

			// fragments need to be destructible
			std::is_destructible<F>::value,

		void>::type> : public std::true_type {};


	// ---------------------------------------------------------------------------------
	//									  Data Items
	// ---------------------------------------------------------------------------------


	template<typename D, typename _ = void>
	struct is_data_item : public std::false_type {};

	template<typename D>
	struct is_data_item<D,typename std::enable_if<
			std::is_same<D,typename D::facade_type>::value &&
			is_facade<D>::value &&
			is_fragment<typename D::fragment_type>::value &&
			is_shared_data<typename D::shared_data_type>::value,
		void>::type> : public std::true_type {};


	template<
		typename Fragment
	>
	struct data_item {

		// make sure the region type is satisfying the concept
		static_assert(is_region<typename Fragment::region_type>::value, "Region type must fit region concept!");
		static_assert(is_fragment<Fragment>::value, "Fragment type must fit fragment concept!");
		static_assert(is_shared_data<typename Fragment::shared_data_type>::value, "Shared data type must fit shared data concept!");

		using fragment_type = Fragment;
		using region_type = typename Fragment::region_type;
		using facade_type = typename Fragment::facade_type;
		using shared_data_type = typename Fragment::shared_data_type;

		// define default init/copy/move support

		data_item() = default;
		data_item(data_item&&) = default;
		data_item(const data_item&) = delete;

		data_item& operator=(const data_item&) = delete;
		data_item& operator=(data_item&&) = default;
	};


	// ---------------------------------------------------------------------------------
	//									  Utilities
	// ---------------------------------------------------------------------------------


	/**
	 * A generic utility to compute whether a region a is covering a sub-set of a region b.
	 */
	template<typename R>
	typename std::enable_if<is_region<R>::value,bool>::type
	isSubRegion(const R& a, const R& b) {
		return R::difference(a,b).empty();
	}

	/**
	 * A convenience wrapper for computing the span (e.g. convex hull) between two data regions.
	 */
	template<typename R>
	typename std::enable_if<is_region<R>::value,R>::type
	span(const R& a, const R& b) {
		return R::span(a,b);
	}

	/**
	 * A convince wrapper for merging a number of regions (single element base-case).
	 */
	template<typename R>
	typename std::enable_if<is_region<R>::value,R>::type
	merge(const R& a) {
		return a;
	}

	/**
	 * A convince wrapper for merging a number of regions (multiple element step-case).
	 */
	template<typename R, typename ... Rs>
	typename std::enable_if<is_region<R>::value,R>::type
	merge(const R& a, const Rs& ... rest) {
		return R::merge(a,merge(rest...));
	}

	/**
	 * A default implementation of shared data for data items that do not need shared any shared data.
	 */
	struct no_shared_data {

		void store(utils::ArchiveWriter&) const {
			// nothing to do
		}

		static no_shared_data load(utils::ArchiveReader&) {
			return no_shared_data();
		}

	};

	// make sure the no_shared_data is a shared data instance
	static_assert(is_shared_data<no_shared_data>::value, "no_shared_data type does not fulfill shared data concept!");

} // end namespace core
} // end namespace api
} // end namespace allscale
