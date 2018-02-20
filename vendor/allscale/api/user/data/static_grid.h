#pragma once

#include "allscale/api/user/data/grid.h"

namespace allscale {
namespace api {
namespace user {
namespace data {


	// ---------------------------------------------------------------------------------
	//								 Declarations
	// ---------------------------------------------------------------------------------


	using coordinate_type = std::int64_t;

	template<std::size_t Dims>
	using StaticGridPoint = GridPoint<Dims>;

	template<std::size_t Dims>
	using StaticGridBox = GridBox<Dims>;

	template<std::size_t Dims>
	using StaticGridRegion = GridRegion<Dims>;

	template<typename T, std::size_t ... Sizes>
	class StaticGridFragment;

	template<typename T, std::size_t ... Sizes>
	class StaticGrid;




	// ---------------------------------------------------------------------------------
	//								  Definitions
	// ---------------------------------------------------------------------------------


	template<typename T, std::size_t ... Sizes>
	class StaticGridFragment {
	public:

		enum { Dims = sizeof...(Sizes) };

		using shared_data_type = core::no_shared_data;
		using facade_type = StaticGrid<T,Sizes...>;
		using region_type = StaticGridRegion<Dims>;

	private:

		using point = StaticGridPoint<Dims>;
		using box = StaticGridBox<Dims>;

		region_type size;

		utils::LargeArray<T> data;

	public:

		StaticGridFragment(const region_type& size = region_type())
			: StaticGridFragment(core::no_shared_data(), size) {}

		StaticGridFragment(const core::no_shared_data&, const region_type& size = region_type()) : size(size), data(area(totalSize())) {
			// allocate covered data space
			size.scanByLines([&](const point& a, const point& b) {
				data.allocate(flatten(a),flatten(b));
			});
		}

		bool operator==(const StaticGridFragment& other) const {
			return data == other.data;
		}

		T& operator[](const point& pos) {
			return data[flatten(pos)];
		}

		const T& operator[](const point& pos) const {
			return data[flatten(pos)];
		}

		StaticGrid<T,Sizes...> mask() {
			return StaticGrid<T,Sizes...>(*this);
		}

		const region_type& getCoveredRegion() const {
			return size;
		}

		point totalSize() const {
			return point({ Sizes... });
		}

		void resize(const region_type& newSize) {

			// get the difference
			region_type plus  = region_type::difference(newSize,size);
			region_type minus = region_type::difference(size,newSize);

			// update the size
			size = newSize;

			// allocated new data
			plus.scanByLines([&](const point& a, const point& b){
				data.allocate(flatten(a),flatten(b));
			});

			// free excessive memory
			minus.scanByLines([&](const point& a, const point& b){
				data.free(flatten(a),flatten(b));
			});
		}

		void insert(const StaticGridFragment& other, const region_type& area) {
			assert_true(core::isSubRegion(area,other.size)) << "New data " << area << " not covered by source of size " << size << "\n";
			assert_true(core::isSubRegion(area,size))       << "New data " << area << " not covered by target of size " << size << "\n";

			// copy data line by line using memcpy
			area.scanByLines([&](const point& a, const point& b){
				auto start = flatten(a);
				auto length = (flatten(b) - start) * sizeof(T);
				std::memcpy(&data[start],&other.data[start],length);
			});
		}

		void extract(utils::ArchiveWriter& writer, const region_type& region) const {

			// make sure the region is covered
			assert_pred2(core::isSubRegion, region, getCoveredRegion())
				<< "This fragment does not contain all of the requested data!";

			// write the requested region to the archive
			writer.write(region);

			// add the data
			region.scan([&](const point& p){
				writer.write((*this)[p]);
			});
		}

		void insert(utils::ArchiveReader& reader) {

			// extract the covered region contained in the archive
			auto region = reader.read<region_type>();

			// check that it is fitting
			assert_pred2(core::isSubRegion, region, getCoveredRegion())
				<< "Targeted fragment does not cover data to be inserted!";

			// insert the data
			region.scan([&](const point& p){
				(*this)[p] = reader.read<T>();
			});
		}

	private:

		static std::size_t area(const StaticGridPoint<Dims>& pos) {
			std::size_t res = 1;
			for(std::size_t i=0; i<Dims; ++i) {
				res *= pos[i];
			}
			return res;
		}

		coordinate_type flatten(const StaticGridPoint<Dims>& pos) const {

			static const std::array<coordinate_type, Dims> totalSize{ { Sizes ... } };

			coordinate_type res = 0;
			coordinate_type size = 1;

			for(int i=Dims-1; i>=0; i--) {
				res += pos[i] * size;
				size *= totalSize[i];
			}

			return res;
		}

	};

	template<typename T, std::size_t ... Sizes>
	class StaticGrid : public core::data_item<StaticGridFragment<T,Sizes...>> {

		/**
		 * A pointer to an underlying fragment owned if used in an unmanaged state.
		 */
		std::unique_ptr<StaticGridFragment<T,Sizes...>> owned;

		/**
		 * A reference to the fragment instance operating on, referencing the owned fragment or an externally managed one.
		 */
		StaticGridFragment<T,Sizes...>* base;

		/**
		 * Enables fragments to use the private constructor below.
		 */
		friend class StaticGridFragment<T,Sizes...>;

		/**
		 * The constructor to be utilized by the fragment to create a facade for an existing fragment.
		 */
		StaticGrid(StaticGridFragment<T,Sizes...>& base) : base(&base) {}

	public:

		/**
		 * The number of dimensions.
		 */
		enum { dimensions = sizeof...(Sizes) };

		/**
		 * The type of coordinate utilized by this type.
		 */
		using coordinate_type = StaticGridPoint<dimensions>;

		/**
		 * The type of region utilized by this type.
		 */
		using region_type = StaticGridRegion<dimensions>;

		/**
		 * Creates a new map covering the given region.
		 */
		StaticGrid()
			: owned(std::make_unique<StaticGridFragment<T,Sizes...>>(region_type(0,size()))), base(owned.get()) {}

		/**
		 * A constructor for static grids accepting a size parameter, to be compatible to the dynamic sized grid.
		 */
		StaticGrid(const StaticGridPoint<dimensions>& size)
			: owned(std::make_unique<StaticGridFragment<T,Sizes...>>(region_type(0,size))), base(owned.get()) {
			assert_eq(size,this->size()) << "Initialization of invalid sized static grid.";
		}

		/**
		 * Disable copy construction.
		 */
		StaticGrid(const StaticGrid&) = delete;

		/**
		 * Enable move construction.
		 */
		StaticGrid(StaticGrid&&) = default;

		/**
		 * Disable copy-assignments.
		 */
		StaticGrid& operator=(const StaticGrid&) = delete;

		/**
		 * Enable move assignments.
		 */
		StaticGrid& operator=(StaticGrid&&) = default;

		/**
		 * Obtains the full size of this grid.
		 */
		coordinate_type size() const {
			return coordinate_type({ Sizes ... });
		}

		/**
		 * Compare the full content of the grid.
		 */
		bool operator==(const StaticGrid& other) const {
			return *base == *other.base;
		}

		/**
		 * Provides read/write access to one of the values stored within this grid.
		 */
		T& operator[](const coordinate_type& index) {
			allscale_check_bounds(index, (*this));
			return data_item_element_access(*this, region_type::single(index), (*base)[index]);
		}

		/**
		 * Provides read access to one of the values stored within this grid.
		 */
		const T& operator[](const coordinate_type& index) const {
			allscale_check_bounds(index, (*this));
			return data_item_element_access(*this, region_type::single(index), (*base)[index]);
		}

		/**
		 * A sequential scan over all elements within this grid, providing
		 * read-only access.
		 */
		template<typename Op>
		void forEach(const Op& op) const {
			allscale::api::user::algorithm::detail::forEach(
					coordinate_type(0),
					size(),
					[&](const auto& pos){
						op((*this)[pos]);
					}
			);
		}

		/**
		 * A sequential scan over all elements within this grid, providing
		 * read/write access.
		 */
		template<typename Op>
		void forEach(const Op& op) {
			allscale::api::user::algorithm::detail::forEach(
					coordinate_type(0),
					size(),
					[&](const auto& pos){
						op((*this)[pos]);
					}
			);
		}

		/**
		 * A sequential scan over all elements within this grid, providing
		 * read-only access.
		 */
		template<typename Op>
		auto pforEach(const Op& op) const {
			return algorithm::pfor(coordinate_type(0), size(), [&](const auto& pos) { op((*this)[pos]); });
		}

		/**
		 * A parallel scan over all elements within this grid, providing
		 * read/write access.
		 */
		template<typename Op>
		auto pforEach(const Op& op) {
			return algorithm::pfor(coordinate_type(0), size(), [&](const auto& pos) { op((*this)[pos]); });
		}

	};

} // end namespace data
} // end namespace user
} // end namespace api
} // end namespace allscale
