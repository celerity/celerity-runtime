#pragma once

#include <cstring>
#include <memory>

#include "allscale/api/core/data.h"

#include "allscale/api/user/algorithm/pfor.h"

#include "allscale/utils/assert.h"
#include "allscale/utils/large_array.h"
#include "allscale/utils/printer/join.h"
#include "allscale/utils/serializer/vectors.h"
#include "allscale/utils/vector.h"

namespace allscale {
namespace api {
namespace user {
namespace data {


	// ---------------------------------------------------------------------------------
	//								 Declarations
	// ---------------------------------------------------------------------------------


	using coordinate_type = std::int64_t;

	template<std::size_t Dims>
	using GridPoint = utils::Vector<coordinate_type,Dims>;

	template<std::size_t Dims>
	class GridBox;

	template<std::size_t Dims>
	class GridRegion;

	template<typename T, std::size_t Dims>
	class GridFragment;

	template<typename T, std::size_t Dims = 2>
	class Grid;




	// ---------------------------------------------------------------------------------
	//								  Definitions
	// ---------------------------------------------------------------------------------

	namespace detail {

		template<std::size_t I>
		struct difference_computer {

			template<std::size_t Dims>
			void collectDifferences(const GridBox<Dims>& a, const GridBox<Dims>& b, GridBox<Dims>& cur, std::vector<GridBox<Dims>>& res) {
				std::size_t i = I-1;

				// if b is within a
				if (a.min[i] <= b.min[i] && b.max[i] <= a.max[i]) {

					// cover left part
					cur.min[i] = a.min[i]; cur.max[i] = b.min[i];
					if (cur.min[i] < cur.max[i]) difference_computer<I-1>().collectDifferences(a,b,cur,res);

					// cover center part
					cur.min[i] = b.min[i]; cur.max[i] = b.max[i];
					if (cur.min[i] < cur.max[i]) difference_computer<I-1>().collectDifferences(a,b,cur,res);

					// cover right part
					cur.min[i] = b.max[i]; cur.max[i] = a.max[i];
					if (cur.min[i] < cur.max[i]) difference_computer<I-1>().collectDifferences(a,b,cur,res);

				// if a is within b
				} else if (b.min[i] <= a.min[i] && a.max[i] <= b.max[i]) {

					// cover inner part
					cur.min[i] = a.min[i]; cur.max[i] = a.max[i];
					if (cur.min[i] < cur.max[i]) difference_computer<I-1>().collectDifferences(a,b,cur,res);

				// if a is on the left
				} else if (a.min[i] <= b.min[i]) {

					// cover left part
					cur.min[i] = a.min[i]; cur.max[i] = b.min[i];
					if (cur.min[i] < cur.max[i]) difference_computer<I-1>().collectDifferences(a,b,cur,res);

					// cover right part
					cur.min[i] = b.min[i]; cur.max[i] = a.max[i];
					if (cur.min[i] < cur.max[i]) difference_computer<I-1>().collectDifferences(a,b,cur,res);

				// otherwise a is on the right
				} else {

					// cover left part
					cur.min[i] = a.min[i]; cur.max[i] = b.max[i];
					if (cur.min[i] < cur.max[i]) difference_computer<I-1>().collectDifferences(a,b,cur,res);

					// cover right part
					cur.min[i] = b.max[i]; cur.max[i] = a.max[i];
					if (cur.min[i] < cur.max[i]) difference_computer<I-1>().collectDifferences(a,b,cur,res);

				}

			}

		};

		template<>
		struct difference_computer<0> {

			template<std::size_t Dims>
			void collectDifferences(const GridBox<Dims>&, const GridBox<Dims>& b, GridBox<Dims>& cur, std::vector<GridBox<Dims>>& res) {
				if(!b.covers(cur) && !cur.empty()) res.push_back(cur);
			}
		};

		template<std::size_t I>
		struct box_fuser {
			template<std::size_t Dims>
			bool apply(std::vector<GridBox<Dims>>& boxes) {

				// try fuse I-th dimension
				for(std::size_t i = 0; i<boxes.size(); i++) {
					for(std::size_t j = i+1; j<boxes.size(); j++) {

						// check whether a fusion is possible
						GridBox<Dims>& a = boxes[i];
						GridBox<Dims>& b = boxes[j];
						if (GridBox<Dims>::template areFusable<I-1>(a,b)) {

							// fuse the boxes
							GridBox<Dims> f = GridBox<Dims>::template fuse<I-1>(a,b);
							boxes.erase(boxes.begin() + j);
							boxes[i] = f;

							// start over again
							apply(boxes);
							return true;
						}
					}
				}

				// fuse smaller dimensions
				if (box_fuser<I-1>().apply(boxes)) {
					// start over again
					apply(boxes);
					return true;
				}

				// no more changes
				return false;
			}
		};

		template<>
		struct box_fuser<0> {
			template<std::size_t Dims>
			bool apply(std::vector<GridBox<Dims>>&) { return false; }
		};

		template<std::size_t I>
		struct line_scanner {
			template<std::size_t Dims, typename Lambda>
			void apply(const GridBox<Dims>& box, GridPoint<Dims>& a, GridPoint<Dims>& b, const Lambda& body) {
				for(coordinate_type i = box.min[Dims-I]; i < box.max[Dims-I]; ++i ) {
					a[Dims-I] = i;
					b[Dims-I] = i;
					line_scanner<I-1>().template apply<Dims>(box,a,b,body);
				}
			}
		};

		template<>
		struct line_scanner<1> {
			template<std::size_t Dims, typename Lambda>
			void apply(const GridBox<Dims>& box, GridPoint<Dims>& a, GridPoint<Dims>& b, const Lambda& body) {
				a[Dims-1] = box.min[Dims-1];
				b[Dims-1] = box.max[Dims-1];
				body(a,b);
			}
		};
	}


	template<std::size_t Dims>
	class GridBox {

		static_assert(Dims >= 1, "0-dimension Grids (=Scalars) not yet supported.");

		template<std::size_t I>
		friend struct detail::difference_computer;

		template<std::size_t I>
		friend struct detail::line_scanner;

		template<std::size_t D>
		friend class GridRegion;

		using point_type = GridPoint<Dims>;

		point_type min;
		point_type max;

	public:
		GridBox() {}

		GridBox(coordinate_type N)
			: min(0), max(N) {}

		GridBox(coordinate_type A, coordinate_type B)
			: min(A), max(B) {}

		GridBox(const point_type& N)
			: min(0), max(N) {}

		GridBox(const point_type& A, const point_type& B)
			: min(A), max(B) {}

		bool empty() const {
			return !min.strictlyDominatedBy(max);
		}

		std::size_t area() const {
			std::size_t res = 1;
			for(std::size_t i=0; i<Dims; i++) {
				if (max[i] <= min[i]) return 0;
				res *= max[i] - min[i];
			}
			return res;
		}

		bool operator==(const GridBox& other) const {
			if (this == &other) return true;
			if (empty() && other.empty()) return true;
			return min == other.min && max == other.max;
		}

		bool operator!=(const GridBox& other) const {
			return !(*this == other);
		}

		bool covers(const point_type& point) const {
			for(std::size_t i = 0; i<Dims; i++) {
				if (!(min[i] <= point[i] && point[i] < max[i])) return false;
			}
			return true;
		}

		bool covers(const GridBox& other) const {
			if (other.empty()) return true;
			if (empty()) return false;
			return min.dominatedBy(other.min) && other.max.dominatedBy(max);
		}

		bool intersectsWith(const GridBox& other) const {
			// empty sets do not intersect with any set
			if (other.empty() || empty()) return false;

			// check each dimension
			for(std::size_t i = 0; i<Dims; i++) {
				// the minimum of the one has to be between the min and max of the other
				if (!(
						(min[i] <= other.min[i] && other.min[i] < max[i]) ||
						(other.min[i] <= min[i] && min[i] < other.max[i])
					)) {
					return false;		// if not, there is no intersection
				}
			}
			return true;
		}

		static std::vector<GridBox> merge(const GridBox& a, const GridBox& b) {

			// handle empty sets
			if (a.empty() && b.empty()) return std::vector<GridBox>();
			if (a.empty()) return std::vector<GridBox>({b});
			if (b.empty()) return std::vector<GridBox>({a});

			// boxes are intersecting => we have to do some work
			auto res = difference(a,b);
			res.push_back(b);
			return res;
		}

		static GridBox intersect(const GridBox& a, const GridBox& b) {
			// compute the intersection
			GridBox res = a;
			for(std::size_t i = 0; i<Dims; i++) {
				res.min[i] = std::max(res.min[i],b.min[i]);
				res.max[i] = std::min(res.max[i],b.max[i]);
			}
			return res;
		}

		static std::vector<GridBox> difference(const GridBox& a, const GridBox& b) {

			// handle case where b covers whole a
			if (b.covers(a)) return std::vector<GridBox>();

			// check whether there is an actual intersection
			if (!a.intersectsWith(b)) {
				return std::vector<GridBox>({a});
			}

			// slice up every single dimension
			GridBox cur;
			std::vector<GridBox> res;
			detail::difference_computer<Dims>().collectDifferences(a,b,cur,res);
			return res;
		}

		static GridBox span(const GridBox& a, const GridBox& b) {
			return GridBox(
				allscale::utils::elementwiseMin(a.min,b.min),
				allscale::utils::elementwiseMax(a.max,b.max)
			);
		}

		template<typename Lambda>
		void scanByLines(const Lambda& body) const {
			if (empty()) return;
			point_type a;
			point_type b;
			detail::line_scanner<Dims>().template apply<Dims>(*this,a,b,body);
		}

		template<std::size_t D>
		static bool areFusable(const GridBox& a, const GridBox& b) {
			static_assert(D < Dims, "Can not fuse on non-existing dimension.");
			if (a.min > b.min) return areFusable<D>(b,a);
			if (a.max[D] != b.min[D]) return false;
			for(std::size_t i = 0; i<Dims; i++) {
				if (i == D) continue;
				if (a.min[i] != b.min[i]) return false;
				if (a.max[i] != b.max[i]) return false;
			}
			return true;
		}

		template<std::size_t D>
		static GridBox fuse(const GridBox& a, const GridBox& b) {
			assert_true(areFusable<D>(a,b));
			if (a.min[D] > b.min[D]) return fuse<D>(b,a);
			GridBox res = a;
			res.max[D] = b.max[D];
			return res;
		}

		friend std::ostream& operator<<(std::ostream& out, const GridBox& box) {
			return out << "[" << box.min << " - " << box.max << "]";
		}

		/**
		 * An operator to load an instance of this range from the given archive.
		 */
		static GridBox load(utils::ArchiveReader& reader) {
			auto min = reader.read<point_type>();
			auto max = reader.read<point_type>();
			return { min, max };
		}

		/**
		 * An operator to store an instance of this range into the given archive.
		 */
		void store(utils::ArchiveWriter& writer) const {
			writer.write(min);
			writer.write(max);
		}

	};

	template<std::size_t Dims>
	class GridRegion {

		static_assert(Dims > 0, "0-dimensional grids are not supported yet");

		using point_type = GridPoint<Dims>;
		using box_type = GridBox<Dims>;

		std::vector<box_type> regions;

	public:

		GridRegion() {}

		GridRegion(coordinate_type N)
			: regions({box_type(N)}) {
			if (0 >= N) regions.clear();
		}

		GridRegion(const point_type& size)
			: regions({box_type(0,size)}) {
			if (regions[0].empty()) regions.clear();
		}

		GridRegion(const point_type& min, const point_type& max)
			: regions({box_type(min,max)}) {
			assert_true(min.dominatedBy(max));
			if (regions[0].empty()) regions.clear();
		}

		GridRegion(const box_type& box)
			: regions({box}) {
			if (regions[0].empty()) regions.clear();
		}

		GridRegion(const GridRegion&) = default;
		GridRegion(GridRegion&&) = default;

		GridRegion& operator=(const GridRegion&) = default;
		GridRegion& operator=(GridRegion&&) = default;

		static GridRegion single(const point_type& p) {
			return GridRegion(p,p+point_type(1));
		}

		box_type boundingBox() const {
			// handle empty region
			if (regions.empty()) return box_type(0);

			// if there is a single element
			if (regions.size() == 1u) return regions.front();

			// compute the bounding box
			box_type res = regions.front();
			for(const box_type& cur : regions) {
				res.min = utils::elementwiseMin(res.min, cur.min);
				res.max = utils::elementwiseMax(res.max, cur.max);
			}
			return res;
		}

		bool operator==(const GridRegion& other) const {
			return difference(*this,other).empty() && other.difference(other,*this).empty();
		}

		bool operator!=(const GridRegion& other) const {
			return regions != other.regions;
		}

		bool empty() const {
			return regions.empty();
		}

		std::size_t area() const {
			std::size_t res = 0;
			for(const auto& cur : regions) {
				res += cur.area();
			}
			return res;
		}

		static GridRegion merge(const GridRegion& a, const GridRegion& b) {

			// if both sets are empty => done
			if(a.empty() && b.empty()) return a;

			// build result
			GridRegion res = a;

			// combine regions
			for(const auto& cur : difference(b,a).regions) {
				res.regions.push_back(cur);
			}

			// compress result
			res.compress();

			// done
			return res;
		}

		template<typename ... Rest>
		static GridRegion merge(const GridRegion& a, const GridRegion& b, const Rest& ... rest) {
			return merge(merge(a,b),rest...);
		}

		static GridRegion intersect(const GridRegion& a, const GridRegion& b) {

			// if one of the sets is empty => done
			if(a.empty()) return a;
			if(b.empty()) return b;

			// build result
			GridRegion res;

			// combine regions
			for(const auto& curA : a.regions) {
				for(const auto& curB : b.regions) {
					box_type diff = box_type::intersect(curA,curB);
					if (!diff.empty()) {
						res.regions.push_back(diff);
					}
				}
			}

			// compress result
			res.compress();

			// done
			return res;
		}

		static GridRegion difference(const GridRegion& a, const GridRegion& b) {

			// handle empty sets
			if(a.empty() || b.empty()) return a;


			// build result
			GridRegion res = a;

			// combine regions
			for(const auto& curB : b.regions) {
				std::vector<box_type> next;
				for(const auto& curA : res.regions) {
					for(const auto& n : box_type::difference(curA,curB)) {
						next.push_back(n);
					}
				}
				res.regions.swap(next);
			}

			// compress result
			res.compress();

			// done
			return res;
		}

		static GridRegion span(const GridRegion& a, const GridRegion& b) {
			GridRegion res;
			for(const auto& ba : a.regions) {
				for(const auto& bb : b.regions) {
					res = merge(res,GridRegion(box_type::span(ba,bb)));
				}
			}
			return res;
		}

		/**
		 * Scans the covered range, line by line.
		 */
		template<typename Lambda>
		void scanByLines(const Lambda& body) const {
			for(const auto& cur : regions) {
				cur.scanByLines(body);
			}
		}

		/**
		 * Scan the covered range, point by point.
		 */
		template<typename Lambda>
		void scan(const Lambda& body) const {
			scanByLines([&](point_type a, const point_type& b) {
				for(; a[Dims-1]<b[Dims-1]; a[Dims-1]++) {
					body(a);
				}
			});
		}

		/**
		 * An operator to load an instance of this range from the given archive.
		 */
		static GridRegion load(utils::ArchiveReader& reader) {
			// start with an empty region
			GridRegion res;

			// read the box entries
			res.regions = std::move(reader.read<std::vector<box_type>>());

			// done
			return res;
		}

		/**
		 * An operator to store an instance of this range into the given archive.
		 */
		void store(utils::ArchiveWriter& writer) const {
			// just save the regions
			writer.write(regions);
		}

		friend std::ostream& operator<<(std::ostream& out, const GridRegion& region) {
			return out << "{" << utils::join(",",region.regions) << "}";
		}

	private:

		void compress() {
			// try to fuse boxes
			detail::box_fuser<Dims>().apply(regions);
		}

	};


	template<std::size_t Dims>
	struct GridSharedData {

		using size_type = GridPoint<Dims>;

		size_type size;

        GridSharedData()
        {}

		GridSharedData(const size_type& size)
			: size(size) {}

		/**
		 * An operator to load an instance of this shared data from the given archive.
		 */
		static GridSharedData load(utils::ArchiveReader& reader) {
			return { reader.read<size_type>() };
		}

		/**
		 * An operator to store an instance of this shared data into the given archive.
		 */
		void store(utils::ArchiveWriter& writer) const {
			writer.write(size);
		}

	};


	template<typename T, std::size_t Dims>
	class GridFragment {
	public:

		using shared_data_type = GridSharedData<Dims>;
		using facade_type = Grid<T,Dims>;
		using region_type = GridRegion<Dims>;
		using size_type = GridPoint<Dims>;

	private:

		using point = GridPoint<Dims>;
		using box = GridBox<Dims>;

		size_type totalSize;
		region_type coveredRegion;

		utils::LargeArray<T> data;

	public:

		GridFragment(const size_type& size)
			: GridFragment(shared_data_type{ size }, size) {}

		GridFragment(const size_type& size, const region_type& region)
			: GridFragment(shared_data_type{ size }, region) {
			assert_pred2(core::isSubRegion, region, region_type(totalSize));
		}

		GridFragment(const shared_data_type& sharedData, const region_type& region = region_type())
				: totalSize(sharedData.size), coveredRegion(region_type::intersect(totalSize,region)), data(area(totalSize)) {
			// allocate covered data space
			coveredRegion.scanByLines([&](const point& a, const point& b) {
				data.allocate(flatten(a),flatten(b));
			});
		}

		T& operator[](const point& pos) {
			return data[flatten(pos)];
		}

		const T& operator[](const point& pos) const {
			return data[flatten(pos)];
		}

		Grid<T,Dims> mask() {
			return Grid<T,Dims>(*this);
		}

		const region_type& getCoveredRegion() const {
			return coveredRegion;
		}

		const size_type& getTotalSize() const {
			return totalSize;
		}

		void resize(const region_type& newCoveredRegion) {

			// get region to be covered within the requested region
			region_type newRegion = region_type::intersect(getTotalSize(),newCoveredRegion);

			// get the difference
			region_type plus  = region_type::difference(newRegion,coveredRegion);
			region_type minus = region_type::difference(coveredRegion,newRegion);

			// update the size
			coveredRegion = newRegion;

			// allocated new data
			plus.scanByLines([&](const point& a, const point& b){
				data.allocate(flatten(a),flatten(b));
			});

			// free excessive memory
			minus.scanByLines([&](const point& a, const point& b){
				data.free(flatten(a),flatten(b));
			});
		}

		void insert(const GridFragment& other, const region_type& area) {
			assert_true(core::isSubRegion(area,other.coveredRegion)) << "New data " << area << " not covered by source of size " << other.coveredRegion << "\n";
			assert_true(core::isSubRegion(area,coveredRegion))       << "New data " << area << " not covered by target of size " << coveredRegion << "\n";

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

		static std::size_t area(const GridPoint<Dims>& pos) {
			std::size_t res = 1;
			for(std::size_t i=0; i<Dims; ++i) {
				res *= pos[i];
			}
			return res;
		}

		coordinate_type flatten(const GridPoint<Dims>& pos) const {
			coordinate_type res = 0;
			coordinate_type size = 1;

			for(int i=Dims-1; i>=0; i--) {
				res += pos[i] * size;
				size *= getTotalSize()[i];
			}

			return res;
		}

	};

	template<typename T, std::size_t Dims>
	class Grid : public core::data_item<GridFragment<T,Dims>> {

		/**
		 * A pointer to an underlying fragment owned if used in an unmanaged state.
		 */
		std::unique_ptr<GridFragment<T,Dims>> owned;

		/**
		 * A reference to the fragment instance operating on, referencing the owned fragment or an externally managed one.
		 */
		GridFragment<T,Dims>* base;

		/**
		 * Enables fragments to use the private constructor below.
		 */
		friend class GridFragment<T,Dims>;

		/**
		 * The constructor to be utilized by the fragment to create a facade for an existing fragment.
		 */
		Grid(GridFragment<T,Dims>& base) : base(&base) {}

	public:

		/**
		 * The type of element stored in this grid.
		 */
		using element_type = T;

		/**
		 * The type of coordinate utilized by this type.
		 */
		using coordinate_type = GridPoint<Dims>;

		/**
		 * The type of region utilized by this type.
		 */
		using region_type = GridRegion<Dims>;

		/**
		 * Creates a new Grid covering the given region.
		 */
		Grid(const coordinate_type& size)
			: owned(std::make_unique<GridFragment<T,Dims>>(GridSharedData<Dims>{ size },region_type(0,size))), base(owned.get()) {}

		/**
		 * Disable copy construction.
		 */
		Grid(const Grid&) = delete;

		/**
		 * Enable move construction.
		 */
		Grid(Grid&&) = default;

		/**
		 * Disable copy-assignments.
		 */
		Grid& operator=(const Grid&) = delete;

		/**
		 * Enable move assignments.
		 */
		Grid& operator=(Grid&&) = default;

		/**
		 * Obtains the full size of this grid.
		 */
		coordinate_type size() const {
			return base->getTotalSize();
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
		 * A parallel scan over all elements within this grid, providing
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

