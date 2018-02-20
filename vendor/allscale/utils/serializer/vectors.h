#pragma once

#ifdef ALLSCALE_WITH_HPX
	#include "allscale/utils/serializer.h"
#endif

#include <vector>

#include "allscale/utils/serializer/arrays.h"

namespace allscale {
namespace utils {

	/**
	 * Add support for serializing / de-serializing std::vectors.
	 */
	template<typename T, typename Allocator>
	struct serializer<std::vector<T,Allocator>,typename std::enable_if<is_serializable<T>::value,void>::type> {

		static std::vector<T,Allocator> load(ArchiveReader& reader) {

			// create the result
			std::vector<T,Allocator> res;

			// load the size
			auto size = reader.read<std::size_t>();

			// make some space
			res.reserve(size);

			// load the elements
			for(std::size_t i=0; i<size; i++) {
				res.emplace_back(reader.read<T>());
			}

			// done
			return res;
		}
		static void store(ArchiveWriter& writer, const std::vector<T,Allocator>& value) {

			// start with the size
			writer.write(value.size());

			// followed by all the elements
			for(const auto& cur : value) {
				writer.write(cur);
			}
		}
	};

} // end namespace utils
} // end namespace allscale
