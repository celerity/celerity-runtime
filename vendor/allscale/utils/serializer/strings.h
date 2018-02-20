#pragma once

#ifdef ALLSCALE_WITH_HPX
#include <hpx/runtime/serialization/string.hpp>
#endif

#include "allscale/utils/serializer.h"

#include <string>

namespace allscale {
namespace utils {

	/**
	 * Add support for serializing / de-serializing strings.
	 */
	template<>
	struct serializer<std::string> {

		static std::string load(ArchiveReader& reader) {
			auto size = reader.read<std::size_t>();
			std::string res;
			res.resize(size);
			reader.read(&res[0],size);
			return res;
		}
		static void store(ArchiveWriter& writer, const std::string& value) {
			writer.write<std::size_t>(value.size());
			writer.write(&value[0],value.size());
		}
	};

} // end namespace utils
} // end namespace allscale
