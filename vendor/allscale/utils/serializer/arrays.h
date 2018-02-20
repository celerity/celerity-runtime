#pragma once

#ifdef ALLSCALE_WITH_HPX
	#include <hpx/runtime/serialization/array.hpp>
#endif

#include "allscale/utils/serializer.h"

#include <array>

namespace allscale {
namespace utils {


	namespace detail {

		template<typename T, std::size_t len, std::size_t pos>
		struct array_load_helper {

			template<typename ... Args>
			std::array<T,len> operator()(ArchiveReader& reader, Args&& ... args) {
				return array_load_helper<T,len,pos-1>()(reader,args...,reader.read<T>());
			}
		};

		template<typename T, std::size_t len>
		struct array_load_helper<T,len,0> {

			template<typename ... Args>
			std::array<T,len> operator()(ArchiveReader&, Args&& ... args) {
				return std::array<T,len>{
					{ args... }
				};
			}

		};

	}


	/**
	 * Add support for serializing / de-serializing arrays.
	 */
	template<typename T, std::size_t size>
	struct serializer<std::array<T,size>,typename std::enable_if<is_serializable<T>::value,void>::type> {

		static std::array<T,size> load(ArchiveReader& reader) {
			// support loading of array for elements without default constructor
			return detail::array_load_helper<T,size,size>()(reader);
		}
		static void store(ArchiveWriter& writer, const std::array<T,size>& value) {
			for(const auto& cur : value) {
				writer.write(cur);
			}
		}
	};

} // end namespace utils
} // end namespace allscale

