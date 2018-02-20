#pragma once

namespace allscale {
namespace utils {

	/**
	 * A utility for interpreting raw buffers.
	 */
	class RawBuffer {

		char* cur;

	public:

		/**
		 * Creates a buffer based on the given memory location.
		 */
		template<typename T>
		RawBuffer(T* base) : cur(reinterpret_cast<char*>(base)) {}

		/**
		 * Consumes an element of type T from the underlying buffer.
		 */
		template<typename T>
		T& consume() {
			return consumeArray<T>(1)[0];
		}

		/**
		 * Consumes an array of elements of type T form the underlying buffer.
		 */
		template<typename T>
		T* consumeArray(std::size_t numElements) {

			// check that the given type allows this kind of operations
			static_assert(
				std::is_trivially_copy_assignable<T>::value ||
				std::is_trivially_move_assignable<T>::value,
				"Invalid reinterpretation of raw data!"
			);

			// 'parse' initial elements
			auto res = reinterpret_cast<T*>(cur);
			// progress position
			cur += sizeof(T) * numElements;
			// return result
			return res;
		}

	};

} // end namespace utils
} // end namespace allscale
