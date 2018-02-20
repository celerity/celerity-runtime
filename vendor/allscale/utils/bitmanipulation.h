#pragma once

#ifdef _MSC_VER
	#include <intrin.h>
#endif

namespace allscale {
namespace utils {

	/**
	 * A wrapper function for counting leading zeros
	 */
	inline int countLeadingZeros(unsigned value) {
		#ifdef _MSC_VER
			unsigned long retVal = 0;
			if(_BitScanReverse(&retVal, value))
				return 31-retVal;
			// all zeros is undefined behavior, we simply return 32
			return 32;
		#else
			return __builtin_clz(value);
		#endif
	}

	/**
	* A wrapper function for counting trailing zeros
	*/
	inline int countTrailingZeros(unsigned value) {
		#ifdef _MSC_VER
			unsigned long retVal = 0;
			if(_BitScanForward(&retVal, value))
				return retVal;
			// all zeros is undefined behavior, we simply return 32
			return 32;
		#else
			return __builtin_ctz(value);
		#endif
	}
	
	/**
	* A wrapper function for counting 1-bits
	*/
	inline int countOnes(unsigned value) {
		#ifdef _MSC_VER
			return __popcnt(value);
		#else
			return __builtin_popcount(value);
		#endif
	}

} // end namespace utils
} // end namespace allscale
