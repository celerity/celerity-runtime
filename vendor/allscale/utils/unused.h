#pragma once

/**
 * This header defines a macro to mark knowingly unused variables as being
 * unused, so that the compiler is not issuing warnings about those.
 */

#ifdef __GNUC__
	#define __allscale_unused __attribute__((unused))
#else
	#define __allscale_unused
#endif
