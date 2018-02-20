#pragma once

#include <array>
#include <iostream>

#include "allscale/utils/printer/join.h"

namespace std {

	template<typename E, std::size_t N>
	ostream& operator<<(ostream& out, const array<E,N>& data) {
		return out << "[" << allscale::utils::join(",", data) << "]";
	}

}
