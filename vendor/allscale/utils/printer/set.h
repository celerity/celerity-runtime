#pragma once

#include <set>
#include <iostream>

#include "allscale/utils/printer/join.h"

namespace std {

	template<typename E, typename C, typename A>
	ostream& operator<<(ostream& out, const set<E,C,A>& data) {
		return out << "{" << allscale::utils::join(",", data) << "}";
	}

}
