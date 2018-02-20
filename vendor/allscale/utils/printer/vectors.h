#pragma once

#include <vector>
#include <iostream>

#include "allscale/utils/printer/join.h"

namespace std {

	template<typename E,typename A>
	ostream& operator<<(ostream& out, const vector<E,A>& data) {
		return out << "[" << allscale::utils::join(",", data) << "]";
	}

}
