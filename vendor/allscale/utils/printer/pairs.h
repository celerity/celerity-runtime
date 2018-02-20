#pragma once

#include <utility>
#include <iostream>

namespace std {

	template<typename A,typename B>
	ostream& operator<<(ostream& out, const pair<A,B>& data) {
		return out << "[" << data.first << "," << data.second << "]";
	}

}
