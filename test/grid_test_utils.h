#include "grid.h"
#include "test_utils.h"

namespace celerity::test_utils {

struct partition_vector_order {
	template <int Dims>
	bool operator()(const detail::box_vector<Dims>& lhs, const detail::box_vector<Dims>& rhs) {
		if(lhs.size() < rhs.size()) return true;
		if(lhs.size() > rhs.size()) return false;
		constexpr detail::box_coordinate_order box_order;
		for(size_t i = 0; i < lhs.size(); ++i) {
			if(box_order(lhs[i], rhs[i])) return true;
			if(box_order(rhs[i], lhs[i])) return false;
		}
		return false;
	}
};

void render_boxes(const detail::box_vector<2>& boxes, const std::string_view suffix = "region");

}
