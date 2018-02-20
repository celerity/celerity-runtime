#pragma once

#include <algorithm>
#include <iterator>
#include <vector>

namespace allscale {
namespace utils {

	namespace detail {

		template<typename Iter>
		struct get_size {
			std::size_t operator()(const Iter& a, const Iter& b) {
				return std::distance(a,b);
			}
		};

		template<typename T>
		struct get_size<T*> {
			std::size_t operator()(const T* a, const T* b) {
				return b - a;
			}
		};
	}


	template<typename Iter>
	struct range {
		Iter _begin;
		Iter _end;

		Iter begin() const {
			return _begin;
		}

		Iter end() const {
			return _end;
		}

		bool empty() const {
			return _begin == _end;
		}

		std::size_t size() const {
			return detail::get_size<Iter>()(_begin,_end);
		}

		const typename std::iterator_traits<Iter>::value_type& front() const {
			return *_begin;
		}

		const typename std::iterator_traits<Iter>::value_type& back() const {
			return *(_end - 1);
		}
	};

	template<typename A, typename B>
	bool operator==(const std::vector<A>& data, const range<B>& range) {
		if (data.size() != range.size()) return false;
		return std::equal(data.begin(), data.end(), range.begin());
	}

	template<typename A, typename B>
	bool operator==(const range<B>& range, const std::vector<A>& data) {
		return data == range;
	}

	template<typename A, typename B>
	bool operator!=(const std::vector<A>& data, const range<B>& range) {
		return !(data == range);
	}

	template<typename A, typename B>
	bool operator!=(const range<B>& range, const std::vector<A>& data) {
		return data != range;
	}

} // end namespace utils
} // end namespace allscale
