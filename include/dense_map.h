#pragma once

#include <cassert>
#include <cstdlib>
#include <vector>

namespace celerity::detail {

/// Like a simple std::unordered_map, but implemented by indexing into a vector with the integral key type.
template <typename KeyId, typename Value>
class dense_map : private std::vector<Value> {
  private:
	using vector = std::vector<Value>;

  public:
	dense_map() = default;

	explicit dense_map(const size_t size) : vector(size) {}

	explicit dense_map(const size_t size, const Value& init) : vector(size, init) {}

	template <typename InputIterator, typename std::iterator_traits<InputIterator>::pointer Enable = nullptr>
	explicit dense_map(const InputIterator begin, const InputIterator end) : vector(begin, end) {}

	using vector::begin, vector::end, vector::cbegin, vector::cend, vector::empty, vector::size, vector::resize;

	Value& operator[](const KeyId key) {
		assert(key < size());
		return vector::operator[](static_cast<size_t>(key));
	}

	const Value& operator[](const KeyId key) const {
		assert(key < size());
		return vector::operator[](static_cast<size_t>(key));
	}
};

} // namespace celerity::detail
