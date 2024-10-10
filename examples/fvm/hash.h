#pragma once

#include <cstddef>
#include <cstdint>

// FNV-1a hash, 64 bit length
class hash {
  public:
	using digest = uint64_t;

	template <typename T>
	void add(const T& value) {
		const unsigned char* bytes = reinterpret_cast<const unsigned char*>(&value);
		for(size_t i = 0; i < sizeof(T); ++i) {
			m_d = (m_d ^ bytes[i]) * 0x100000001b3ull;
		}
	}

	digest get() const { return m_d; }

  private:
	digest m_d = 0xcbf29ce484222325ull;
};
