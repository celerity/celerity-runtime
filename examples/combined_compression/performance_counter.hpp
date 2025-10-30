#pragma once

#include <array>
#include <chrono>
#include <iostream>

using namespace std::chrono_literals;

template <int T>
class counter_stub {
  public:
	template <int I>
	void record() {}

	void reset() {}

	void print() {}
};


template <int N>
class performance_counter {
  public:
	performance_counter() = default;
	~performance_counter() = default;

	explicit performance_counter(std::array<std::chrono::time_point<std::chrono::steady_clock>, N> time_points) : m_time_points(std::move(time_points)) {}

	performance_counter(const performance_counter&) = delete;
	performance_counter(performance_counter&&) = delete;
	performance_counter& operator=(const performance_counter&) = delete;
	performance_counter& operator=(performance_counter&&) = delete;


	template <int I>
	constexpr void record() {
		static_assert(I < N, "I must be smaller than N");
		m_time_points[I] = std::chrono::steady_clock::now();
	}

	constexpr void reset() { m_time_points.fill({}); }

	void print() {
		for(size_t i = 0; i < N - 1; i++) {
			std::cout << "Time difference between " << i << " and " << i + 1 << ": " << get_time_difference(i, i + 1) << "ms" << std::endl;
		}
	}

  private:
	long double get_time_difference(int i, int j) {
		if(N < 1) { return 0; }

		return (m_time_points[j] - m_time_points[i]) / 1.0ms;
	}

	std::array<std::chrono::time_point<std::chrono::steady_clock>, N> m_time_points;
};