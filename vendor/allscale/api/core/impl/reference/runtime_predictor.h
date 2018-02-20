#pragma once

#include <array>
#include <cstdint>
#include <cmath>
#include <thread>
#include <chrono>
#include <ostream>

#if defined _MSC_VER
#include <intrin.h>
#elif defined (__ppc64__) || defined (_ARCH_PPC64) || defined(__powerpc__) || defined(__ppc__)
static __inline__ unsigned long long __rdtsc(void)
{
  int64_t tb;
  asm("mfspr %0, 268" : "=r"(tb));
  return tb;
}
#else
#include <x86intrin.h>
#endif

namespace allscale {
namespace api {
namespace core {
namespace impl {
namespace reference {

	/**
	 * A type to represent a type safe cycle count.
	 */
	class CycleCount {

		using time_t = unsigned long long;

		time_t value;

	public:

		CycleCount() {}

		CycleCount(time_t value) : value(value) {}

		bool operator==(const CycleCount& other) const {
			return value == other.value;
		}

		bool operator!=(const CycleCount& other) const {
			return value != other.value;
		}

		bool operator<(const CycleCount& other) const {
			return value < other.value;
		}

		bool operator>(const CycleCount& other) const {
			return value > other.value;
		}

		CycleCount operator+(const CycleCount& other) const {
			return value + other.value;
		}

		CycleCount operator-(const CycleCount& other) const {
			return value - other.value;
		}

		time_t count() const {
			return value;
		}

		static CycleCount zero() {
			return 0;
		}

		static CycleCount max() {
			return std::numeric_limits<time_t>::max();
		}

	};

	inline CycleCount operator*(long unsigned int f, const CycleCount& count) {
		return f * count.count();
	}

	inline CycleCount operator*(const CycleCount& count, long unsigned int f) {
		return count.count() * f;
	}

	inline CycleCount operator/(const CycleCount& count, long unsigned int div) {
		return count.count() / div;
	}

	/**
	 * A cycle clock for the time prediction.
	 */
	struct CycleClock {

		using time_point = CycleCount;
		using duration = CycleCount;

		static time_point now() {
			return __rdtsc();
		}

	};


	/**
	 * A utility to estimate the execution time of tasks on different
	 * levels of task-decomposition steps.
	 */
	class RuntimePredictor {

	public:

		using clock = CycleClock;

		using duration = clock::duration;

		enum { MAX_LEVELS = 100 };

	private:

		/**
		 * The number of samples recorded per task level.
		 */
		std::array<std::size_t,MAX_LEVELS> samples;

		/**
		 * The current estimates of execution times of tasks.
		 */
		std::array<duration,MAX_LEVELS> times;

	public:

		RuntimePredictor(unsigned numWorkers = std::thread::hardware_concurrency()) {
			// reset number of collected samples
			samples.fill(0);

			// initialize time estimates
			times.fill(duration::zero());

			// initialize execution times up to a given level
			for(int i=0; i<std::log2(numWorkers) + 4; ++i) {
				times[i] = duration::max();
			}
		}

		/**
		 * Obtain a prediction of a given level.
		 */
		duration predictTime(std::size_t level) const {
			if (level >= MAX_LEVELS) return duration::zero();
			return times[level];
		}

		/**
		 * Update the predictions for a level.
		 */
		void registerTime(std::size_t level, const duration& time) {

			// update matching level
			updateTime(level,time);

			// update higher levels (with reduced weight)
			auto smallerTime = time / 2;
			auto largerTime = time * 2;
			for(std::size_t d = 1; d < 5; d++) {

				// update higher element
				if (d <= level) {
					updateTime(level-d,largerTime);
				}

				// update smaller element
				if (level+d < MAX_LEVELS) {
					updateTime(level+d,smallerTime);
				}

				// update parameters
				smallerTime = smallerTime / 2;
				largerTime = largerTime * 2;
			}

		}

		/**
		 * Enable the printing of the predictor state.
		 */
		friend std::ostream& operator<<(std::ostream& out, const RuntimePredictor& pred) {
			out << "Predictions:\n";
			for(int i = 0; i<MAX_LEVELS; i++) {
				auto us = pred.times[i].count();
				out << "\t" << i << ": " << us << "\n";
				if (us == 0) return out;
			}
			return out;
		}

	private:

		void updateTime(std::size_t level, const duration& time) {

			// update estimate of time of a task on this level
			long unsigned N = (long unsigned)samples[level];
			times[level] = (N * times[level] + time) / (N+1);

			// update sample count
			++samples[level];
		}

	};


	/**
	 * A global singleton dispatcher associating to each task type
	 * a thread local runtime predictor.
	 */
	template<typename TaskType>
	inline RuntimePredictor& getRuntimePredictor() {
		static thread_local RuntimePredictor predictor = RuntimePredictor();
		return predictor;
	}


} // end namespace reference
} // end namespace impl
} // end namespace core
} // end namespace api
} // end namespace allscale
