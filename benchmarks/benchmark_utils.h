#pragma once

#include <args.hxx>

#include <chrono>
#include <cmath>
#include <iomanip>
#include <memory>
#include <numeric>
#include <vector>

template <class Duration>
inline double toMsDouble(Duration d) {
	std::chrono::microseconds us = std::chrono::duration_cast<std::chrono::microseconds>(d);
	return us.count() / 1000.0;
}

class Benchmark {
	static inline std::unique_ptr<Benchmark> instance;

	// timing
	std::chrono::steady_clock clock;
	using TimePoint = std::chrono::steady_clock::time_point;
	TimePoint programStart = clock.now();
	std::vector<double> runTimes;

	// command line argument handling
	std::unique_ptr<args::Group> argGroup;
	std::unique_ptr<args::ValueFlag<int>> numRepeats;

  public:
	static Benchmark& get() {
		if(!instance) instance = std::make_unique<Benchmark>();
		return *instance;
	}

	void addArguments(args::ArgumentParser& parser) {
		argGroup.reset(new args::Group(parser, "benchmarking arguments", args::Group::Validators::DontCare, args::Options::Global));
		numRepeats.reset(new args::ValueFlag<int>(*argGroup, "repeats", "The number of times to repeat the measurement", {'r'}, 5));
	}

	template <typename Functor>
	void run(Functor fun) {
		int repeats = numRepeats ? numRepeats->Get() : 5;
		for(int i = 0; i < repeats; ++i) {
			TimePoint currentStart = clock.now();
			fun();
			TimePoint currentEnd = clock.now();
			runTimes.emplace_back(toMsDouble(currentEnd - currentStart));
		}
	}

	~Benchmark() {
		if(runTimes.size() <= 0) {
			std::cout << "No benchmarks performed.\n";
			return;
		}

		double sum = std::accumulate(runTimes.begin(), runTimes.end(), 0.0);
		double mean = sum / runTimes.size();

		double stddev = 0.0;
		for(double t : runTimes) {
			stddev += pow(t - mean, 2.0);
		}
		stddev = sqrt(stddev / runTimes.size());

		std::cout << std::fixed << std::setprecision(3)                                                         //
		          << "Benchmark results (times in milliseconds):\n"                                             //
		          << "Total execution time: " << std::setw(8) << toMsDouble(clock.now() - programStart) << "\n" //
		          << "Mean Benchmark time:  " << std::setw(8) << mean << "\n"                                   //
		          << "Standard Deviation:   " << std::setw(8) << stddev << "\n"                                 //
				  << "Individual times:\n\t";
		for(double t : runTimes) {
			std::cout << t << ", ";			
		}
		std::cout << "\n";
	}
};
