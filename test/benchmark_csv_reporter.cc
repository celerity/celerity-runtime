#include <algorithm>
#include <regex>

#include <catch2/catch_test_case_info.hpp>
#include <catch2/reporters/catch_reporter_registrars.hpp>
#include <catch2/reporters/catch_reporter_streaming_base.hpp>
#include <spdlog/fmt/fmt.h>
#include <spdlog/fmt/ostr.h>

// Escape according to RFC 4180 (w/o line break support)
static std::string escape_csv(const std::string& str) {
	assert(str.find_first_of("\r\n") == std::string::npos);
	// Determine if we need escaping at all (string contains comma or quote).
	if(str.find_first_of("\",") == std::string::npos) return str;

	const std::regex re("\"");
	// Prefix quotes with a another quote, wrap whole string in quotes.
	return fmt::format("\"{}\"", std::regex_replace(str, re, "\"\""));
}

class benchmark_reporter_base : public Catch::StreamingReporterBase {
  public:
	using StreamingReporterBase::StreamingReporterBase;

	// TODO: Do we want to somehow report this?
	void benchmarkFailed(Catch::StringRef benchmarkName) override { StreamingReporterBase::benchmarkFailed(benchmarkName); }

	void sectionStarting(Catch::SectionInfo const& sectionInfo) override {
		StreamingReporterBase::sectionStarting(sectionInfo);
		// Each test case has an implicit section with the name of the test case itself,
		// so there is no need to capture that separately.
		active_sections.push_back(sectionInfo.name);
	}

	void testCasePartialEnded(Catch::TestCaseStats const& testCaseStats, uint64_t partNumber) override {
		StreamingReporterBase::testCasePartialEnded(testCaseStats, partNumber);
		// If the exact same set of sections was active as before, generators must have been involved.
		if(active_sections == previous_active_sections) {
			if(!did_print_generators_warning) {
				fmt::print("WARNING: Using generators will result in indistinguishable test case columns.\n");
				did_print_generators_warning = true;
			}
		}
		std::swap(active_sections, previous_active_sections);
		active_sections.clear();
	}

  protected:
	std::string get_test_case_name() const { return fmt::format("{}", fmt::join(active_sections, " > ")); }

  private:
	std::vector<std::string> active_sections;
	std::vector<std::string> previous_active_sections;
	bool did_print_generators_warning = false;
};

/**
 * Prints benchmark results in CSV format.
 * All timings are in nanoseconds.
 *
 * Note that unlike in some other reporters, sections that precede a BENCHMARK and are
 * active in the current invocation will be printed as well (not just those directly
 * surrounding a BENCHMARK).
 */
class benchmark_csv_reporter : public benchmark_reporter_base {
  public:
	using benchmark_reporter_base::benchmark_reporter_base;

	static std::string getDescription() { return "Reporter for benchmarks in CSV format"; }

	void testRunStarting(Catch::TestRunInfo const& testRunInfo) override {
		StreamingReporterBase::testRunStarting(testRunInfo);
		fmt::print(m_stream, "test case,benchmark name,samples,iterations,estimated,mean,low mean,high mean,std dev,low std dev,high std dev,raw\n");
	}

	void benchmarkEnded(Catch::BenchmarkStats<> const& benchmarkStats) override {
		StreamingReporterBase::benchmarkEnded(benchmarkStats);
		auto& info = benchmarkStats.info;
		fmt::print(m_stream, "{},{},{},{},{},", escape_csv(get_test_case_name()), escape_csv(info.name), info.samples, info.iterations, info.estimatedDuration);
		fmt::print(m_stream, "{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},", benchmarkStats.mean.point.count(), benchmarkStats.mean.lower_bound.count(),
		    benchmarkStats.mean.upper_bound.count(), benchmarkStats.standardDeviation.point.count(), benchmarkStats.standardDeviation.lower_bound.count(),
		    benchmarkStats.standardDeviation.upper_bound.count());
		// Finally print all raw values for custom analyses (as quoted comma-separated values).
		std::vector<double> raw;
		raw.reserve(benchmarkStats.samples.size());
		std::transform(benchmarkStats.samples.cbegin(), benchmarkStats.samples.cend(), std::back_inserter(raw), [](auto& d) { return d.count(); });
		fmt::print(m_stream, "\"{:.4f}\"\n", fmt::join(raw, ","));
		// Flush so we can watch results come in when writing to file
		m_stream.flush();
	}
};

CATCH_REGISTER_REPORTER("celerity-benchmark-csv", benchmark_csv_reporter)
