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

/**
 * Prints benchmark results in CSV format.
 * All timings are in nanoseconds.
 *
 * Note that unlike in some other reporters, sections that precede a BENCHMARK and are
 * active in the current invocation will be printed as well (not just those directly
 * surrounding a BENCHMARK).
 */
class benchmark_csv_reporter : public Catch::StreamingReporterBase {
  public:
	using StreamingReporterBase::StreamingReporterBase;

	static std::string getDescription() { return "Reporter for benchmarks in CSV format"; }

	void testRunStarting(Catch::TestRunInfo const& testRunInfo) override {
		StreamingReporterBase::testRunStarting(testRunInfo);
		fmt::print(m_stream, "test case,part,benchmark name,samples,iterations,estimated,mean,low mean,high mean,std dev,low std dev,high std dev\n");
	}

	void benchmarkStarting(Catch::BenchmarkInfo const& benchmarkInfo) override {
		StreamingReporterBase::benchmarkStarting(benchmarkInfo);
		fmt::print(m_stream, "{},{},{},{},{},{},", fmt::join(active_sections, " > "), active_part, escape_csv(benchmarkInfo.name), benchmarkInfo.samples,
		    benchmarkInfo.iterations, benchmarkInfo.estimatedDuration);
	}

	void benchmarkEnded(Catch::BenchmarkStats<> const& benchmarkStats) override {
		StreamingReporterBase::benchmarkEnded(benchmarkStats);
		fmt::print(m_stream, "{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n", benchmarkStats.mean.point.count(), benchmarkStats.mean.lower_bound.count(),
		    benchmarkStats.mean.upper_bound.count(), benchmarkStats.standardDeviation.point.count(), benchmarkStats.standardDeviation.lower_bound.count(),
		    benchmarkStats.standardDeviation.upper_bound.count());
		// Flush so we can watch results come in when writing to file
		m_stream.flush();
	}

	// It looks like benchmarkStarting is not called in case a benchmark fails, so we don't need to print the remaining columns either.
	// TODO: Do we want to somehow report this anyway?
	void benchmarkFailed(Catch::StringRef benchmarkName) override { StreamingReporterBase::benchmarkFailed(benchmarkName); }

	void testCasePartialStarting(Catch::TestCaseInfo const& testInfo, uint64_t partNumber) override {
		StreamingReporterBase::testCasePartialStarting(testInfo, partNumber);
		active_part = partNumber;
	}

	void sectionStarting(Catch::SectionInfo const& sectionInfo) override {
		StreamingReporterBase::sectionStarting(sectionInfo);
		// Each test case has an implicit section with the name of the test case itself,
		// so there is no need to capture that separately.
		active_sections.push_back(escape_csv(sectionInfo.name));
	}

	void testCasePartialEnded(Catch::TestCaseStats const& testCaseStats, uint64_t partNumber) override {
		StreamingReporterBase::testCasePartialEnded(testCaseStats, partNumber);
		active_part = 0;
		active_sections.clear();
	}

  private:
	// It is difficult to distinguish test case invocations for different generator values, since
	// as of Catch2 v.3.0.0-preview4 there is unfortunately no real metadata available (for example,
	// it is not possible to get the current value returned for each of the generators).
	// The only piece of information we have is the "part number", which we emit to distinguish
	// different generator values. Note however that sections also increment the part number.
	uint64_t active_part;
	std::vector<std::string> active_sections;
};

CATCH_REGISTER_REPORTER("celerity-benchmark-csv", benchmark_csv_reporter)
