#include <algorithm>
#include <chrono>
#include <ctime>
#include <locale>
#include <ostream>
#include <regex>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <catch2/catch_test_case_info.hpp>
#include <catch2/reporters/catch_reporter_registrars.hpp>
#include <catch2/reporters/catch_reporter_streaming_base.hpp>
#include <spdlog/fmt/chrono.h>
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

	void benchmarkPreparing(Catch::StringRef benchmarkName) override {
		StreamingReporterBase::benchmarkPreparing(benchmarkName);
		test_case_benchmark_combinations.insert(get_test_case_name() + ": " + benchmarkName);
	}

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
		active_sections.clear();
	}

	void testRunEnded(Catch::TestRunStats const& testRunStats) override {
		StreamingReporterBase::testRunEnded(testRunStats);
		bool warning_printed = false;
		for(auto it = test_case_benchmark_combinations.cbegin(); it != test_case_benchmark_combinations.cend(); ++it) {
			const auto id = *it;
			const auto count = test_case_benchmark_combinations.count(id);
			if(count > 1) {
				if(!warning_printed) {
					fmt::print(stderr, "WARNING: Using generators will result in indistinguishable test cases. The following cases are ambiguous:\n");
					warning_printed = true;
				}
				fmt::print(stderr, "\t{}\n", id);
			}
			// Same values are guaranteed to be contiguous; skip ahead.
			std::advance(it, count - 1);
		}
		if(warning_printed) { fmt::print(stderr, "Consider naming benchmarks dynamically to avoid this.\n"); }
	}

  protected:
	[[nodiscard]] std::string get_test_case_name() const { return fmt::format("{}", fmt::join(active_sections, " > ")); }

  private:
	std::vector<std::string> active_sections;
	std::unordered_multiset<std::string> test_case_benchmark_combinations;
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
		benchmark_reporter_base::testRunStarting(testRunInfo);
		fmt::print(m_stream, "test case,benchmark name,samples,iterations,estimated,mean,low mean,high mean,std dev,low std dev,high std dev,raw\n");
	}

	void benchmarkEnded(Catch::BenchmarkStats<> const& benchmarkStats) override {
		benchmark_reporter_base::benchmarkEnded(benchmarkStats);
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

enum class align : char { left = '<', right = '>', center = '^' };

class markdown_table_printer {
	struct column {
		static constexpr size_t min_width = 3;
		std::string header;
		size_t width;
		align alignment;
	};

  public:
	markdown_table_printer(const std::vector<std::pair<std::string, align>>& columns) {
		std::transform(columns.cbegin(), columns.end(), std::back_inserter(this->columns), [](const auto& c) {
			const auto [name, alignment] = c;
			return column{name, std::max(column::min_width, name.length()), alignment};
		});
	}

	void add_row(std::vector<std::string> cells) {
		if(cells.size() != columns.size()) { throw std::runtime_error("Column mismatch"); }
		for(size_t i = 0; i < columns.size(); ++i) {
			columns[i].width = std::max(columns[i].width, cells[i].length());
		}
		rows.push_back(std::move(cells));
	}

	void print(std::ostream& os) const {
		// fmt does not allow to set alignment dynamically, so we need a helper function.
		// Replaces 'A' in fmt_str with '<','>' or '^'.
		constexpr auto align_fmt = [](std::string fmt_str, align a) {
			std::replace(fmt_str.begin(), fmt_str.end(), 'A', static_cast<char>(a));
			return fmt_str;
		};

		// Print column headers
		fmt::print(os, "|");
		for(const auto& [header, width, a] : columns) {
			fmt::print(os, align_fmt(" {: A{}} |", a), header, width);
		}
		fmt::print(os, "\n");

		// Print separators
		fmt::print(os, "|");
		for(const auto& [_, width, a] : columns) {
			const char align_left = a != align::right ? ':' : '-';
			const char align_right = a != align::left ? ':' : '-';
			fmt::print(os, align_fmt(" {}{:-A{}}{} |", a), align_left, "", width - 2, align_right);
		}
		fmt::print(os, "\n");

		// Print rows
		for(const auto& r : rows) {
			fmt::print(os, "|");
			for(size_t i = 0; i < r.size(); ++i) {
				fmt::print(os, align_fmt(" {: A{}} |", columns[i].alignment), r[i], columns[i].width);
			}
			fmt::print(os, "\n");
		}
	}

  private:
	std::vector<column> columns;
	std::vector<std::vector<std::string>> rows;
};

class benchmark_md_reporter : public benchmark_reporter_base {
  public:
	using benchmark_reporter_base::benchmark_reporter_base;

	static std::string getDescription() { return "Generates a Markdown report for benchmark results"; }

	void testRunStarting(Catch::TestRunInfo const& testRunInfo) override {
		benchmark_reporter_base::testRunStarting(testRunInfo);

		fmt::print(m_stream, "# Benchmark Results\n\n");

		markdown_table_printer meta_printer({{"Metadata", align::left}, {"", align::left}});
		const auto now_gmt = fmt::gmtime(std::time(nullptr));
		// TODO: It would be cool to also have celerity version, hostname, argv, ...
		meta_printer.add_row({"Created", fmt::format("{:%FT%TZ}", now_gmt)});

		meta_printer.print(m_stream);
	}

	void testRunEnded(Catch::TestRunStats const& testRunStats) override {
		benchmark_reporter_base::testRunEnded(testRunStats);
		fmt::print(m_stream, "\n\n");
		results_printer.print(m_stream);
		fmt::print(m_stream, "\nAll numbers are in nanoseconds.\n");
	}

	void benchmarkEnded(Catch::BenchmarkStats<> const& benchmarkStats) override {
		benchmark_reporter_base::benchmarkEnded(benchmarkStats);

		// Format numbers with ' as thousand separator and . as decimal separator.
		constexpr auto format_result = [](std::chrono::duration<double, std::nano> ns) {
			// fmt can only do thousands separators based on locale, so we need to do a character replacement afterwards.
			// Also it only works on integral types, so we need to format the fractional part separately.
			double integral;
			const double fractional = std::modf(ns.count(), &integral);
			auto integral_formatted = fmt::format(std::locale("en_US.UTF-8"), "{:L}", static_cast<int64_t>(integral));
			std::replace(integral_formatted.begin(), integral_formatted.end(), ',', '\'');
			const auto fractional_formatted = fmt::format("{:.2f}", fractional).substr(2);
			return fmt::format("{}.{}", integral_formatted, fractional_formatted);
		};

		results_printer.add_row({fmt::format("{}", get_test_case_name()), // Test case
		    benchmarkStats.info.name,                                     // Benchmark name
		    format_result(benchmarkStats.mean.point),                     // Mean
		    format_result(benchmarkStats.standardDeviation.point)});      // Std dev
	}

  private:
	markdown_table_printer results_printer{{{"Test case", align::left}, {"Benchmark name", align::left}, {"Mean", align::right}, {"Std dev", align::right}}};
};

CATCH_REGISTER_REPORTER("celerity-benchmark-md", benchmark_md_reporter)
