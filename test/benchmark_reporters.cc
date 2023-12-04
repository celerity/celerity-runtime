#include <algorithm>
#include <chrono>
#include <ctime>
#include <limits>
#include <numeric>
#include <ostream>
#include <regex>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <catch2/catch_test_case_info.hpp>
#include <catch2/reporters/catch_reporter_registrars.hpp>
#include <catch2/reporters/catch_reporter_streaming_base.hpp>

#include <fmt/chrono.h>
#include <fmt/format.h>
#include <fmt/ostream.h>


// Escape according to RFC 4180 (w/o line break support)
static std::string escape_csv(const std::string& str) {
	assert(str.find_first_of("\r\n") == std::string::npos);
	// Determine if we need escaping at all (string contains comma or quote).
	if(str.find_first_of("\",") == std::string::npos) return str;

	const std::regex re("\"");
	// Prefix quotes with a another quote, wrap whole string in quotes.
	return fmt::format("\"{}\"", std::regex_replace(str, re, "\"\""));
}

static std::string escape_md_partial(const std::string& str) {
	// We only escape characters that
	// - are likely to occur in a test case / benchmark name
	// - have inline semantics (e.g. we don't expect a test case to start with a '-' or '>')
	// - are likely not intended for formatting (e.g. using backticks to denote types/code is fine)
	// - have a meaning in popular Markdown implementations (e.g. {} is reserved but not used)
	if(str.find_first_of("*_|[]\\") == std::string::npos) return str;
	const std::regex re(R"(([*_|[\]\\]))");
	return std::regex_replace(str, re, "\\$1");
}

class benchmark_reporter_base : public Catch::StreamingReporterBase {
  public:
	using StreamingReporterBase::StreamingReporterBase;

	void benchmarkPreparing(Catch::StringRef benchmark_name) override {
		StreamingReporterBase::benchmarkPreparing(benchmark_name);
		m_test_case_benchmark_combinations.insert(get_test_case_name() + ": " + benchmark_name);
	}

	// TODO: Do we want to somehow report this?
	void benchmarkFailed(Catch::StringRef benchmark_name) override { StreamingReporterBase::benchmarkFailed(benchmark_name); }

	void sectionStarting(const Catch::SectionInfo& section_info) override {
		StreamingReporterBase::sectionStarting(section_info);
		// Each test case has an implicit section with the name of the test case itself,
		// so there is no need to capture that separately.
		m_active_sections.push_back(section_info.name);
	}

	void testCasePartialEnded(const Catch::TestCaseStats& test_case_stats, uint64_t part_number) override {
		StreamingReporterBase::testCasePartialEnded(test_case_stats, part_number);
		m_active_sections.clear();
	}

	void testRunEnded(const Catch::TestRunStats& test_run_stats) override {
		StreamingReporterBase::testRunEnded(test_run_stats);
		bool warning_printed = false;
		for(auto it = m_test_case_benchmark_combinations.cbegin(); it != m_test_case_benchmark_combinations.cend(); ++it) {
			const auto id = *it;
			const auto count = m_test_case_benchmark_combinations.count(id);
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
	[[nodiscard]] std::string get_test_case_name() const { return fmt::format("{}", fmt::join(m_active_sections, " > ")); }

  private:
	std::vector<std::string> m_active_sections;
	std::unordered_multiset<std::string> m_test_case_benchmark_combinations;
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

	static std::string getDescription() { return "Reporter for benchmarks in CSV format"; } // NOLINT(readability-identifier-naming)

	void testRunStarting(const Catch::TestRunInfo& test_run_info) override {
		benchmark_reporter_base::testRunStarting(test_run_info);
		fmt::print(m_stream, "test case,benchmark name,samples,iterations,estimated,mean,low mean,high mean,std dev,low std dev,high std dev,tags,raw\n");
	}

	void benchmarkEnded(const Catch::BenchmarkStats<>& benchmark_stats) override {
		benchmark_reporter_base::benchmarkEnded(benchmark_stats);
		auto& info = benchmark_stats.info;
		fmt::print(m_stream, "{},{},{},{},{},", escape_csv(get_test_case_name()), escape_csv(info.name), info.samples, info.iterations, info.estimatedDuration);
		fmt::print(m_stream, "{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},", benchmark_stats.mean.point.count(), benchmark_stats.mean.lower_bound.count(),
		    benchmark_stats.mean.upper_bound.count(), benchmark_stats.standardDeviation.point.count(), benchmark_stats.standardDeviation.lower_bound.count(),
		    benchmark_stats.standardDeviation.upper_bound.count());
		// Print the benchmark tags for tool-based processing and categorization (as quoted comma-separated values)
		const auto& tci = currentTestCaseInfo;
		std::vector<std::string> tags;
		std::transform(tci->tags.cbegin(), tci->tags.cend(), std::back_inserter(tags), [](const Catch::Tag& t) { return std::string(t.original); });
		fmt::print(m_stream, "\"{}\",", fmt::join(tags, ","));
		// Finally print all raw values for custom analyses (as quoted comma-separated values).
		std::vector<double> raw;
		raw.reserve(benchmark_stats.samples.size());
		std::transform(benchmark_stats.samples.cbegin(), benchmark_stats.samples.cend(), std::back_inserter(raw), [](auto& d) { return d.count(); });
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
		std::transform(columns.cbegin(), columns.end(), std::back_inserter(this->m_columns), [](const auto& c) {
			const auto [name, alignment] = c;
			return column{name, std::max(column::min_width, name.length()), alignment};
		});
	}

	void add_row(std::vector<std::string> cells) {
		if(cells.size() != m_columns.size()) { throw std::runtime_error("Column mismatch"); }
		for(size_t i = 0; i < m_columns.size(); ++i) {
			m_columns[i].width = std::max(m_columns[i].width, cells[i].length());
		}
		m_rows.push_back(std::move(cells));
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
		for(const auto& [header, width, a] : m_columns) {
			fmt::print(os, align_fmt(" {: A{}} |", a), header, width);
		}
		fmt::print(os, "\n");

		// Print separators
		fmt::print(os, "|");
		for(const auto& [_, width, a] : m_columns) {
			const char align_left = a != align::right ? ':' : '-';
			const char align_right = a != align::left ? ':' : '-';
			fmt::print(os, align_fmt(" {}{:-A{}}{} |", a), align_left, "", width - 2, align_right);
		}
		fmt::print(os, "\n");

		// Print rows
		for(const auto& r : m_rows) {
			fmt::print(os, "|");
			for(size_t i = 0; i < r.size(); ++i) {
				fmt::print(os, align_fmt(" {: A{}} |", m_columns[i].alignment), r[i], m_columns[i].width);
			}
			fmt::print(os, "\n");
		}
	}

  private:
	std::vector<column> m_columns;
	std::vector<std::vector<std::string>> m_rows;
};

class benchmark_md_reporter : public benchmark_reporter_base {
  public:
	using benchmark_reporter_base::benchmark_reporter_base;

	static std::string getDescription() { return "Generates a Markdown report for benchmark results"; } // NOLINT(readability-identifier-naming)

	void testRunStarting(const Catch::TestRunInfo& test_run_info) override {
		benchmark_reporter_base::testRunStarting(test_run_info);

		fmt::print(m_stream, "# Benchmark Results\n\n");

		markdown_table_printer meta_printer({{"Metadata", align::left}, {"", align::left}});
		const auto now_gmt = fmt::gmtime(std::time(nullptr));
		// TODO: It would be cool to also have celerity version, hostname, argv, ...
		meta_printer.add_row({"Created", fmt::format("{:%FT%TZ}", now_gmt)});

		meta_printer.print(m_stream);
	}

	void testRunEnded(const Catch::TestRunStats& test_run_stats) override {
		benchmark_reporter_base::testRunEnded(test_run_stats);
		fmt::print(m_stream, "\n\n");
		m_results_printer.print(m_stream);
		fmt::print(m_stream, "\nAll numbers are in nanoseconds.\n");
	}

	void benchmarkEnded(const Catch::BenchmarkStats<>& benchmark_stats) override {
		benchmark_reporter_base::benchmarkEnded(benchmark_stats);

		const auto min = std::reduce(benchmark_stats.samples.cbegin(), benchmark_stats.samples.cend(),
		    std::chrono::duration<double, std::nano>(std::numeric_limits<double>::max()), [](const auto& a, const auto& b) { return std::min(a, b); });

		m_results_printer.add_row({fmt::format("{}", escape_md_partial(get_test_case_name())), // Test case
		    escape_md_partial(benchmark_stats.info.name),                                      // Benchmark name
		    format_result(min),                                                                // Min
		    format_result(benchmark_stats.mean.point),                                         // Mean
		    format_result(benchmark_stats.standardDeviation.point)});                          // Std dev
	}

  private:
	markdown_table_printer m_results_printer{
	    {{"Test case", align::left}, {"Benchmark name", align::left}, {"Min", align::right}, {"Mean", align::right}, {"Std dev", align::right}}};

	// Format numbers with ' as thousands separator and . as decimal separator.
	static std::string format_result(std::chrono::duration<double, std::nano> duration) {
		const auto ns = duration.count();
		// Manually insert thousands separators into integral part to avoid relying on non-C locale
		auto str = fmt::format("{:.2f}", ns);
		const size_t first_separator = 3 /* integral digits */ + 1 /* dot */ + 2 /* decimal digits */;
		const size_t separator_step = 3 /* integral digits */ + 1 /* previous thousands separator */;
		for(size_t separator = first_separator; separator < str.length() - std::signbit(ns); separator += separator_step) {
			str.insert(str.length() - separator, 1, '\'');
		}
		return str;
	}
};

CATCH_REGISTER_REPORTER("celerity-benchmark-md", benchmark_md_reporter)
