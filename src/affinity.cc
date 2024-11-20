// non-platform-specific code for thread pinning

#include "affinity.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include <libenvpp/env.hpp>

namespace celerity::detail::thread_pinning {

std::string thread_type_to_string(const thread_type t_type) {
	switch(t_type) {
	case thread_type::application: return "application";
	case thread_type::scheduler: return "scheduler";
	case thread_type::executor: return "executor";
	default: break;
	}
	if(t_type >= thread_type::first_device_submitter && t_type < thread_type::first_host_queue) {
		return fmt::format("device_submitter_{}", t_type - thread_type::first_device_submitter);
	}
	if(t_type >= thread_type::first_host_queue && t_type < thread_type::max) { return fmt::format("host_queue_{}", t_type - thread_type::first_host_queue); }
	return fmt::format("unknown({})", static_cast<uint32_t>(t_type));
}

namespace {
	// When we no longer need to support compilers without a working std::views::split, get rid of this function
	std::vector<std::string> split(const std::string_view str, const char delim) {
		std::vector<std::string> result;
		size_t start = 0;
		size_t end = str.find(delim);
		while(end != std::string::npos) {
			result.push_back(std::string(str.substr(start, end - start)));
			start = end + 1;
			end = str.find(delim, start);
		}
		if(start < str.size()) result.push_back(std::string(str.substr(start)));
		return result;
	}
} // namespace

environment_configuration parse_validate_env(const std::string_view str) {
	using namespace std::string_view_literals;
	constexpr const char* error_msg =
	    "Cannot parse CELERITY_THREAD_PINNING setting, needs to be either 'auto', 'from:#', comma-separated core list, or bool: {}";

	if(str.empty()) return {};

	// "auto" case
	constexpr uint32_t auto_start_from_core = 1;
	if(str == "auto") { return {true, auto_start_from_core, {}}; }

	// "from:" case
	constexpr auto from_prefix = "from:"sv;
	if(str.starts_with(from_prefix)) {
		try {
			const auto from = env::default_parser<uint32_t>{}(std::string(str.substr(from_prefix.size())));
			return {true, from, {}};
		} catch(const env::parser_error& e) { throw env::parser_error{fmt::format(error_msg, e.what())}; }
	}

	// core list case
	if(str.find(',') != std::string::npos) {
		std::vector<uint32_t> core_ids;
		for(auto cs : split(str, ',')) {
			try {
				core_ids.push_back(env::default_parser<uint32_t>{}(std::string(cs.begin(), cs.end())));
			} catch(const env::parser_error& e) { throw env::parser_error{fmt::format(error_msg, e.what())}; }
		}
		return {true, 0, core_ids};
	}

	// if all else fails, assume we have a boolean
	try {
		return {env::default_parser<bool>{}(str), auto_start_from_core, {}};
	} catch(const env::parser_error& e) { throw env::parser_error{fmt::format(error_msg, e.what())}; }
}

} // namespace celerity::detail::thread_pinning
