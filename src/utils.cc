#include "utils.h"
#include "log.h"

#include <atomic>
#include <regex>
#include <stdexcept>

#if !defined(_MSC_VER)
// Required for kernel name demangling in Clang
#include <cxxabi.h>
#endif


namespace celerity::detail::utils {

std::string get_simplified_type_name_from_pointer(const std::type_info& pointer_type_info) {
#if !defined(_MSC_VER)
	const std::unique_ptr<char, void (*)(void*)> demangle_buffer(abi::__cxa_demangle(pointer_type_info.name(), nullptr, nullptr, nullptr), std::free);
	std::string demangled_type_name = demangle_buffer.get();
#else
	std::string demangled_type_name = pointer_type_info.name();
#endif

	// get rid of the pointer "*"
	if(!demangled_type_name.empty() && demangled_type_name.back() == '*') { demangled_type_name.pop_back(); }

	if(demangled_type_name.length() < 2) return demangled_type_name;
	bool templated = false;
	// there are two options:
	// 1. the type is templated; in this case, the last character is ">" and we go back to the matching "<"
	std::string::size_type last_idx = demangled_type_name.length() - 1;
	if(demangled_type_name[last_idx] == '>') {
		templated = true;
		int open = 0;
		while(last_idx > 1) {
			last_idx--;
			if(demangled_type_name[last_idx] == '>') { open++; }
			if(demangled_type_name[last_idx] == '<') {
				if(open > 0) {
					open--;
				} else {
					last_idx--;
					break;
				}
			}
		}
	}
	// 2. the type isn't templated (or we just removed the template); in this case, we are interested in the part from the end to the last ":" (or the start)
	std::string::size_type start_idx = last_idx - 1;
	while(start_idx > 0 && demangled_type_name[start_idx - 1] != ':') {
		start_idx--;
	}
	// if the type was templated, we add a "<...>" to indicate that
	return demangled_type_name.substr(start_idx, last_idx - start_idx + 1) + (templated ? "<...>" : "");
}

std::string escape_for_dot_label(std::string str) {
	str = std::regex_replace(str, std::regex("&"), "&amp;");
	str = std::regex_replace(str, std::regex("<"), "&lt;");
	str = std::regex_replace(str, std::regex(">"), "&gt;");
	return str;
}

[[noreturn]] void unreachable() {
	assert(!"executed unreachable code");
	abort();
}

// The panic solution defaults to `log_and_abort`, but is set to `throw_logic_error` in test binaries. Since panics are triggered from celerity library code, we
// manage it in a global and decide which path to take at runtime. We have also considered deciding this at link time by defining a weak symbol (GCC
// __attribute__((weak))) in the library which is overwritten by a strong symbol in the test library, but decided against this because there is no equivalent in
// MSVC and we would have to resort to even dirtier linker hacks for that target.
std::atomic<panic_solution> g_panic_solution = panic_solution::log_and_abort; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

void set_panic_solution(panic_solution solution) { g_panic_solution.store(solution, std::memory_order_relaxed); }

[[noreturn]] void panic(const std::string& msg) {
	switch(g_panic_solution.load(std::memory_order_relaxed)) {
	case celerity::detail::utils::panic_solution::throw_logic_error: {
		throw std::logic_error(msg);
	}
	case celerity::detail::utils::panic_solution::log_and_abort:
	default: {
		// Print directly instead of logging: The abort message must not be hidden by log level setting, and in tests would be captured without the logging
		// infrastructure having a chance of dumping the logs due to the abort.
		fmt::print(stderr, "celerity-runtime panic: {}\n", msg);
		std::abort();
	}
	}
}

void report_error(const error_policy policy, const std::string& msg) {
	switch(policy) {
	case error_policy::ignore: break;
	case error_policy::log_warning: CELERITY_WARN("{}", msg); break;
	case error_policy::log_error: CELERITY_ERROR("{}", msg); break;
	case error_policy::panic: panic(msg); break;
	}
}

std::string make_buffer_debug_label(const buffer_id bid, const std::string& name) {
	// if there is no name defined, the name will be the buffer id.
	// if there is a name we want "id name"
	return !name.empty() ? fmt::format("B{} \"{}\"", bid, name) : fmt::format("B{}", bid);
}

std::string make_task_debug_label(const task_type tt, const task_id tid, const std::string& debug_name, bool title_case) {
	const auto type_string = [tt] {
		switch(tt) {
		case task_type::epoch: return "epoch";
		case task_type::host_compute: return "host-compute task";
		case task_type::device_compute: return "device kernel";
		case task_type::collective: return "collective host task";
		case task_type::master_node: return "master-node host task";
		case task_type::horizon: return "horizon";
		case task_type::fence: return "fence";
		default: return "unknown task";
		}
	}();

	auto label = fmt::format("{} T{}", type_string, tid);
	if(title_case) { label[0] = static_cast<char>(std::toupper(label[0])); }
	if(!debug_name.empty()) { fmt::format_to(std::back_inserter(label), " \"{}\"", debug_name); }
	return label;
}

} // namespace celerity::detail::utils


// implemented here because types.h must not depend on utils.h
std::size_t std::hash<celerity::detail::transfer_id>::operator()(const celerity::detail::transfer_id& t) const noexcept {
	auto hash = std::hash<celerity::detail::task_id>{}(t.consumer_tid);
	celerity::detail::utils::hash_combine(hash, std::hash<celerity::detail::buffer_id>{}(t.bid));
	celerity::detail::utils::hash_combine(hash, std::hash<celerity::detail::reduction_id>{}(t.rid));
	return hash;
}
