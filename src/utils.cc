#include "utils.h"

#include <regex>

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

} // namespace celerity::detail::utils
