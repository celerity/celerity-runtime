#include "utils.h"

#include <regex>

namespace celerity::detail::utils {

std::string simplify_task_name(const std::string& demangled_type_name) {
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
