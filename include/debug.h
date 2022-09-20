#include <string>

#include "buffer.h"

namespace celerity {
namespace debug {
	template <typename DataT, int Dims>
	void set_buffer_name(const celerity::buffer<DataT, Dims>& buff, const std::string& debug_name) {
		detail::set_buffer_name(buff, debug_name);
		detail::runtime::get_instance().get_buffer_manager().set_debug_name(detail::get_buffer_id(buff), debug_name);
	}
	template <typename DataT, int Dims>
	std::string get_buffer_name(const celerity::buffer<DataT, Dims>& buff) {
		return detail::get_buffer_name(buff);
	}

} // namespace debug
} // namespace celerity