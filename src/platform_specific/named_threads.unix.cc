#include "named_threads.h"

namespace celerity::detail {

void set_thread_name(std::thread& thread, const std::string& name) {}

void set_current_thread_name(const std::string& name) {}

std::string get_thread_name(std::thread& thread) {}

std::string get_current_thread_name() {}

} // namespace celerity::detail
