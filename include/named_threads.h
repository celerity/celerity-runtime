#pragma once

#include <string>
#include <thread>


namespace celerity::detail {

std::thread::native_handle_type get_current_thread_handle();

void set_thread_name(const std::thread::native_handle_type thread_handle, const std::string& name);

std::string get_thread_name(const std::thread::native_handle_type thread_handle);

} // namespace celerity::detail
