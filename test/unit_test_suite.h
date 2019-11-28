#pragma once

// Printing of graphs can be enabled using the "--print-graphs" command line flag
extern bool print_graphs;

// This translation unit (unit_test_suite) needs to be compiled without Celerity/SYCL
// to prevent Catch2 conflicts with some SYCL compilers.
// On the other hand, it also needs to provide the Catch2 LISTENER, since that's only
// possible here. As such, we provide wrapper functions declarations to call within
// the listener, which will be linked in from unit_test_suite_celerity, which is
// compiled as SYCL code.
namespace detail {
void test_started_callback();
void test_ended_callback();
} // namespace detail
