add_executable(pretty_printables pretty_printables.cc)

set_property(TARGET pretty_printables PROPERTY CXX_STANDARD ${CELERITY_CXX_STANDARD})
set_property(TARGET pretty_printables PROPERTY FOLDER "tests/debug")

add_celerity_to_target(TARGET pretty_printables SOURCES pretty_printables.cc)
