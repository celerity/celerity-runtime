# Adapted from https://github.com/Crascit/DownloadProject/blob/master/CMakeLists.txt
#
# CAVEAT: use DownloadProject.cmake
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
if (CMAKE_VERSION VERSION_LESS 3.2)
    set(UPDATE_DISCONNECTED_IF_AVAILABLE "")
else()
    set(UPDATE_DISCONNECTED_IF_AVAILABLE "UPDATE_DISCONNECTED 1")
endif()

include(DownloadProject)
download_project(PROJ                catch2
                 GIT_REPOSITORY      https://github.com/catchorg/Catch2
                 GIT_TAG             v2.2.1
                 ${UPDATE_DISCONNECTED_IF_AVAILABLE}
)

#add_subdirectory(${catch2_SOURCE_DIR} ${catch2_BINARY_DIR})

include_directories("${catch2_SOURCE_DIR}/single_include")