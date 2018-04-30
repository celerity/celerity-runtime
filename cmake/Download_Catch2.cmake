# Download Catch2
# See https://github.com/Crascit/DownloadProject

include(DownloadProject)
download_project(PROJ                Catch2
                 GIT_REPOSITORY      https://github.com/catchorg/Catch2
                 GIT_TAG             v2.2.1
                 UPDATE_DISCONNECTED 1
)

# Use single header version
set(Catch2_INCLUDE_DIRS "${Catch2_SOURCE_DIR}/single_include")

# Add cmake modules
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${Catch2_SOURCE_DIR}/contrib" )
