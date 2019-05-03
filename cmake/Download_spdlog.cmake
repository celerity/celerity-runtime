# Download spdlog
# See https://github.com/Crascit/DownloadProject

include(DownloadProject)
download_project(PROJ                spdlog
                 GIT_REPOSITORY      https://github.com/gabime/spdlog
                 GIT_TAG             v1.3.1
                 UPDATE_DISCONNECTED 1
)

set(spdlog_INCLUDE_DIRS "${spdlog_SOURCE_DIR}/include")
