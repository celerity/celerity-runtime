# Download spdlog
# See https://github.com/Crascit/DownloadProject

include(DownloadProject)
download_project(PROJ                spdlog
                 GIT_REPOSITORY      https://github.com/gabime/spdlog
                 GIT_TAG             8cc0997f796d2295b0ccc9caaf0abcca25d89525
                 UPDATE_DISCONNECTED 1
)

set(spdlog_INCLUDE_DIRS "${spdlog_SOURCE_DIR}/include")
