# Download spdlog
# See https://github.com/Crascit/DownloadProject

include(DownloadProject)
download_project(PROJ                spdlog
                 GIT_REPOSITORY      https://github.com/gabime/spdlog
                 GIT_TAG             74dbf4cf702b49c98642c9afe74d114a238a6a07
                 UPDATE_DISCONNECTED 1
)

set(spdlog_INCLUDE_DIRS "${spdlog_SOURCE_DIR}/include")
