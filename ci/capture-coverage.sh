#!/bin/bash
# usage: capture-coverage.sh source-dir build-dir <options...>

set -eu -o pipefail

SOURCE_DIR="$(realpath "$1")"
BUILD_DIR="$(realpath "$2")"

exec fastcov \
	--compiler-directory "$SOURCE_DIR" \
	--search-directory "$BUILD_DIR" \
	--process-gcno \
	--include "$SOURCE_DIR/include" "$SOURCE_DIR/src" \
	--branch-coverage \
	--exclude-br-lines-starting-with assert CELERITY_DETAIL_ASSERT_ON_HOST \
	--lcov \
	--output lcov.info
