#!/bin/bash

set -o errexit -o pipefail -o noclobber -o nounset

if [[ $# -gt 0 && $1 == "--help" ]]; then
    {
        echo "Run clang-tidy on all Celerity source files"
        echo "Usage: $0 [<clang-tidy arguments>...]"
        echo "Set CLANG_TIDY environment variable to override default executable (\`clang-tidy\`)."
    } >&2
    exit 1
fi

if [[ ! -d src ]]; then
    echo "Error: This script should be run from within the repository root directory." >&2
    exit 1
fi

CLANG_TIDY=${CLANG_TIDY:-clang-tidy}
if [[ ! -x "$CLANG_TIDY" ]]; then
    echo "Clang tidy executable \`$CLANG_TIDY\` does not exist. Set CLANG_TIDY environment variable to override."
    exit 1
fi

RUN_PARALLEL=1
if [[ ! -x $(command -v parallel) ]]; then
    echo "\`parallel\` not found. Running sequentially." >&2
    RUN_PARALLEL=0
fi
if grep -q -- "--fix\(-errors\)\?" <<< "$@" ; then
    echo "\`--fix\` or \`--fix-errors\` is set. Running sequentially." >&2
    RUN_PARALLEL=0
fi

INCLUDE_DIR=$(readlink -f include)
TEST_DIR=$(readlink -f test)
SOURCES=$(find examples src test -name "*.cc")

if [[ $RUN_PARALLEL -eq 1 ]]; then
    set -x
    parallel "$CLANG_TIDY" --header-filter="$INCLUDE_DIR|$TEST_DIR" "$@" -- $SOURCES
else
    set -x
    "$CLANG_TIDY" --header-filter="$INCLUDE_DIR|$TEST_DIR" "$@" $SOURCES
fi
