#!/bin/bash

set -o errexit -o pipefail -o nounset

if [[ ! -d CMakeFiles ]]; then
    echo "Warning: This script should be run from within a build directory" >&2
fi

if [[ $# -ne 0 ]]; then
    echo "Usage: $0" >&2
    exit 1
fi

# Running "make test" is slow because CTest will spawn one process per test case, adding significant start-up costs.
# We just call the all_tests executable manually to get around this.
bash /root/capture-backtrace.sh test/all_tests
