#!/bin/bash

set -o errexit -o pipefail -o nounset

if [[ ! -d CMakeFiles ]]; then
    echo "Warning: This script should be run from within a build directory" >&2
fi

if [[ $# -ne 0 ]]; then
    echo "Usage: $0" >&2
    exit 1
fi

bash /root/capture-backtrace.sh test/benchmarks \
    --reporter celerity-benchmark-md::out=gpuc2_bench.md \
    --reporter celerity-benchmark-csv::out=gpuc2_bench.csv
