#!/bin/bash

set -o errexit -o pipefail -o nounset

if [[ ! -d CMakeFiles ]]; then
    echo "Warning: This script should be run from within a build directory" >&2
fi

if [[ $# -ne 0 ]]; then
    echo "Usage: $0" >&2
    exit 1
fi

# we use taskset to pin the process to a specific set of cores and their HTs to reduce benchmark result variance
# this set is chosen to all be located on the second die of the Threadripper 2920X CPU in our current CI benchmark system "gpuc2"
# since this is a very rare system with only 3 cores per CC, we use a hardcoded thread pinning strategy which places the 4 BE worker threads on HTs
export CELERITY_THREAD_PINNING=7,8,9,10,11,22,23
bash /root/capture-backtrace.sh taskset -c 6-11,18-23 test/benchmarks \
    --reporter celerity-benchmark-md::out=gpuc2_bench.md \
    --reporter celerity-benchmark-csv::out=gpuc2_bench.csv
