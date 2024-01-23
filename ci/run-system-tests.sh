#!/bin/bash

if [[ ! -d examples || ! -d CMakeFiles ]]; then
    echo "Warning: This script should be run from within a build directory" >&2
fi

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <num nodes> [<num nodes>...]" >&2
    exit 1
fi

NUM_NODES=("$@")

SYSTEM_TESTS=(
    "distr_tests"
)

for e in "${!SYSTEM_TESTS[@]}"; do
    EXE="${SYSTEM_TESTS[$e]}"
    CMD="./test/system/${EXE}"

    for n in "${NUM_NODES[@]}"; do
        echo -e "\n\n ---- Running \"$EXE\" on $n node(s) ----\n\n" >&2
        mpirun --bind-to none -n ${n} bash /root/capture-backtrace.sh "$CMD" || exit 1
    done
done
