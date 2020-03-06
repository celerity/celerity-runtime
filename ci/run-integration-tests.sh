#!/bin/bash

if [[ ! -d examples || ! -d CMakeFiles ]]; then
    echo "Warning: This script should be run from within a build directory" 1>&2
fi

if [[ $# -le 1 ]]; then
    echo "Usage: $0 <convolution image file> <num nodes> [<num nodes>...]"
    exit 1
fi

CONV_IMG="$1"; shift
NUM_NODES=("$@")

EXAMPLES=(
    "matmul"
    "convolution"
    "wave_sim"
    "syncing"
)
PARAMS=(
    ""
    $CONV_IMG
    "-T 15 --dt 0.5 --sample-rate 2"
    ""
)
ARTIFACTS=(
    ""
    "output.png"
    "wave_sim_result.csv"
    ""
)

expected_checksum=""
for e in "${!EXAMPLES[@]}"; do
    EXE="${EXAMPLES[$e]}"
    CMD="./examples/${EXE}/${EXE} ${PARAMS[$e]}"
    ARTIFACT="${ARTIFACTS[$e]}"

    for n in "${NUM_NODES[@]}"; do
        echo -e "\n\n ---- Running \"$EXE\" on $n node(s) ----\n\n"
        rm -rf ${ARTIFACT} # Delete artifact before each run to make sure it is actually created
        mpirun -n ${n} ${CMD} || exit 1
        if [ ! -z "${ARTIFACT}" ]; then
            if [ "${n}" -eq "${NUM_NODES[0]}" ]; then
                expected_checksum=$(md5sum ${ARTIFACT})
            else
                # We don't check whether tests with outputs produce the exact same result for every
                # configuration (debug/release, hipSYCL/ComputeCpp, etc) because they don't. Instead
                # we just check whether they produce the same result across runs with different nodes.
                if [[ $(md5sum ${ARTIFACT}) != "${expected_checksum}" ]]; then
                    echo "${EXE}: Wrong ARTIFACT checksum after running with ${n} nodes." && exit 1
                fi
            fi
        fi
    done
done
