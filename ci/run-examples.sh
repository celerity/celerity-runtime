#!/bin/bash

set -o errexit -o pipefail -o nounset

if [[ ! -d CMakeFiles ]]; then
    echo "Warning: This script should be run from within a build directory" >&2
fi

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <convolution image file> <num nodes> [<num nodes>...]" >&2
    exit 1
fi

CONV_IMG="$1"; shift
NUM_NODES=("$@")

EXAMPLES=(
    "hello_world"
    "matmul"
    "convolution"
    "wave_sim"
    "syncing"
    "reduction"
)
IS_OPTIONAL=(
    ""
    ""
    ""
    ""
    ""
    "yes"
)
PARAMS=(
    ""
    ""
    "$CONV_IMG"
    "-T 15 --dt 0.5 --sample-rate 2"
    ""
    "$CONV_IMG"
)
ARTIFACTS=(
    ""
    ""
    "output.png"
    "wave_sim_result.bin"
    ""
    "output.jpg"
)

EXAMPLES_DIR="$(pwd)"
if [[ ! -x "${EXAMPLES[0]}/${EXAMPLES[0]}" ]]; then
  EXAMPLES_DIR="$EXAMPLES_DIR/examples"
fi

expected_checksum=""
for e in "${!EXAMPLES[@]}"; do
    NAME="${EXAMPLES[$e]}"

    EXE="$EXAMPLES_DIR/$NAME/$NAME"
    if [ -n "${IS_OPTIONAL[$e]}" ] && ! [ -f "$EXE" ]; then
      echo -e "\n\n ---- (Skipping optional \"$NAME\" because it has not been built) ----\n\n" >&2
      continue
    fi

    # shellcheck disable=SC2206
    CMD=("$EXE" ${PARAMS[$e]})
    ARTIFACT="${ARTIFACTS[$e]}"

    for n in "${NUM_NODES[@]}"; do
        echo -e "\n\n ---- Running \"$NAME\" on $n node(s) ----\n\n" >&2
        rm -rf "$ARTIFACT" # Delete artifact before each run to make sure it is actually created
        mpirun --bind-to none -n "$n" bash /root/capture-backtrace.sh "${CMD[@]}"
        if [ -n "$ARTIFACT" ]; then
            if [ "$n" -eq "${NUM_NODES[0]}" ]; then
                expected_checksum=$(md5sum "$ARTIFACT")
            else
                # We don't check whether tests with outputs produce the exact same result for every
                # configuration (debug/release, AdaptiveCpp/DPC++, etc) because they don't. Instead
                # we just check whether they produce the same result across runs with different nodes.
                if [[ $(md5sum "$ARTIFACT") != "$expected_checksum" ]]; then
                    echo "$NAME: Wrong ARTIFACT checksum after running with $n nodes." >&2
                    exit 1
                fi
            fi
        fi
    done
done
