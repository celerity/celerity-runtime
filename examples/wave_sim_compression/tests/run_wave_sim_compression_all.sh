#!/usr/bin/env bash

# Usage:
#   ./run_wave_sim_compression_all.sh [N] [T] [sample_rate] [epsilon]
# Example:
#   ./run_wave_sim_compression_all.sh 128 100 4 0.01

N="${1:-128}"
T="${2:-100}"
SAMPLE_RATE="${3:-4}"
EPSILON="${4:-0.01}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/../../../build_acpp_release"
COMPARE_SCRIPT="${SCRIPT_DIR}/compare_wave_sim.py"

echo "Running wave_sim tests with N=${N}, T=${T}, sample_rate=${SAMPLE_RATE}, epsilon=${EPSILON} using executables from ${BUILD_DIR}"

if [[ ! -d "${BUILD_DIR}" ]]; then
  echo "ERROR: Build directory not found: ${BUILD_DIR}" >&2
  exit 2
fi

if [[ ! -f "${COMPARE_SCRIPT}" ]]; then
  echo "ERROR: Missing compare script: ${COMPARE_SCRIPT}" >&2
  exit 2
fi

cd "${BUILD_DIR}"

run_integration() {
  local exe="./examples/wave_sim/wave_sim"
  local out="./wave_sim_result_integration.bin"

  if [[ ! -x "${exe}" ]]; then
    echo "ERROR: Missing executable: ${exe}" >&2
    exit 2
  fi

  echo "Running ${exe} ..."
  "${exe}" -N "${N}" -T "${T}" --sample-rate "${SAMPLE_RATE}"

  if [[ ! -f ./wave_sim_result.bin ]]; then
    echo "ERROR: Expected output ./wave_sim_result.bin was not produced by ${exe}" >&2
    exit 2
  fi

  mv -f ./wave_sim_result.bin "${out}"
  echo "Wrote ${out}"
}

run_variant() {
  local exe_suffix="$1"
  local out_suffix="$2"
  local exe="./examples/wave_sim_compression/wave_sim_compression_${exe_suffix}"
  local out="./wave_sim_result_${out_suffix}.bin"

  if [[ ! -x "${exe}" ]]; then
    echo "ERROR: Missing executable: ${exe}" >&2
    exit 2
  fi

  echo "Running ${exe} ..."
  "${exe}" -N "${N}" -T "${T}" --sample-rate "${SAMPLE_RATE}"

  if [[ ! -f ./wave_sim_result.bin ]]; then
    echo "ERROR: Expected output ./wave_sim_result.bin was not produced by ${exe}" >&2
    exit 2
  fi

  mv -f ./wave_sim_result.bin "${out}"
  echo "Wrote ${out}"
}

run_variant "local" "local"
run_variant "global" "global"
run_variant "element" "element_wise"

run_integration

echo "Running comparisons against integration output..."
python3 "${COMPARE_SCRIPT}" -r ./wave_sim_result_integration.bin -c ./wave_sim_result_local.bin -e "${EPSILON}"
python3 "${COMPARE_SCRIPT}" -r ./wave_sim_result_integration.bin -c ./wave_sim_result_global.bin -e "${EPSILON}"
python3 "${COMPARE_SCRIPT}" -r ./wave_sim_result_integration.bin -c ./wave_sim_result_element_wise.bin -e "${EPSILON}"

echo "All wave_sim runs and comparisons completed successfully."


