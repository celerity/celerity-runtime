#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_compressed_all.sh [x_range_start] [x_range_end] [y_range_start] [y_range_end]
# Example:
#   ./run_compressed_all.sh -100 99 -100 99

X_START="${1:--100}"
X_END="${2:-99}"
Y_START="${3:--100}"
Y_END="${4:-99}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/../../../build_acpp_release"
CHECK_SCRIPT="${SCRIPT_DIR}/check_tiling.py"
CHECK_SHAPE_SCRIPT="${SCRIPT_DIR}/check_shape_factor_calculation.py"
INPUT_FILE="${SCRIPT_DIR}/plane_2D_200_200.bin"
GT_TILES_FILE="${BUILD_DIR}/tiles_test_file.bin"

# Keep compatibility with both historical spellings.
if [[ -f "${BUILD_DIR}/shapefactors_test_file.bin" ]]; then
  GT_SHAPE_FILE="${BUILD_DIR}/shapefactors_test_file.bin"
else
  GT_SHAPE_FILE="${BUILD_DIR}/shapefaktors_test_file.bin"
fi

echo "Running compressed tests with x_range [${X_START}, ${X_END}], y_range [${Y_START}, ${Y_END}]"

if [[ ! -d "${BUILD_DIR}" ]]; then
  echo "ERROR: Build directory not found: ${BUILD_DIR}" >&2
  exit 2
fi

if [[ ! -f "${CHECK_SCRIPT}" ]]; then
  echo "ERROR: Missing check script: ${CHECK_SCRIPT}" >&2
  exit 2
fi

if [[ ! -f "${CHECK_SHAPE_SCRIPT}" ]]; then
  echo "ERROR: Missing shape-factor check script: ${CHECK_SHAPE_SCRIPT}" >&2
  exit 2
fi

if [[ ! -f "${INPUT_FILE}" ]]; then
  echo "ERROR: Missing input file: ${INPUT_FILE}" >&2
  exit 2
fi

if [[ ! -f "${GT_TILES_FILE}" ]]; then
  echo "ERROR: Missing ground-truth tiles file: ${GT_TILES_FILE}" >&2
  exit 2
fi

if [[ ! -f "${GT_SHAPE_FILE}" ]]; then
  echo "ERROR: Missing ground-truth shape-factor file: ${GT_SHAPE_FILE}" >&2
  exit 2
fi

cd "${BUILD_DIR}"

run_variant() {
  local exe_suffix="$1"
  local out_suffix="$2"
  local exe="./examples/compressed_${exe_suffix}/compressed_${exe_suffix}"
  local tiles_out="./plane_tiles_${out_suffix}.bin"
  local shape_out="./plane_shape_${out_suffix}.bin"
  local tmp_tiles="./plane_tiles.bin"
  local tmp_shape="./plane_shape.bin"

  if [[ ! -x "${exe}" ]]; then
    echo "ERROR: Missing executable: ${exe}" >&2
    exit 2
  fi

  echo "Running ${exe} ..."
  "${exe}" "${INPUT_FILE}" "${tmp_tiles}" "${tmp_shape}"

  if [[ ! -f "${tmp_tiles}" ]] || [[ ! -f "${tmp_shape}" ]]; then
    echo "ERROR: Expected output files were not produced by ${exe}" >&2
    exit 2
  fi

  mv -f "${tmp_tiles}" "${tiles_out}"
  mv -f "${tmp_shape}" "${shape_out}"

  echo "Wrote ${tiles_out} and ${shape_out}"

  echo "Running check_tiling.py on ${tiles_out} ..."
  python3 "${CHECK_SCRIPT}" --x_range "${X_START}" "${X_END}" --y_range "${Y_START}" "${Y_END}" "${tiles_out}"

  echo "Running check_shape_factor_calculation.py on ${shape_out} ..."
  python3 "${CHECK_SHAPE_SCRIPT}" "${GT_TILES_FILE}" "${GT_SHAPE_FILE}" "${tiles_out}" "${shape_out}"
}

# run_variant "local" "local"
run_variant "global" "global"
run_variant "zcurve" "zcurve"
run_variant "element" "element_wise"

echo "All compressed runs and checks completed successfully."
