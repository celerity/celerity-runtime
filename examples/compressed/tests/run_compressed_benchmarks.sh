#!/usr/bin/env bash
set -euo pipefail

# Runs all compressed benchmark executables and records timings for:
#   Time difference between 0 and 1
#   Time difference between 1 and 2
#
# Defaults are set for this repository layout, but can be overridden.
#
# Usage:
#   ./run_compressed_benchmarks.sh [--build-dir DIR] [--input-file FILE] [--output-dir DIR] [--warmup-runs N] [--runs N]
#
# Environment overrides for CUDA visible devices per profile:
#   CUDA_VISIBLE_DEVICES_1 (default: 0)
#   CUDA_VISIBLE_DEVICES_2 (default: 0,1)
#   CUDA_VISIBLE_DEVICES_4 (default: 0,1,2,3)
#
# Backward compatibility is preserved for:
#   GPU_MASK_1, GPU_MASK_2, GPU_MASK_4

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

BUILD_DIR="${REPO_ROOT}/build_acpp_release"
INPUT_FILE="${SCRIPT_DIR}/plane_2D_200_200.bin"
WARMUP_RUNS=1
RUNS=5
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-dir)
      BUILD_DIR="$2"
      shift 2
      ;;
    --input-file)
      INPUT_FILE="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --warmup-runs)
      WARMUP_RUNS="$2"
      shift 2
      ;;
    --runs)
      RUNS="$2"
      shift 2
      ;;
    -h|--help)
      sed -n '1,40p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="${BUILD_DIR}/benchmark_results/compressed_$(date +%Y%m%d_%H%M%S)"
fi

if [[ ! -d "${BUILD_DIR}" ]]; then
  echo "ERROR: Build directory not found: ${BUILD_DIR}" >&2
  exit 2
fi

if [[ ! -f "${INPUT_FILE}" ]]; then
  echo "ERROR: Input file not found: ${INPUT_FILE}" >&2
  exit 2
fi

if ! [[ "${WARMUP_RUNS}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: --warmup-runs must be a non-negative integer" >&2
  exit 2
fi

if ! [[ "${RUNS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --runs must be a positive integer" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}/logs" "${OUTPUT_DIR}/outputs"

TIMING_01_CSV="${OUTPUT_DIR}/timing_0_1_ms.csv"
TIMING_12_CSV="${OUTPUT_DIR}/timing_1_2_ms.csv"
SUMMARY_TXT="${OUTPUT_DIR}/summary.txt"

echo "gpu_count,cuda_visible_devices,run,executable,time_ms,log_file" > "${TIMING_01_CSV}"
echo "gpu_count,cuda_visible_devices,run,executable,time_ms,log_file" > "${TIMING_12_CSV}"

# All compressed variants produced by examples/compressed/CMakeLists.txt
EXECUTABLES=(
  "compressed_point_cloud_element"
  "compressed_point_cloud_local"
  "compressed_point_cloud_global"
  "compressed_point_cloud_with_dep_local"
  "compressed_point_cloud_with_dep_global"
  "compressed_zcurve_local"
  "compressed_zcurve_global"
  "compressed_uncompressed"
)

declare -A CUDA_VISIBLE_DEVICES_PROFILES
CUDA_VISIBLE_DEVICES_PROFILES[1]="${CUDA_VISIBLE_DEVICES_1:-${GPU_MASK_1:-0}}"
CUDA_VISIBLE_DEVICES_PROFILES[2]="${CUDA_VISIBLE_DEVICES_2:-${GPU_MASK_2:-0,1}}"
CUDA_VISIBLE_DEVICES_PROFILES[4]="${CUDA_VISIBLE_DEVICES_4:-${GPU_MASK_4:-0,1,2,3}}"

extract_timing_ms() {
  local pattern="$1"
  local file="$2"
  local line
  line="$(grep -m1 "${pattern}" "${file}" || true)"
  if [[ -z "${line}" ]]; then
    echo ""
    return 0
  fi
  sed -E 's/.*: *([0-9]+(\.[0-9]+)?)ms.*/\1/' <<< "${line}"
}

{
  echo "Compressed benchmark run"
  echo "Build dir : ${BUILD_DIR}"
  echo "Input file: ${INPUT_FILE}"
  echo "Warmup runs: ${WARMUP_RUNS}"
  echo "Runs      : ${RUNS}"
  echo "Output dir: ${OUTPUT_DIR}"
  echo "CUDA_VISIBLE_DEVICES profiles: 1=>${CUDA_VISIBLE_DEVICES_PROFILES[1]} 2=>${CUDA_VISIBLE_DEVICES_PROFILES[2]} 4=>${CUDA_VISIBLE_DEVICES_PROFILES[4]}"
  echo
} | tee "${SUMMARY_TXT}"

cd "${BUILD_DIR}"

for gpu_count in 1 2 4; do
  cuda_visible_devices="${CUDA_VISIBLE_DEVICES_PROFILES[${gpu_count}]}"
  echo "=== GPU profile: ${gpu_count} (CUDA_VISIBLE_DEVICES=${cuda_visible_devices}) ===" | tee -a "${SUMMARY_TXT}"

  for exe_name in "${EXECUTABLES[@]}"; do
    exe_path="./compress/${exe_name}"
    if [[ ! -x "${exe_path}" ]]; then
      echo "ERROR: Missing executable: ${exe_path}" | tee -a "${SUMMARY_TXT}" >&2
      exit 2
    fi

    for warmup_idx in $(seq 1 "${WARMUP_RUNS}"); do
      warmup_tile="${OUTPUT_DIR}/outputs/${exe_name}_gpu${gpu_count}_warmup${warmup_idx}_tiles.bin"
      warmup_shape="${OUTPUT_DIR}/outputs/${exe_name}_gpu${gpu_count}_warmup${warmup_idx}_shape.bin"
      warmup_log="${OUTPUT_DIR}/logs/${exe_name}_gpu${gpu_count}_warmup${warmup_idx}.log"

      echo "Warmup ${exe_name} | gpu=${gpu_count} warmup=${warmup_idx}" | tee -a "${SUMMARY_TXT}"
      if ! CUDA_VISIBLE_DEVICES="${cuda_visible_devices}" "${exe_path}" "${INPUT_FILE}" "${warmup_tile}" "${warmup_shape}" > "${warmup_log}" 2>&1; then
        echo "ERROR: Warmup failed for ${exe_name} (gpu=${gpu_count}, warmup=${warmup_idx}). See ${warmup_log}" | tee -a "${SUMMARY_TXT}" >&2
        exit 1
      fi
    done

    for run_idx in $(seq 1 "${RUNS}"); do
      out_tile="${OUTPUT_DIR}/outputs/${exe_name}_gpu${gpu_count}_run${run_idx}_tiles.bin"
      out_shape="${OUTPUT_DIR}/outputs/${exe_name}_gpu${gpu_count}_run${run_idx}_shape.bin"
      log_file="${OUTPUT_DIR}/logs/${exe_name}_gpu${gpu_count}_run${run_idx}.log"

      echo "Running ${exe_name} | gpu=${gpu_count} run=${run_idx}" | tee -a "${SUMMARY_TXT}"
      if ! CUDA_VISIBLE_DEVICES="${cuda_visible_devices}" "${exe_path}" "${INPUT_FILE}" "${out_tile}" "${out_shape}" > "${log_file}" 2>&1; then
        echo "ERROR: Benchmark failed for ${exe_name} (gpu=${gpu_count}, run=${run_idx}). See ${log_file}" | tee -a "${SUMMARY_TXT}" >&2
        exit 1
      fi

      t01="$(extract_timing_ms "Time difference between 0 and 1:" "${log_file}")"
      t12="$(extract_timing_ms "Time difference between 1 and 2:" "${log_file}")"

      if [[ -z "${t01}" || -z "${t12}" ]]; then
        echo "ERROR: Timing lines not found in ${log_file}" | tee -a "${SUMMARY_TXT}" >&2
        exit 1
      fi

      printf '%s,"%s",%s,%s,%s,"%s"\n' "${gpu_count}" "${cuda_visible_devices}" "${run_idx}" "${exe_name}" "${t01}" "${log_file}" >> "${TIMING_01_CSV}"
      printf '%s,"%s",%s,%s,%s,"%s"\n' "${gpu_count}" "${cuda_visible_devices}" "${run_idx}" "${exe_name}" "${t12}" "${log_file}" >> "${TIMING_12_CSV}"

      {
        echo "  Time difference between 0 and 1: ${t01}ms"
        echo "  Time difference between 1 and 2: ${t12}ms"
      } | tee -a "${SUMMARY_TXT}"
    done
  done

  echo | tee -a "${SUMMARY_TXT}"
done

echo "Done. Timing files:" | tee -a "${SUMMARY_TXT}"
echo "  ${TIMING_01_CSV}" | tee -a "${SUMMARY_TXT}"
echo "  ${TIMING_12_CSV}" | tee -a "${SUMMARY_TXT}"
echo "  ${SUMMARY_TXT}" | tee -a "${SUMMARY_TXT}"
