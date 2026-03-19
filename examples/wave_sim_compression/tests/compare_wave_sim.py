#!/usr/bin/env python3
"""Compare two wave simulation binary dumps with an epsilon tolerance.

Format (same as plot.rb):
- uint64: N (grid width/height)
- uint64: T (number of stored frames)
- float32 array: T * N * N values

Usage examples:
  python3 compare_wave_sim.py --reference ref.bin --candidate out.bin --epsilon 1e-3
  python3 compare_wave_sim.py -r wave_sim_result.bin -c wave_sim_result_test.bin -e 1e-4
"""

from __future__ import annotations

import argparse
import array
import math
import struct
import sys
from dataclasses import dataclass


GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"


@dataclass
class WaveSimData:
    n: int
    t: int
    values: array.array


class WaveSimReadError(RuntimeError):
    pass


def read_wave_sim_file(path: str) -> WaveSimData:
    with open(path, "rb") as f:
        data = f.read()

    if len(data) < 16:
        raise WaveSimReadError(f"{path}: file too small (needs at least 16 bytes header)")

    # Match the current C++ writer on Linux systems (little-endian).
    n, t = struct.unpack_from("<QQ", data, 0)
    payload = data[16:]

    if len(payload) % 4 != 0:
        raise WaveSimReadError(f"{path}: payload size is not a multiple of 4 bytes")

    values = array.array("f")
    values.frombytes(payload)

    expected = n * n * t
    if len(values) != expected:
        raise WaveSimReadError(
            f"{path}: value count mismatch, expected {expected} floats from header, got {len(values)}"
        )

    return WaveSimData(n=n, t=t, values=values)


def linear_to_coords(index: int, n: int) -> tuple[int, int, int]:
    frame_size = n * n
    frame = index // frame_size
    in_frame = index % frame_size
    y = in_frame // n
    x = in_frame % n
    return frame, y, x


def compare(reference: WaveSimData, candidate: WaveSimData, epsilon: float) -> tuple[bool, str]:
    if reference.n != candidate.n or reference.t != candidate.t:
        return (
            False,
            "Header mismatch: "
            f"reference (N={reference.n}, T={reference.t}) vs "
            f"candidate (N={candidate.n}, T={candidate.t})",
        )

    if len(reference.values) != len(candidate.values):
        return (
            False,
            f"Value count mismatch: reference={len(reference.values)} candidate={len(candidate.values)}",
        )

    max_abs_diff = -1.0
    max_idx = -1

    for idx, (ref, cand) in enumerate(zip(reference.values, candidate.values)):
        abs_diff = abs(ref - cand)
        if abs_diff > max_abs_diff:
            max_abs_diff = abs_diff
            max_idx = idx

    if max_abs_diff <= epsilon:
        frame, y, x = linear_to_coords(max_idx, reference.n)
        return (
            True,
            f"All values within epsilon={epsilon:g}. max_abs_diff={max_abs_diff:.6e} at frame={frame}, y={y}, x={x}",
        )

    frame, y, x = linear_to_coords(max_idx, reference.n)
    ref_val = reference.values[max_idx]
    cand_val = candidate.values[max_idx]
    return (
        False,
        "Exceeded epsilon: "
        f"max_abs_diff={max_abs_diff:.6e} > epsilon={epsilon:g} "
        f"at frame={frame}, y={y}, x={x}, reference={ref_val:.6e}, candidate={cand_val:.6e}",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare two wave-sim binary result files with an epsilon tolerance."
    )
    parser.add_argument("-r", "--reference", required=True, help="Path to reference .bin file")
    parser.add_argument("-c", "--candidate", required=True, help="Path to candidate .bin file")
    parser.add_argument("-e", "--epsilon", required=True, type=float, help="Allowed absolute error")
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI colors in output",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if not math.isfinite(args.epsilon) or args.epsilon < 0.0:
        print("ERROR: epsilon must be a finite, non-negative number", file=sys.stderr)
        return 2

    try:
        reference = read_wave_sim_file(args.reference)
        candidate = read_wave_sim_file(args.candidate)
    except (OSError, WaveSimReadError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    ok, message = compare(reference, candidate, args.epsilon)

    if args.no_color:
        prefix = "PASS" if ok else "FAIL"
    else:
        prefix = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"

    print(f"{prefix}: {message}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
