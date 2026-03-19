#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


@dataclass
class TimingRow:
    gpu_count: int
    cuda_visible_devices: str
    run: int
    executable: str
    time_ms: float
    phase: str


def parse_timing_file(path: Path, phase: str) -> List[TimingRow]:
    rows: List[TimingRow] = []

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for record in reader:
            rows.append(
                TimingRow(
                    gpu_count=int(record["gpu_count"]),
                    cuda_visible_devices=record["cuda_visible_devices"],
                    run=int(record["run"]),
                    executable=record["executable"],
                    time_ms=float(record["time_ms"]),
                    phase=phase,
                )
            )

    return rows


def aggregate_phase(rows: List[TimingRow]) -> Dict[tuple[str, int], List[float]]:
    grouped: Dict[tuple[str, int], List[float]] = {}
    for row in rows:
        grouped.setdefault((row.executable, row.gpu_count), []).append(row.time_ms)
    return grouped


def pretty_name(exe: str) -> str:
    mapping = {
        "compressed_point_cloud_element": "PointCloud (Elem)",
        "compressed_point_cloud_local": "PointCloud (Local)",
        "compressed_point_cloud_global": "PointCloud (Global)",
        "compressed_point_cloud_with_dep_local": "PointCloud+Dep (Local)",
        "compressed_point_cloud_with_dep_global": "PointCloud+Dep (Global)",
        "compressed_zcurve_local": "ZCurve Hybrid (Local)",
        "compressed_zcurve_global": "ZCurve Hybrid (Global)",
        "compressed_uncompressed": "Uncompressed Baseline",
    }
    return mapping.get(exe, exe)


def lighten_color(hex_color: str, amount: float = 0.28) -> str:
    hex_color = hex_color.lstrip("#")
    red = int(hex_color[0:2], 16)
    green = int(hex_color[2:4], 16)
    blue = int(hex_color[4:6], 16)
    red = int(red + (255 - red) * amount)
    green = int(green + (255 - green) * amount)
    blue = int(blue + (255 - blue) * amount)
    return f"#{red:02X}{green:02X}{blue:02X}"


def build_plot(rows_01: List[TimingRow], rows_12: List[TimingRow], output_file: Path) -> None:
    grouped_01 = aggregate_phase(rows_01)
    grouped_12 = aggregate_phase(rows_12)

    phase_means = {
        "0->1": {key: mean(values) for key, values in grouped_01.items()},
        "1->2": {key: mean(values) for key, values in grouped_12.items()},
    }
    phase_stddevs = {
        "0->1": {key: (stdev(values) if len(values) > 1 else 0.0) for key, values in grouped_01.items()},
        "1->2": {key: (stdev(values) if len(values) > 1 else 0.0) for key, values in grouped_12.items()},
    }

    runs_01: Dict[tuple[str, int], Dict[int, float]] = {}
    runs_12: Dict[tuple[str, int], Dict[int, float]] = {}
    for row in rows_01:
        runs_01.setdefault((row.executable, row.gpu_count), {})[row.run] = row.time_ms
    for row in rows_12:
        runs_12.setdefault((row.executable, row.gpu_count), {})[row.run] = row.time_ms

    total_grouped: Dict[tuple[str, int], List[float]] = {}
    for key, phase_01_runs in runs_01.items():
        phase_12_runs = runs_12.get(key, {})
        common_runs = sorted(set(phase_01_runs).intersection(phase_12_runs))
        total_grouped[key] = [phase_01_runs[run] + phase_12_runs[run] for run in common_runs]

    total_means = {key: mean(values) for key, values in total_grouped.items()}
    total_stddevs = {key: (stdev(values) if len(values) > 1 else 0.0) for key, values in total_grouped.items()}

    executable_order = [
        "compressed_point_cloud_element",
        "compressed_point_cloud_local",
        "compressed_point_cloud_global",
        "compressed_point_cloud_with_dep_local",
        "compressed_point_cloud_with_dep_global",
        "compressed_zcurve_local",
        "compressed_zcurve_global",
        "compressed_uncompressed",
    ]
    available_pairs = set(total_grouped.keys())
    executables = [exe for exe in executable_order if any((exe, g) in available_pairs for g in [1, 2, 4])]
    gpu_counts = sorted({gpu for _, gpu in available_pairs})
    phases = ["0->1", "1->2", "total"]
    phase_display_name = {
        "0->1": "Tiling",
        "1->2": "Shape Factors",
        "total": "Whole Pipeline",
    }
    executable_hatch = {
        "compressed_point_cloud_element": "///",
        "compressed_point_cloud_local": "\\\\",
        "compressed_point_cloud_global": "|||",
        "compressed_point_cloud_with_dep_local": "---",
        "compressed_point_cloud_with_dep_global": "+++",
        "compressed_zcurve_local": "xxx",
        "compressed_zcurve_global": "ooo",
        "compressed_uncompressed": "...",
    }
    phase_segment_hatch = {
        "0->1": "///",
        "1->2": "xx",
    }
    colors = {
        "compressed_point_cloud_element": "#0072B2",
        "compressed_point_cloud_local": "#009E73",
        "compressed_point_cloud_global": "#56B4E9",
        "compressed_point_cloud_with_dep_local": "#E69F00",
        "compressed_point_cloud_with_dep_global": "#D55E00",
        "compressed_zcurve_local": "#CC79A7",
        "compressed_zcurve_global": "#9B59B6",
        "compressed_uncompressed": "#374151",
    }
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(25, 8), sharey=False)
    fig.patch.set_facecolor("white")

    bar_width = 0.11
    x_positions = list(range(len(gpu_counts)))

    for ax, phase in zip(axes, phases):
        ax.set_facecolor("white")

        for idx, exe in enumerate(executables):
            base_color = colors[exe]
            light_color = lighten_color(base_color)
            if phase == "total":
                tiling = [phase_means["0->1"][(exe, gpu)] for gpu in gpu_counts]
                shaping = [phase_means["1->2"][(exe, gpu)] for gpu in gpu_counts]
                total = [t + s for t, s in zip(tiling, shaping)]
                yerr = [total_stddevs[(exe, gpu)] for gpu in gpu_counts]
            else:
                ys = [phase_means[phase][(exe, gpu)] for gpu in gpu_counts]
                yerr = [phase_stddevs[phase][(exe, gpu)] for gpu in gpu_counts]
            bar_x = [x + (idx - (len(executables) - 1) / 2) * bar_width for x in x_positions]
            if phase == "total":
                bars_tiling = ax.bar(
                    bar_x,
                    tiling,
                    width=bar_width,
                    color=base_color,
                    hatch=executable_hatch[exe] + phase_segment_hatch["0->1"],
                    edgecolor="#1F2937",
                    linewidth=0.55,
                    label=pretty_name(exe),
                )
                ax.bar(
                    bar_x,
                    shaping,
                    width=bar_width,
                    bottom=tiling,
                    color=light_color,
                    hatch=executable_hatch[exe] + phase_segment_hatch["1->2"],
                    edgecolor="#1F2937",
                    linewidth=0.55,
                )
                ax.errorbar(
                    bar_x,
                    total,
                    yerr=yerr,
                    fmt="none",
                    ecolor="#1F2937",
                    elinewidth=0.8,
                    capsize=3,
                )

                end_bar = list(bars_tiling)[-1]
                ax.text(
                    end_bar.get_x() + end_bar.get_width() / 2,
                    total[-1] + max(total) * 0.01,
                    f"{total[-1]:.0f}",
                    color="#111827",
                    fontsize=7,
                    ha="center",
                    va="bottom",
                )
            else:
                bars = ax.bar(
                    bar_x,
                    ys,
                    width=bar_width,
                    yerr=yerr,
                    color=base_color,
                    hatch=executable_hatch[exe],
                    label=pretty_name(exe),
                    alpha=1.0,
                    capsize=4,
                    edgecolor="#1F2937",
                    linewidth=0.55,
                )

                # Annotate only the 4-GPU bar to keep labels readable.
                end_bar = list(bars)[-1]
                ax.text(
                    end_bar.get_x() + end_bar.get_width() / 2,
                    end_bar.get_height() + max(ys) * 0.01,
                    f"{ys[-1]:.1f}",
                    color="#111827",
                    fontsize=8,
                    ha="center",
                    va="bottom",
                )

        ax.set_title(
            f"{phase_display_name[phase]} Runtime",
            fontsize=15,
            weight="bold",
            color="#1F2937",
        )
        ax.set_xlabel("GPU Count", fontsize=12, color="#111827")
        ax.set_ylabel("Time (ms)", fontsize=12, color="#111827")
        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(g) for g in gpu_counts])
        ax.tick_params(axis="both", labelsize=10)
        ax.grid(True, alpha=0.35, linestyle=":")

    legend_handles = [
        Patch(facecolor=colors[exe], edgecolor="#1F2937", hatch=executable_hatch[exe], label=pretty_name(exe))
        for exe in executables
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
        fontsize=10,
    )

    plt.tight_layout(rect=[0.02, 0.08, 1, 1.0])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a styled plot from compressed benchmark CSV outputs.")
    parser.add_argument("--timing-01", type=Path, required=True, help="Path to timing_0_1_ms.csv")
    parser.add_argument("--timing-12", type=Path, required=True, help="Path to timing_1_2_ms.csv")
    parser.add_argument("--output", type=Path, required=True, help="Output plot path (e.g., benchmark_plot.png)")
    args = parser.parse_args()

    rows = []
    rows_01 = parse_timing_file(args.timing_01, "0->1")
    rows_12 = parse_timing_file(args.timing_12, "1->2")

    build_plot(rows_01, rows_12, args.output)
    print(f"Wrote plot: {args.output}")


if __name__ == "__main__":
    main()
