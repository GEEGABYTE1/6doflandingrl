from __future__ import annotations
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_rows(args.input_dir / "summary_by_controller_level.csv")
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["controller_label"], []).append(row)

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.8), constrained_layout=True)
    metric_specs = [
        ("landing_position_error_m_mean", "landing error [m]"),
        ("vertical_touchdown_velocity_mps_mean", "vertical touchdown velocity [m/s]"),
        ("fuel_used_kg_mean", "fuel used [kg]"),
    ]
    for ax, (metric, ylabel) in zip(axes, metric_specs):
        for label, group in grouped.items():
            ordered = sorted(group, key=lambda row: int(row["disturbance_index"]))
            ax.plot(
                [int(row["disturbance_index"]) for row in ordered],
                [float(row[metric]) for row in ordered],
                marker="o",
                label=label,
            )
        ax.set_xlabel("disturbance level")
        ax.set_ylabel(ylabel)
        ax.set_xticks([0, 1, 2, 3])
    axes[0].legend(fontsize=8)
    fig.savefig(args.output_dir / "dispersion_panel.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()

