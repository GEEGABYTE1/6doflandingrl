from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--phase1-dir", type=Path, default=Path("outputs/phase1_evaluation"))
    parser.add_argument("--phase2a-dir", type=Path, default=Path("outputs/phase2_rl/eval_go_no_go_baseline_v2_default_seed7_50k"))
    parser.add_argument("--phase2b-dir", type=Path, default=Path("outputs/phase2b_hierarchical_rl/eval_nominal_seed7_40k_50k_flare_tracking_v1"))
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()

def read_csv_rows(path: Path) -> list[dict[str, float]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]

def lateral_error(rows: list[dict[str, float]]) -> np.ndarray:
    return np.sqrt(
        np.square(np.array([row["x_m"] for row in rows], dtype=float))
        + np.square(np.array([row["y_m"] for row in rows], dtype=float))
    )

def main() -> None:
    args = parse_args()
    hybrid_rows = read_csv_rows(args.input_dir / "trajectory.csv")
    phase2b_rows = read_csv_rows(args.phase2b_dir / "trajectory.csv")
    phase2a_rows = read_csv_rows(args.phase2a_dir / "trajectory.csv")
    phase1_rows = read_csv_rows(args.phase1_dir / "runs" / "scenario_table__nominal" / "trajectory.csv")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), constrained_layout=True)
    for rows, label, style in [
        (hybrid_rows, "Phase 2C hybrid residual", "-"),
        (phase2b_rows, "Phase 2B hierarchical", "--"),
        (phase2a_rows, "Phase 2A flat PPO", "-."),
        (phase1_rows, "Phase 1 LQR", ":"),
    ]:
        t = np.array([row["time_s"] for row in rows], dtype=float)
        z = np.array([row["z_m"] for row in rows], dtype=float)
        vz = np.array([row["vz_mps"] for row in rows], dtype=float)
        lat = lateral_error(rows)
        axes[0, 0].plot(t, z, linestyle=style, label=label)
        axes[0, 1].plot(t, vz, linestyle=style, label=label)
        axes[1, 0].plot(t, lat, linestyle=style, label=label)
        axes[1, 1].plot([row["x_m"] for row in rows], [row["z_m"] for row in rows], linestyle=style, label=label)

    axes[0, 0].set_xlabel("time [s]")
    axes[0, 0].set_ylabel("altitude [m]")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].set_xlabel("time [s]")
    axes[0, 1].set_ylabel("vertical velocity [m/s]")
    axes[1, 0].set_xlabel("time [s]")
    axes[1, 0].set_ylabel("lateral error [m]")
    axes[1, 1].set_xlabel("x [m]")
    axes[1, 1].set_ylabel("altitude [m]")
    fig.savefig(args.output_dir / "hybrid_vs_baselines_panel.png", dpi=200)
    plt.close(fig)

if __name__ == "__main__":
    main()
