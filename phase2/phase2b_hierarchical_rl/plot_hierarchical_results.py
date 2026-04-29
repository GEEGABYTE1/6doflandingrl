
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Phase 2B evaluation artifact directory.")
    parser.add_argument(
        "--phase1-dir",
        type=Path,
        default=Path("outputs/phase1_evaluation"),
        help="Phase 1 evaluation root for baseline comparison.",
    )
    parser.add_argument(
        "--phase2a-dir",
        type=Path,
        default=Path("outputs/phase2_rl/eval_go_no_go_baseline_v2_default_seed7_50k"),
        help="Phase 2A flat-PPO evaluation directory for baseline comparison.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for PNG outputs.")
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
    hierarchical_rows = read_csv_rows(args.input_dir / "trajectory.csv")
    phase1_rows = read_csv_rows(args.phase1_dir / "runs" / "scenario_table__nominal" / "trajectory.csv")
    phase2a_rows = read_csv_rows(args.phase2a_dir / "trajectory.csv")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    t_h = np.array([row["time_s"] for row in hierarchical_rows], dtype=float)
    z_h = np.array([row["z_m"] for row in hierarchical_rows], dtype=float)
    vz_h = np.array([row["vz_mps"] for row in hierarchical_rows], dtype=float)
    lat_h = lateral_error(hierarchical_rows)

    t_lqr = np.array([row["time_s"] for row in phase1_rows], dtype=float)
    z_lqr = np.array([row["z_m"] for row in phase1_rows], dtype=float)
    vz_lqr = np.array([row["vz_mps"] for row in phase1_rows], dtype=float)
    lat_lqr = lateral_error(phase1_rows)

    t_flat = np.array([row["time_s"] for row in phase2a_rows], dtype=float)
    z_flat = np.array([row["z_m"] for row in phase2a_rows], dtype=float)
    vz_flat = np.array([row["vz_mps"] for row in phase2a_rows], dtype=float)
    lat_flat = lateral_error(phase2a_rows)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), constrained_layout=True)
    axes[0, 0].plot(t_h, z_h, label="Phase 2B hierarchical PPO")
    axes[0, 0].plot(t_flat, z_flat, linestyle="--", label="Phase 2A flat PPO")
    axes[0, 0].plot(t_lqr, z_lqr, linestyle=":", label="Phase 1 LQR")
    axes[0, 0].set_xlabel("time [s]")
    axes[0, 0].set_ylabel("altitude [m]")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(t_h, vz_h, label="Hierarchical PPO")
    axes[0, 1].plot(t_flat, vz_flat, linestyle="--", label="Flat PPO")
    axes[0, 1].plot(t_lqr, vz_lqr, linestyle=":", label="LQR")
    axes[0, 1].set_xlabel("time [s]")
    axes[0, 1].set_ylabel("vertical velocity [m/s]")

    axes[1, 0].plot(t_h, lat_h, label="Hierarchical PPO")
    axes[1, 0].plot(t_flat, lat_flat, linestyle="--", label="Flat PPO")
    axes[1, 0].plot(t_lqr, lat_lqr, linestyle=":", label="LQR")
    axes[1, 0].set_xlabel("time [s]")
    axes[1, 0].set_ylabel("lateral error [m]")

    axes[1, 1].plot(
        [row["x_m"] for row in hierarchical_rows],
        [row["z_m"] for row in hierarchical_rows],
        label="Hierarchical PPO",
    )
    axes[1, 1].plot(
        [row["x_m"] for row in phase2a_rows],
        [row["z_m"] for row in phase2a_rows],
        linestyle="--",
        label="Flat PPO",
    )
    axes[1, 1].plot(
        [row["x_m"] for row in phase1_rows],
        [row["z_m"] for row in phase1_rows],
        linestyle=":",
        label="LQR",
    )
    axes[1, 1].set_xlabel("x [m]")
    axes[1, 1].set_ylabel("altitude [m]")
    axes[1, 1].legend(fontsize=8)

    fig.savefig(args.output_dir / "hierarchical_vs_baselines_panel.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
