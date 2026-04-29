from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Phase 2A evaluation artifact directory.")
    parser.add_argument(
        "--phase1-dir",
        type=Path,
        default=Path("outputs/phase1_evaluation"),
        help="Phase 1 evaluation root for baseline comparison.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for PNG outputs.")
    return parser.parse_args()

def read_csv_rows(path: Path) -> list[dict[str, float]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]

def main() -> None:
    args = parse_args()
    ppo_rows = read_csv_rows(args.input_dir / "trajectory.csv")
    phase1_rows = read_csv_rows(args.phase1_dir / "runs" / "scenario_table__nominal" / "trajectory.csv")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    t_ppo = np.array([row["time_s"] for row in ppo_rows], dtype=float)
    z_ppo = np.array([row["z_m"] for row in ppo_rows], dtype=float)
    vz_ppo = np.array([row["vz_mps"] for row in ppo_rows], dtype=float)
    lat_ppo = np.sqrt(
        np.square(np.array([row["x_m"] for row in ppo_rows], dtype=float))
        + np.square(np.array([row["y_m"] for row in ppo_rows], dtype=float))
    )

    t_lqr = np.array([row["time_s"] for row in phase1_rows], dtype=float)
    z_lqr = np.array([row["z_m"] for row in phase1_rows], dtype=float)
    vz_lqr = np.array([row["vz_mps"] for row in phase1_rows], dtype=float)
    lat_lqr = np.sqrt(
        np.square(np.array([row["x_m"] for row in phase1_rows], dtype=float))
        + np.square(np.array([row["y_m"] for row in phase1_rows], dtype=float))
    )

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    axes[0, 0].plot(t_ppo, z_ppo, label="PPO")
    axes[0, 0].plot(t_lqr, z_lqr, linestyle="--", label="Phase 1 LQR")
    axes[0, 0].set_xlabel("time [s]")
    axes[0, 0].set_ylabel("altitude [m]")
    axes[0, 0].legend()

    axes[0, 1].plot(t_ppo, vz_ppo, label="PPO")
    axes[0, 1].plot(t_lqr, vz_lqr, linestyle="--", label="Phase 1 LQR")
    axes[0, 1].set_xlabel("time [s]")
    axes[0, 1].set_ylabel("vertical velocity [m/s]")

    axes[1, 0].plot(t_ppo, lat_ppo, label="PPO")
    axes[1, 0].plot(t_lqr, lat_lqr, linestyle="--", label="Phase 1 LQR")
    axes[1, 0].set_xlabel("time [s]")
    axes[1, 0].set_ylabel("lateral error [m]")

    axes[1, 1].plot(
        [row["x_m"] for row in ppo_rows],
        [row["z_m"] for row in ppo_rows],
        label="PPO",
    )
    axes[1, 1].plot(
        [row["x_m"] for row in phase1_rows],
        [row["z_m"] for row in phase1_rows],
        linestyle="--",
        label="Phase 1 LQR",
    )
    axes[1, 1].set_xlabel("x [m]")
    axes[1, 1].set_ylabel("altitude [m]")
    axes[1, 1].legend()

    fig.savefig(args.output_dir / "ppo_vs_phase1_panel.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
