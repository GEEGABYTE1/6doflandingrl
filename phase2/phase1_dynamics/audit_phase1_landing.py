"""Audit whether a saved Phase 1 trajectory represents a complete landing."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/.cache/matplotlib").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("outputs/.cache").resolve()))

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=Path("outputs/phase1_evaluation/runs/scenario_table__nominal/trajectory.csv"),
        help="Saved trajectory CSV to audit.",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("outputs/phase1_evaluation/runs/scenario_table__nominal/metrics.json"),
        help="Saved metrics JSON for the trajectory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/phase1_audit"),
        help="Directory for audit plots and final-state summary.",
    )
    return parser.parse_args()


def load_trajectory(path: Path) -> dict[str, np.ndarray]:
    """Load a trajectory CSV into arrays."""
    with path.open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return {key: np.array([float(row[key]) for row in rows], dtype=float) for key in rows[0].keys()}


def save_time_plot(time: np.ndarray, value: np.ndarray, ylabel: str, title: str, path: Path) -> None:
    """Save one time-history plot."""
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.plot(time, value)
    ax.set_xlabel("time [s]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_trajectory(args.trajectory)
    with args.metrics.open("r", encoding="utf-8") as stream:
        metrics = json.load(stream)

    final_summary = {
        "trajectory": str(args.trajectory),
        "metrics": str(args.metrics),
        "initial_altitude_m": float(data["z_m"][0]),
        "max_altitude_m": float(np.max(data["z_m"])),
        "final_altitude_m": float(data["z_m"][-1]),
        "final_vertical_velocity_mps": float(data["vz_mps"][-1]),
        "final_horizontal_velocity_mps": float(np.hypot(data["vx_mps"][-1], data["vy_mps"][-1])),
        "final_tilt_deg": float(data["tilt_deg"][-1]) if "tilt_deg" in data else float(metrics["tilt_angle_deg"]),
        "final_angular_rate_norm_radps": float(
            np.linalg.norm([data["p_radps"][-1], data["q_radps"][-1], data["r_radps"][-1]])
        ),
        "success": bool(metrics["success"]),
        "failure_modes": metrics["failure_modes"],
        "termination_reason": metrics.get("termination_reason", "unknown"),
        "touchdown_detected": bool(metrics.get("touchdown_detected", metrics.get("final_altitude_m", 1.0) <= 0.05)),
        "yaw_excluded_from_success_criteria": bool(metrics.get("yaw_excluded_from_success_criteria", True)),
    }
    with (args.output_dir / "final_state_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(final_summary, stream, indent=2, sort_keys=True)

    save_time_plot(
        data["time_s"],
        data["z_m"],
        "altitude [m]",
        "Phase 1 Representative Altitude vs Time",
        args.output_dir / "altitude_vs_time.png",
    )
    save_time_plot(
        data["time_s"],
        data["vz_mps"],
        "vertical velocity [m/s]",
        "Phase 1 Representative Vertical Velocity vs Time",
        args.output_dir / "vertical_velocity_vs_time.png",
    )
    print(json.dumps(final_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
