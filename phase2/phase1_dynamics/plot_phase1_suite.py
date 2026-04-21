"""Generate the reproducible Phase 1 paper figure suite from saved artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/.cache/matplotlib").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("outputs/.cache").resolve()))

import matplotlib.pyplot as plt

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from phase1_dynamics.metrics import row_tilt_deg
else:
    from .metrics import row_tilt_deg


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/phase1_evaluation"))
    parser.add_argument("--validation-dir", type=Path, default=Path("outputs/phase1_validation"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase1_figures"))
    return parser.parse_args()


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    """Load raw CSV rows as dictionaries."""
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def load_trajectory(path: Path) -> dict[str, np.ndarray]:
    """Load a trajectory CSV into named arrays."""
    rows = load_csv_rows(path)
    if not rows:
        raise ValueError(f"No rows in {path}")
    data = {key: np.array([float(row[key]) for row in rows], dtype=float) for key in rows[0].keys()}
    if "tilt_deg" not in data:
        data["tilt_deg"] = np.array([row_tilt_deg({k: float(v) for k, v in row.items()}) for row in rows])
    return data


def save_single_axis(
    time: np.ndarray,
    values: list[tuple[np.ndarray, str]],
    ylabel: str,
    title: str,
    path: Path,
) -> Path:
    """Save a simple time-history figure."""
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    for series, label in values:
        ax.plot(time, series, label=label)
    ax.set_xlabel("time [s]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if len(values) > 1:
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_trajectory_evidence_panel(data: dict[str, np.ndarray], output_dir: Path) -> Path:
    """Save the primary paper-facing figure proving touchdown behavior."""
    time = data["time_s"]
    lateral_error = np.hypot(data["x_m"], data["y_m"])
    downrange = np.hypot(data["x_m"] - data["x_m"][0], data["y_m"] - data["y_m"][0])

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.2))
    axes[0, 0].plot(time, data["z_m"])
    axes[0, 0].scatter(time[0], data["z_m"][0], label=f"start z={data['z_m'][0]:.1f} m")
    axes[0, 0].scatter(time[-1], data["z_m"][-1], label=f"touchdown z={data['z_m'][-1]:.1f} m")
    axes[0, 0].set_ylabel("altitude [m]")
    axes[0, 0].set_title("Altitude")
    axes[0, 0].legend(loc="best")

    axes[0, 1].plot(time, data["vz_mps"])
    axes[0, 1].axhline(0.0, linewidth=0.8)
    axes[0, 1].set_ylabel("vertical velocity [m/s]")
    axes[0, 1].set_title("Vertical Velocity")

    axes[1, 0].plot(time, lateral_error)
    axes[1, 0].set_xlabel("time [s]")
    axes[1, 0].set_ylabel("lateral error [m]")
    axes[1, 0].set_title("Lateral Error")

    axes[1, 1].plot(downrange, data["z_m"])
    axes[1, 1].scatter(downrange[0], data["z_m"][0], label="start")
    axes[1, 1].scatter(downrange[-1], data["z_m"][-1], label="touchdown")
    axes[1, 1].set_xlabel("ground-track distance [m]")
    axes[1, 1].set_ylabel("altitude [m]")
    axes[1, 1].set_title("Side View")
    axes[1, 1].legend(loc="best")

    for ax in axes.ravel():
        ax.grid(True, alpha=0.3)
    fig.suptitle("Phase 1 Representative Landing Evidence")
    fig.tight_layout()
    path = output_dir / "trajectory_evidence_panel.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def run_dir(input_dir: Path, run_id: str) -> Path:
    """Return the artifact directory for a run id."""
    return input_dir / "runs" / run_id


def plot_nominal_suite(input_dir: Path, output_dir: Path, nominal_run_id: str) -> list[Path]:
    """Plot trajectory and state histories for the nominal representative run."""
    data = load_trajectory(run_dir(input_dir, nominal_run_id) / "trajectory.csv")
    saved: list[Path] = []
    time = data["time_s"]
    saved.append(save_trajectory_evidence_panel(data, output_dir))

    fig = plt.figure(figsize=(7.0, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_proj_type("ortho")
    points = ax.scatter(
        data["x_m"],
        data["y_m"],
        data["z_m"],
        c=time,
        s=6.0,
        label="trajectory",
    )
    ax.plot(data["x_m"], data["y_m"], data["z_m"], linewidth=1.0, alpha=0.7)
    ax.scatter(data["x_m"][0], data["y_m"][0], data["z_m"][0], marker="o", label="start")
    ax.scatter(data["x_m"][-1], data["y_m"][-1], data["z_m"][-1], marker="x", label="touchdown")
    ax.text(data["x_m"][0], data["y_m"][0], data["z_m"][0], f" start z={data['z_m'][0]:.0f} m")
    ax.text(data["x_m"][-1], data["y_m"][-1], data["z_m"][-1], f" touchdown z={data['z_m'][-1]:.0f} m")
    x_min, x_max = float(np.min(data["x_m"])), float(np.max(data["x_m"]))
    y_min, y_max = float(np.min(data["y_m"])), float(np.max(data["y_m"]))
    ground_x, ground_y = np.meshgrid(
        np.linspace(x_min, x_max, 2),
        np.linspace(y_min, y_max, 2),
    )
    ax.plot_surface(
        ground_x,
        ground_y,
        np.zeros_like(ground_x),
        alpha=0.15,
        linewidth=0.0,
    )
    ax.set_xlabel("x inertial [m]")
    ax.set_ylabel("y inertial [m]")
    ax.set_zlabel("altitude [m]")
    ax.set_zlim(bottom=0.0, top=max(1.0, float(np.max(data["z_m"]))))
    ax.view_init(elev=24.0, azim=-58.0)
    ax.set_title("Supplemental 3D View: Gain-Scheduled LQR Trajectory")
    ax.legend(loc="best")
    fig.colorbar(points, ax=ax, shrink=0.65, label="time [s]")
    path = output_dir / "trajectory_3d.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved.append(path)

    saved.append(save_single_axis(time, [(data["z_m"], "altitude")], "altitude [m]", "Altitude vs Time", output_dir / "altitude_time.png"))
    saved.append(save_single_axis(time, [(data["vz_mps"], "vz")], "vertical velocity [m/s]", "Vertical Velocity vs Time", output_dir / "vertical_velocity_time.png"))
    saved.append(save_single_axis(time, [(data["x_m"], "x"), (data["y_m"], "y")], "lateral position [m]", "Lateral Position vs Time", output_dir / "lateral_position_time.png"))
    saved.append(save_single_axis(time, [(data["tilt_deg"], "tilt")], "tilt [deg]", "Tilt vs Time", output_dir / "tilt_time.png"))
    saved.append(save_single_axis(time, [(data["p_radps"], "p"), (data["q_radps"], "q"), (data["r_radps"], "r")], "angular rate [rad/s]", "Angular Rates vs Time", output_dir / "angular_rates_time.png"))
    saved.append(save_single_axis(time, [(data["throttle"], "throttle"), (np.rad2deg(data["gimbal_pitch_rad"]), "pitch deg"), (np.rad2deg(data["gimbal_yaw_rad"]), "yaw deg")], "control input", "Control Inputs vs Time", output_dir / "control_inputs_time.png"))
    fuel_used = data["mass_kg"][0] - data["mass_kg"]
    saved.append(save_single_axis(time, [(data["mass_kg"], "mass"), (fuel_used, "fuel used")], "kg", "Mass and Fuel vs Time", output_dir / "mass_fuel_time.png"))

    zoom_mask = time >= max(0.0, time[-1] - 5.0)
    saved.append(save_single_axis(time[zoom_mask], [(data["z_m"][zoom_mask], "altitude"), (-data["vz_mps"][zoom_mask], "descent speed")], "m or m/s", "Touchdown Zoom-In", output_dir / "touchdown_zoom.png"))
    return saved


def plot_sweep_summary(input_dir: Path, output_dir: Path) -> list[Path]:
    """Plot mass and disturbance sweep summaries from aggregate metrics."""
    metrics_rows = load_csv_rows(input_dir / "metrics.csv")
    saved: list[Path] = []

    mass_rows = [row for row in metrics_rows if row["group"] == "mass_sweep"]
    fig, ax1 = plt.subplots(figsize=(8.0, 4.8))
    labels = [row["scenario"].replace("mass_", "").replace("kg", "") for row in mass_rows]
    x = np.arange(len(labels))
    ax1.plot(x, [float(row["landing_position_error_m"]) for row in mass_rows], marker="o", label="position error")
    ax1.plot(x, [float(row["touchdown_speed_mps"]) for row in mass_rows], marker="s", label="touchdown speed")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_xlabel("initial mass [kg]")
    ax1.set_ylabel("metric value")
    ax1.set_title("Mass Sweep Summary")
    ax1.legend(loc="best")
    path = output_dir / "mass_sweep_summary.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved.append(path)

    disturbance_rows = [row for row in metrics_rows if row["group"] == "disturbance_sweep"]
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    labels = [row["scenario"] for row in disturbance_rows]
    x = np.arange(len(labels))
    ax.plot(x, [float(row["landing_position_error_m"]) for row in disturbance_rows], marker="o", label="position error")
    ax.plot(x, [float(row["touchdown_speed_mps"]) for row in disturbance_rows], marker="s", label="touchdown speed")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("metric value")
    ax.set_title("Disturbance Sweep Summary")
    ax.legend(loc="best")
    path = output_dir / "disturbance_sweep_summary.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved.append(path)
    return saved


def plot_representatives(input_dir: Path, output_dir: Path) -> list[Path]:
    """Plot representative success and failure cases."""
    with (input_dir / "summary.json").open("r", encoding="utf-8") as stream:
        summary = json.load(stream)
    saved: list[Path] = []
    success_id = summary.get("representative_success_run_id", "")
    if success_id:
        data = load_trajectory(run_dir(input_dir, success_id) / "trajectory.csv")
        saved.append(save_single_axis(data["time_s"], [(data["z_m"], "altitude"), (np.hypot(data["x_m"], data["y_m"]), "lateral error")], "m", f"Representative Success: {success_id}", output_dir / "representative_success.png"))

    failure_id = summary.get("representative_failure_run_id", "")
    if failure_id:
        data = load_trajectory(run_dir(input_dir, failure_id) / "trajectory.csv")
        saved.append(save_single_axis(data["time_s"], [(data["z_m"], "altitude"), (np.hypot(data["x_m"], data["y_m"]), "lateral error")], "m", f"Representative Failure: {failure_id}", output_dir / "representative_failure.png"))
    else:
        fig, ax = plt.subplots(figsize=(7.0, 3.5))
        ax.axis("off")
        ax.text(0.5, 0.5, "No failure observed in configured Phase 1 suite", ha="center", va="center")
        path = output_dir / "representative_failure.png"
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        saved.append(path)
    return saved


def plot_validation(validation_dir: Path, output_dir: Path) -> list[Path]:
    """Plot timestep sensitivity from validation artifacts."""
    rows = load_csv_rows(validation_dir / "timestep_sensitivity.csv")
    x = np.array([float(row["dt_s"]) for row in rows], dtype=float)
    speed = np.array([float(row["touchdown_speed_mps"]) for row in rows], dtype=float)
    error = np.array([float(row["landing_position_error_m"]) for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(x, speed, marker="o", label="touchdown speed")
    ax.plot(x, error, marker="s", label="landing error")
    ax.set_xlabel("RK4 step size [s]")
    ax.set_ylabel("metric value")
    ax.set_title("RK4 Timestep Sensitivity")
    ax.invert_xaxis()
    ax.legend(loc="best")
    path = output_dir / "timestep_sensitivity.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return [path]


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    nominal_id = "scenario_table__nominal"
    saved = []
    saved.extend(plot_nominal_suite(args.input_dir, args.output_dir, nominal_id))
    saved.extend(plot_sweep_summary(args.input_dir, args.output_dir))
    saved.extend(plot_representatives(args.input_dir, args.output_dir))
    saved.extend(plot_validation(args.validation_dir, args.output_dir))
    for path in saved:
        print(path)


if __name__ == "__main__":
    main()
