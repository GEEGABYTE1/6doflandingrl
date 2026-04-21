
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/phase1_evaluation"))
    parser.add_argument("--validation-dir", type=Path, default=Path("outputs/phase1_validation"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase1_figures"))
    return parser.parse_args()


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def load_trajectory(path: Path) -> dict[str, np.ndarray]:
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


def save_3d_trajectory_view(
    data: dict[str, np.ndarray],
    output_dir: Path,
    filename: str,
    elev_deg: float,
    azim_deg: float,
    title_suffix: str,
    include_side_projections: bool = False,
    selected: bool = False,
) -> Path:
    time = data["time_s"]
    x = data["x_m"]
    y = data["y_m"]
    z = data["z_m"]
    if not np.all(np.diff(time) > 0.0):
        raise ValueError("Trajectory time stamps must be strictly increasing for 3D plotting.")
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    z_min, z_max = 0.0, max(1.0, float(np.max(z)))
    x_center = 0.5 * (x_min + x_max)
    y_center = 0.5 * (y_min + y_max)
    z_center = 0.5 * (z_min + z_max)
    data_range = max(x_max - x_min, y_max - y_min, z_max - z_min, 1.0)

    half_range = 0.55 * data_range
    x_plot_min, x_plot_max = x_center - half_range, x_center + half_range
    y_plot_min, y_plot_max = y_center - half_range, y_center + half_range
    z_plot_min, z_plot_max = max(0.0, z_center - half_range), z_center + half_range

    fig = plt.figure(figsize=(8.0, 6.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_proj_type("ortho")
    ax.plot(x, y, z, color="0.12", linewidth=1.6)

    # faint projection anchors the curve to the actual ground track.
    ax.plot(x, y, np.zeros_like(z), color="0.55", alpha=0.35, linewidth=1.0, linestyle="--")
    if include_side_projections:
        ax.plot(x, np.full_like(y, y_plot_max), z, color="0.65", alpha=0.20, linewidth=0.9)
        ax.plot(np.full_like(x, x_plot_min), y, z, color="0.65", alpha=0.20, linewidth=0.9)

    marker_indices = np.unique(np.round(np.linspace(0, len(time) - 1, 5)).astype(int))
    for idx in marker_indices:
        ax.scatter(x[idx], y[idx], z[idx], s=22.0, color="0.12", edgecolors="white", linewidths=0.5)

    ax.scatter(x[0], y[0], z[0], marker="o", s=46.0, color="white", edgecolors="0.05", linewidths=1.2)
    ax.scatter(x[-1], y[-1], z[-1], marker="x", s=58.0, color="0.05", linewidths=1.6)
    ax.text(x[0] + 0.03 * data_range, y[0], z[0] - 0.04 * data_range, "start", fontsize=7, color="0.10")
    ax.text(
        x[-1] + 0.03 * data_range,
        y[-1] + 0.04 * data_range,
        z[-1] + 0.04 * data_range,
        "touchdown",
        fontsize=7,
        color="0.10",
    )
    ground_x, ground_y = np.meshgrid(
        np.linspace(x_plot_min, x_plot_max, 2),
        np.linspace(y_plot_min, y_plot_max, 2),
    )
    ax.plot_wireframe(ground_x, ground_y, np.zeros_like(ground_x), color="0.82", linewidth=0.7, alpha=0.8)

    ax.set_xlim(x_plot_min, x_plot_max)
    ax.set_ylim(y_plot_min, y_plot_max)
    ax.set_zlim(z_plot_min, z_plot_max)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.view_init(elev=elev_deg, azim=azim_deg)
    ax.set_xlabel("x [m]", labelpad=14)
    ax.set_ylabel("y [m]", labelpad=14)
    ax.set_zlabel("altitude [m]", labelpad=18)
    ax.set_xticks(np.linspace(x_plot_min, x_plot_max, 5))
    ax.set_yticks(np.linspace(y_plot_min, y_plot_max, 5))
    ax.set_zticks(np.linspace(z_plot_min, z_plot_max, 5))
    if not selected:
        ax.set_title(f"Supplementary 3D candidate {title_suffix}", pad=12)
    ax.grid(True, alpha=0.20)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_alpha(0.02)
        axis._axinfo["grid"]["color"] = (0.72, 0.72, 0.72, 0.35)
        axis._axinfo["grid"]["linewidth"] = 0.6
    ax.tick_params(labelsize=7, pad=2)
    path = output_dir / filename
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def run_dir(input_dir: Path, run_id: str) -> Path:
    return input_dir / "runs" / run_id


def plot_nominal_suite(input_dir: Path, output_dir: Path, nominal_run_id: str) -> list[Path]:
    data = load_trajectory(run_dir(input_dir, nominal_run_id) / "trajectory.csv")
    saved: list[Path] = []
    time = data["time_s"]
    saved.append(save_trajectory_evidence_panel(data, output_dir))

    saved.append(save_3d_trajectory_view(data, output_dir, "trajectory_3d_candidate_a.png", 18.0, -58.0, "A"))
    saved.append(save_3d_trajectory_view(data, output_dir, "trajectory_3d_candidate_b.png", 10.0, -88.0, "B"))
    saved.append(save_3d_trajectory_view(data, output_dir, "trajectory_3d_candidate_c.png", 28.0, -38.0, "C", include_side_projections=True))
    saved.append(save_3d_trajectory_view(data, output_dir, "trajectory_3d_supplement.png", 12.0, -72.0, "selected", selected=True))
    saved.append(save_3d_trajectory_view(data, output_dir, "trajectory_3d.png", 12.0, -72.0, "selected", selected=True))

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
