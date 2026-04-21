"""Touchdown metrics and failure taxonomy for Phase 1 evaluations."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

from .quaternion_utils import rotation_matrix_body_to_inertial


Array = NDArray[np.float64]


@dataclass(frozen=True)
class SuccessCriteria:
    """Default touchdown success thresholds for Phase 1 paper tables."""

    max_vertical_speed_mps: float = 2.0
    max_horizontal_speed_mps: float = 1.0
    max_lateral_error_m: float = 2.0
    max_tilt_deg: float = 10.0
    max_angular_rate_radps: float = 0.5
    min_fuel_margin_kg: float = 1.0
    max_final_altitude_m: float = 0.05
    max_position_norm_m: float = 10_000.0
    max_speed_norm_mps: float = 1_000.0
    max_saturation_fraction: float = 0.35

    def to_dict(self) -> dict[str, float]:
        """Return JSON-serializable thresholds."""
        return asdict(self)


def quaternion_tilt_deg(q_bi: Array) -> float:
    """Return tilt angle between body ``+z`` and inertial ``+z`` in degrees."""
    body_z_inertial = rotation_matrix_body_to_inertial(np.asarray(q_bi, dtype=float))[:, 2]
    cos_tilt = float(np.clip(body_z_inertial[2], -1.0, 1.0))
    return float(np.rad2deg(np.arccos(cos_tilt)))


def row_tilt_deg(row: dict[str, float]) -> float:
    """Return tilt angle for a saved trajectory row."""
    return quaternion_tilt_deg(np.array([row["qw"], row["qx"], row["qy"], row["qz"]], dtype=float))


def trajectory_arrays(rows: list[dict[str, float]]) -> dict[str, Array]:
    """Convert trajectory rows into named arrays."""
    if not rows:
        raise ValueError("Trajectory rows must be non-empty.")
    return {
        key: np.array([float(row[key]) for row in rows], dtype=float)
        for key in rows[0].keys()
    }


def touchdown_metrics(
    rows: list[dict[str, float]],
    dry_mass_kg: float,
    criteria: SuccessCriteria | None = None,
) -> dict[str, float | str | bool]:
    """Compute touchdown metrics and classify failure modes."""
    limits = criteria or SuccessCriteria()
    data = trajectory_arrays(rows)
    final = {key: float(values[-1]) for key, values in data.items()}
    initial_mass = float(data["mass_kg"][0])
    final_q = np.array([final["qw"], final["qx"], final["qy"], final["qz"]], dtype=float)
    vertical_speed = abs(final["vz_mps"])
    horizontal_speed = float(np.linalg.norm([final["vx_mps"], final["vy_mps"]]))
    lateral_error = float(np.linalg.norm([final["x_m"], final["y_m"]]))
    angular_rate_norm = float(np.linalg.norm([final["p_radps"], final["q_radps"], final["r_radps"]]))
    tilt_deg = quaternion_tilt_deg(final_q)
    throttle_saturation = np.logical_or(data["throttle"] <= 1.0e-8, data["throttle"] >= 1.0 - 1.0e-8)
    gimbal_saturation = np.logical_or(
        np.abs(data["gimbal_pitch_rad"]) >= np.deg2rad(8.0) - 1.0e-8,
        np.abs(data["gimbal_yaw_rad"]) >= np.deg2rad(8.0) - 1.0e-8,
    )
    saturation_fraction = float(np.mean(np.logical_or(throttle_saturation, gimbal_saturation)))
    altitude = data["z_m"]
    late_start = max(0, int(0.5 * len(altitude)))
    oscillation_count = int(np.sum(np.diff(np.signbit(np.diff(altitude[late_start:]))) != 0))

    failure_modes: list[str] = []
    if final["z_m"] > limits.max_final_altitude_m:
        failure_modes.append("no_touchdown")
    if vertical_speed > limits.max_vertical_speed_mps:
        failure_modes.append("hard_touchdown")
    if horizontal_speed > limits.max_horizontal_speed_mps:
        failure_modes.append("horizontal_speed")
    if lateral_error > limits.max_lateral_error_m:
        failure_modes.append("lateral_miss")
    if tilt_deg > limits.max_tilt_deg:
        failure_modes.append("tilt_over")
    if angular_rate_norm > limits.max_angular_rate_radps:
        failure_modes.append("excess_angular_rate")
    if final["mass_kg"] <= dry_mass_kg + limits.min_fuel_margin_kg:
        failure_modes.append("fuel_exhaustion")
    if (
        np.max(np.linalg.norm(np.column_stack((data["x_m"], data["y_m"], data["z_m"])), axis=1))
        > limits.max_position_norm_m
        or np.max(np.linalg.norm(np.column_stack((data["vx_mps"], data["vy_mps"], data["vz_mps"])), axis=1))
        > limits.max_speed_norm_mps
    ):
        failure_modes.append("divergence")
    if saturation_fraction > limits.max_saturation_fraction:
        failure_modes.append("saturation_driven_instability")
    if oscillation_count >= 3:
        failure_modes.append("oscillatory_descent")

    return {
        "success": len(failure_modes) == 0,
        "failure_modes": "none" if not failure_modes else ";".join(failure_modes),
        "touchdown_time_s": final["time_s"],
        "vertical_touchdown_velocity_mps": final["vz_mps"],
        "horizontal_touchdown_velocity_mps": horizontal_speed,
        "touchdown_speed_mps": float(np.linalg.norm([final["vx_mps"], final["vy_mps"], final["vz_mps"]])),
        "tilt_angle_deg": tilt_deg,
        "angular_rate_norm_radps": angular_rate_norm,
        "landing_position_error_m": lateral_error,
        "fuel_used_kg": initial_mass - final["mass_kg"],
        "final_mass_kg": final["mass_kg"],
        "fuel_margin_kg": final["mass_kg"] - dry_mass_kg,
        "max_dynamic_pressure_pa": float(np.max(data["dynamic_pressure_pa"])),
        "saturation_fraction": saturation_fraction,
        "oscillation_count": float(oscillation_count),
        "final_altitude_m": final["z_m"],
    }

