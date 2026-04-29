#reward shaping
from __future__ import annotations
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray
from phase1_dynamics.metrics import quaternion_tilt_deg

Array = NDArray[np.float64]

@dataclass(frozen=True)
class RewardWeights:
    altitude_progress: float = 0.05
    lateral_progress: float = 0.0
    lateral_error: float = 1.10
    outward_velocity: float = 0.0
    vertical_speed: float = 0.80
    horizontal_speed: float = 1.00
    tilt: float = 0.20
    angular_rate: float = 0.12
    control_effort: float = 0.02
    success_bonus: float = 450.0
    touchdown_failure_penalty: float = 180.0
    timeout_penalty: float = 220.0
    divergence_penalty: float = 320.0
    terminal_vertical_speed: float = 8.0
    terminal_horizontal_speed: float = 12.0
    terminal_lateral_error: float = 10.0
    terminal_tilt: float = 2.0


@dataclass(frozen=True)
class RewardBreakdown:
    altitude_progress: float
    lateral_progress: float
    lateral_penalty: float
    outward_velocity_penalty: float
    vertical_speed_penalty: float
    horizontal_speed_penalty: float
    tilt_penalty: float
    angular_rate_penalty: float
    control_effort_penalty: float
    terminal_bonus: float
    total: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def compute_reward(
    prev_state: Array,
    next_state: Array,
    action: Array,
    terminated: bool,
    truncated: bool,
    touchdown_success: bool,
    divergence: bool,
    weights: RewardWeights | None = None,
) -> RewardBreakdown:
    w = weights or RewardWeights()
    prev_altitude = max(float(prev_state[2]), 0.0)
    next_altitude = max(float(next_state[2]), 0.0)
    prev_lateral_error = float(np.linalg.norm(prev_state[0:2]))
    altitude_progress = prev_altitude - next_altitude
    lateral_error = float(np.linalg.norm(next_state[0:2]))
    vertical_speed = abs(float(next_state[5]))
    horizontal_speed = float(np.linalg.norm(next_state[3:5]))
    tilt_deg = quaternion_tilt_deg(next_state[6:10])
    angular_rate_norm = float(np.linalg.norm(next_state[10:13]))
    control_effort = float(np.linalg.norm(action))
    lateral_progress = prev_lateral_error - lateral_error
    if lateral_error > 1.0e-6:
        radial_direction = next_state[0:2] / lateral_error
        outward_velocity = float(np.dot(radial_direction, next_state[3:5]))
    else:
        outward_velocity = 0.0

    altitude_term = w.altitude_progress * altitude_progress
    lateral_progress_term = w.lateral_progress * lateral_progress
    lateral_term = -w.lateral_error * (lateral_error / 20.0)
    outward_velocity_term = -w.outward_velocity * max(outward_velocity, 0.0)
    vertical_term = -w.vertical_speed * (vertical_speed / 10.0)
    horizontal_term = -w.horizontal_speed * (horizontal_speed / 10.0)
    tilt_term = -w.tilt * (tilt_deg / 15.0)
    angular_rate_term = -w.angular_rate * angular_rate_norm
    control_term = -w.control_effort * control_effort

    terminal_term = 0.0
    if divergence:
        terminal_term -= w.divergence_penalty
    elif terminated:
        terminal_term += w.success_bonus if touchdown_success else -w.touchdown_failure_penalty
    elif truncated:
        terminal_term -= w.timeout_penalty
    total = (
        altitude_term
        + lateral_progress_term
        + lateral_term
        + outward_velocity_term
        + vertical_term
        + horizontal_term
        + tilt_term
        + angular_rate_term
        + control_term
        + terminal_term
    )
    return RewardBreakdown(
        altitude_progress=float(altitude_term),
        lateral_progress=float(lateral_progress_term),
        lateral_penalty=float(lateral_term),
        outward_velocity_penalty=float(outward_velocity_term),
        vertical_speed_penalty=float(vertical_term),
        horizontal_speed_penalty=float(horizontal_term),
        tilt_penalty=float(tilt_term),
        angular_rate_penalty=float(angular_rate_term),
        control_effort_penalty=float(control_term),
        terminal_bonus=float(terminal_term),
        total=float(total),
    )


def terminal_metric_reward(metrics: dict[str, float | str | bool], weights: RewardWeights | None = None) -> float:
    w = weights or RewardWeights()
    if bool(metrics.get("success", False)):
        return 0.0
    vertical_speed = abs(float(metrics.get("vertical_touchdown_velocity_mps", 0.0)))
    horizontal_speed = float(metrics.get("horizontal_touchdown_velocity_mps", 0.0))
    lateral_error = float(metrics.get("landing_position_error_m", 0.0))
    tilt_deg = float(metrics.get("tilt_angle_deg", 0.0))
    penalty = 0.0
    penalty -= w.terminal_vertical_speed * vertical_speed
    penalty -= w.terminal_horizontal_speed * horizontal_speed
    penalty -= w.terminal_lateral_error * lateral_error
    penalty -= w.terminal_tilt * tilt_deg
    return float(penalty)
