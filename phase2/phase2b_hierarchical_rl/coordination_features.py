#vertical coordination features shared by phase 2b training and inf

from __future__ import annotations
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True)
class CoordinationFeatureConfig:
    vertical_speed_scale_mps: float = 40.0
    flare_altitude_scale_m: float = 30.0
    terminal_descent_rate_mps: float = 0.8
    braking_accel_mps2: float = 1.5
    throttle_delta_limit: float = 0.35
    dt_s: float = 0.05

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @property
    def throttle_rate_scale_per_s(self) -> float:
        return max(self.throttle_delta_limit / max(self.dt_s, 1.0e-6), 1.0e-6)


def flare_reference_vertical_speed(altitude_m: float, config: CoordinationFeatureConfig) -> float:
    altitude_m = max(float(altitude_m), 0.0)
    ref_speed = np.sqrt(
        config.terminal_descent_rate_mps**2 + 2.0 * config.braking_accel_mps2 * altitude_m
    )
    return -float(ref_speed)


def build_coordination_features(
    state: Array,
    throttle: float,
    previous_throttle: float,
    config: CoordinationFeatureConfig,
) -> np.ndarray:
    altitude_m = float(state[2])
    vertical_speed_mps = float(state[5])
    throttle_rate = (float(throttle) - float(previous_throttle)) / max(config.dt_s, 1.0e-6)
    vz_ref = flare_reference_vertical_speed(altitude_m, config)
    tracking_error = vertical_speed_mps - vz_ref
    flare_progress = np.clip(1.0 - altitude_m / max(config.flare_altitude_scale_m, 1.0e-6), 0.0, 1.0)
    return np.array(
        [
            2.0 * float(throttle) - 1.0,
            float(np.clip(throttle_rate / config.throttle_rate_scale_per_s, -1.0, 1.0)),
            float(np.clip(tracking_error / config.vertical_speed_scale_mps, -2.0, 2.0)),
            float(np.clip(vz_ref / config.vertical_speed_scale_mps, -2.0, 2.0)),
            float(flare_progress),
        ],
        dtype=np.float32,
    )

