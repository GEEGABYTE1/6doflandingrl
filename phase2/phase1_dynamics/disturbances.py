"""disturbance models"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

@dataclass(frozen=True)
class WindModel:
    steady_wind_inertial_mps: Array = field(default_factory=lambda: np.zeros(3, dtype=float))
    gust_amplitude_mps: Array = field(default_factory=lambda: np.zeros(3, dtype=float))
    gust_frequency_hz: float = 0.0

    def velocity(self, time_s: float, position_inertial_m: Array) -> Array:
        """Return inertial wind velocity at the current time and position."""
        del position_inertial_m
        gust = self.gust_amplitude_mps * np.sin(2.0 * np.pi * self.gust_frequency_hz * time_s)
        return np.asarray(self.steady_wind_inertial_mps, dtype=float) + gust

@dataclass(frozen=True)
class ThrustMisalignmentModel:
    #fixed thrust direction bias expressed as pitch/yaw gimbal offsets.

    pitch_bias_rad: float = 0.0
    yaw_bias_rad: float = 0.0

    def angles(self, time_s: float) -> Array:
        del time_s
        return np.array([self.pitch_bias_rad, self.yaw_bias_rad], dtype=float)

@dataclass
class SensorNoiseModel:
    #gaussian sensor noise

    position_std_m: float = 0.0
    velocity_std_mps: float = 0.0
    attitude_std_rad: float = 0.0
    angular_rate_std_radps: float = 0.0
    seed: int = 0
    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def noisy_state(self, state: Array) -> Array:
        noisy = np.array(state, dtype=float, copy=True)
        noisy[0:3] += self.rng.normal(0.0, self.position_std_m, size=3)
        noisy[3:6] += self.rng.normal(0.0, self.velocity_std_mps, size=3)
        noisy[6:10] += self.rng.normal(0.0, self.attitude_std_rad, size=4)
        noisy[10:13] += self.rng.normal(0.0, self.angular_rate_std_radps, size=3)
        return noisy


@dataclass(frozen=True)
class DisturbanceModel:
    wind: WindModel = field(default_factory=WindModel)
    thrust_misalignment: ThrustMisalignmentModel = field(default_factory=ThrustMisalignmentModel)
    sensor_noise: SensorNoiseModel = field(default_factory=SensorNoiseModel)

