"""Named Phase 1 evaluation scenarios."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from numpy.typing import NDArray

from .disturbances import DisturbanceModel, SensorNoiseModel, ThrustMisalignmentModel, WindModel
from .quaternion_utils import normalize_quaternion


Array = NDArray[np.float64]


@dataclass(frozen=True)
class ScenarioConfig:
    """Initial condition and disturbance configuration for one evaluation run."""

    name: str
    description: str
    initial_position_m: tuple[float, float, float] = (12.0, -8.0, 120.0)
    initial_velocity_mps: tuple[float, float, float] = (-1.0, 0.6, -18.0)
    initial_attitude_error_deg: tuple[float, float, float] = (2.0, -1.5, 0.5)
    initial_mass_kg: float = 1_250.0
    wind_mps: tuple[float, float, float] = (1.5, -0.5, 0.0)
    gust_mps: tuple[float, float, float] = (0.4, 0.2, 0.0)
    gust_frequency_hz: float = 0.08
    misalignment_deg: tuple[float, float] = (0.15, -0.10)
    duration_s: float = 45.0
    dt_s: float = 0.02

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable scenario dictionary."""
        return asdict(self)

    def initial_state(self, seed: int) -> Array:
        """Return the deterministic 14-state initial condition for this scenario."""
        rng = np.random.default_rng(seed)
        state = np.zeros(14, dtype=float)
        state[0:3] = np.array(self.initial_position_m, dtype=float)
        state[3:6] = np.array(self.initial_velocity_mps, dtype=float)
        attitude = np.deg2rad(np.array(self.initial_attitude_error_deg, dtype=float))
        state[6:10] = normalize_quaternion(np.array([1.0, *attitude], dtype=float))
        state[10:13] = rng.normal(0.0, 0.015, size=3)
        state[13] = self.initial_mass_kg
        return state

    def disturbances(self, seed: int) -> DisturbanceModel:
        """Return modular disturbance models for this scenario."""
        return DisturbanceModel(
            wind=WindModel(
                steady_wind_inertial_mps=np.array(self.wind_mps, dtype=float),
                gust_amplitude_mps=np.array(self.gust_mps, dtype=float),
                gust_frequency_hz=self.gust_frequency_hz,
            ),
            thrust_misalignment=ThrustMisalignmentModel(
                pitch_bias_rad=float(np.deg2rad(self.misalignment_deg[0])),
                yaw_bias_rad=float(np.deg2rad(self.misalignment_deg[1])),
            ),
            sensor_noise=SensorNoiseModel(seed=seed),
        )


def named_scenarios() -> list[ScenarioConfig]:
    """Return the Phase 1 nominal and off-nominal scenario set."""
    return [
        ScenarioConfig(name="nominal", description="Nominal descent with mild wind and thrust bias."),
        ScenarioConfig(
            name="heavy",
            description="Heavier initial mass near upper fuel state.",
            initial_mass_kg=1_350.0,
        ),
        ScenarioConfig(
            name="light",
            description="Lower initial fuel state with reduced touchdown margin.",
            initial_mass_kg=950.0,
        ),
        ScenarioConfig(
            name="crosswind",
            description="Persistent crosswind from inertial +x.",
            wind_mps=(5.0, 0.0, 0.0),
            gust_mps=(0.0, 0.0, 0.0),
        ),
        ScenarioConfig(
            name="gust",
            description="Sinusoidal gusting wind in lateral axes.",
            wind_mps=(2.0, -1.0, 0.0),
            gust_mps=(2.0, 1.0, 0.0),
            gust_frequency_hz=0.12,
        ),
        ScenarioConfig(
            name="misalignment",
            description="One-degree fixed thrust misalignment.",
            misalignment_deg=(1.0, -1.0),
        ),
        ScenarioConfig(
            name="high_lateral_offset",
            description="Large initial lateral miss distance.",
            initial_position_m=(35.0, -25.0, 120.0),
        ),
        ScenarioConfig(
            name="high_descent_rate",
            description="Initial vertical speed faster than nominal.",
            initial_velocity_mps=(-1.0, 0.6, -24.0),
        ),
        ScenarioConfig(
            name="combined_off_nominal",
            description="Combined mass, wind, misalignment, and lateral offset challenge.",
            initial_position_m=(30.0, -20.0, 130.0),
            initial_velocity_mps=(-1.5, 1.0, -22.0),
            initial_mass_kg=1_325.0,
            wind_mps=(6.0, -2.0, 0.0),
            gust_mps=(1.5, 1.0, 0.0),
            misalignment_deg=(0.75, -0.75),
        ),
    ]


def mass_sweep_scenarios() -> list[ScenarioConfig]:
    """Return default initial-mass sweep scenarios."""
    return [
        ScenarioConfig(
            name=f"mass_{int(mass)}kg",
            description=f"Mass sweep initial mass {mass:.0f} kg.",
            initial_mass_kg=float(mass),
        )
        for mass in [950.0, 1_050.0, 1_150.0, 1_250.0, 1_350.0]
    ]


def disturbance_sweep_scenarios() -> list[ScenarioConfig]:
    """Return wind and thrust-misalignment severity sweep scenarios."""
    scenarios: list[ScenarioConfig] = []
    for wind in [0.0, 2.0, 5.0, 8.0]:
        scenarios.append(
            ScenarioConfig(
                name=f"wind_{wind:.0f}mps",
                description=f"Disturbance sweep steady crosswind {wind:.0f} m/s.",
                wind_mps=(float(wind), 0.0, 0.0),
                gust_mps=(0.0, 0.0, 0.0),
                misalignment_deg=(0.0, 0.0),
            )
        )
    for angle in [0.0, 0.25, 0.5, 1.0]:
        scenarios.append(
            ScenarioConfig(
                name=f"misalign_{angle:.2f}deg".replace(".", "p"),
                description=f"Disturbance sweep fixed thrust misalignment {angle:.2f} deg.",
                wind_mps=(0.0, 0.0, 0.0),
                gust_mps=(0.0, 0.0, 0.0),
                misalignment_deg=(float(angle), -float(angle)),
            )
        )
    return scenarios

