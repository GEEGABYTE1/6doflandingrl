from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True)
class ObservationNormalizer:
    #mapping sim state to a dimensionless rl observation vector
    position_scale_m: Array = field(default_factory=lambda: np.array([50.0, 50.0, 150.0], dtype=float))
    velocity_scale_mps: Array = field(default_factory=lambda: np.array([20.0, 20.0, 40.0], dtype=float))
    angular_rate_scale_radps: Array = field(default_factory=lambda: np.array([1.0, 1.0, 1.0], dtype=float))

    def encode(
        self,
        state: Array,
        target_position_m: Array,
        target_velocity_mps: Array,
        dry_mass_kg: float,
        initial_mass_kg: float,
    ) -> Array:
        relative_position = (state[0:3] - target_position_m) / self.position_scale_m
        relative_velocity = (state[3:6] - target_velocity_mps) / self.velocity_scale_mps
        quaternion = np.asarray(state[6:10], dtype=float)
        angular_rate = state[10:13] / self.angular_rate_scale_radps
        fuel_capacity = max(initial_mass_kg - dry_mass_kg, 1.0)
        fuel_fraction = np.clip((state[13] - dry_mass_kg) / fuel_capacity, 0.0, 1.0)
        return np.concatenate(
            (
                relative_position,
                relative_velocity,
                quaternion,
                angular_rate,
                np.array([fuel_fraction], dtype=float),
            )
        ).astype(np.float32)

