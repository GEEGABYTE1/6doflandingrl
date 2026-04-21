"""aerodynamic force and moment model """

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray
from .quaternion_utils import rotate_body_to_inertial, rotate_inertial_to_body

Array = NDArray[np.float64]

@dataclass(frozen=True)
class AeroConfig:
    reference_area_m2: float = 1.2
    drag_coefficient: float = 0.65
    reference_length_m: float = 4.0
    center_of_pressure_body_m: Array = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.75], dtype=float)
    )
    rotational_damping_coefficients: Array = field(
        default_factory=lambda: np.array([0.08, 0.08, 0.02], dtype=float)
    )
@dataclass(frozen=True)
class AeroOutput:
    force_inertial_n: Array
    moment_body_nm: Array
    dynamic_pressure_pa: float
    relative_velocity_inertial_mps: Array

@dataclass(frozen=True)
class AerodynamicModel:
    config: AeroConfig = field(default_factory=AeroConfig)
    def evaluate(
        self,
        density_kgpm3: float,
        velocity_inertial_mps: Array,
        wind_inertial_mps: Array,
        q_bi: Array,
        omega_body_radps: Array,
    ) -> AeroOutput:
        rel_velocity = np.asarray(velocity_inertial_mps, dtype=float) - np.asarray(
            wind_inertial_mps, dtype=float
        )
        speed = float(np.linalg.norm(rel_velocity))
        q_dyn = 0.5 * density_kgpm3 * speed * speed
        if speed > 1.0e-9:
            drag_force_inertial = (
                -0.5
                * density_kgpm3
                * self.config.drag_coefficient
                * self.config.reference_area_m2
                * speed
                * rel_velocity
            )
        else:
            drag_force_inertial = np.zeros(3, dtype=float)
        drag_force_body = rotate_inertial_to_body(q_bi, drag_force_inertial)
        cp_moment = np.cross(self.config.center_of_pressure_body_m, drag_force_body)
        damping = (
            -self.config.rotational_damping_coefficients
            * q_dyn
            * self.config.reference_area_m2
            * self.config.reference_length_m
            * np.asarray(omega_body_radps, dtype=float)
        )
        moment_body = cp_moment + damping
        return AeroOutput(
            force_inertial_n=drag_force_inertial,
            moment_body_nm=moment_body,
            dynamic_pressure_pa=float(q_dyn),
            relative_velocity_inertial_mps=rel_velocity,
        )

