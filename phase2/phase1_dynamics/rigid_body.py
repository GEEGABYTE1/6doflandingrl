"""Rigid-body 6DOF equations for the rocket landing simulator.

State vector
------------
The simulator state is a 14-element vector:

``[x_I, y_I, z_I, vx_I, vy_I, vz_I, qw, qx, qy, qz, p_B, q_B, r_B, mass]``.

Frame conventions
-----------------
The inertial frame I is locally flat with ``+z_I`` upward. Gravity is
``[0, 0, -g]``. The body frame B has ``+z_B`` along the rocket longitudinal
axis, pointing from engine toward nose. For an upright vehicle at identity
attitude, ``+z_B`` aligns with ``+z_I``. Quaternions are scalar-first and map
body vectors to inertial vectors.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .aero import AerodynamicModel
from .atmosphere import ISAAtmosphere
from .disturbances import DisturbanceModel
from .propulsion import PropulsionModel, TVCCommand
from .quaternion_utils import (
    normalize_quaternion,
    quaternion_derivative,
    rotate_body_to_inertial,
)


Array = NDArray[np.float64]


@dataclass(frozen=True)
class RocketConfig:
    """Mass and inertia properties for the Phase 1 lander."""

    inertia_body_kgm2: Array = field(
        default_factory=lambda: np.diag(np.array([1_200.0, 1_200.0, 120.0], dtype=float))
    )
    gravity_mps2: float = 9.80665
    dry_mass_kg: float = 850.0


@dataclass(frozen=True)
class DerivativeDiagnostics:
    """Diagnostics from one dynamics evaluation."""

    thrust_force_body_n: Array
    thrust_moment_body_nm: Array
    aero_force_inertial_n: Array
    aero_moment_body_nm: Array
    gravity_force_inertial_n: Array
    mass_flow_kgps: float
    dynamic_pressure_pa: float


@dataclass(frozen=True)
class RocketDynamics:
    """Composable 6DOF rocket dynamics model."""

    rocket: RocketConfig = field(default_factory=RocketConfig)
    atmosphere: ISAAtmosphere = field(default_factory=ISAAtmosphere)
    aerodynamics: AerodynamicModel = field(default_factory=AerodynamicModel)
    propulsion: PropulsionModel = field(default_factory=PropulsionModel)
    disturbances: DisturbanceModel = field(default_factory=DisturbanceModel)

    def state_derivative(
        self,
        time_s: float,
        state: Array,
        command: TVCCommand,
        return_diagnostics: bool = False,
    ) -> Array | tuple[Array, DerivativeDiagnostics]:
        """Return the time derivative of the full 6DOF state."""
        position = np.asarray(state[0:3], dtype=float)
        velocity = np.asarray(state[3:6], dtype=float)
        q_bi = normalize_quaternion(np.asarray(state[6:10], dtype=float))
        omega_body = np.asarray(state[10:13], dtype=float)
        mass = max(float(state[13]), self.rocket.dry_mass_kg)

        altitude = max(0.0, float(position[2]))
        atmosphere = self.atmosphere.sample(altitude)
        wind = self.disturbances.wind.velocity(time_s, position)
        misalignment = self.disturbances.thrust_misalignment.angles(time_s)

        prop = self.propulsion.evaluate(command, mass, misalignment)
        aero = self.aerodynamics.evaluate(
            atmosphere.density,
            velocity,
            wind,
            q_bi,
            omega_body,
        )

        thrust_force_inertial = rotate_body_to_inertial(q_bi, prop.force_body_n)
        gravity_force = np.array([0.0, 0.0, -mass * self.rocket.gravity_mps2], dtype=float)
        total_force_inertial = thrust_force_inertial + gravity_force + aero.force_inertial_n
        acceleration = total_force_inertial / mass

        inertia = self.rocket.inertia_body_kgm2
        inertia_inv = np.linalg.inv(inertia)
        total_moment_body = prop.moment_body_nm + aero.moment_body_nm
        # Euler rigid-body equation in body axes: I*w_dot = M - w x (I*w).
        omega_dot = inertia_inv @ (
            total_moment_body - np.cross(omega_body, inertia @ omega_body)
        )

        derivative = np.zeros(14, dtype=float)
        derivative[0:3] = velocity
        derivative[3:6] = acceleration
        derivative[6:10] = quaternion_derivative(q_bi, omega_body)
        derivative[10:13] = omega_dot
        derivative[13] = prop.mass_flow_kgps if mass > self.rocket.dry_mass_kg else 0.0

        if not return_diagnostics:
            return derivative

        diagnostics = DerivativeDiagnostics(
            thrust_force_body_n=prop.force_body_n,
            thrust_moment_body_nm=prop.moment_body_nm,
            aero_force_inertial_n=aero.force_inertial_n,
            aero_moment_body_nm=aero.moment_body_nm,
            gravity_force_inertial_n=gravity_force,
            mass_flow_kgps=prop.mass_flow_kgps,
            dynamic_pressure_pa=aero.dynamic_pressure_pa,
        )
        return derivative, diagnostics

