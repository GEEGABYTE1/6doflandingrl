"""Classical landing baseline based on hover-trim 6DOF LQR."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_continuous_are

from .propulsion import EngineConfig, TVCCommand
from .quaternion_utils import attitude_error_vector


Array = NDArray[np.float64]


def hover_trim_linearization(
    engine: EngineConfig,
    nominal_mass_kg: float,
    inertia_body_kgm2: Array,
) -> tuple[Array, Array]:
    """Return continuous-time small-error matrices around upright hover trim.

    The error state is ``[r_I, v_I, eta_BI, omega_B]`` where ``eta_BI`` is the
    small rotation vector associated with the body-to-inertial quaternion. The
    control perturbation is ``[delta_throttle, pitch_gimbal, yaw_gimbal]``.

    Small-angle equations used here:
    ``ax ~= g * (theta + pitch_gimbal)``,
    ``ay ~= g * (yaw_gimbal - phi)``,
    ``az ~= (Tmax / m) * delta_throttle``,
    ``pdot ~= L * T_hover / Ixx * yaw_gimbal``,
    ``qdot ~= -L * T_hover / Iyy * pitch_gimbal``.
    """
    mass = float(nominal_mass_kg)
    hover_thrust = mass * engine.standard_gravity_mps2
    a = np.zeros((12, 12), dtype=float)
    b = np.zeros((12, 3), dtype=float)

    a[0:3, 3:6] = np.eye(3)
    a[6:9, 9:12] = np.eye(3)

    # Translational coupling from attitude near upright hover.
    a[3, 7] = engine.standard_gravity_mps2
    a[4, 6] = -engine.standard_gravity_mps2

    b[3, 1] = engine.standard_gravity_mps2
    b[4, 2] = engine.standard_gravity_mps2
    b[5, 0] = engine.max_thrust_n / mass

    ixx = float(inertia_body_kgm2[0, 0])
    iyy = float(inertia_body_kgm2[1, 1])
    b[9, 2] = engine.lever_arm_m * hover_thrust / ixx
    b[10, 1] = -engine.lever_arm_m * hover_thrust / iyy
    return a, b


def hover_trim_lqr_gain(
    engine: EngineConfig,
    nominal_mass_kg: float,
    inertia_body_kgm2: Array,
    q_weights: Array,
    r_weights: Array,
) -> Array:
    """Solve the hover-trim LQR on the controllable TVC state subset.

    A single centered TVC engine controls pitch and roll moments through thrust
    vectoring, but it does not generate moment about body ``+z``. The yaw angle
    and yaw rate states are therefore kept in the public 12-state model with
    zero feedback gains rather than forcing an invalid Riccati solve.
    """
    a_full, b_full = hover_trim_linearization(engine, nominal_mass_kg, inertia_body_kgm2)
    controlled_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 9, 10], dtype=int)
    a = a_full[np.ix_(controlled_indices, controlled_indices)]
    b = b_full[controlled_indices, :]
    q = np.diag(q_weights[controlled_indices])
    r = np.diag(r_weights)
    p = solve_continuous_are(a, b, q, r)
    reduced_gain = np.linalg.solve(r, b.T @ p)
    full_gain = np.zeros((3, 12), dtype=float)
    full_gain[:, controlled_indices] = reduced_gain
    return full_gain


@dataclass(frozen=True)
class LandingTarget:
    """Desired terminal landing state."""

    position_inertial_m: Array = field(default_factory=lambda: np.zeros(3, dtype=float))
    velocity_inertial_mps: Array = field(default_factory=lambda: np.zeros(3, dtype=float))


@dataclass
class HoverTrimLQRController:
    """Full hover-trim 6DOF LQR baseline for vertical landing.

    The controller is designed from a small-angle hover linearization of the
    6DOF rocket dynamics. It tracks a descent-rate reference in the vertical
    channel and zero lateral displacement at the pad. This is a stronger
    classical baseline than the previous decoupled controller, while still
    documenting the key approximation: the gain is a local hover-trim gain, not
    a time-varying LQR along a full fuel-varying descent trajectory.
    """

    engine: EngineConfig = field(default_factory=EngineConfig)
    target: LandingTarget = field(default_factory=LandingTarget)
    nominal_mass_kg: float = 1_150.0
    inertia_body_kgm2: Array = field(
        default_factory=lambda: np.diag(np.array([1_200.0, 1_200.0, 120.0], dtype=float))
    )
    q_weights: Array = field(
        default_factory=lambda: np.array(
            [
                1.2,
                1.2,
                0.8,
                2.0,
                2.0,
                3.0,
                35.0,
                35.0,
                0.2,
                8.0,
                8.0,
                0.2,
            ],
            dtype=float,
        )
    )
    r_weights: Array = field(
        default_factory=lambda: np.array([0.8, 35.0, 35.0], dtype=float)
    )
    glide_slope_gain: float = 0.075
    min_descent_rate_mps: float = 0.6
    max_descent_rate_mps: float = 7.0
    max_throttle_delta: float = 0.55
    command_gimbal_limit_rad: float = np.deg2rad(8.0)
    gain: Array = field(init=False)

    def __post_init__(self) -> None:
        """Solve the continuous-time LQR problem for the hover trim."""
        self.gain = hover_trim_lqr_gain(
            self.engine,
            self.nominal_mass_kg,
            self.inertia_body_kgm2,
            self.q_weights,
            self.r_weights,
        )

    def command(self, time_s: float, state: Array) -> TVCCommand:
        """Compute a TVC command from the current simulator state."""
        del time_s
        position = np.asarray(state[0:3], dtype=float)
        velocity = np.asarray(state[3:6], dtype=float)
        q_bi = np.asarray(state[6:10], dtype=float)
        omega_body = np.asarray(state[10:13], dtype=float)
        mass = float(state[13])

        altitude = max(0.0, float(position[2] - self.target.position_inertial_m[2]))
        desired_vz = -float(
            np.clip(
                self.glide_slope_gain * altitude + self.min_descent_rate_mps,
                self.min_descent_rate_mps,
                self.max_descent_rate_mps,
            )
        )
        reference_position = np.array(
            [
                self.target.position_inertial_m[0],
                self.target.position_inertial_m[1],
                position[2],
            ],
            dtype=float,
        )
        reference_velocity = np.array([0.0, 0.0, desired_vz], dtype=float)
        attitude_error = attitude_error_vector(q_bi)
        error_state = np.concatenate(
            (
                position - reference_position,
                velocity - reference_velocity,
                attitude_error,
                omega_body,
            )
        )

        control_delta = -self.gain @ error_state
        hover_throttle = mass * self.engine.standard_gravity_mps2 / self.engine.max_thrust_n
        throttle = hover_throttle + float(
            np.clip(control_delta[0], -self.max_throttle_delta, self.max_throttle_delta)
        )
        pitch = float(control_delta[1])
        yaw = float(control_delta[2])

        return TVCCommand(
            throttle=float(np.clip(throttle, self.engine.min_throttle, self.engine.max_throttle)),
            pitch_rad=float(np.clip(pitch, -self.command_gimbal_limit_rad, self.command_gimbal_limit_rad)),
            yaw_rad=float(np.clip(yaw, -self.command_gimbal_limit_rad, self.command_gimbal_limit_rad)),
        )


SimplifiedLQRController = HoverTrimLQRController
