"""Classical landing baseline based on simplified LQR gains."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_continuous_are

from .propulsion import EngineConfig, TVCCommand
from .quaternion_utils import attitude_error_vector


Array = NDArray[np.float64]


def double_integrator_lqr(q_position: float, q_velocity: float, r_accel: float) -> Array:
    """Return LQR feedback gains for ``x_dot = v, v_dot = u``."""
    a = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=float)
    b = np.array([[0.0], [1.0]], dtype=float)
    q = np.diag([q_position, q_velocity])
    r = np.array([[r_accel]], dtype=float)
    p = solve_continuous_are(a, b, q, r)
    gain = np.linalg.solve(r, b.T @ p)
    return gain.reshape(2)


@dataclass(frozen=True)
class LandingTarget:
    """Desired terminal landing state."""

    position_inertial_m: Array = field(default_factory=lambda: np.zeros(3, dtype=float))
    velocity_inertial_mps: Array = field(default_factory=lambda: np.zeros(3, dtype=float))


@dataclass
class SimplifiedLQRController:
    """Stabilizing classical baseline for vertical landing.

    This controller uses continuous-time LQR gains from decoupled translational
    double-integrator approximations. Lateral acceleration commands are mapped
    to gimbal angles, while vertical acceleration commands are mapped to
    throttle. Attitude rates are damped with a PD term. This is intentionally
    documented as a Phase 1 baseline rather than a full nonlinear TVC LQR.
    """

    engine: EngineConfig = field(default_factory=EngineConfig)
    target: LandingTarget = field(default_factory=LandingTarget)
    vertical_gain: Array = field(default_factory=lambda: double_integrator_lqr(0.04, 0.8, 1.0))
    lateral_gain: Array = field(default_factory=lambda: double_integrator_lqr(0.03, 0.5, 2.0))
    attitude_kp_nm_per_rad: float = 6_000.0
    attitude_kd_nm_per_radps: float = 3_500.0
    lateral_tvc_scale: float = 0.0
    vertical_velocity_gain: float = 2.2
    glide_slope_gain: float = 0.09
    min_descent_rate_mps: float = 0.6
    max_descent_rate_mps: float = 10.0
    max_vertical_accel_mps2: float = 18.0
    command_gimbal_limit_rad: float = np.deg2rad(8.0)

    def command(self, time_s: float, state: Array) -> TVCCommand:
        """Compute a TVC command from the current simulator state."""
        del time_s
        position = np.asarray(state[0:3], dtype=float)
        velocity = np.asarray(state[3:6], dtype=float)
        q_bi = np.asarray(state[6:10], dtype=float)
        omega_body = np.asarray(state[10:13], dtype=float)
        mass = float(state[13])

        pos_error = position - self.target.position_inertial_m
        vel_error = velocity - self.target.velocity_inertial_mps

        lateral_accel_cmd = np.zeros(2, dtype=float)
        for axis in range(2):
            lateral_accel_cmd[axis] = -float(
                self.lateral_gain @ np.array([pos_error[axis], vel_error[axis]], dtype=float)
            )

        # A pure double-integrator regulator to z=0 can command additional
        # downward acceleration when the vehicle is high above the pad. For a
        # landing baseline we keep the LQR gains for the lateral channel and
        # use a glide-slope vertical target that brakes hard when descending
        # faster than the altitude-dependent reference speed.
        altitude = max(0.0, float(position[2] - self.target.position_inertial_m[2]))
        desired_vz = -float(
            np.clip(
                self.glide_slope_gain * altitude + self.min_descent_rate_mps,
                self.min_descent_rate_mps,
                self.max_descent_rate_mps,
            )
        )
        vertical_accel_cmd = self.vertical_velocity_gain * (desired_vz - velocity[2])
        vertical_accel_cmd = float(
            np.clip(vertical_accel_cmd, -self.max_vertical_accel_mps2, self.max_vertical_accel_mps2)
        )

        required_thrust = mass * (self.engine.standard_gravity_mps2 + vertical_accel_cmd)
        throttle = required_thrust / self.engine.max_thrust_n

        # Small-angle thrust tilt: lateral acceleration ~= g * tilt angle near hover.
        pitch = self.lateral_tvc_scale * lateral_accel_cmd[0] / max(
            self.engine.standard_gravity_mps2, 1.0
        )
        yaw = self.lateral_tvc_scale * lateral_accel_cmd[1] / max(
            self.engine.standard_gravity_mps2, 1.0
        )

        attitude_error = attitude_error_vector(q_bi)
        thrust_estimate = max(throttle * self.engine.max_thrust_n, 1.0)
        moment_arm = max(self.engine.lever_arm_m * thrust_estimate, 1.0)
        desired_moment_x = (
            -self.attitude_kp_nm_per_rad * attitude_error[0]
            - self.attitude_kd_nm_per_radps * omega_body[0]
        )
        desired_moment_y = (
            -self.attitude_kp_nm_per_rad * attitude_error[1]
            - self.attitude_kd_nm_per_radps * omega_body[1]
        )
        # For r_engine=[0,0,-L], small angles give Mx ~= L*T*yaw
        # and My ~= -L*T*pitch.
        yaw += desired_moment_x / moment_arm
        pitch += -desired_moment_y / moment_arm

        return TVCCommand(
            throttle=float(np.clip(throttle, self.engine.min_throttle, self.engine.max_throttle)),
            pitch_rad=float(np.clip(pitch, -self.command_gimbal_limit_rad, self.command_gimbal_limit_rad)),
            yaw_rad=float(np.clip(yaw, -self.command_gimbal_limit_rad, self.command_gimbal_limit_rad)),
        )
