"""Quaternion utilities for the 6DOF rocket simulator.

Conventions
-----------
Quaternions are scalar-first arrays ``q = [w, x, y, z]``. The simulator uses
``q_BI`` to denote the attitude that maps vectors from the body frame B to the
inertial frame I. Body angular velocity ``omega_B = [p, q, r]`` is expressed in
the body frame, and the kinematic equation is

    q_dot = 0.5 * q_BI ⊗ [0, omega_B].

Frames are documented in ``design_decisions.md`` and repeated in the dynamics
module where state derivatives are computed.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


Array = NDArray[np.float64]


def normalize_quaternion(q: Array) -> Array:
    """Return a unit-length scalar-first quaternion."""
    norm = float(np.linalg.norm(q))
    if norm <= 0.0:
        raise ValueError("Quaternion norm must be positive.")
    return np.asarray(q, dtype=float) / norm


def quaternion_multiply(q1: Array, q2: Array) -> Array:
    """Return the Hamilton product ``q1 ⊗ q2`` for scalar-first quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


def quaternion_conjugate(q: Array) -> Array:
    """Return the conjugate of a scalar-first quaternion."""
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def quaternion_derivative(q_bi: Array, omega_body: Array) -> Array:
    """Compute ``q_dot`` for a body-to-inertial attitude quaternion."""
    omega_quat = np.array([0.0, *omega_body], dtype=float)
    return 0.5 * quaternion_multiply(q_bi, omega_quat)


def rotation_matrix_body_to_inertial(q_bi: Array) -> Array:
    """Return the 3x3 rotation matrix mapping body-frame vectors to inertial."""
    w, x, y, z = normalize_quaternion(q_bi)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def rotate_body_to_inertial(q_bi: Array, vector_body: Array) -> Array:
    """Rotate a vector from body frame B to inertial frame I."""
    return rotation_matrix_body_to_inertial(q_bi) @ np.asarray(vector_body, dtype=float)


def rotate_inertial_to_body(q_bi: Array, vector_inertial: Array) -> Array:
    """Rotate a vector from inertial frame I to body frame B."""
    return rotation_matrix_body_to_inertial(q_bi).T @ np.asarray(vector_inertial, dtype=float)


def attitude_error_vector(q_bi: Array) -> Array:
    """Return a small-angle body attitude error relative to upright identity.

    For the Phase 1 baseline, upright means body ``+z`` aligned with inertial
    ``+z`` and no yaw error. With scalar-first quaternions close to identity,
    ``2 * q_vec`` is the first-order rotation vector.
    """
    q = normalize_quaternion(q_bi)
    if q[0] < 0.0:
        q = -q
    return 2.0 * q[1:4]

