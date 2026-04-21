""" body angular velocity ``omega_B = [p, q, r]`` is expressed in
the body frame. """

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

def normalize_quaternion(q: Array) -> Array:
    norm = float(np.linalg.norm(q))
    if norm <= 0.0:
        raise ValueError("Quaternion norm must be positive.")
    return np.asarray(q, dtype=float) / norm


def quaternion_multiply(q1: Array, q2: Array) -> Array:
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
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def quaternion_derivative(q_bi: Array, omega_body: Array) -> Array:
    omega_quat = np.array([0.0, *omega_body], dtype=float)
    return 0.5 * quaternion_multiply(q_bi, omega_quat)


def rotation_matrix_body_to_inertial(q_bi: Array) -> Array:
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
    return rotation_matrix_body_to_inertial(q_bi) @ np.asarray(vector_body, dtype=float)


def rotate_inertial_to_body(q_bi: Array, vector_inertial: Array) -> Array:
    return rotation_matrix_body_to_inertial(q_bi).T @ np.asarray(vector_inertial, dtype=float)


def attitude_error_vector(q_bi: Array) -> Array:
    """returns a small-angle body attitude error relative to upright identity."""
    q = normalize_quaternion(q_bi)
    if q[0] < 0.0:
        q = -q
    return 2.0 * q[1:4]

