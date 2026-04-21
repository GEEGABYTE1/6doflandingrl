"""Numerical integration utilities for simulator state propagation."""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from .quaternion_utils import normalize_quaternion


Array = NDArray[np.float64]
DerivativeFunction = Callable[[float, Array], Array]


def rk4_step(derivative_fn: DerivativeFunction, time_s: float, state: Array, dt_s: float) -> Array:
    """Advance ``state`` by one fixed-step fourth-order Runge-Kutta update."""
    y = np.asarray(state, dtype=float)
    dt = float(dt_s)
    k1 = derivative_fn(time_s, y)
    k2 = derivative_fn(time_s + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = derivative_fn(time_s + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = derivative_fn(time_s + dt, y + dt * k3)
    next_state = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    next_state[6:10] = normalize_quaternion(next_state[6:10])
    return next_state

