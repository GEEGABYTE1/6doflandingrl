#Gym dependent phase 2a - compatibility layer
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces

    GYMNASIUM_AVAILABLE = True
except ModuleNotFoundError:  
    GYMNASIUM_AVAILABLE = False

    class Env:
        observation_space: Any
        action_space: Any

        def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
            del seed, options
            raise NotImplementedError

        def step(self, action: np.ndarray):
            raise NotImplementedError

    @dataclass
    class Box:
        #box = space used for local smoke test
        low: np.ndarray
        high: np.ndarray
        shape: tuple[int, ...]
        dtype: type[np.floating[Any]]

        def sample(self) -> np.ndarray:
            return np.random.uniform(self.low, self.high).astype(self.dtype)

        def contains(self, value: np.ndarray) -> bool:
            arr = np.asarray(value, dtype=self.dtype)
            return arr.shape == self.shape and np.all(arr >= self.low) and np.all(arr <= self.high)

    class _Spaces:
        Box = Box

    class _Gym:
        Env = Env

    gym = _Gym()
    spaces = _Spaces()

