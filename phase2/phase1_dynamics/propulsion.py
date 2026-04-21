"""thrust and tvc model."""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

@dataclass(frozen=True)
class EngineConfig:
    """single-engine configuration for a throttleable landing rocket."""
    max_thrust_n: float = 18_000.0
    min_throttle: float = 0.0
    max_throttle: float = 1.0
    specific_impulse_s: float = 285.0
    gimbal_limit_rad: float = np.deg2rad(8.0)
    lever_arm_m: float = 2.0
    dry_mass_kg: float = 850.0
    standard_gravity_mps2: float = 9.80665

@dataclass(frozen=True)
class TVCCommand:
    """
    ``pitch_rad`` tilts thrust toward body ``+x``. ``yaw_rad`` tilts thrust
    toward body ``+y``. Nominal thrust points along body ``+z``.
    """
    throttle: float
    pitch_rad: float
    yaw_rad: float

@dataclass(frozen=True)
class PropulsionOutput:
    force_body_n: Array
    moment_body_nm: Array
    mass_flow_kgps: float
    throttle: float
    pitch_rad: float
    yaw_rad: float

@dataclass(frozen=True)
class PropulsionModel:
    config: EngineConfig = field(default_factory=EngineConfig)

    def evaluate(
        self,
        command: TVCCommand,
        mass_kg: float,
        misalignment_pitch_yaw_rad: Array | None = None,
    ) -> PropulsionOutput:
        """ the engine is located below the center of mass at ``r = [0, 0, -L]`` in
        body coordinates. The moment is ``M_B = r_B x F_B``.
        """
        cfg = self.config
        throttle = float(np.clip(command.throttle, cfg.min_throttle, cfg.max_throttle))
        pitch = float(np.clip(command.pitch_rad, -cfg.gimbal_limit_rad, cfg.gimbal_limit_rad))
        yaw = float(np.clip(command.yaw_rad, -cfg.gimbal_limit_rad, cfg.gimbal_limit_rad))
        if misalignment_pitch_yaw_rad is not None:
            pitch += float(misalignment_pitch_yaw_rad[0])
            yaw += float(misalignment_pitch_yaw_rad[1])

        thrust_n = throttle * cfg.max_thrust_n if mass_kg > cfg.dry_mass_kg else 0.0
        direction_body = np.array(
            [
                np.sin(pitch),
                np.sin(yaw),
                np.cos(pitch) * np.cos(yaw),
            ],
            dtype=float,
        )
        direction_body /= np.linalg.norm(direction_body)
        force_body = thrust_n * direction_body
        engine_position_body = np.array([0.0, 0.0, -cfg.lever_arm_m], dtype=float)
        moment_body = np.cross(engine_position_body, force_body)

        if thrust_n <= 0.0:
            mass_flow = 0.0
        else:
            mass_flow = -thrust_n / (cfg.specific_impulse_s * cfg.standard_gravity_mps2)

        return PropulsionOutput(
            force_body_n=force_body,
            moment_body_nm=moment_body,
            mass_flow_kgps=float(mass_flow),
            throttle=throttle,
            pitch_rad=pitch,
            yaw_rad=yaw,
        )

