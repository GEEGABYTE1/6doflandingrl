# hierarch controller

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Protocol

import numpy as np

from phase1_dynamics.propulsion import EngineConfig, TVCCommand
from phase2_rl.observations import ObservationNormalizer
from .coordination_features import CoordinationFeatureConfig, build_coordination_features
from .throttle_env import ThrottleObservationNormalizer


class ThrottlePolicy(Protocol):

    def command_throttle(self, state: np.ndarray, dry_mass_kg: float, initial_mass_kg: float) -> float:
        """Return physical throttle in `[0, 1]`."""

class TVCPolicy(Protocol):
    def command_gimbal(
        self,
        state: np.ndarray,
        throttle: float,
        dry_mass_kg: float,
        initial_mass_kg: float,
    ) -> tuple[float, float]:
        """Return `(pitch_rad, yaw_rad)` within gimbal limits."""

    def reset(self) -> None:
        """Reset any internal coordination state before a fresh rollout."""


@dataclass
class FrozenThrottlePolicy:
    model: object
    normalizer: ThrottleObservationNormalizer
    engine: EngineConfig
    throttle_delta_limit: float
    observation_mode: str = "baseline_v1"

    @classmethod
    def from_path(cls, model_path: Path, altitude_scale_m: float = 150.0, vertical_speed_scale_mps: float = 40.0) -> "FrozenThrottlePolicy":
        from stable_baselines3 import PPO

        engine = EngineConfig()
        delta_limit = 0.35
        normalizer_altitude = altitude_scale_m
        normalizer_vspeed = vertical_speed_scale_mps
        observation_mode = "baseline_v1"
        feature_eps = 1.0e-3
        config_path = model_path.resolve().parent / "config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            env_cfg = config.get("throttle_env_config", {})
            normalizer_altitude = float(env_cfg.get("altitude_scale_m", altitude_scale_m))
            normalizer_vspeed = float(env_cfg.get("vertical_speed_scale_mps", vertical_speed_scale_mps))
            delta_limit = float(env_cfg.get("throttle_delta_limit", delta_limit))
            observation_mode = str(env_cfg.get("observation_mode", observation_mode))
            feature_eps = float(env_cfg.get("feature_eps", feature_eps))
        return cls(
            model=PPO.load(str(model_path)),
            normalizer=ThrottleObservationNormalizer(
                altitude_scale_m=normalizer_altitude,
                vertical_speed_scale_mps=normalizer_vspeed,
                observation_mode=observation_mode,
                feature_eps=feature_eps,
            ),
            engine=engine,
            throttle_delta_limit=delta_limit,
            observation_mode=observation_mode,
        )

    def command_throttle(self, state: np.ndarray, dry_mass_kg: float, initial_mass_kg: float) -> float:
        reduced_state = type("ReducedState", (), {
            "altitude_m": float(state[2]),
            "vertical_speed_mps": float(state[5]),
            "mass_kg": float(state[13]),
        })()
        observation = self.normalizer.encode(reduced_state, dry_mass_kg, initial_mass_kg, self.engine)
        action, _ = self.model.predict(observation, deterministic=True)
        scalar = float(np.clip(np.asarray(action, dtype=float).reshape(-1)[0], -1.0, 1.0))
        hover_throttle = float(state[13]) * self.engine.standard_gravity_mps2 / self.engine.max_thrust_n
        throttle = hover_throttle + scalar * self.throttle_delta_limit
        return float(np.clip(throttle, self.engine.min_throttle, self.engine.max_throttle))


@dataclass
class FrozenTVCPolicy:
    model: object
    observation_normalizer: ObservationNormalizer
    engine: EngineConfig
    coordination_config: CoordinationFeatureConfig
    observation_mode: str = "coordination_v1"
    previous_throttle: float = 0.0

    @classmethod
    def from_path(cls, model_path: Path, engine: EngineConfig | None = None) -> "FrozenTVCPolicy":
        """Load a PPO TVC policy from disk."""
        from stable_baselines3 import PPO

        coordination_config = CoordinationFeatureConfig()
        observation_mode = "legacy_throttle_only_v1"
        config_path = model_path.resolve().parent / "config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            env_cfg = config.get("tvc_env_config", {})
            if "coordination_features" in env_cfg:
                coordination_config = CoordinationFeatureConfig(**env_cfg["coordination_features"])
                observation_mode = "coordination_v1"
        return cls(
            model=PPO.load(str(model_path)),
            observation_normalizer=ObservationNormalizer(),
            engine=engine or EngineConfig(),
            coordination_config=coordination_config,
            observation_mode=observation_mode,
        )

    def reset(self) -> None:
        self.previous_throttle = 0.0

    def command_gimbal(
        self,
        state: np.ndarray,
        throttle: float,
        dry_mass_kg: float,
        initial_mass_kg: float,
    ) -> tuple[float, float]:
        base_observation = self.observation_normalizer.encode(
            state=state,
            target_position_m=np.zeros(3, dtype=float),
            target_velocity_mps=np.zeros(3, dtype=float),
            dry_mass_kg=dry_mass_kg,
            initial_mass_kg=initial_mass_kg,
        )
        if self.observation_mode == "legacy_throttle_only_v1":
            observation = np.concatenate(
                (base_observation, np.array([2.0 * float(throttle) - 1.0], dtype=np.float32))
            ).astype(np.float32)
        else:
            prior_throttle = float(self.previous_throttle if self.previous_throttle > 0.0 else throttle)
            coordination = build_coordination_features(
                state=state,
                throttle=throttle,
                previous_throttle=prior_throttle,
                config=self.coordination_config,
            )
            observation = np.concatenate((base_observation, coordination)).astype(np.float32)
        action, _ = self.model.predict(observation, deterministic=True)
        self.previous_throttle = float(throttle)
        action = np.clip(np.asarray(action, dtype=float).reshape(2), -1.0, 1.0)
        limit = self.engine.gimbal_limit_rad
        return float(action[0] * limit), float(action[1] * limit)


@dataclass
class HierarchicalPolicyController:
    #compose frozen throttle and TVC PPO policies into one controller.

    throttle_policy: ThrottlePolicy
    tvc_policy: TVCPolicy
    engine: EngineConfig

    def command(self, state: np.ndarray, initial_mass_kg: float) -> TVCCommand:
        dry_mass_kg = self.engine.dry_mass_kg
        throttle = self.throttle_policy.command_throttle(state, dry_mass_kg, initial_mass_kg)
        pitch_rad, yaw_rad = self.tvc_policy.command_gimbal(state, throttle, dry_mass_kg, initial_mass_kg)
        return TVCCommand(
            throttle=float(np.clip(throttle, self.engine.min_throttle, self.engine.max_throttle)),
            pitch_rad=float(np.clip(pitch_rad, -self.engine.gimbal_limit_rad, self.engine.gimbal_limit_rad)),
            yaw_rad=float(np.clip(yaw_rad, -self.engine.gimbal_limit_rad, self.engine.gimbal_limit_rad)),
        )

    def reset(self) -> None:
        if hasattr(self.tvc_policy, "reset"):
            self.tvc_policy.reset()
