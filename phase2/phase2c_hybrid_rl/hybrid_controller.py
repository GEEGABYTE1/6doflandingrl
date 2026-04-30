from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from stable_baselines3 import PPO
import numpy as np

from phase1_dynamics.lqr_controller import GainScheduledLQRController
from phase1_dynamics.propulsion import EngineConfig, TVCCommand
from phase2_rl.observations import ObservationNormalizer
from phase2b_hierarchical_rl.coordination_features import (
    CoordinationFeatureConfig,
    build_coordination_features,
)
from phase2b_hierarchical_rl.hierarchical_controller import FrozenThrottlePolicy

from .terminal_braking import (
    energy_assist_delta,
    guidance_brake_throttle,
    overspeed_brake_assist_delta,
    overspeed_brake_floor_throttle,
    stopping_distance_floor_throttle,
    terminal_throttle_residual_gate,
)

@dataclass
class FrozenResidualPolicy:
    model: object
    observation_normalizer: ObservationNormalizer
    engine: EngineConfig
    coordination_config: CoordinationFeatureConfig
    residual_gimbal_limit_rad: float
    residual_throttle_delta_limit: float
    action_mode: str
    include_prior_throttle_feature: bool
    terminal_throttle_gate_altitude_m: float
    terminal_throttle_gate_power: float
    throttle_residual_positive_only: bool
    overspeed_brake_assist_enabled: bool
    overspeed_brake_assist_trigger_mps: float
    overspeed_brake_assist_full_scale_mps: float
    overspeed_brake_assist_max_delta: float
    overspeed_brake_assist_late_stage_altitude_m: float
    overspeed_brake_assist_late_stage_extra_delta: float
    energy_assist_enabled: bool
    energy_gate_altitude_m: float
    energy_gate_power: float
    energy_full_scale: float
    energy_shape_power: float
    energy_max_delta: float
    energy_touchdown_speed_mps: float
    energy_braking_accel_mps2: float
    brake_floor_enabled: bool
    brake_floor_altitude_m: float
    brake_floor_power: float
    brake_floor_safe_margin_mps: float
    brake_floor_trigger_mps: float
    brake_floor_full_scale_mps: float
    brake_floor_base_throttle: float
    brake_floor_max_throttle: float
    brake_floor_shape_power: float
    brake_floor_late_stage_altitude_m: float
    brake_floor_late_stage_extra_throttle: float
    guidance_enabled: bool
    guidance_gate_altitude_m: float
    guidance_gate_power: float
    guidance_target_touchdown_speed_mps: float
    guidance_altitude_floor_m: float
    guidance_base_throttle: float
    guidance_late_stage_altitude_m: float
    guidance_late_stage_extra_throttle: float
    stopping_floor_enabled: bool
    stopping_floor_gate_altitude_m: float
    stopping_floor_gate_power: float
    stopping_floor_altitude_floor_m: float
    stopping_floor_touchdown_speed_mps: float
    stopping_floor_min_downward_speed_mps: float
    previous_throttle: float = 0.0

    @classmethod
    def from_path(cls, model_path: Path, engine: EngineConfig | None = None) -> "FrozenResidualPolicy":
        coordination = CoordinationFeatureConfig()
        residual_limit = np.deg2rad(3.0)
        residual_throttle_limit = 0.08
        action_mode = "tvc_only"
        include_prior_throttle_feature = False
        terminal_throttle_gate_altitude_m = 15.0
        terminal_throttle_gate_power = 2.0
        throttle_residual_positive_only = False
        overspeed_brake_assist_enabled = False
        overspeed_brake_assist_trigger_mps = 1.0
        overspeed_brake_assist_full_scale_mps = 6.0
        overspeed_brake_assist_max_delta = 0.08
        overspeed_brake_assist_late_stage_altitude_m = 0.0
        overspeed_brake_assist_late_stage_extra_delta = 0.0
        energy_assist_enabled = False
        energy_gate_altitude_m = 35.0
        energy_gate_power = 1.5
        energy_full_scale = 40.0
        energy_shape_power = 0.8
        energy_max_delta = 0.10
        energy_touchdown_speed_mps = 0.8
        energy_braking_accel_mps2 = 1.5
        brake_floor_enabled = False
        brake_floor_altitude_m = 5.0
        brake_floor_power = 2.0
        brake_floor_safe_margin_mps = 0.5
        brake_floor_trigger_mps = 0.25
        brake_floor_full_scale_mps = 3.0
        brake_floor_base_throttle = 0.88
        brake_floor_max_throttle = 0.98
        brake_floor_shape_power = 1.0
        brake_floor_late_stage_altitude_m = 0.0
        brake_floor_late_stage_extra_throttle = 0.0
        guidance_enabled = False
        guidance_gate_altitude_m = 20.0
        guidance_gate_power = 1.5
        guidance_target_touchdown_speed_mps = 0.8
        guidance_altitude_floor_m = 0.5
        guidance_base_throttle = 0.0
        guidance_late_stage_altitude_m = 0.0
        guidance_late_stage_extra_throttle = 0.0
        stopping_floor_enabled = False
        stopping_floor_gate_altitude_m = 25.0
        stopping_floor_gate_power = 1.5
        stopping_floor_altitude_floor_m = 0.5
        stopping_floor_touchdown_speed_mps = 0.8
        stopping_floor_min_downward_speed_mps = 1.0
        config_path = model_path.resolve().parent / "config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            env_cfg = config.get("residual_env_config", {})
            if "coordination_features" in env_cfg:
                coordination = CoordinationFeatureConfig(**env_cfg["coordination_features"])
            if "residual_gimbal_limit_rad" in env_cfg:
                residual_limit = float(env_cfg["residual_gimbal_limit_rad"])
            if "residual_throttle_delta_limit" in env_cfg:
                residual_throttle_limit = float(env_cfg["residual_throttle_delta_limit"])
            action_mode = str(env_cfg.get("action_mode", action_mode))
            include_prior_throttle_feature = bool(
                env_cfg.get("include_prior_throttle_feature", include_prior_throttle_feature)
            )
            terminal_throttle_gate_altitude_m = float(
                env_cfg.get("terminal_throttle_gate_altitude_m", terminal_throttle_gate_altitude_m)
            )
            terminal_throttle_gate_power = float(
                env_cfg.get("terminal_throttle_gate_power", terminal_throttle_gate_power)
            )
            throttle_residual_positive_only = bool(
                env_cfg.get("throttle_residual_positive_only", throttle_residual_positive_only)
            )
            overspeed_brake_assist_enabled = bool(
                env_cfg.get("overspeed_brake_assist_enabled", overspeed_brake_assist_enabled)
            )
            overspeed_brake_assist_trigger_mps = float(
                env_cfg.get("overspeed_brake_assist_trigger_mps", overspeed_brake_assist_trigger_mps)
            )
            overspeed_brake_assist_full_scale_mps = float(
                env_cfg.get("overspeed_brake_assist_full_scale_mps", overspeed_brake_assist_full_scale_mps)
            )
            overspeed_brake_assist_max_delta = float(
                env_cfg.get("overspeed_brake_assist_max_delta", overspeed_brake_assist_max_delta)
            )
            overspeed_brake_assist_late_stage_altitude_m = float(
                env_cfg.get(
                    "overspeed_brake_assist_late_stage_altitude_m",
                    overspeed_brake_assist_late_stage_altitude_m,
                )
            )
            overspeed_brake_assist_late_stage_extra_delta = float(
                env_cfg.get(
                    "overspeed_brake_assist_late_stage_extra_delta",
                    overspeed_brake_assist_late_stage_extra_delta,
                )
            )
            energy_assist_enabled = bool(env_cfg.get("energy_assist_enabled", energy_assist_enabled))
            energy_gate_altitude_m = float(env_cfg.get("energy_gate_altitude_m", energy_gate_altitude_m))
            energy_gate_power = float(env_cfg.get("energy_gate_power", energy_gate_power))
            energy_full_scale = float(env_cfg.get("energy_full_scale", energy_full_scale))
            energy_shape_power = float(env_cfg.get("energy_shape_power", energy_shape_power))
            energy_max_delta = float(env_cfg.get("energy_max_delta", energy_max_delta))
            energy_touchdown_speed_mps = float(
                env_cfg.get("energy_touchdown_speed_mps", energy_touchdown_speed_mps)
            )
            energy_braking_accel_mps2 = float(
                env_cfg.get("energy_braking_accel_mps2", energy_braking_accel_mps2)
            )
            brake_floor_enabled = bool(env_cfg.get("brake_floor_enabled", brake_floor_enabled))
            brake_floor_altitude_m = float(env_cfg.get("brake_floor_altitude_m", brake_floor_altitude_m))
            brake_floor_power = float(env_cfg.get("brake_floor_power", brake_floor_power))
            brake_floor_safe_margin_mps = float(
                env_cfg.get("brake_floor_safe_margin_mps", brake_floor_safe_margin_mps)
            )
            brake_floor_trigger_mps = float(env_cfg.get("brake_floor_trigger_mps", brake_floor_trigger_mps))
            brake_floor_full_scale_mps = float(
                env_cfg.get("brake_floor_full_scale_mps", brake_floor_full_scale_mps)
            )
            brake_floor_base_throttle = float(
                env_cfg.get("brake_floor_base_throttle", brake_floor_base_throttle)
            )
            brake_floor_max_throttle = float(env_cfg.get("brake_floor_max_throttle", brake_floor_max_throttle))
            brake_floor_shape_power = float(env_cfg.get("brake_floor_shape_power", brake_floor_shape_power))
            brake_floor_late_stage_altitude_m = float(
                env_cfg.get("brake_floor_late_stage_altitude_m", brake_floor_late_stage_altitude_m)
            )
            brake_floor_late_stage_extra_throttle = float(
                env_cfg.get("brake_floor_late_stage_extra_throttle", brake_floor_late_stage_extra_throttle)
            )
            guidance_enabled = bool(env_cfg.get("guidance_enabled", guidance_enabled))
            guidance_gate_altitude_m = float(
                env_cfg.get("guidance_gate_altitude_m", guidance_gate_altitude_m)
            )
            guidance_gate_power = float(env_cfg.get("guidance_gate_power", guidance_gate_power))
            guidance_target_touchdown_speed_mps = float(
                env_cfg.get("guidance_target_touchdown_speed_mps", guidance_target_touchdown_speed_mps)
            )
            guidance_altitude_floor_m = float(
                env_cfg.get("guidance_altitude_floor_m", guidance_altitude_floor_m)
            )
            guidance_base_throttle = float(env_cfg.get("guidance_base_throttle", guidance_base_throttle))
            guidance_late_stage_altitude_m = float(
                env_cfg.get("guidance_late_stage_altitude_m", guidance_late_stage_altitude_m)
            )
            guidance_late_stage_extra_throttle = float(
                env_cfg.get("guidance_late_stage_extra_throttle", guidance_late_stage_extra_throttle)
            )
            stopping_floor_enabled = bool(env_cfg.get("stopping_floor_enabled", stopping_floor_enabled))
            stopping_floor_gate_altitude_m = float(
                env_cfg.get("stopping_floor_gate_altitude_m", stopping_floor_gate_altitude_m)
            )
            stopping_floor_gate_power = float(
                env_cfg.get("stopping_floor_gate_power", stopping_floor_gate_power)
            )
            stopping_floor_altitude_floor_m = float(
                env_cfg.get("stopping_floor_altitude_floor_m", stopping_floor_altitude_floor_m)
            )
            stopping_floor_touchdown_speed_mps = float(
                env_cfg.get("stopping_floor_touchdown_speed_mps", stopping_floor_touchdown_speed_mps)
            )
            stopping_floor_min_downward_speed_mps = float(
                env_cfg.get("stopping_floor_min_downward_speed_mps", stopping_floor_min_downward_speed_mps)
            )
        return cls(
            model=PPO.load(str(model_path)),
            observation_normalizer=ObservationNormalizer(),
            engine=engine or EngineConfig(),
            coordination_config=coordination,
            residual_gimbal_limit_rad=residual_limit,
            residual_throttle_delta_limit=residual_throttle_limit,
            action_mode=action_mode,
            include_prior_throttle_feature=include_prior_throttle_feature,
            terminal_throttle_gate_altitude_m=terminal_throttle_gate_altitude_m,
            terminal_throttle_gate_power=terminal_throttle_gate_power,
            throttle_residual_positive_only=throttle_residual_positive_only,
            overspeed_brake_assist_enabled=overspeed_brake_assist_enabled,
            overspeed_brake_assist_trigger_mps=overspeed_brake_assist_trigger_mps,
            overspeed_brake_assist_full_scale_mps=overspeed_brake_assist_full_scale_mps,
            overspeed_brake_assist_max_delta=overspeed_brake_assist_max_delta,
            overspeed_brake_assist_late_stage_altitude_m=overspeed_brake_assist_late_stage_altitude_m,
            overspeed_brake_assist_late_stage_extra_delta=overspeed_brake_assist_late_stage_extra_delta,
            energy_assist_enabled=energy_assist_enabled,
            energy_gate_altitude_m=energy_gate_altitude_m,
            energy_gate_power=energy_gate_power,
            energy_full_scale=energy_full_scale,
            energy_shape_power=energy_shape_power,
            energy_max_delta=energy_max_delta,
            energy_touchdown_speed_mps=energy_touchdown_speed_mps,
            energy_braking_accel_mps2=energy_braking_accel_mps2,
            brake_floor_enabled=brake_floor_enabled,
            brake_floor_altitude_m=brake_floor_altitude_m,
            brake_floor_power=brake_floor_power,
            brake_floor_safe_margin_mps=brake_floor_safe_margin_mps,
            brake_floor_trigger_mps=brake_floor_trigger_mps,
            brake_floor_full_scale_mps=brake_floor_full_scale_mps,
            brake_floor_base_throttle=brake_floor_base_throttle,
            brake_floor_max_throttle=brake_floor_max_throttle,
            brake_floor_shape_power=brake_floor_shape_power,
            brake_floor_late_stage_altitude_m=brake_floor_late_stage_altitude_m,
            brake_floor_late_stage_extra_throttle=brake_floor_late_stage_extra_throttle,
            guidance_enabled=guidance_enabled,
            guidance_gate_altitude_m=guidance_gate_altitude_m,
            guidance_gate_power=guidance_gate_power,
            guidance_target_touchdown_speed_mps=guidance_target_touchdown_speed_mps,
            guidance_altitude_floor_m=guidance_altitude_floor_m,
            guidance_base_throttle=guidance_base_throttle,
            guidance_late_stage_altitude_m=guidance_late_stage_altitude_m,
            guidance_late_stage_extra_throttle=guidance_late_stage_extra_throttle,
            stopping_floor_enabled=stopping_floor_enabled,
            stopping_floor_gate_altitude_m=stopping_floor_gate_altitude_m,
            stopping_floor_gate_power=stopping_floor_gate_power,
            stopping_floor_altitude_floor_m=stopping_floor_altitude_floor_m,
            stopping_floor_touchdown_speed_mps=stopping_floor_touchdown_speed_mps,
            stopping_floor_min_downward_speed_mps=stopping_floor_min_downward_speed_mps,
        )

    def reset(self) -> None:
        self.previous_throttle = 0.0

    def residual_action(
        self,
        state: np.ndarray,
        throttle: float,
        base_command: TVCCommand,
        dry_mass_kg: float,
        initial_mass_kg: float,
    ) -> tuple[float, float, float]:
        base = self.observation_normalizer.encode(
            state=state,
            target_position_m=np.zeros(3, dtype=float),
            target_velocity_mps=np.zeros(3, dtype=float),
            dry_mass_kg=dry_mass_kg,
            initial_mass_kg=initial_mass_kg,
        )
        prior_throttle = float(self.previous_throttle if self.previous_throttle > 0.0 else throttle)
        coordination = build_coordination_features(
            state=state,
            throttle=throttle,
            previous_throttle=prior_throttle,
            config=self.coordination_config,
        )
        gimbal_limit = max(self.engine.gimbal_limit_rad, 1.0e-6)
        parts = [base, coordination]
        if self.include_prior_throttle_feature:
            parts.append(np.array([2.0 * float(throttle) - 1.0], dtype=np.float32))
        base_features = np.array(
            [
                float(np.clip(base_command.pitch_rad / gimbal_limit, -1.0, 1.0)),
                float(np.clip(base_command.yaw_rad / gimbal_limit, -1.0, 1.0)),
            ],
            dtype=np.float32,
        )
        parts.append(base_features)
        observation = np.concatenate(parts).astype(np.float32)
        action, _ = self.model.predict(observation, deterministic=True)
        self.previous_throttle = float(throttle)
        if self.action_mode == "tvc_throttle":
            action = np.clip(np.asarray(action, dtype=float).reshape(3), -1.0, 1.0)
            throttle_action = float(action[0])
            if self.throttle_residual_positive_only:
                throttle_action = max(throttle_action, 0.0)
            return (
                float(throttle_action * self.residual_throttle_delta_limit),
                float(action[1] * self.residual_gimbal_limit_rad),
                float(action[2] * self.residual_gimbal_limit_rad),
            )
        action = np.clip(np.asarray(action, dtype=float).reshape(2), -1.0, 1.0)
        return (
            0.0,
            float(action[0] * self.residual_gimbal_limit_rad),
            float(action[1] * self.residual_gimbal_limit_rad),
        )


@dataclass
class HybridResidualController:
    throttle_policy: FrozenThrottlePolicy
    residual_policy: FrozenResidualPolicy
    lqr_controller: GainScheduledLQRController
    engine: EngineConfig

    def reset(self) -> None:
        self.residual_policy.reset()

    def command(self, time_s: float, state: np.ndarray, initial_mass_kg: float) -> TVCCommand:
        dry_mass_kg = self.engine.dry_mass_kg
        throttle = self.prior_throttle(state, initial_mass_kg)
        base_command = self.lqr_controller.command(time_s, state)
        d_throttle, d_pitch, d_yaw = self.residual_policy.residual_action(
            state=state,
            throttle=throttle,
            base_command=base_command,
            dry_mass_kg=dry_mass_kg,
            initial_mass_kg=initial_mass_kg,
        )
        if bool(getattr(self.residual_policy, "throttle_residual_positive_only", False)):
            d_throttle = max(float(d_throttle), 0.0)
        throttle_gate = terminal_throttle_residual_gate(
            altitude_m=float(state[2]),
            gate_altitude_m=float(getattr(self.residual_policy, "terminal_throttle_gate_altitude_m", 15.0)),
            gate_power=float(getattr(self.residual_policy, "terminal_throttle_gate_power", 2.0)),
        )
        return TVCCommand(
            throttle=float(
                np.clip(
                    throttle + d_throttle * throttle_gate,
                    self.engine.min_throttle,
                    self.engine.max_throttle,
                )
            ),
            pitch_rad=float(np.clip(base_command.pitch_rad + d_pitch, -self.engine.gimbal_limit_rad, self.engine.gimbal_limit_rad)),
            yaw_rad=float(np.clip(base_command.yaw_rad + d_yaw, -self.engine.gimbal_limit_rad, self.engine.gimbal_limit_rad)),
        )

    def prior_throttle(self, state: np.ndarray, initial_mass_kg: float) -> float:
        dry_mass_kg = self.engine.dry_mass_kg
        base = self.throttle_policy.command_throttle(state, dry_mass_kg, initial_mass_kg)
        assist = 0.0
        if bool(getattr(self.residual_policy, "overspeed_brake_assist_enabled", False)):
            assist = overspeed_brake_assist_delta(
                altitude_m=float(state[2]),
                vertical_speed_mps=float(state[5]),
                config=getattr(self.residual_policy, "coordination_config"),
                gate_altitude_m=float(getattr(self.residual_policy, "terminal_throttle_gate_altitude_m", 15.0)),
                gate_power=float(getattr(self.residual_policy, "terminal_throttle_gate_power", 2.0)),
                trigger_mps=float(getattr(self.residual_policy, "overspeed_brake_assist_trigger_mps", 1.0)),
                full_scale_mps=float(getattr(self.residual_policy, "overspeed_brake_assist_full_scale_mps", 6.0)),
                max_delta=float(getattr(self.residual_policy, "overspeed_brake_assist_max_delta", 0.08)),
                late_stage_altitude_m=float(
                    getattr(self.residual_policy, "overspeed_brake_assist_late_stage_altitude_m", 0.0)
                ),
                late_stage_extra_delta=float(
                    getattr(self.residual_policy, "overspeed_brake_assist_late_stage_extra_delta", 0.0)
                ),
            )
        energy_assist = 0.0
        if bool(getattr(self.residual_policy, "energy_assist_enabled", False)):
            energy_assist = energy_assist_delta(
                altitude_m=float(state[2]),
                vertical_speed_mps=float(state[5]),
                gate_altitude_m=float(getattr(self.residual_policy, "energy_gate_altitude_m", 35.0)),
                gate_power=float(getattr(self.residual_policy, "energy_gate_power", 1.5)),
                full_scale=float(getattr(self.residual_policy, "energy_full_scale", 40.0)),
                shape_power=float(getattr(self.residual_policy, "energy_shape_power", 0.8)),
                max_delta=float(getattr(self.residual_policy, "energy_max_delta", 0.10)),
                touchdown_speed_mps=float(getattr(self.residual_policy, "energy_touchdown_speed_mps", 0.8)),
                braking_accel_mps2=float(getattr(self.residual_policy, "energy_braking_accel_mps2", 1.5)),
                standard_gravity_mps2=self.engine.standard_gravity_mps2,
            )
        assist_prior = float(
            np.clip(base + assist + energy_assist, self.engine.min_throttle, self.engine.max_throttle)
        )
        floor_prior = 0.0
        if bool(getattr(self.residual_policy, "brake_floor_enabled", False)):
            floor_prior = overspeed_brake_floor_throttle(
                altitude_m=float(state[2]),
                vertical_speed_mps=float(state[5]),
                config=getattr(self.residual_policy, "coordination_config"),
                gate_altitude_m=float(getattr(self.residual_policy, "brake_floor_altitude_m", 5.0)),
                gate_power=float(getattr(self.residual_policy, "brake_floor_power", 2.0)),
                safe_margin_mps=float(getattr(self.residual_policy, "brake_floor_safe_margin_mps", 0.5)),
                trigger_mps=float(getattr(self.residual_policy, "brake_floor_trigger_mps", 0.25)),
                full_scale_mps=float(getattr(self.residual_policy, "brake_floor_full_scale_mps", 3.0)),
                base_throttle=float(getattr(self.residual_policy, "brake_floor_base_throttle", 0.88)),
                max_throttle=float(getattr(self.residual_policy, "brake_floor_max_throttle", 0.98)),
                shape_power=float(getattr(self.residual_policy, "brake_floor_shape_power", 1.0)),
                late_stage_altitude_m=float(
                    getattr(self.residual_policy, "brake_floor_late_stage_altitude_m", 0.0)
                ),
                late_stage_extra_throttle=float(
                    getattr(self.residual_policy, "brake_floor_late_stage_extra_throttle", 0.0)
                ),
            )
        guidance_prior = 0.0
        if bool(getattr(self.residual_policy, "guidance_enabled", False)):
            guidance_prior = guidance_brake_throttle(
                altitude_m=float(state[2]),
                vertical_speed_mps=float(state[5]),
                mass_kg=float(state[13]),
                min_throttle=self.engine.min_throttle,
                max_throttle=self.engine.max_throttle,
                max_thrust_n=self.engine.max_thrust_n,
                standard_gravity_mps2=self.engine.standard_gravity_mps2,
                gate_altitude_m=float(getattr(self.residual_policy, "guidance_gate_altitude_m", 20.0)),
                gate_power=float(getattr(self.residual_policy, "guidance_gate_power", 1.5)),
                target_touchdown_speed_mps=float(
                    getattr(self.residual_policy, "guidance_target_touchdown_speed_mps", 0.8)
                ),
                altitude_floor_m=float(getattr(self.residual_policy, "guidance_altitude_floor_m", 0.5)),
                base_throttle=float(getattr(self.residual_policy, "guidance_base_throttle", 0.0)),
                late_stage_altitude_m=float(
                    getattr(self.residual_policy, "guidance_late_stage_altitude_m", 0.0)
                ),
                late_stage_extra_throttle=float(
                    getattr(self.residual_policy, "guidance_late_stage_extra_throttle", 0.0)
                ),
            )
        stopping_floor_prior = 0.0
        if bool(getattr(self.residual_policy, "stopping_floor_enabled", False)):
            stopping_floor_prior = stopping_distance_floor_throttle(
                altitude_m=float(state[2]),
                vertical_speed_mps=float(state[5]),
                mass_kg=float(state[13]),
                min_throttle=self.engine.min_throttle,
                max_throttle=self.engine.max_throttle,
                max_thrust_n=self.engine.max_thrust_n,
                standard_gravity_mps2=self.engine.standard_gravity_mps2,
                gate_altitude_m=float(getattr(self.residual_policy, "stopping_floor_gate_altitude_m", 25.0)),
                gate_power=float(getattr(self.residual_policy, "stopping_floor_gate_power", 1.5)),
                altitude_floor_m=float(getattr(self.residual_policy, "stopping_floor_altitude_floor_m", 0.5)),
                touchdown_speed_mps=float(
                    getattr(self.residual_policy, "stopping_floor_touchdown_speed_mps", 0.8)
                ),
                min_downward_speed_mps=float(
                    getattr(self.residual_policy, "stopping_floor_min_downward_speed_mps", 1.0)
                ),
            )
        return float(max(assist_prior, floor_prior, guidance_prior, stopping_floor_prior))
