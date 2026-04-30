from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
import numpy as np
from numpy.typing import NDArray

from phase1_dynamics.integrator import rk4_step
from phase1_dynamics.lqr_controller import GainScheduledLQRController
from phase1_dynamics.metrics import SuccessCriteria, quaternion_tilt_deg, touchdown_metrics
from phase1_dynamics.propulsion import TVCCommand
from phase1_dynamics.scenarios import ScenarioConfig, named_scenarios
from phase1_dynamics.simulate_lqr import build_dynamics
from phase2_rl.curriculum import CurriculumBounds, TrainingCurriculum
from phase2_rl.gym_compat import gym, spaces
from phase2_rl.observations import ObservationNormalizer
from phase2b_hierarchical_rl.coordination_features import (
    CoordinationFeatureConfig,
    build_coordination_features,
    flare_reference_vertical_speed,
)
from phase2b_hierarchical_rl.hierarchical_controller import FrozenThrottlePolicy
from .experiment_configs import ResidualRewardWeights
from .terminal_braking import (
    energy_assist_delta,
    guidance_brake_throttle,
    overspeed_brake_assist_delta,
    overspeed_brake_floor_throttle,
    stopping_distance_floor_throttle,
    terminal_throttle_residual_gate,
)

Array = NDArray[np.float64]


def scenario_by_name(name: str) -> ScenarioConfig:
    for scenario in named_scenarios():
        if scenario.name == name:
            return scenario
    raise KeyError(f"Unknown Phase 2C scenario: {name}")


@dataclass(frozen=True)
class ResidualEnvConfig:
    scenario_name: str = "nominal"
    dt_s: float = 0.05
    max_duration_s: float = 30.0
    success_criteria: SuccessCriteria = field(default_factory=SuccessCriteria)
    reward_weights: ResidualRewardWeights = field(default_factory=ResidualRewardWeights)
    curriculum_enabled: bool = True
    staged_wind_enabled: bool = True
    curriculum_bounds: CurriculumBounds = field(default_factory=CurriculumBounds)
    coordination_features: CoordinationFeatureConfig = field(default_factory=CoordinationFeatureConfig)
    residual_gimbal_limit_rad: float = np.deg2rad(3.0)
    residual_throttle_delta_limit: float = 0.08
    action_mode: str = "tvc_only"
    include_prior_throttle_feature: bool = False
    terminal_throttle_gate_altitude_m: float = 15.0
    terminal_throttle_gate_power: float = 2.0
    near_ground_gate_altitude_m: float = 5.0
    near_ground_gate_power: float = 2.0
    near_ground_safe_margin_mps: float = 0.5
    brake_floor_enabled: bool = False
    brake_floor_altitude_m: float = 5.0
    brake_floor_power: float = 2.0
    brake_floor_safe_margin_mps: float = 0.5
    brake_floor_trigger_mps: float = 0.25
    brake_floor_full_scale_mps: float = 3.0
    brake_floor_base_throttle: float = 0.88
    brake_floor_max_throttle: float = 0.98
    brake_floor_shape_power: float = 1.0
    brake_floor_late_stage_altitude_m: float = 0.0
    brake_floor_late_stage_extra_throttle: float = 0.0
    throttle_residual_positive_only: bool = False
    overspeed_brake_assist_enabled: bool = False
    overspeed_brake_assist_trigger_mps: float = 1.0
    overspeed_brake_assist_full_scale_mps: float = 6.0
    overspeed_brake_assist_max_delta: float = 0.08
    overspeed_brake_assist_late_stage_altitude_m: float = 0.0
    overspeed_brake_assist_late_stage_extra_delta: float = 0.0
    energy_assist_enabled: bool = False
    energy_gate_altitude_m: float = 35.0
    energy_gate_power: float = 1.5
    energy_full_scale: float = 40.0
    energy_shape_power: float = 0.8
    energy_max_delta: float = 0.10
    energy_touchdown_speed_mps: float = 0.8
    energy_braking_accel_mps2: float = 1.5
    guidance_enabled: bool = False
    guidance_gate_altitude_m: float = 20.0
    guidance_gate_power: float = 1.5
    guidance_target_touchdown_speed_mps: float = 0.8
    guidance_altitude_floor_m: float = 0.5
    guidance_base_throttle: float = 0.0
    guidance_late_stage_altitude_m: float = 0.0
    guidance_late_stage_extra_throttle: float = 0.0
    stopping_floor_enabled: bool = False
    stopping_floor_gate_altitude_m: float = 25.0
    stopping_floor_gate_power: float = 1.5
    stopping_floor_altitude_floor_m: float = 0.5
    stopping_floor_touchdown_speed_mps: float = 0.8
    stopping_floor_min_downward_speed_mps: float = 1.0

    def to_dict(self) -> dict[str, object]:
        return {
            "scenario_name": self.scenario_name,
            "dt_s": self.dt_s,
            "max_duration_s": self.max_duration_s,
            "success_criteria": self.success_criteria.to_dict(),
            "reward_weights": asdict(self.reward_weights),
            "curriculum_enabled": self.curriculum_enabled,
            "staged_wind_enabled": self.staged_wind_enabled,
            "curriculum_bounds": asdict(self.curriculum_bounds),
            "coordination_features": self.coordination_features.to_dict(),
            "residual_gimbal_limit_rad": self.residual_gimbal_limit_rad,
            "residual_throttle_delta_limit": self.residual_throttle_delta_limit,
            "action_mode": self.action_mode,
            "include_prior_throttle_feature": self.include_prior_throttle_feature,
            "terminal_throttle_gate_altitude_m": self.terminal_throttle_gate_altitude_m,
            "terminal_throttle_gate_power": self.terminal_throttle_gate_power,
            "near_ground_gate_altitude_m": self.near_ground_gate_altitude_m,
            "near_ground_gate_power": self.near_ground_gate_power,
            "near_ground_safe_margin_mps": self.near_ground_safe_margin_mps,
            "brake_floor_enabled": self.brake_floor_enabled,
            "brake_floor_altitude_m": self.brake_floor_altitude_m,
            "brake_floor_power": self.brake_floor_power,
            "brake_floor_safe_margin_mps": self.brake_floor_safe_margin_mps,
            "brake_floor_trigger_mps": self.brake_floor_trigger_mps,
            "brake_floor_full_scale_mps": self.brake_floor_full_scale_mps,
            "brake_floor_base_throttle": self.brake_floor_base_throttle,
            "brake_floor_max_throttle": self.brake_floor_max_throttle,
            "brake_floor_shape_power": self.brake_floor_shape_power,
            "brake_floor_late_stage_altitude_m": self.brake_floor_late_stage_altitude_m,
            "brake_floor_late_stage_extra_throttle": self.brake_floor_late_stage_extra_throttle,
            "throttle_residual_positive_only": self.throttle_residual_positive_only,
            "overspeed_brake_assist_enabled": self.overspeed_brake_assist_enabled,
            "overspeed_brake_assist_trigger_mps": self.overspeed_brake_assist_trigger_mps,
            "overspeed_brake_assist_full_scale_mps": self.overspeed_brake_assist_full_scale_mps,
            "overspeed_brake_assist_max_delta": self.overspeed_brake_assist_max_delta,
            "overspeed_brake_assist_late_stage_altitude_m": self.overspeed_brake_assist_late_stage_altitude_m,
            "overspeed_brake_assist_late_stage_extra_delta": self.overspeed_brake_assist_late_stage_extra_delta,
            "energy_assist_enabled": self.energy_assist_enabled,
            "energy_gate_altitude_m": self.energy_gate_altitude_m,
            "energy_gate_power": self.energy_gate_power,
            "energy_full_scale": self.energy_full_scale,
            "energy_shape_power": self.energy_shape_power,
            "energy_max_delta": self.energy_max_delta,
            "energy_touchdown_speed_mps": self.energy_touchdown_speed_mps,
            "energy_braking_accel_mps2": self.energy_braking_accel_mps2,
            "guidance_enabled": self.guidance_enabled,
            "guidance_gate_altitude_m": self.guidance_gate_altitude_m,
            "guidance_gate_power": self.guidance_gate_power,
            "guidance_target_touchdown_speed_mps": self.guidance_target_touchdown_speed_mps,
            "guidance_altitude_floor_m": self.guidance_altitude_floor_m,
            "guidance_base_throttle": self.guidance_base_throttle,
            "guidance_late_stage_altitude_m": self.guidance_late_stage_altitude_m,
            "guidance_late_stage_extra_throttle": self.guidance_late_stage_extra_throttle,
            "stopping_floor_enabled": self.stopping_floor_enabled,
            "stopping_floor_gate_altitude_m": self.stopping_floor_gate_altitude_m,
            "stopping_floor_gate_power": self.stopping_floor_gate_power,
            "stopping_floor_altitude_floor_m": self.stopping_floor_altitude_floor_m,
            "stopping_floor_touchdown_speed_mps": self.stopping_floor_touchdown_speed_mps,
            "stopping_floor_min_downward_speed_mps": self.stopping_floor_min_downward_speed_mps,
        }


class HybridResidualEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        throttle_policy: FrozenThrottlePolicy,
        lqr_controller: GainScheduledLQRController | None = None,
        config: ResidualEnvConfig | None = None,
        seed: int = 7,
    ) -> None:
        self.throttle_policy = throttle_policy
        self.lqr_controller = lqr_controller or GainScheduledLQRController()
        self.config = config or ResidualEnvConfig()
        self.seed_value = int(seed)
        self.base_scenario = scenario_by_name(self.config.scenario_name)
        self.scenario = self.base_scenario
        self.rng = np.random.default_rng(self.seed_value)
        self.curriculum = (
            TrainingCurriculum(
                bounds=self.config.curriculum_bounds,
                staged_wind_enabled=self.config.staged_wind_enabled,
            )
            if self.config.curriculum_enabled
            else None
        )
        self.curriculum_progress = 0.0
        self.dynamics = build_dynamics(self.seed_value, self.scenario)
        self.normalizer = ObservationNormalizer()
        self.initial_mass_kg = float(self.scenario.initial_mass_kg)
        self.max_steps = int(np.ceil(self.config.max_duration_s / self.config.dt_s))
        self.action_dim = 3 if self.config.action_mode == "tvc_throttle" else 2
        self.observation_dim = 22 if self.config.include_prior_throttle_feature else 21
        self.action_space = spaces.Box(
            low=np.full(self.action_dim, -1.0, dtype=np.float32),
            high=np.full(self.action_dim, 1.0, dtype=np.float32),
            shape=(self.action_dim,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=np.full(self.observation_dim, -np.inf, dtype=np.float32),
            high=np.full(self.observation_dim, np.inf, dtype=np.float32),
            shape=(self.observation_dim,),
            dtype=np.float32,
        )
        self.state = np.zeros(14, dtype=float)
        self.time_s = 0.0
        self.step_count = 0
        self.last_throttle = 0.0
        self.last_prior_throttle = 0.0
        self.previous_throttle = 0.0
        self.last_residual_action = np.zeros(self.action_dim, dtype=float)
        self.previous_residual_action = np.zeros(self.action_dim, dtype=float)
        self.trajectory_rows: list[dict[str, float]] = []
        self.last_metrics: dict[str, Any] = {}

    def set_curriculum_progress(self, progress: float) -> None:
        self.curriculum_progress = float(np.clip(progress, 0.0, 1.0))
        if self.curriculum is not None:
            self.curriculum.set_progress(self.curriculum_progress)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del options
        if seed is not None:
            self.seed_value = int(seed)
            self.rng = np.random.default_rng(self.seed_value)
        if self.curriculum is not None:
            self.scenario = self.curriculum.sample(self.base_scenario, self.rng)
        else:
            self.scenario = self.base_scenario
        self.dynamics = build_dynamics(self.seed_value, self.scenario)
        self.state = self.scenario.initial_state(self.seed_value).astype(float)
        self.initial_mass_kg = float(self.state[13])
        self.time_s = 0.0
        self.step_count = 0
        self.last_throttle = self._prior_throttle(self.state)
        self.last_prior_throttle = float(self.last_throttle)
        self.previous_throttle = float(self.last_throttle)
        self.last_residual_action = np.zeros(self.action_dim, dtype=float)
        self.previous_residual_action = np.zeros(self.action_dim, dtype=float)
        self.trajectory_rows = []
        self.last_metrics = {}
        return self._observation(), {
            "scenario": self.scenario.name,
            "curriculum_progress": self.curriculum_progress,
        }

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        residual_action = np.clip(np.asarray(action, dtype=float).reshape(self.action_dim), -1.0, 1.0)
        self.previous_residual_action = self.last_residual_action.copy()
        self.last_residual_action = residual_action.copy()
        self.previous_throttle = float(self.last_throttle)
        self.last_prior_throttle = self._prior_throttle(self.state)
        base_command = self.lqr_controller.command(self.time_s, self.state)
        residual_limit = self.config.residual_gimbal_limit_rad
        throttle_gate = self._terminal_throttle_gate(float(self.state[2]))
        throttle_action = float(residual_action[0]) if self.config.action_mode == "tvc_throttle" else 0.0
        if self.config.throttle_residual_positive_only:
            throttle_action = max(throttle_action, 0.0)
        residual_throttle = (
            throttle_action * self.config.residual_throttle_delta_limit * throttle_gate
            if self.config.action_mode == "tvc_throttle"
            else 0.0
        )
        pitch_index = 1 if self.config.action_mode == "tvc_throttle" else 0
        yaw_index = 2 if self.config.action_mode == "tvc_throttle" else 1
        self.last_throttle = float(
            np.clip(
                self.last_prior_throttle + residual_throttle,
                self.dynamics.propulsion.config.min_throttle,
                self.dynamics.propulsion.config.max_throttle,
            )
        )
        command = TVCCommand(
            throttle=float(self.last_throttle),
            pitch_rad=float(
                np.clip(
                    base_command.pitch_rad + residual_action[pitch_index] * residual_limit,
                    -self.dynamics.propulsion.config.gimbal_limit_rad,
                    self.dynamics.propulsion.config.gimbal_limit_rad,
                )
            ),
            yaw_rad=float(
                np.clip(
                    base_command.yaw_rad + residual_action[yaw_index] * residual_limit,
                    -self.dynamics.propulsion.config.gimbal_limit_rad,
                    self.dynamics.propulsion.config.gimbal_limit_rad,
                )
            ),
        )
        prev_state = self.state.copy()
        _, diagnostics = self.dynamics.state_derivative(
            self.time_s,
            self.state,
            command,
            return_diagnostics=True,
        )
        self._append_row(self.state, command, base_command, diagnostics.dynamic_pressure_pa)

        def derivative(local_time: float, local_state: np.ndarray) -> np.ndarray:
            return self.dynamics.state_derivative(local_time, local_state, command)
        
        next_state = rk4_step(derivative, self.time_s, self.state, self.config.dt_s)
        if next_state[2] < 0.0:
            next_state[2] = 0.0
        self.state = next_state
        self.time_s += self.config.dt_s
        self.step_count += 1
        terminated = False
        truncated = False
        termination_reason = "running"
        divergence = self._is_divergent(self.state)
        if self.state[2] <= 0.0 and self.step_count > 0:
            terminated = True
            termination_reason = "touchdown"
        elif divergence:
            terminated = True
            termination_reason = "divergence"
        elif self.step_count >= self.max_steps:
            truncated = True
            termination_reason = "timeout"
        terminal_bonus = 0.0
        if terminated or truncated:
            _, final_diagnostics = self.dynamics.state_derivative(
                self.time_s,
                self.state,
                command,
                return_diagnostics=True,
            )
            self._append_row(self.state, command, base_command, final_diagnostics.dynamic_pressure_pa)
            metrics = touchdown_metrics(
                self.trajectory_rows,
                self.dynamics.rocket.dry_mass_kg,
                criteria=self.config.success_criteria,
            )
            metrics["termination_reason"] = termination_reason
            metrics["touchdown_detected"] = termination_reason == "touchdown"
            metrics["duration_s"] = float(self.time_s)
            metrics["seed"] = float(self.seed_value)
            self.last_metrics = metrics
            terminal_bonus = self._terminal_metric_bonus(metrics)
        reward = self._compute_reward(prev_state, self.state, residual_action, terminated, truncated, divergence)
        total_reward = reward + terminal_bonus
        info: dict[str, Any] = {
            "time_s": float(self.time_s),
            "termination_reason": termination_reason,
            "throttle": float(self.last_throttle),
            "prior_throttle": float(self.last_prior_throttle),
            "curriculum_progress": self.curriculum_progress,
        }
        if self.last_metrics:
            info["metrics"] = self.last_metrics
        return self._observation(), float(total_reward), terminated, truncated, info

    def _observation(self) -> np.ndarray:
        base = self.normalizer.encode(
            state=self.state,
            target_position_m=np.zeros(3, dtype=float),
            target_velocity_mps=np.zeros(3, dtype=float),
            dry_mass_kg=self.dynamics.rocket.dry_mass_kg,
            initial_mass_kg=self.initial_mass_kg,
        )
        coordination = build_coordination_features(
            state=self.state,
            throttle=self.last_throttle,
            previous_throttle=self.previous_throttle,
            config=self.config.coordination_features,
        )
        base_command = self.lqr_controller.command(self.time_s, self.state)
        limit = max(self.dynamics.propulsion.config.gimbal_limit_rad, 1.0e-6)
        features = [
            base,
            coordination,
        ]
        if self.config.include_prior_throttle_feature:
            features.append(np.array([2.0 * float(self._prior_throttle(self.state)) - 1.0], dtype=np.float32))
        base_features = np.array(
            [
                float(np.clip(base_command.pitch_rad / limit, -1.0, 1.0)),
                float(np.clip(base_command.yaw_rad / limit, -1.0, 1.0)),
            ],
            dtype=np.float32,
        )
        features.append(base_features)
        return np.concatenate(features).astype(np.float32)

    def _compute_reward(
        self,
        prev_state: Array,
        next_state: Array,
        residual_action: Array,
        terminated: bool,
        truncated: bool,
        divergence: bool,
    ) -> float:
        del prev_state
        w = self.config.reward_weights
        lateral_error = float(np.linalg.norm(next_state[0:2]))
        horizontal_speed = float(np.linalg.norm(next_state[3:5]))
        vertical_speed = abs(float(next_state[5]))
        tilt = float(quaternion_tilt_deg(next_state[6:10]))
        angular_rate = float(np.linalg.norm(next_state[10:13]))
        residual_effort = float(np.linalg.norm(residual_action))
        residual_rate = float(np.linalg.norm(residual_action - self.previous_residual_action))
        terminal_gate = self._terminal_throttle_gate(float(next_state[2]))
        near_ground_gate = self._near_ground_touchdown_gate(float(next_state[2]))
        vz_ref = flare_reference_vertical_speed(float(next_state[2]), self.config.coordination_features)
        vertical_tracking_error = abs(float(next_state[5]) - vz_ref)
        v_safe = abs(float(vz_ref)) + float(self.config.near_ground_safe_margin_mps)
        near_ground_overspeed = max(0.0, vertical_speed - v_safe)
        reward = 0.0
        reward -= w.lateral_error * (lateral_error / 20.0)
        reward -= w.horizontal_speed * (horizontal_speed / 10.0)
        reward -= w.vertical_speed_coupling * (vertical_speed / 10.0)
        reward -= w.terminal_vertical_tracking * terminal_gate * (vertical_tracking_error / 10.0)
        reward -= w.near_ground_overspeed * near_ground_gate * near_ground_overspeed**2
        reward -= w.tilt * (tilt / 15.0)
        reward -= w.angular_rate * angular_rate
        reward -= w.residual_effort * residual_effort
        reward -= w.residual_rate * residual_rate
        if divergence:
            reward -= w.divergence_penalty
        elif terminated:
            reward += w.success_bonus if bool(self.last_metrics.get("success", False)) else -w.touchdown_failure_penalty
        elif truncated:
            reward -= w.timeout_penalty
        return float(reward)

    def _terminal_metric_bonus(self, metrics: dict[str, float | str | bool]) -> float:
        if bool(metrics.get("success", False)):
            return 0.0
        w = self.config.reward_weights
        return float(
            -w.terminal_horizontal_speed * float(metrics.get("horizontal_touchdown_velocity_mps", 0.0))
            -w.terminal_lateral_error * float(metrics.get("landing_position_error_m", 0.0))
            -w.terminal_tilt * float(metrics.get("tilt_angle_deg", 0.0))
            -w.terminal_vertical_speed * abs(float(metrics.get("vertical_touchdown_velocity_mps", 0.0)))
        )

    def _append_row(
        self,
        state: Array,
        command: TVCCommand,
        base_command: TVCCommand,
        dynamic_pressure_pa: float,
    ) -> None:
        vz_ref = flare_reference_vertical_speed(float(state[2]), self.config.coordination_features)
        self.trajectory_rows.append(
            {
                "time_s": float(self.time_s),
                "x_m": float(state[0]),
                "y_m": float(state[1]),
                "z_m": float(state[2]),
                "vx_mps": float(state[3]),
                "vy_mps": float(state[4]),
                "vz_mps": float(state[5]),
                "qw": float(state[6]),
                "qx": float(state[7]),
                "qy": float(state[8]),
                "qz": float(state[9]),
                "p_radps": float(state[10]),
                "q_radps": float(state[11]),
                "r_radps": float(state[12]),
                "mass_kg": float(state[13]),
                "throttle": float(command.throttle),
                "prior_throttle": float(self.last_prior_throttle),
                "throttle_rate_per_s": float((command.throttle - self.previous_throttle) / max(self.config.dt_s, 1.0e-6)),
                "vz_ref_mps": float(vz_ref),
                "vertical_tracking_error_mps": float(state[5] - vz_ref),
                "base_gimbal_pitch_rad": float(base_command.pitch_rad),
                "base_gimbal_yaw_rad": float(base_command.yaw_rad),
                "gimbal_pitch_rad": float(command.pitch_rad),
                "gimbal_yaw_rad": float(command.yaw_rad),
                "residual_throttle": float(command.throttle - self.last_prior_throttle),
                "residual_pitch_rad": float(command.pitch_rad - base_command.pitch_rad),
                "residual_yaw_rad": float(command.yaw_rad - base_command.yaw_rad),
                "terminal_throttle_gate": float(self._terminal_throttle_gate(float(state[2]))),
                "dynamic_pressure_pa": float(dynamic_pressure_pa),
            }
        )

    def _terminal_throttle_gate(self, altitude_m: float) -> float:
        return terminal_throttle_residual_gate(
            altitude_m=altitude_m,
            gate_altitude_m=self.config.terminal_throttle_gate_altitude_m,
            gate_power=self.config.terminal_throttle_gate_power,
        )

    def _near_ground_touchdown_gate(self, altitude_m: float) -> float:
        return terminal_throttle_residual_gate(
            altitude_m=altitude_m,
            gate_altitude_m=self.config.near_ground_gate_altitude_m,
            gate_power=self.config.near_ground_gate_power,
        )

    def _overspeed_brake_assist(self, state: Array) -> float:
        if not self.config.overspeed_brake_assist_enabled:
            return 0.0
        return overspeed_brake_assist_delta(
            altitude_m=float(state[2]),
            vertical_speed_mps=float(state[5]),
            config=self.config.coordination_features,
            gate_altitude_m=self.config.terminal_throttle_gate_altitude_m,
            gate_power=self.config.terminal_throttle_gate_power,
            trigger_mps=self.config.overspeed_brake_assist_trigger_mps,
            full_scale_mps=self.config.overspeed_brake_assist_full_scale_mps,
            max_delta=self.config.overspeed_brake_assist_max_delta,
            late_stage_altitude_m=self.config.overspeed_brake_assist_late_stage_altitude_m,
            late_stage_extra_delta=self.config.overspeed_brake_assist_late_stage_extra_delta,
        )

    def _overspeed_brake_floor(self, state: Array) -> float:
        if not self.config.brake_floor_enabled:
            return 0.0
        return overspeed_brake_floor_throttle(
            altitude_m=float(state[2]),
            vertical_speed_mps=float(state[5]),
            config=self.config.coordination_features,
            gate_altitude_m=self.config.brake_floor_altitude_m,
            gate_power=self.config.brake_floor_power,
            safe_margin_mps=self.config.brake_floor_safe_margin_mps,
            trigger_mps=self.config.brake_floor_trigger_mps,
            full_scale_mps=self.config.brake_floor_full_scale_mps,
            base_throttle=self.config.brake_floor_base_throttle,
            max_throttle=self.config.brake_floor_max_throttle,
            shape_power=self.config.brake_floor_shape_power,
            late_stage_altitude_m=self.config.brake_floor_late_stage_altitude_m,
            late_stage_extra_throttle=self.config.brake_floor_late_stage_extra_throttle,
        )

    def _energy_assist(self, state: Array) -> float:
        if not self.config.energy_assist_enabled:
            return 0.0
        return energy_assist_delta(
            altitude_m=float(state[2]),
            vertical_speed_mps=float(state[5]),
            gate_altitude_m=self.config.energy_gate_altitude_m,
            gate_power=self.config.energy_gate_power,
            full_scale=self.config.energy_full_scale,
            shape_power=self.config.energy_shape_power,
            max_delta=self.config.energy_max_delta,
            touchdown_speed_mps=self.config.energy_touchdown_speed_mps,
            braking_accel_mps2=self.config.energy_braking_accel_mps2,
            standard_gravity_mps2=self.dynamics.propulsion.config.standard_gravity_mps2,
        )

    def _guidance_brake_throttle(self, state: Array) -> float:
        if not self.config.guidance_enabled:
            return 0.0
        return guidance_brake_throttle(
            altitude_m=float(state[2]),
            vertical_speed_mps=float(state[5]),
            mass_kg=float(state[13]),
            min_throttle=self.dynamics.propulsion.config.min_throttle,
            max_throttle=self.dynamics.propulsion.config.max_throttle,
            max_thrust_n=self.dynamics.propulsion.config.max_thrust_n,
            standard_gravity_mps2=self.dynamics.propulsion.config.standard_gravity_mps2,
            gate_altitude_m=self.config.guidance_gate_altitude_m,
            gate_power=self.config.guidance_gate_power,
            target_touchdown_speed_mps=self.config.guidance_target_touchdown_speed_mps,
            altitude_floor_m=self.config.guidance_altitude_floor_m,
            base_throttle=self.config.guidance_base_throttle,
            late_stage_altitude_m=self.config.guidance_late_stage_altitude_m,
            late_stage_extra_throttle=self.config.guidance_late_stage_extra_throttle,
        )

    def _stopping_distance_floor(self, state: Array) -> float:
        if not self.config.stopping_floor_enabled:
            return 0.0
        return stopping_distance_floor_throttle(
            altitude_m=float(state[2]),
            vertical_speed_mps=float(state[5]),
            mass_kg=float(state[13]),
            min_throttle=self.dynamics.propulsion.config.min_throttle,
            max_throttle=self.dynamics.propulsion.config.max_throttle,
            max_thrust_n=self.dynamics.propulsion.config.max_thrust_n,
            standard_gravity_mps2=self.dynamics.propulsion.config.standard_gravity_mps2,
            gate_altitude_m=self.config.stopping_floor_gate_altitude_m,
            gate_power=self.config.stopping_floor_gate_power,
            altitude_floor_m=self.config.stopping_floor_altitude_floor_m,
            touchdown_speed_mps=self.config.stopping_floor_touchdown_speed_mps,
            min_downward_speed_mps=self.config.stopping_floor_min_downward_speed_mps,
        )

    def _prior_throttle(self, state: Array) -> float:
        base = self.throttle_policy.command_throttle(
            state,
            self.dynamics.rocket.dry_mass_kg,
            self.initial_mass_kg,
        )
        assist_prior = float(
            np.clip(
                base + self._overspeed_brake_assist(state) + self._energy_assist(state),
                self.dynamics.propulsion.config.min_throttle,
                self.dynamics.propulsion.config.max_throttle,
            )
        )
        return float(
            max(
                assist_prior,
                self._overspeed_brake_floor(state),
                self._guidance_brake_throttle(state),
                self._stopping_distance_floor(state),
            )
        )

    def _is_divergent(self, state: Array) -> bool:
        if not np.all(np.isfinite(state)):
            return True
        if float(np.linalg.norm(state[0:3])) > 1500.0:
            return True
        if float(np.linalg.norm(state[3:6])) > 200.0:
            return True
        return False
