#tvc env for 2b

from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from phase1_dynamics.integrator import rk4_step
from phase1_dynamics.metrics import SuccessCriteria, quaternion_tilt_deg, touchdown_metrics
from phase1_dynamics.propulsion import TVCCommand
from phase1_dynamics.scenarios import ScenarioConfig, named_scenarios
from phase1_dynamics.simulate_lqr import build_dynamics
from phase2_rl.curriculum import CurriculumBounds, TrainingCurriculum
from phase2_rl.gym_compat import gym, spaces
from phase2_rl.observations import ObservationNormalizer
from .coordination_features import CoordinationFeatureConfig, build_coordination_features
from .hierarchical_controller import ThrottlePolicy

Array = NDArray[np.float64]

def scenario_by_name(name: str) -> ScenarioConfig:
    for scenario in named_scenarios():
        if scenario.name == name:
            return scenario
    raise KeyError(f"Unknown Phase 2B TVC scenario: {name}")


@dataclass(frozen=True)
class TVCRewardWeights:
    lateral_progress: float = 0.0
    lateral_error: float = 1.25
    outward_velocity: float = 0.0
    horizontal_speed: float = 1.10
    tilt: float = 0.30
    angular_rate: float = 0.16
    control_effort: float = 0.03
    vertical_speed_coupling: float = 0.15
    success_bonus: float = 450.0
    touchdown_failure_penalty: float = 180.0
    timeout_penalty: float = 220.0
    divergence_penalty: float = 320.0
    terminal_horizontal_speed: float = 12.0
    terminal_lateral_error: float = 12.0
    terminal_tilt: float = 4.0
    terminal_vertical_speed: float = 2.0

@dataclass(frozen=True)
class TVCEnvConfig:
    scenario_name: str = "nominal"
    dt_s: float = 0.05
    max_duration_s: float = 30.0
    success_criteria: SuccessCriteria = field(default_factory=SuccessCriteria)
    reward_weights: TVCRewardWeights = field(default_factory=TVCRewardWeights)
    curriculum_enabled: bool = True
    staged_wind_enabled: bool = True
    curriculum_bounds: CurriculumBounds = field(default_factory=CurriculumBounds)
    coordination_features: CoordinationFeatureConfig = field(default_factory=CoordinationFeatureConfig)

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
        }


class TVCPolicyEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, throttle_policy: ThrottlePolicy, config: TVCEnvConfig | None = None, seed: int = 7) -> None:
        self.throttle_policy = throttle_policy
        self.config = config or TVCEnvConfig()
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
        self.action_space = spaces.Box(
            low=np.full(2, -1.0, dtype=np.float32),
            high=np.full(2, 1.0, dtype=np.float32),
            shape=(2,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=np.full(19, -np.inf, dtype=np.float32),
            high=np.full(19, np.inf, dtype=np.float32),
            shape=(19,),
            dtype=np.float32,
        )
        self.state = np.zeros(14, dtype=float)
        self.time_s = 0.0
        self.step_count = 0
        self.last_throttle = 0.0
        self.previous_throttle = 0.0
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
        self.last_throttle = self.throttle_policy.command_throttle(
            self.state,
            self.dynamics.rocket.dry_mass_kg,
            self.initial_mass_kg,
        )
        self.previous_throttle = float(self.last_throttle)
        self.trajectory_rows = []
        self.last_metrics = {}
        return self._observation(), {
            "scenario": self.scenario.name,
            "curriculum_progress": self.curriculum_progress,
        }

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        clipped_action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        self.previous_throttle = float(self.last_throttle)
        self.last_throttle = self.throttle_policy.command_throttle(
            self.state,
            self.dynamics.rocket.dry_mass_kg,
            self.initial_mass_kg,
        )
        limit = self.dynamics.propulsion.config.gimbal_limit_rad
        command = TVCCommand(
            throttle=float(self.last_throttle),
            pitch_rad=float(clipped_action[0]) * limit,
            yaw_rad=float(clipped_action[1]) * limit,
        )
        prev_state = self.state.copy()
        _, diagnostics = self.dynamics.state_derivative(
            self.time_s,
            self.state,
            command,
            return_diagnostics=True,
        )
        self._append_row(self.state, command, diagnostics.dynamic_pressure_pa)

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
        divergence = self._is_divergent(self.state)
        termination_reason = "running"
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
            self._append_row(self.state, command, final_diagnostics.dynamic_pressure_pa)
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
        reward = self._compute_reward(prev_state, self.state, clipped_action, terminated, truncated, divergence)
        total_reward = reward + terminal_bonus
        info: dict[str, Any] = {
            "time_s": float(self.time_s),
            "termination_reason": termination_reason,
            "throttle": float(self.last_throttle),
            "throttle_rate": float((self.last_throttle - self.previous_throttle) / max(self.config.dt_s, 1.0e-6)),
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
        return np.concatenate((base, coordination)).astype(np.float32)

    def _compute_reward(
        self,
        prev_state: Array,
        next_state: Array,
        action: Array,
        terminated: bool,
        truncated: bool,
        divergence: bool,
    ) -> float:
        w = self.config.reward_weights
        lateral_error = float(np.linalg.norm(next_state[0:2]))
        prev_lateral_error = float(np.linalg.norm(prev_state[0:2]))
        lateral_progress = prev_lateral_error - lateral_error
        horizontal_speed = float(np.linalg.norm(next_state[3:5]))
        if lateral_error > 1.0e-6:
            radial_direction = next_state[0:2] / lateral_error
            outward_velocity = float(np.dot(radial_direction, next_state[3:5]))
        else:
            outward_velocity = 0.0
        vertical_speed = abs(float(next_state[5]))
        tilt_deg = quaternion_tilt_deg(next_state[6:10])
        angular_rate_norm = float(np.linalg.norm(next_state[10:13]))
        control_effort = float(np.linalg.norm(action))
        reward = 0.0
        reward += w.lateral_progress * lateral_progress
        reward -= w.lateral_error * (lateral_error / 20.0)
        reward -= w.outward_velocity * max(outward_velocity, 0.0)
        reward -= w.horizontal_speed * (horizontal_speed / 10.0)
        reward -= w.vertical_speed_coupling * (vertical_speed / 10.0)
        reward -= w.tilt * (tilt_deg / 15.0)
        reward -= w.angular_rate * angular_rate_norm
        reward -= w.control_effort * control_effort
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

    def _append_row(self, state: Array, command: TVCCommand, dynamic_pressure_pa: float) -> None:
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
                "throttle_rate_per_s": float(
                    (command.throttle - self.previous_throttle) / max(self.config.dt_s, 1.0e-6)
                ),
                "gimbal_pitch_rad": float(command.pitch_rad),
                "gimbal_yaw_rad": float(command.yaw_rad),
                "dynamic_pressure_pa": float(dynamic_pressure_pa),
            }
        )

    def _is_divergent(self, state: Array) -> bool:
        limits = self.config.success_criteria
        position_norm = float(np.linalg.norm(state[0:3]))
        speed_norm = float(np.linalg.norm(state[3:6]))
        return (
            not np.all(np.isfinite(state))
            or position_norm > limits.max_position_norm_m
            or speed_norm > limits.max_speed_norm_mps
        )
