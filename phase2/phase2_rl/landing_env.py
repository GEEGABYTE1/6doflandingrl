#gym env
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
from .curriculum import TrainingCurriculum
from .gym_compat import gym, spaces
from .observations import ObservationNormalizer
from .reward import RewardWeights, compute_reward, terminal_metric_reward

Array = NDArray[np.float64]

def scenario_by_name(name: str) -> ScenarioConfig:
    for scenario in named_scenarios():
        if scenario.name == name:
            return scenario
    raise KeyError(f"Unknown Phase 2A scenario: {name}")


@dataclass(frozen=True)
class Phase2RLConfig:
    scenario_name: str = "nominal"
    dt_s: float = 0.05
    max_duration_s: float = 30.0
    success_criteria: SuccessCriteria = field(default_factory=SuccessCriteria)
    reward_weights: RewardWeights = field(default_factory=RewardWeights)
    max_position_norm_m: float = 1_500.0
    max_speed_norm_mps: float = 200.0
    throttle_delta_limit: float = 0.35
    curriculum_enabled: bool = False
    staged_wind_enabled: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "scenario_name": self.scenario_name,
            "dt_s": self.dt_s,
            "max_duration_s": self.max_duration_s,
            "success_criteria": self.success_criteria.to_dict(),
            "reward_weights": asdict(self.reward_weights),
            "max_position_norm_m": self.max_position_norm_m,
            "max_speed_norm_mps": self.max_speed_norm_mps,
            "throttle_delta_limit": self.throttle_delta_limit,
            "curriculum_enabled": self.curriculum_enabled,
            "staged_wind_enabled": self.staged_wind_enabled,
        }


class RocketLandingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: Phase2RLConfig | None = None, seed: int = 7) -> None:
        self.config = config or Phase2RLConfig()
        self.seed_value = int(seed)
        self.base_scenario = scenario_by_name(self.config.scenario_name)
        self.scenario = self.base_scenario
        self.rng = np.random.default_rng(self.seed_value)
        self.curriculum = (
            TrainingCurriculum(staged_wind_enabled=self.config.staged_wind_enabled)
            if self.config.curriculum_enabled
            else None
        )
        self.curriculum_progress = 0.0
        self.dynamics = build_dynamics(self.seed_value, self.scenario)
        self.normalizer = ObservationNormalizer()
        self.target_position_m = np.zeros(3, dtype=float)
        self.target_velocity_mps = np.zeros(3, dtype=float)
        self.initial_mass_kg = float(self.scenario.initial_mass_kg)
        self.max_steps = int(np.ceil(self.config.max_duration_s / self.config.dt_s))
        self.action_space = spaces.Box(
            low=np.full(3, -1.0, dtype=np.float32),
            high=np.full(3, 1.0, dtype=np.float32),
            shape=(3,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=np.full(14, -np.inf, dtype=np.float32),
            high=np.full(14, np.inf, dtype=np.float32),
            shape=(14,),
            dtype=np.float32,
        )
        self.state = np.zeros(14, dtype=float)
        self.time_s = 0.0
        self.step_count = 0
        self.trajectory_rows: list[dict[str, float]] = []
        self.last_metrics: dict[str, Any] = {}
        self.last_reward_breakdown: dict[str, float] = {}

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
        self.time_s = 0.0
        self.step_count = 0
        self.trajectory_rows = []
        self.last_metrics = {}
        self.last_reward_breakdown = {}
        observation = self._observation()
        return observation, {
            "scenario": self.scenario.name,
            "curriculum_progress": self.curriculum_progress,
        }

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        clipped_action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        command = self._action_to_command(clipped_action)
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
            terminal_bonus = terminal_metric_reward(metrics, self.config.reward_weights)
        reward_breakdown = compute_reward(
            prev_state=prev_state,
            next_state=self.state,
            action=clipped_action,
            terminated=terminated,
            truncated=truncated,
            touchdown_success=bool(self.last_metrics.get("success", False)),
            divergence=divergence,
            weights=self.config.reward_weights,
        )
        total_reward = reward_breakdown.total + terminal_bonus
        self.last_reward_breakdown = reward_breakdown.to_dict()
        self.last_reward_breakdown["terminal_metric_bonus"] = float(terminal_bonus)
        self.last_reward_breakdown["total"] = float(total_reward)
        info: dict[str, Any] = {
            "time_s": float(self.time_s),
            "termination_reason": termination_reason,
            "reward_breakdown": self.last_reward_breakdown,
            "curriculum_progress": self.curriculum_progress,
        }
        if self.last_metrics:
            info["metrics"] = self.last_metrics
        return self._observation(), float(total_reward), terminated, truncated, info

    def _observation(self) -> np.ndarray:
        return self.normalizer.encode(
            state=self.state,
            target_position_m=self.target_position_m,
            target_velocity_mps=self.target_velocity_mps,
            dry_mass_kg=self.dynamics.rocket.dry_mass_kg,
            initial_mass_kg=self.initial_mass_kg,
        )

    def _action_to_command(self, action: Array) -> TVCCommand:
        mass = max(float(self.state[13]), self.dynamics.rocket.dry_mass_kg)
        hover_throttle = mass * self.dynamics.rocket.gravity_mps2 / self.dynamics.propulsion.config.max_thrust_n
        throttle = hover_throttle + float(action[0]) * self.config.throttle_delta_limit
        gimbal_limit = self.dynamics.propulsion.config.gimbal_limit_rad
        return TVCCommand(
            throttle=float(np.clip(throttle, self.dynamics.propulsion.config.min_throttle, self.dynamics.propulsion.config.max_throttle)),
            pitch_rad=float(action[1]) * gimbal_limit,
            yaw_rad=float(action[2]) * gimbal_limit,
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
                "gimbal_pitch_rad": float(command.pitch_rad),
                "gimbal_yaw_rad": float(command.yaw_rad),
                "tilt_deg": float(quaternion_tilt_deg(state[6:10])),
                "dynamic_pressure_pa": float(dynamic_pressure_pa),
            }
        )

    def _is_divergent(self, state: Array) -> bool:
        position_norm = float(np.linalg.norm(state[0:3]))
        speed_norm = float(np.linalg.norm(state[3:6]))
        if position_norm > self.config.max_position_norm_m:
            return True
        if speed_norm > self.config.max_speed_norm_mps:
            return True
        if not np.all(np.isfinite(state)):
            return True
        return False
