# reduced-order vertical descent environment for throttle policy.

from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from phase1_dynamics.metrics import SuccessCriteria
from phase1_dynamics.propulsion import EngineConfig
from phase1_dynamics.scenarios import ScenarioConfig, named_scenarios
from phase2_rl.gym_compat import gym, spaces

Array = NDArray[np.float64]

def scenario_by_name(name: str) -> ScenarioConfig:
    for scenario in named_scenarios():
        if scenario.name == name:
            return scenario
    raise KeyError(f"Unknown Phase 2B throttle scenario: {name}")


@dataclass(frozen=True)
class ThrottleRewardWeights:
    altitude_progress: float = 0.08
    vertical_speed: float = 1.10
    vertical_tracking: float = 0.0
    low_altitude_vertical_speed: float = 1.50
    touchdown_vertical_speed: float = 0.0
    touchdown_tracking: float = 0.0
    touchdown_underdescent: float = 0.0
    terminal_envelope_overspeed: float = 0.0
    terminal_envelope_underspeed: float = 0.0
    throttle_effort: float = 0.03
    throttle_rate: float = 0.00
    success_bonus: float = 250.0
    touchdown_failure_penalty: float = 180.0
    timeout_penalty: float = 150.0
    divergence_penalty: float = 250.0
    terminal_vertical_speed: float = 8.0
    potential_shaping: float = 0.0


@dataclass(frozen=True)
class ThrottleEnvConfig:
    scenario_name: str = "nominal"
    dt_s: float = 0.05
    max_duration_s: float = 25.0
    success_criteria: SuccessCriteria = field(default_factory=SuccessCriteria)
    reward_weights: ThrottleRewardWeights = field(default_factory=ThrottleRewardWeights)
    altitude_scale_m: float = 150.0
    vertical_speed_scale_mps: float = 40.0
    throttle_delta_limit: float = 0.35
    terminal_descent_rate_mps: float = 0.8
    braking_accel_mps2: float = 1.5
    touchdown_zone_altitude_m: float = 12.0
    touchdown_gate_power: float = 2.0
    terminal_envelope_altitude_m: float = 20.0
    terminal_envelope_wide_margin_mps: float = 2.0
    terminal_envelope_tight_margin_mps: float = 0.4
    timeout_above_ground_penalty: float = 0.0
    timeout_underdescent_penalty: float = 0.0
    observation_mode: str = "baseline_v1"
    potential_mode: str = "disabled"
    potential_gamma: float = 0.99
    feature_eps: float = 1.0e-3
    randomize_reset: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "scenario_name": self.scenario_name,
            "dt_s": self.dt_s,
            "max_duration_s": self.max_duration_s,
            "success_criteria": self.success_criteria.to_dict(),
            "reward_weights": asdict(self.reward_weights),
            "altitude_scale_m": self.altitude_scale_m,
            "vertical_speed_scale_mps": self.vertical_speed_scale_mps,
            "throttle_delta_limit": self.throttle_delta_limit,
            "terminal_descent_rate_mps": self.terminal_descent_rate_mps,
            "braking_accel_mps2": self.braking_accel_mps2,
            "touchdown_zone_altitude_m": self.touchdown_zone_altitude_m,
            "touchdown_gate_power": self.touchdown_gate_power,
            "terminal_envelope_altitude_m": self.terminal_envelope_altitude_m,
            "terminal_envelope_wide_margin_mps": self.terminal_envelope_wide_margin_mps,
            "terminal_envelope_tight_margin_mps": self.terminal_envelope_tight_margin_mps,
            "timeout_above_ground_penalty": self.timeout_above_ground_penalty,
            "timeout_underdescent_penalty": self.timeout_underdescent_penalty,
            "observation_mode": self.observation_mode,
            "potential_mode": self.potential_mode,
            "potential_gamma": self.potential_gamma,
            "feature_eps": self.feature_eps,
            "randomize_reset": self.randomize_reset,
        }

@dataclass(frozen=True)
class VerticalState:
    altitude_m: float
    vertical_speed_mps: float
    mass_kg: float


@dataclass(frozen=True)
class VerticalMetrics:
    success: bool
    termination_reason: str
    final_altitude_m: float
    vertical_touchdown_velocity_mps: float
    touchdown_time_s: float
    fuel_used_kg: float
    final_mass_kg: float
    fuel_margin_kg: float

    def to_dict(self) -> dict[str, float | str | bool]:
        return asdict(self)


class ThrottleObservationNormalizer:
    def __init__(
        self,
        altitude_scale_m: float,
        vertical_speed_scale_mps: float,
        observation_mode: str = "baseline_v1",
        time_to_ground_scale_s: float | None = None,
        stopping_distance_ratio_scale: float = 2.0,
        feature_eps: float = 1.0e-3,
    ) -> None:
        self.altitude_scale_m = float(altitude_scale_m)
        self.vertical_speed_scale_mps = float(vertical_speed_scale_mps)
        self.observation_mode = str(observation_mode)
        self.time_to_ground_scale_s = float(time_to_ground_scale_s or max(self.altitude_scale_m / 5.0, 1.0))
        self.stopping_distance_ratio_scale = float(stopping_distance_ratio_scale)
        self.feature_eps = float(feature_eps)

    def braking_features(self, state: VerticalState, engine: EngineConfig) -> np.ndarray:
        downward_speed = max(-float(state.vertical_speed_mps), 0.0)
        eps = max(self.feature_eps, 1.0e-6)
        time_to_ground_s = float(state.altitude_m) / max(downward_speed, eps)
        available_braking_accel = max(engine.max_thrust_n / max(float(state.mass_kg), engine.dry_mass_kg) - engine.standard_gravity_mps2, eps)
        stopping_distance_ratio = (downward_speed * downward_speed) / (2.0 * available_braking_accel * max(float(state.altitude_m), eps))
        return np.array(
            [
                float(np.clip(time_to_ground_s / self.time_to_ground_scale_s, 0.0, 5.0)),
                float(np.clip(stopping_distance_ratio / max(self.stopping_distance_ratio_scale, eps), 0.0, 5.0)),
            ],
            dtype=np.float32,
        )

    def encode(self, state: VerticalState, dry_mass_kg: float, initial_mass_kg: float, engine: EngineConfig | None = None) -> np.ndarray:
        fuel_capacity = max(float(initial_mass_kg) - float(dry_mass_kg), 1.0)
        fuel_fraction = np.clip((state.mass_kg - dry_mass_kg) / fuel_capacity, 0.0, 1.0)
        base = np.array(
            [
                state.altitude_m / self.altitude_scale_m,
                state.vertical_speed_mps / self.vertical_speed_scale_mps,
                fuel_fraction,
            ],
            dtype=np.float32,
        )
        if self.observation_mode == "baseline_v1":
            return base
        if self.observation_mode == "braking_awareness_v1":
            if engine is None:
                raise ValueError("EngineConfig is required for braking-aware throttle observations.")
            return np.concatenate((base, self.braking_features(state, engine))).astype(np.float32)
        raise KeyError(f"Unknown throttle observation mode: {self.observation_mode}")


class VerticalThrottleEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: ThrottleEnvConfig | None = None, seed: int = 7) -> None:
        self.config = config or ThrottleEnvConfig()
        self.seed_value = int(seed)
        self.base_scenario = scenario_by_name(self.config.scenario_name)
        self.engine = EngineConfig()
        self.normalizer = ThrottleObservationNormalizer(
            altitude_scale_m=self.config.altitude_scale_m,
            vertical_speed_scale_mps=self.config.vertical_speed_scale_mps,
            observation_mode=self.config.observation_mode,
            feature_eps=self.config.feature_eps,
        )
        self.rng = np.random.default_rng(self.seed_value)
        self.initial_mass_kg = float(self.base_scenario.initial_mass_kg)
        self.max_steps = int(np.ceil(self.config.max_duration_s / self.config.dt_s))
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            shape=(1,),
            dtype=np.float32,
        )
        observation_dim = 3 if self.config.observation_mode == "baseline_v1" else 5
        self.observation_space = spaces.Box(
            low=np.full(observation_dim, -np.inf, dtype=np.float32),
            high=np.full(observation_dim, np.inf, dtype=np.float32),
            shape=(observation_dim,),
            dtype=np.float32,
        )
        self.state = VerticalState(
            altitude_m=float(self.base_scenario.initial_position_m[2]),
            vertical_speed_mps=float(self.base_scenario.initial_velocity_mps[2]),
            mass_kg=float(self.base_scenario.initial_mass_kg),
        )
        self.time_s = 0.0
        self.step_count = 0
        self.last_throttle = 0.0
        self.trajectory_rows: list[dict[str, float]] = []
        self.last_metrics: dict[str, float | str | bool] = {}

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
        self.initial_mass_kg = float(self.base_scenario.initial_mass_kg)
        if self.config.randomize_reset:
            altitude = float(self.rng.uniform(20.0, self.base_scenario.initial_position_m[2]))
            vertical_speed = -float(self.rng.uniform(3.0, max(abs(self.base_scenario.initial_velocity_mps[2]), 4.0)))
            mass = float(self.rng.uniform(self.engine.dry_mass_kg + 30.0, self.base_scenario.initial_mass_kg))
        else:
            altitude = float(self.base_scenario.initial_position_m[2])
            vertical_speed = float(self.base_scenario.initial_velocity_mps[2])
            mass = float(self.base_scenario.initial_mass_kg)
        self.state = VerticalState(altitude_m=altitude, vertical_speed_mps=vertical_speed, mass_kg=mass)
        self.initial_mass_kg = mass
        self.time_s = 0.0
        self.step_count = 0
        hover_throttle = mass * self.engine.standard_gravity_mps2 / self.engine.max_thrust_n
        self.last_throttle = float(np.clip(hover_throttle, self.engine.min_throttle, self.engine.max_throttle))
        self.trajectory_rows = []
        self.last_metrics = {}
        return self._observation(), {"scenario": self.base_scenario.name}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        throttle = self._action_to_throttle(action)
        prev_state = self.state
        thrust_n = throttle * self.engine.max_thrust_n if prev_state.mass_kg > self.engine.dry_mass_kg else 0.0
        acceleration = thrust_n / max(prev_state.mass_kg, self.engine.dry_mass_kg) - self.engine.standard_gravity_mps2
        if thrust_n > 0.0:
            mass_flow = -thrust_n / (self.engine.specific_impulse_s * self.engine.standard_gravity_mps2)
        else:
            mass_flow = 0.0
        next_altitude = prev_state.altitude_m + self.config.dt_s * prev_state.vertical_speed_mps
        next_vertical_speed = prev_state.vertical_speed_mps + self.config.dt_s * acceleration
        next_mass = max(prev_state.mass_kg + self.config.dt_s * mass_flow, self.engine.dry_mass_kg)
        if next_altitude < 0.0:
            next_altitude = 0.0
        self._append_row(prev_state, throttle)
        self.state = VerticalState(
            altitude_m=float(next_altitude),
            vertical_speed_mps=float(next_vertical_speed),
            mass_kg=float(next_mass),
        )
        self.time_s += self.config.dt_s
        self.step_count += 1
        terminated = False
        truncated = False
        divergence = not np.isfinite(next_vertical_speed) or abs(next_vertical_speed) > self.config.success_criteria.max_speed_norm_mps
        termination_reason = "running"
        if self.state.altitude_m <= 0.0:
            terminated = True
            termination_reason = "touchdown"
        elif divergence:
            terminated = True
            termination_reason = "divergence"
        elif self.step_count >= self.max_steps:
            truncated = True
            termination_reason = "timeout"
        reward = self._compute_reward(prev_state, self.state, throttle, terminated, truncated, divergence)
        self.last_throttle = float(throttle)
        info: dict[str, Any] = {"time_s": float(self.time_s), "termination_reason": termination_reason}
        if terminated or truncated:
            self._append_row(self.state, throttle)
            metrics = self._metrics(termination_reason)
            self.last_metrics = metrics.to_dict()
            reward += self._terminal_metric_penalty(metrics)
            info["metrics"] = self.last_metrics
        return self._observation(), float(reward), terminated, truncated, info

    def _observation(self) -> np.ndarray:
        return self.normalizer.encode(self.state, self.engine.dry_mass_kg, self.initial_mass_kg, self.engine)

    def _action_to_throttle(self, action: np.ndarray) -> float:
        scalar = float(np.clip(np.asarray(action, dtype=float).reshape(-1)[0], -1.0, 1.0))
        hover_throttle = self.state.mass_kg * self.engine.standard_gravity_mps2 / self.engine.max_thrust_n
        throttle = hover_throttle + scalar * self.config.throttle_delta_limit
        return float(np.clip(throttle, self.engine.min_throttle, self.engine.max_throttle))

    def _compute_reward(
        self,
        prev_state: VerticalState,
        next_state: VerticalState,
        throttle: float,
        terminated: bool,
        truncated: bool,
        divergence: bool,
    ) -> float:
        w = self.config.reward_weights
        altitude_progress = prev_state.altitude_m - next_state.altitude_m
        desired_speed = np.sqrt(
            self.config.terminal_descent_rate_mps * self.config.terminal_descent_rate_mps
            + 2.0 * self.config.braking_accel_mps2 * max(next_state.altitude_m, 0.0)
        )
        desired_vz = -float(desired_speed)
        tracking_error = abs(next_state.vertical_speed_mps - desired_vz)
        reward = w.altitude_progress * altitude_progress
        reward -= w.vertical_speed * (abs(next_state.vertical_speed_mps) / 10.0)
        reward -= w.vertical_tracking * (tracking_error / 10.0)
        if next_state.altitude_m < 25.0:
            flare_weight = (25.0 - next_state.altitude_m) / 25.0
            reward -= w.low_altitude_vertical_speed * flare_weight * (abs(next_state.vertical_speed_mps) / 8.0)
        touchdown_gate = self._touchdown_gate(next_state.altitude_m)
        if touchdown_gate > 0.0:
            reward -= w.touchdown_vertical_speed * touchdown_gate * (abs(next_state.vertical_speed_mps) / 8.0)
            reward -= w.touchdown_tracking * touchdown_gate * (tracking_error / 8.0)
            underdescent = max(next_state.vertical_speed_mps - desired_vz, 0.0)
            reward -= w.touchdown_underdescent * touchdown_gate * (underdescent / 6.0)
        envelope_gate = self._terminal_envelope_gate(next_state.altitude_m)
        if envelope_gate > 0.0:
            margin = self._terminal_envelope_margin(envelope_gate)
            lower_bound = desired_vz - margin
            upper_bound = desired_vz + margin
            overspeed = max(lower_bound - next_state.vertical_speed_mps, 0.0)
            underspeed = max(next_state.vertical_speed_mps - upper_bound, 0.0)
            reward -= w.terminal_envelope_overspeed * envelope_gate * (overspeed / 6.0)
            reward -= w.terminal_envelope_underspeed * envelope_gate * (underspeed / 6.0)
        reward -= w.throttle_effort * abs(throttle)
        reward -= w.throttle_rate * abs(throttle - self.last_throttle)
        reward += w.potential_shaping * self._potential_shaping(prev_state, next_state)
        if divergence:
            reward -= w.divergence_penalty
        elif terminated:
            success = abs(next_state.vertical_speed_mps) <= self.config.success_criteria.max_vertical_speed_mps
            reward += w.success_bonus if success else -w.touchdown_failure_penalty
        elif truncated:
            reward -= w.timeout_penalty
        return float(reward)

    def _braking_potential(self, state: VerticalState) -> float:
        if self.config.potential_mode == "disabled":
            return 0.0
        if self.config.potential_mode != "stopping_distance_ratio_v1":
            raise KeyError(f"Unknown throttle potential mode: {self.config.potential_mode}")
        ratio = float(self.normalizer.braking_features(state, self.engine)[1]) * self.normalizer.stopping_distance_ratio_scale
        return -ratio

    def _potential_shaping(self, prev_state: VerticalState, next_state: VerticalState) -> float:
        if self.config.potential_mode == "disabled":
            return 0.0
        return float(self.config.potential_gamma * self._braking_potential(next_state) - self._braking_potential(prev_state))

    def _metrics(self, termination_reason: str) -> VerticalMetrics:
        criteria = self.config.success_criteria
        vertical_speed = float(self.state.vertical_speed_mps)
        success = (
            termination_reason == "touchdown"
            and self.state.altitude_m <= criteria.max_final_altitude_m
            and abs(vertical_speed) <= criteria.max_vertical_speed_mps
            and self.state.mass_kg > self.engine.dry_mass_kg + criteria.min_fuel_margin_kg
        )
        return VerticalMetrics(
            success=success,
            termination_reason=termination_reason,
            final_altitude_m=float(self.state.altitude_m),
            vertical_touchdown_velocity_mps=vertical_speed,
            touchdown_time_s=float(self.time_s),
            fuel_used_kg=float(self.initial_mass_kg - self.state.mass_kg),
            final_mass_kg=float(self.state.mass_kg),
            fuel_margin_kg=float(self.state.mass_kg - self.engine.dry_mass_kg),
        )

    def _terminal_metric_penalty(self, metrics: VerticalMetrics) -> float:
        if metrics.success:
            return 0.0
        penalty = -self.config.reward_weights.terminal_vertical_speed * abs(metrics.vertical_touchdown_velocity_mps)
        if metrics.termination_reason == "timeout":
            penalty -= self.config.timeout_above_ground_penalty * metrics.final_altitude_m
            timeout_underdescent = max(metrics.vertical_touchdown_velocity_mps + self.config.terminal_descent_rate_mps, 0.0)
            penalty -= self.config.timeout_underdescent_penalty * timeout_underdescent
        return float(penalty)

    def _touchdown_gate(self, altitude_m: float) -> float:
        zone_altitude = max(self.config.touchdown_zone_altitude_m, 1.0e-6)
        normalized = np.clip((zone_altitude - max(float(altitude_m), 0.0)) / zone_altitude, 0.0, 1.0)
        return float(normalized ** self.config.touchdown_gate_power)

    def _terminal_envelope_gate(self, altitude_m: float) -> float:
        zone_altitude = max(self.config.terminal_envelope_altitude_m, 1.0e-6)
        normalized = np.clip((zone_altitude - max(float(altitude_m), 0.0)) / zone_altitude, 0.0, 1.0)
        return float(normalized)

    def _terminal_envelope_margin(self, envelope_gate: float) -> float:
        wide = max(self.config.terminal_envelope_wide_margin_mps, 1.0e-6)
        tight = max(min(self.config.terminal_envelope_tight_margin_mps, wide), 1.0e-6)
        return float((1.0 - envelope_gate) * wide + envelope_gate * tight)

    def _append_row(self, state: VerticalState, throttle: float) -> None:
        self.trajectory_rows.append(
            {
                "time_s": float(self.time_s),
                "z_m": float(state.altitude_m),
                "vz_mps": float(state.vertical_speed_mps),
                "mass_kg": float(state.mass_kg),
                "throttle": float(throttle),
            }
        )
