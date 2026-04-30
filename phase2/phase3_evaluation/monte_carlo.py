#monte carlo sims on all controllers from phase 2
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Protocol
from stable_baselines3 import PPO
import numpy as np

from phase1_dynamics.integrator import rk4_step
from phase1_dynamics.lqr_controller import GainScheduledLQRController
from phase1_dynamics.metrics import SuccessCriteria, touchdown_metrics
from phase1_dynamics.propulsion import EngineConfig, TVCCommand
from phase1_dynamics.scenarios import ScenarioConfig, named_scenarios
from phase1_dynamics.simulate_lqr import build_dynamics, run_closed_loop
from phase2_rl.landing_env import ObservationNormalizer, Phase2RLConfig
from phase2b_hierarchical_rl.hierarchical_controller import (
    FrozenThrottlePolicy,
    FrozenTVCPolicy,
    HierarchicalPolicyController,
)
from phase2c_hybrid_rl.hybrid_controller import FrozenResidualPolicy, HybridResidualController

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from phase3_evaluation.metrics import (
        METRIC_DEFINITIONS,
        RAW_METRICS,
        count_failure_modes,
        summarize_controller_overall,
        summarize_episode_rows,
        write_csv,
        write_json,
    )
else:
    from .metrics import (
        METRIC_DEFINITIONS,
        RAW_METRICS,
        count_failure_modes,
        summarize_controller_overall,
        summarize_episode_rows,
        write_csv,
        write_json,
    )


DEFAULT_PHASE2A_MODEL = Path("outputs/phase2_rl/go_no_go_baseline_v2_default_seed7_50k/model.zip")
DEFAULT_PHASE2B_THROTTLE_MODEL = Path("outputs/phase2b_hierarchical_rl/throttle_seed7_40k_flare_tracking_v1/model.zip")
DEFAULT_PHASE2B_TVC_MODEL = Path("outputs/phase2b_hierarchical_rl/tvc_seed7_50k_flare_tracking_v1/model.zip")
DEFAULT_PHASE2C_THROTTLE_MODEL = Path("outputs/phase2b_hierarchical_rl/throttle_seed7_40k_touchdown_gate_commit_v1/model.zip")
DEFAULT_PHASE2C_RESIDUAL_MODEL = Path("outputs/phase2c_hybrid_rl/residual_seed7_50k_stopping_floor_v1/model.zip")
DEFAULT_OUTPUT_DIR = Path("outputs/phase3_evaluation/mc_seed7_500ep_composite_levels_v1")

@dataclass(frozen=True)
class DisturbanceLevel:
    name: str
    index: int
    wind_mps: tuple[float, float, float]
    gust_mps: tuple[float, float, float]
    misalignment_deg: tuple[float, float]


@dataclass(frozen=True)
class ControllerSpec:
    controller_id: str
    controller_label: str
    controller_type: str


class CommandAdapter(Protocol):
    def reset(self) -> None:
        """Reset rollout-stateful controller internals."""
    def command(self, time_s: float, state: np.ndarray, initial_mass_kg: float) -> TVCCommand:
        """Return a full 6DOF control command."""


def nominal_scenario() -> ScenarioConfig:
    for scenario in named_scenarios():
        if scenario.name == "nominal":
            return scenario
    raise KeyError("Nominal scenario not found.")


def composite_disturbance_levels() -> tuple[DisturbanceLevel, ...]:
    return (
        DisturbanceLevel("level_0", 0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0)),
        DisturbanceLevel("level_1", 1, (2.0, 0.0, 0.0), (0.5, 0.25, 0.0), (0.20, -0.20)),
        DisturbanceLevel("level_2", 2, (5.0, 0.0, 0.0), (1.0, 0.5, 0.0), (0.50, -0.50)),
        DisturbanceLevel("level_3", 3, (8.0, 0.0, 0.0), (1.5, 0.75, 0.0), (1.00, -1.00)),
    )


def scenario_for_level(level: DisturbanceLevel) -> ScenarioConfig:
    base = nominal_scenario()
    return ScenarioConfig(
        name=f"phase3_{level.name}",
        description=f"Phase 3 composite disturbance {level.name}.",
        initial_position_m=base.initial_position_m,
        initial_velocity_mps=base.initial_velocity_mps,
        initial_attitude_error_deg=base.initial_attitude_error_deg,
        initial_mass_kg=base.initial_mass_kg,
        wind_mps=level.wind_mps,
        gust_mps=level.gust_mps,
        gust_frequency_hz=base.gust_frequency_hz,
        misalignment_deg=level.misalignment_deg,
        duration_s=base.duration_s,
        dt_s=base.dt_s,
    )

def episode_seed(base_seed: int, controller_id: str, disturbance_level: str, episode_index: int) -> int:
    token = f"{controller_id}:{disturbance_level}:{episode_index}"
    checksum = sum((idx + 1) * ord(ch) for idx, ch in enumerate(token))
    return int(base_seed + checksum)


class Phase1LQRAdapter:
    def __init__(self, engine) -> None:
        self.controller = GainScheduledLQRController(engine=engine)

    def reset(self) -> None:
        """LQR controller has no rollout state to reset."""

    def command(self, time_s: float, state: np.ndarray, initial_mass_kg: float) -> TVCCommand:
        del initial_mass_kg
        return self.controller.command(time_s, state)

class Phase2AFlatPPOAdapter:
    def __init__(self, model: object, throttle_delta_limit: float) -> None:
        self.model = model
        self.throttle_delta_limit = float(throttle_delta_limit)
        self.normalizer = ObservationNormalizer()
        self.target_position_m = np.zeros(3, dtype=float)
        self.target_velocity_mps = np.zeros(3, dtype=float)
        self.engine = None

    def bind_engine(self, engine) -> None:
        self.engine = engine

    def reset(self) -> None:
        """Flat PPO adapter has no rollout-state history."""

    def command(self, time_s: float, state: np.ndarray, initial_mass_kg: float) -> TVCCommand:
        del time_s
        if self.engine is None:
            raise RuntimeError("Phase2AFlatPPOAdapter requires bind_engine() before rollout.")
        dry_mass_kg = self.engine.dry_mass_kg
        observation = self.normalizer.encode(
            state=state,
            target_position_m=self.target_position_m,
            target_velocity_mps=self.target_velocity_mps,
            dry_mass_kg=dry_mass_kg,
            initial_mass_kg=initial_mass_kg,
        )
        action, _ = self.model.predict(observation, deterministic=True)
        action = np.clip(np.asarray(action, dtype=float).reshape(3), -1.0, 1.0)
        hover_throttle = float(state[13]) * self.engine.standard_gravity_mps2 / self.engine.max_thrust_n
        throttle = hover_throttle + float(action[0]) * self.throttle_delta_limit
        return TVCCommand(
            throttle=float(np.clip(throttle, self.engine.min_throttle, self.engine.max_throttle)),
            pitch_rad=float(action[1]) * self.engine.gimbal_limit_rad,
            yaw_rad=float(action[2]) * self.engine.gimbal_limit_rad,
        )


class Phase2BHierarchicalAdapter:
    """Uniform command adapter around the hierarchical controller."""

    def __init__(self, controller: HierarchicalPolicyController) -> None:
        self.controller = controller

    def reset(self) -> None:
        self.controller.reset()

    def command(self, time_s: float, state: np.ndarray, initial_mass_kg: float) -> TVCCommand:
        del time_s
        return self.controller.command(state, initial_mass_kg)


class Phase2CHybridAdapter:
    def __init__(self, controller: HybridResidualController) -> None:
        self.controller = controller

    def reset(self) -> None:
        self.controller.reset()

    def command(self, time_s: float, state: np.ndarray, initial_mass_kg: float) -> TVCCommand:
        return self.controller.command(time_s, state, initial_mass_kg)


def run_adapter_rollout(
    adapter: CommandAdapter,
    scenario: ScenarioConfig,
    seed: int,
    dt_s: float,
    max_duration_s: float,
    controller_type: str,
) -> dict[str, object]:
    dynamics = build_dynamics(seed, scenario)
    adapter.reset()
    state = scenario.initial_state(seed).astype(float)
    initial_mass_kg = float(state[13])
    rows: list[dict[str, float]] = []
    steps = int(np.ceil(max_duration_s / dt_s))
    termination_reason = "timeout"

    for step in range(steps + 1):
        time_s = step * dt_s
        command = adapter.command(time_s, state, initial_mass_kg)
        _, diagnostics = dynamics.state_derivative(time_s, state, command, return_diagnostics=True)
        rows.append(
            {
                "time_s": float(time_s),
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
                "dynamic_pressure_pa": float(diagnostics.dynamic_pressure_pa),
            }
        )
        if state[2] <= 0.0 and step > 0:
            state[2] = 0.0
            termination_reason = "touchdown"
            break

        def derivative(local_time: float, local_state: np.ndarray) -> np.ndarray:
            local_command = adapter.command(local_time, local_state, initial_mass_kg)
            return dynamics.state_derivative(local_time, local_state, local_command)

        state = rk4_step(derivative, time_s, state, dt_s)
        if state[2] < 0.0:
            state[2] = 0.0

    metrics = touchdown_metrics(rows, dynamics.rocket.dry_mass_kg, criteria=SuccessCriteria())
    metrics["seed"] = float(seed)
    metrics["duration_s"] = float(rows[-1]["time_s"])
    metrics["termination_reason"] = termination_reason
    metrics["touchdown_detected"] = termination_reason == "touchdown"
    metrics["controller_type"] = controller_type
    return metrics


def load_phase2a_model(model_path: Path) -> tuple[object, float]:
    config = json.loads((model_path.resolve().parent / "config.json").read_text(encoding="utf-8"))
    phase2_cfg = config.get("phase2_config", {})
    return PPO.load(str(model_path)), float(phase2_cfg.get("throttle_delta_limit", Phase2RLConfig().throttle_delta_limit))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--episodes-per-level", type=int, default=500)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--max-duration", type=float, default=30.0)
    parser.add_argument(
        "--controllers",
        nargs="+",
        choices=["phase1_lqr", "phase2a_flat_ppo", "phase2b_hierarchical_ppo", "phase2c_hybrid_ppo"],
        default=None,
    )
    parser.add_argument("--phase2a-model", type=Path, default=DEFAULT_PHASE2A_MODEL)
    parser.add_argument("--phase2b-throttle-model", type=Path, default=DEFAULT_PHASE2B_THROTTLE_MODEL)
    parser.add_argument("--phase2b-tvc-model", type=Path, default=DEFAULT_PHASE2B_TVC_MODEL)
    parser.add_argument("--phase2c-throttle-model", type=Path, default=DEFAULT_PHASE2C_THROTTLE_MODEL)
    parser.add_argument("--phase2c-residual-model", type=Path, default=DEFAULT_PHASE2C_RESIDUAL_MODEL)
    return parser.parse_args()


def build_controller_specs(selected: set[str] | None) -> list[ControllerSpec]:
    specs = [
        ControllerSpec("phase1_lqr", "Phase 1 LQR", "gain_scheduled_lqr"),
        ControllerSpec("phase2a_flat_ppo", "Phase 2A Flat PPO", "flat_ppo"),
        ControllerSpec("phase2b_hierarchical_ppo", "Phase 2B Hierarchical PPO", "hierarchical_ppo"),
        ControllerSpec("phase2c_hybrid_ppo", "Phase 2C Hybrid PPO", "hybrid_residual_ppo"),
    ]
    if selected is None:
        return specs
    return [spec for spec in specs if spec.controller_id in selected]


def build_adapter(spec: ControllerSpec, args: argparse.Namespace) -> CommandAdapter:
    engine = EngineConfig()
    if spec.controller_id == "phase1_lqr":
        return Phase1LQRAdapter(engine)
    if spec.controller_id == "phase2a_flat_ppo":
        model, throttle_delta_limit = load_phase2a_model(args.phase2a_model)
        adapter = Phase2AFlatPPOAdapter(model=model, throttle_delta_limit=throttle_delta_limit)
        adapter.bind_engine(engine)
        return adapter
    if spec.controller_id == "phase2b_hierarchical_ppo":
        throttle_policy = FrozenThrottlePolicy.from_path(args.phase2b_throttle_model)
        tvc_policy = FrozenTVCPolicy.from_path(args.phase2b_tvc_model, engine=engine)
        return Phase2BHierarchicalAdapter(
            HierarchicalPolicyController(
                throttle_policy=throttle_policy,
                tvc_policy=tvc_policy,
                engine=engine,
            )
        )
    if spec.controller_id == "phase2c_hybrid_ppo":
        throttle_policy = FrozenThrottlePolicy.from_path(args.phase2c_throttle_model)
        residual_policy = FrozenResidualPolicy.from_path(args.phase2c_residual_model, engine=engine)
        return Phase2CHybridAdapter(
            HybridResidualController(
                throttle_policy=throttle_policy,
                residual_policy=residual_policy,
                lqr_controller=GainScheduledLQRController(engine=engine),
                engine=engine,
            )
        )
    raise KeyError(f"Unknown controller spec: {spec.controller_id}")


def collect_episode_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    levels = composite_disturbance_levels()
    selected = set(args.controllers) if args.controllers else None
    specs = build_controller_specs(selected)
    for spec in specs:
        adapter = build_adapter(spec, args)
        for level in levels:
            scenario = scenario_for_level(level)
            for episode_index in range(args.episodes_per_level):
                seed = episode_seed(args.seed, spec.controller_id, level.name, episode_index)
                if spec.controller_id == "phase1_lqr":
                    metrics = run_closed_loop(
                        seed=seed,
                        duration_s=args.max_duration,
                        dt_s=args.dt,
                        scenario=scenario,
                    )[1]
                    metrics["controller_type"] = spec.controller_type
                else:
                    metrics = run_adapter_rollout(
                        adapter=adapter,
                        scenario=scenario,
                        seed=seed,
                        dt_s=args.dt,
                        max_duration_s=args.max_duration,
                        controller_type=spec.controller_type,
                    )
                row: dict[str, object] = {
                    "controller_id": spec.controller_id,
                    "controller_label": spec.controller_label,
                    "controller_type": spec.controller_type,
                    "disturbance_level": level.name,
                    "disturbance_index": level.index,
                    "episode_index": episode_index,
                    "episode_seed": seed,
                    "wind_x_mps": level.wind_mps[0],
                    "wind_y_mps": level.wind_mps[1],
                    "wind_z_mps": level.wind_mps[2],
                    "gust_x_mps": level.gust_mps[0],
                    "gust_y_mps": level.gust_mps[1],
                    "gust_z_mps": level.gust_mps[2],
                    "misalignment_pitch_deg": level.misalignment_deg[0],
                    "misalignment_yaw_deg": level.misalignment_deg[1],
                }
                for key in RAW_METRICS:
                    row[key] = metrics[key]
                rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    episode_rows = collect_episode_rows(args)
    summary_rows = summarize_episode_rows(episode_rows)
    overall_rows = summarize_controller_overall(episode_rows)
    failure_rows = count_failure_modes(episode_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "seed": args.seed,
        "episodes_per_level": args.episodes_per_level,
        "dt_s": args.dt,
        "max_duration_s": args.max_duration,
        "controllers": args.controllers or [
            "phase1_lqr",
            "phase2a_flat_ppo",
            "phase2b_hierarchical_ppo",
            "phase2c_hybrid_ppo",
        ],
        "disturbance_levels": [level.__dict__ for level in composite_disturbance_levels()],
        "metric_definitions": METRIC_DEFINITIONS,
        "phase2a_model": str(args.phase2a_model),
        "phase2b_throttle_model": str(args.phase2b_throttle_model),
        "phase2b_tvc_model": str(args.phase2b_tvc_model),
        "phase2c_throttle_model": str(args.phase2c_throttle_model),
        "phase2c_residual_model": str(args.phase2c_residual_model),
    }
    write_json(args.output_dir / "config.json", config)
    write_csv(args.output_dir / "episodes.csv", episode_rows)
    write_json(args.output_dir / "episodes.json", episode_rows)
    write_csv(args.output_dir / "summary_by_controller_level.csv", summary_rows)
    write_json(args.output_dir / "summary_by_controller_level.json", summary_rows)
    write_csv(args.output_dir / "controller_overall_summary.csv", overall_rows)
    write_json(args.output_dir / "controller_overall_summary.json", overall_rows)
    write_csv(args.output_dir / "failure_mode_counts.csv", failure_rows)
    write_json(args.output_dir / "failure_mode_counts.json", failure_rows)
    print(json.dumps({"output_dir": str(args.output_dir), "episode_count": len(episode_rows)}, indent=2))


if __name__ == "__main__":
    main()
