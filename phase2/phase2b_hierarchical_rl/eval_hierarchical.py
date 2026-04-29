#evaluating the composed phase 2b hierarchical ppo controller
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import numpy as np

from phase1_dynamics.integrator import rk4_step
from phase1_dynamics.metrics import quaternion_tilt_deg, touchdown_metrics
from phase1_dynamics.scenarios import named_scenarios
from phase1_dynamics.simulate_lqr import build_dynamics
from phase2b_hierarchical_rl.coordination_features import (
    CoordinationFeatureConfig,
    flare_reference_vertical_speed,
)
from phase2b_hierarchical_rl.hierarchical_controller import (
    FrozenThrottlePolicy,
    FrozenTVCPolicy,
    HierarchicalPolicyController,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7, help="Deterministic evaluation seed.")
    parser.add_argument("--scenario", type=str, default="nominal", help="Phase 1 scenario name.")
    parser.add_argument("--throttle-model", type=Path, required=True, help="Trained throttle PPO model.")
    parser.add_argument("--tvc-model", type=Path, required=True, help="Trained TVC PPO model.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Evaluation artifact directory.")
    parser.add_argument("--dt", type=float, default=0.05, help="Rollout step size.")
    parser.add_argument("--max-duration", type=float, default=30.0, help="Rollout duration limit.")
    return parser.parse_args()


def scenario_by_name(name: str):
    for scenario in named_scenarios():
        if scenario.name == name:
            return scenario
    raise KeyError(f"Unknown Phase 2B evaluation scenario: {name}")


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def near_touchdown_summary(
    rows: list[dict[str, float]],
    config: CoordinationFeatureConfig,
    window_s: float = 1.0,
) -> dict[str, object]:
    final_time = float(rows[-1]["time_s"])
    window_rows = [row for row in rows if final_time - float(row["time_s"]) <= window_s]
    horizontal_speed = np.array(
        [np.hypot(row["vx_mps"], row["vy_mps"]) for row in window_rows],
        dtype=float,
    )
    tilt = np.array(
        [
            quaternion_tilt_deg(
                np.array([row["qw"], row["qx"], row["qy"], row["qz"]], dtype=float)
            )
            for row in window_rows
        ],
        dtype=float,
    )
    vz_ref = np.array(
        [flare_reference_vertical_speed(row["z_m"], config) for row in window_rows],
        dtype=float,
    )
    tracking_error = np.array([row["vz_mps"] for row in window_rows], dtype=float) - vz_ref
    throttle = np.array([row["throttle"] for row in window_rows], dtype=float)
    throttle_rate = np.array([row.get("throttle_rate_per_s", 0.0) for row in window_rows], dtype=float)
    return {
        "window_s": float(window_s),
        "samples": int(len(window_rows)),
        "time_start_s": float(window_rows[0]["time_s"]),
        "time_end_s": final_time,
        "mean_tilt_deg": float(np.mean(tilt)),
        "max_tilt_deg": float(np.max(tilt)),
        "mean_horizontal_speed_mps": float(np.mean(horizontal_speed)),
        "max_horizontal_speed_mps": float(np.max(horizontal_speed)),
        "mean_vertical_tracking_error_mps": float(np.mean(tracking_error)),
        "max_abs_vertical_tracking_error_mps": float(np.max(np.abs(tracking_error))),
        "mean_throttle": float(np.mean(throttle)),
        "max_throttle_rate_per_s": float(np.max(np.abs(throttle_rate))),
        "final_vz_ref_mps": float(vz_ref[-1]),
    }


def main() -> None:
    args = parse_args()
    scenario = scenario_by_name(args.scenario)
    dynamics = build_dynamics(args.seed, scenario)
    throttle_policy = FrozenThrottlePolicy.from_path(args.throttle_model)
    tvc_policy = FrozenTVCPolicy.from_path(args.tvc_model, engine=dynamics.propulsion.config)
    controller = HierarchicalPolicyController(
        throttle_policy=throttle_policy,
        tvc_policy=tvc_policy,
        engine=dynamics.propulsion.config,
    )
    controller.reset()
    state = scenario.initial_state(args.seed).astype(float)
    initial_mass_kg = float(state[13])
    rows: list[dict[str, float]] = []
    coordination_config = tvc_policy.coordination_config
    previous_throttle = float(throttle_policy.command_throttle(state, dynamics.rocket.dry_mass_kg, initial_mass_kg))
    steps = int(np.ceil(args.max_duration / args.dt))
    termination_reason = "timeout"
    time_s = 0.0
    for step in range(steps + 1):
        time_s = step * args.dt
        command = controller.command(state, initial_mass_kg=initial_mass_kg)
        _, diagnostics = dynamics.state_derivative(time_s, state, command, return_diagnostics=True)
        vz_ref = flare_reference_vertical_speed(float(state[2]), coordination_config)
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
                "throttle_rate_per_s": float((command.throttle - previous_throttle) / max(args.dt, 1.0e-6)),
                "vz_ref_mps": float(vz_ref),
                "vertical_tracking_error_mps": float(state[5] - vz_ref),
                "gimbal_pitch_rad": float(command.pitch_rad),
                "gimbal_yaw_rad": float(command.yaw_rad),
                "dynamic_pressure_pa": float(diagnostics.dynamic_pressure_pa),
            }
        )
        previous_throttle = float(command.throttle)
        if state[2] <= 0.0 and step > 0:
            state[2] = 0.0
            termination_reason = "touchdown"
            break

        def derivative(local_time: float, local_state: np.ndarray) -> np.ndarray:
            local_command = controller.command(local_state, initial_mass_kg=initial_mass_kg)
            return dynamics.state_derivative(local_time, local_state, local_command)

        state = rk4_step(derivative, time_s, state, args.dt)
        if state[2] < 0.0:
            state[2] = 0.0

    metrics = touchdown_metrics(rows, dynamics.rocket.dry_mass_kg)
    metrics["termination_reason"] = termination_reason
    metrics["touchdown_detected"] = termination_reason == "touchdown"
    metrics["seed"] = float(args.seed)
    metrics["duration_s"] = float(rows[-1]["time_s"])
    metrics["controller_type"] = "hierarchical_ppo"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "trajectory.csv", rows)
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2, sort_keys=True)
    with (args.output_dir / "touchdown_regime_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(near_touchdown_summary(rows, coordination_config), stream, indent=2, sort_keys=True)
    with (args.output_dir / "config.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "seed": args.seed,
                "scenario": args.scenario,
                "dt_s": args.dt,
                "max_duration_s": args.max_duration,
                "throttle_model": str(args.throttle_model),
                "tvc_model": str(args.tvc_model),
                "controller_type": "hierarchical_ppo",
                "coordination_features": coordination_config.to_dict(),
            },
            stream,
            indent=2,
            sort_keys=True,
        )


if __name__ == "__main__":
    main()
