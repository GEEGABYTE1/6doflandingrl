from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

from phase1_dynamics.integrator import rk4_step
from phase1_dynamics.metrics import quaternion_tilt_deg, touchdown_metrics
from phase1_dynamics.scenarios import named_scenarios
from phase1_dynamics.simulate_lqr import build_dynamics
from phase2b_hierarchical_rl.coordination_features import flare_reference_vertical_speed
from phase2b_hierarchical_rl.hierarchical_controller import FrozenThrottlePolicy
from phase1_dynamics.lqr_controller import GainScheduledLQRController

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from phase2c_hybrid_rl.hybrid_controller import FrozenResidualPolicy, HybridResidualController
else:
    from .hybrid_controller import FrozenResidualPolicy, HybridResidualController

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--scenario", type=str, default="nominal")
    parser.add_argument("--throttle-model", type=Path, required=True)
    parser.add_argument("--residual-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--max-duration", type=float, default=30.0)
    return parser.parse_args()

def scenario_by_name(name: str):
    for scenario in named_scenarios():
        if scenario.name == name:
            return scenario
    raise KeyError(f"Unknown Phase 2C evaluation scenario: {name}")

def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def near_touchdown_summary(rows: list[dict[str, float]], terminal_descent_rate_mps: float = 0.8) -> dict[str, object]:
    final_time = float(rows[-1]["time_s"])
    window_rows = [row for row in rows if final_time - float(row["time_s"]) <= 1.0]
    hspeed = np.array([np.hypot(row["vx_mps"], row["vy_mps"]) for row in window_rows], dtype=float)
    tilt = np.array([
        quaternion_tilt_deg(np.array([row["qw"], row["qx"], row["qy"], row["qz"]], dtype=float))
        for row in window_rows
    ])
    tracking = np.array([row["vertical_tracking_error_mps"] for row in window_rows], dtype=float)
    throttle = np.array([row["throttle"] for row in window_rows], dtype=float)
    throttle_rate = np.array([row["throttle_rate_per_s"] for row in window_rows], dtype=float)
    return {
        "window_s": 1.0,
        "samples": len(window_rows),
        "time_start_s": float(window_rows[0]["time_s"]),
        "time_end_s": final_time,
        "mean_tilt_deg": float(np.mean(tilt)),
        "max_tilt_deg": float(np.max(tilt)),
        "mean_horizontal_speed_mps": float(np.mean(hspeed)),
        "max_horizontal_speed_mps": float(np.max(hspeed)),
        "mean_vertical_tracking_error_mps": float(np.mean(tracking)),
        "max_abs_vertical_tracking_error_mps": float(np.max(np.abs(tracking))),
        "mean_throttle": float(np.mean(throttle)),
        "max_throttle_rate_per_s": float(np.max(np.abs(throttle_rate))),
        "final_vz_ref_mps": -float(terminal_descent_rate_mps),
    }

def main() -> None:
    args = parse_args()
    scenario = scenario_by_name(args.scenario)
    dynamics = build_dynamics(args.seed, scenario)
    throttle_policy = FrozenThrottlePolicy.from_path(args.throttle_model)
    residual_policy = FrozenResidualPolicy.from_path(args.residual_model, engine=dynamics.propulsion.config)
    controller = HybridResidualController(
        throttle_policy=throttle_policy,
        residual_policy=residual_policy,
        lqr_controller=GainScheduledLQRController(engine=dynamics.propulsion.config),
        engine=dynamics.propulsion.config,
    )
    controller.reset()
    state = scenario.initial_state(args.seed).astype(float)
    initial_mass_kg = float(state[13])
    rows: list[dict[str, float]] = []
    previous_throttle = float(throttle_policy.command_throttle(state, dynamics.rocket.dry_mass_kg, initial_mass_kg))
    steps = int(np.ceil(args.max_duration / args.dt))
    termination_reason = "timeout"
    for step in range(steps + 1):
        time_s = step * args.dt
        prior_throttle = controller.prior_throttle(state, initial_mass_kg)
        command = controller.command(time_s, state, initial_mass_kg)
        base_command = controller.lqr_controller.command(time_s, state)
        _, diagnostics = dynamics.state_derivative(time_s, state, command, return_diagnostics=True)
        vz_ref = flare_reference_vertical_speed(float(state[2]), residual_policy.coordination_config)
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
                "prior_throttle": float(prior_throttle),
                "throttle_rate_per_s": float((command.throttle - previous_throttle) / max(args.dt, 1.0e-6)),
                "vz_ref_mps": float(vz_ref),
                "vertical_tracking_error_mps": float(state[5] - vz_ref),
                "base_gimbal_pitch_rad": float(base_command.pitch_rad),
                "base_gimbal_yaw_rad": float(base_command.yaw_rad),
                "gimbal_pitch_rad": float(command.pitch_rad),
                "gimbal_yaw_rad": float(command.yaw_rad),
                "residual_throttle": float(command.throttle - prior_throttle),
                "residual_pitch_rad": float(command.pitch_rad - base_command.pitch_rad),
                "residual_yaw_rad": float(command.yaw_rad - base_command.yaw_rad),
                "dynamic_pressure_pa": float(diagnostics.dynamic_pressure_pa),
            }
        )
        previous_throttle = float(command.throttle)
        if state[2] <= 0.0 and step > 0:
            state[2] = 0.0
            termination_reason = "touchdown"
            break

        def derivative(local_time: float, local_state: np.ndarray) -> np.ndarray:
            local_command = controller.command(local_time, local_state, initial_mass_kg)
            return dynamics.state_derivative(local_time, local_state, local_command)

        state = rk4_step(derivative, time_s, state, args.dt)
        if state[2] < 0.0:
            state[2] = 0.0
    metrics = touchdown_metrics(rows, dynamics.rocket.dry_mass_kg)
    metrics["termination_reason"] = termination_reason
    metrics["touchdown_detected"] = termination_reason == "touchdown"
    metrics["seed"] = float(args.seed)
    metrics["duration_s"] = float(rows[-1]["time_s"])
    metrics["controller_type"] = "hybrid_residual_ppo"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "trajectory.csv", rows)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    (args.output_dir / "touchdown_regime_summary.json").write_text(
        json.dumps(near_touchdown_summary(rows), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (args.output_dir / "config.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "scenario": args.scenario,
                "dt_s": args.dt,
                "max_duration_s": args.max_duration,
                "throttle_model": str(args.throttle_model),
                "residual_model": str(args.residual_model),
                "controller_type": "hybrid_residual_ppo",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

if __name__ == "__main__":
    main()
