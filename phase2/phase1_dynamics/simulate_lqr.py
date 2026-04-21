"""Run a reproducible Phase 1 landing simulation with the classical baseline."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from phase1_dynamics.aero import AerodynamicModel
    from phase1_dynamics.atmosphere import ISAAtmosphere
    from phase1_dynamics.disturbances import (
        DisturbanceModel,
        SensorNoiseModel,
        ThrustMisalignmentModel,
        WindModel,
    )
    from phase1_dynamics.integrator import rk4_step
    from phase1_dynamics.lqr_controller import GainScheduledLQRController
    from phase1_dynamics.metrics import quaternion_tilt_deg, touchdown_metrics
    from phase1_dynamics.propulsion import EngineConfig, PropulsionModel
    from phase1_dynamics.quaternion_utils import normalize_quaternion
    from phase1_dynamics.rigid_body import RocketConfig, RocketDynamics
    from phase1_dynamics.scenarios import ScenarioConfig
else:
    from .aero import AerodynamicModel
    from .atmosphere import ISAAtmosphere
    from .disturbances import DisturbanceModel, SensorNoiseModel, ThrustMisalignmentModel, WindModel
    from .integrator import rk4_step
    from .lqr_controller import GainScheduledLQRController
    from .metrics import quaternion_tilt_deg, touchdown_metrics
    from .propulsion import EngineConfig, PropulsionModel
    from .quaternion_utils import normalize_quaternion
    from .rigid_body import RocketConfig, RocketDynamics
    from .scenarios import ScenarioConfig


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducibility.")
    parser.add_argument("--duration", type=float, default=35.0, help="Simulation duration in seconds.")
    parser.add_argument("--dt", type=float, default=0.02, help="Fixed RK4 step size in seconds.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/phase1_lqr"),
        help="Directory where CSV and JSON outputs are saved.",
    )
    return parser.parse_args()


def initial_state(seed: int) -> np.ndarray:
    """Create a deterministic off-nominal initial landing state."""
    rng = np.random.default_rng(seed)
    state = np.zeros(14, dtype=float)
    state[0:3] = np.array([12.0, -8.0, 120.0], dtype=float)
    state[3:6] = np.array([-1.0, 0.6, -18.0], dtype=float)
    state[6:10] = normalize_quaternion(
        np.array([1.0, np.deg2rad(2.0), np.deg2rad(-1.5), np.deg2rad(0.5)], dtype=float)
    )
    state[10:13] = rng.normal(0.0, 0.015, size=3)
    state[13] = 1_250.0
    return state


def build_dynamics(seed: int, scenario: ScenarioConfig | None = None) -> RocketDynamics:
    """Assemble the Phase 1 dynamics model."""
    engine_config = EngineConfig()
    if scenario is None:
        disturbances = DisturbanceModel(
            wind=WindModel(
                steady_wind_inertial_mps=np.array([1.5, -0.5, 0.0], dtype=float),
                gust_amplitude_mps=np.array([0.4, 0.2, 0.0], dtype=float),
                gust_frequency_hz=0.08,
            ),
            thrust_misalignment=ThrustMisalignmentModel(
                pitch_bias_rad=np.deg2rad(0.15),
                yaw_bias_rad=np.deg2rad(-0.10),
            ),
            sensor_noise=SensorNoiseModel(seed=seed),
        )
    else:
        disturbances = scenario.disturbances(seed)
    return RocketDynamics(
        rocket=RocketConfig(dry_mass_kg=engine_config.dry_mass_kg),
        atmosphere=ISAAtmosphere(),
        aerodynamics=AerodynamicModel(),
        propulsion=PropulsionModel(engine_config),
        disturbances=disturbances,
    )


def run_closed_loop(
    seed: int,
    duration_s: float,
    dt_s: float,
    scenario: ScenarioConfig | None = None,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    """Run the closed-loop simulation and return trajectory rows plus metrics."""
    dynamics = build_dynamics(seed, scenario)
    controller = GainScheduledLQRController(engine=dynamics.propulsion.config)
    state = scenario.initial_state(seed) if scenario is not None else initial_state(seed)
    rows: list[dict[str, float]] = []
    steps = int(np.ceil(duration_s / dt_s))
    termination_reason = "timeout"

    for step in range(steps + 1):
        time_s = step * dt_s
        command = controller.command(time_s, state)
        _, diagnostics = dynamics.state_derivative(time_s, state, command, return_diagnostics=True)
        rows.append(
            {
                "time_s": time_s,
                "x_m": state[0],
                "y_m": state[1],
                "z_m": state[2],
                "vx_mps": state[3],
                "vy_mps": state[4],
                "vz_mps": state[5],
                "qw": state[6],
                "qx": state[7],
                "qy": state[8],
                "qz": state[9],
                "p_radps": state[10],
                "q_radps": state[11],
                "r_radps": state[12],
                "mass_kg": state[13],
                "throttle": command.throttle,
                "gimbal_pitch_rad": command.pitch_rad,
                "gimbal_yaw_rad": command.yaw_rad,
                "tilt_deg": quaternion_tilt_deg(state[6:10]),
                "dynamic_pressure_pa": diagnostics.dynamic_pressure_pa,
            }
        )
        if state[2] <= 0.0 and step > 0:
            state[2] = 0.0
            termination_reason = "touchdown"
            break

        def derivative(local_time: float, local_state: np.ndarray) -> np.ndarray:
            local_command = controller.command(local_time, local_state)
            return dynamics.state_derivative(local_time, local_state, local_command)

        state = rk4_step(derivative, time_s, state, dt_s)
        if state[2] < 0.0:
            state[2] = 0.0

    metrics = touchdown_metrics(rows, dynamics.rocket.dry_mass_kg)
    metrics["seed"] = float(seed)
    metrics["duration_s"] = float(rows[-1]["time_s"])
    metrics["requested_duration_s"] = float(duration_s)
    metrics["termination_reason"] = termination_reason
    metrics["touchdown_detected"] = termination_reason == "touchdown"
    metrics["final_downrange_error_m"] = float(metrics["landing_position_error_m"])
    metrics["final_speed_mps"] = float(metrics["touchdown_speed_mps"])
    return rows, metrics


def run_simulation(seed: int, duration_s: float, dt_s: float) -> tuple[list[dict[str, float]], dict[str, float]]:
    """Run the default nominal simulation and return trajectory rows plus metrics."""
    return run_closed_loop(seed=seed, duration_s=duration_s, dt_s=dt_s, scenario=None)


def save_outputs(rows: list[dict[str, float]], metrics: dict[str, float], output_dir: Path) -> None:
    """Save trajectory CSV and metrics JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = output_dir / "trajectory.csv"
    with trajectory_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2, sort_keys=True)


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    rows, metrics = run_simulation(args.seed, args.duration, args.dt)
    save_outputs(rows, metrics, args.output_dir)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
