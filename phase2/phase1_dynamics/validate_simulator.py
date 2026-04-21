
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from phase1_dynamics.propulsion import EngineConfig
    from phase1_dynamics.scenarios import ScenarioConfig
    from phase1_dynamics.simulate_lqr import run_closed_loop, save_outputs
else:
    from .propulsion import EngineConfig
    from .scenarios import ScenarioConfig
    from .simulate_lqr import run_closed_loop, save_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7, help="Deterministic validation seed.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/phase1_validation"),
        help="Directory for validation artifacts.",
    )
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def invariant_report(rows: list[dict[str, float]]) -> dict[str, float | bool]:
    engine = EngineConfig()
    q_norm = np.array(
        [
            np.linalg.norm([row["qw"], row["qx"], row["qy"], row["qz"]])
            for row in rows
        ],
        dtype=float,
    )
    mass = np.array([row["mass_kg"] for row in rows], dtype=float)
    throttle = np.array([row["throttle"] for row in rows], dtype=float)
    pitch = np.array([row["gimbal_pitch_rad"] for row in rows], dtype=float)
    yaw = np.array([row["gimbal_yaw_rad"] for row in rows], dtype=float)
    return {
        "max_quaternion_norm_error": float(np.max(np.abs(q_norm - 1.0))),
        "min_mass_kg": float(np.min(mass)),
        "mass_floor_respected": bool(np.min(mass) >= engine.dry_mass_kg - 1.0e-9),
        "throttle_limits_respected": bool(np.all(throttle >= -1.0e-12) and np.all(throttle <= 1.0 + 1.0e-12)),
        "gimbal_limits_respected": bool(
            np.all(np.abs(pitch) <= engine.gimbal_limit_rad + 1.0e-12)
            and np.all(np.abs(yaw) <= engine.gimbal_limit_rad + 1.0e-12)
        ),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scenario = ScenarioConfig(name="nominal_validation", description="Nominal validation case.")
    dt_values = [0.04, 0.02, 0.01]
    timestep_rows: list[dict[str, object]] = []
    validation_summary: dict[str, object] = {"seed": args.seed, "dt_values": dt_values}

    previous_error: float | None = None
    convergence_trend = True
    for dt in dt_values:
        run_scenario = ScenarioConfig(
            name=f"nominal_dt_{dt:.2f}".replace(".", "p"),
            description=f"Nominal timestep sensitivity case dt={dt:.2f}.",
            dt_s=dt,
        )
        rows, metrics = run_closed_loop(args.seed, scenario.duration_s, dt, run_scenario)
        run_dir = args.output_dir / f"dt_{dt:.2f}".replace(".", "p")
        save_outputs(rows, metrics, run_dir)
        invariants = invariant_report(rows)
        final_error = float(metrics["landing_position_error_m"]) + abs(
            float(metrics["vertical_touchdown_velocity_mps"])
        )
        if previous_error is not None and final_error > previous_error + 0.25:
            convergence_trend = False
        previous_error = final_error
        timestep_rows.append(
            {
                "dt_s": dt,
                **metrics,
                **invariants,
            }
        )

    rows_a, metrics_a = run_closed_loop(args.seed, 45.0, 0.02, scenario)
    rows_b, metrics_b = run_closed_loop(args.seed, 45.0, 0.02, scenario)
    reproducible = metrics_a == metrics_b and rows_a == rows_b
    save_outputs(rows_a, metrics_a, args.output_dir / "nominal_reproducibility")
    write_csv(args.output_dir / "timestep_sensitivity.csv", timestep_rows)
    validation_summary.update(
        {
            "timestep_convergence_trend": convergence_trend,
            "nominal_reproducible": reproducible,
            "max_quaternion_norm_error": max(float(row["max_quaternion_norm_error"]) for row in timestep_rows),
            "mass_floor_respected": all(bool(row["mass_floor_respected"]) for row in timestep_rows),
            "throttle_limits_respected": all(bool(row["throttle_limits_respected"]) for row in timestep_rows),
            "gimbal_limits_respected": all(bool(row["gimbal_limits_respected"]) for row in timestep_rows),
        }
    )
    with (args.output_dir / "validation_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(validation_summary, stream, indent=2, sort_keys=True)
    print(json.dumps(validation_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

