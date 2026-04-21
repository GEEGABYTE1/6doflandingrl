"""Run Phase 1 mass, disturbance, and scenario evaluations."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from phase1_dynamics.metrics import SuccessCriteria
    from phase1_dynamics.scenarios import (
        ScenarioConfig,
        disturbance_sweep_scenarios,
        mass_sweep_scenarios,
        named_scenarios,
    )
    from phase1_dynamics.simulate_lqr import run_closed_loop, save_outputs
else:
    from .metrics import SuccessCriteria
    from .scenarios import ScenarioConfig, disturbance_sweep_scenarios, mass_sweep_scenarios, named_scenarios
    from .simulate_lqr import run_closed_loop, save_outputs


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7, help="Deterministic seed for all scenarios.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/phase1_evaluation"),
        help="Directory for per-run and aggregate evaluation artifacts.",
    )
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write dictionaries to CSV with stable columns."""
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_one(
    scenario: ScenarioConfig,
    group: str,
    seed: int,
    output_dir: Path,
) -> dict[str, object]:
    """Run one scenario, save artifacts, and return aggregate metrics."""
    run_id = f"{group}__{scenario.name}"
    run_dir = output_dir / "runs" / run_id
    rows, metrics = run_closed_loop(
        seed=seed,
        duration_s=scenario.duration_s,
        dt_s=scenario.dt_s,
        scenario=scenario,
    )
    enriched = {
        "run_id": run_id,
        "group": group,
        "scenario": scenario.name,
        "description": scenario.description,
        **metrics,
    }
    save_outputs(rows, metrics, run_dir)
    with (run_dir / "config.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "seed": seed,
                "group": group,
                "run_id": run_id,
                "scenario": scenario.to_dict(),
                "success_criteria": SuccessCriteria().to_dict(),
            },
            stream,
            indent=2,
            sort_keys=True,
        )
    return enriched


def scenario_table_rows(scenarios: list[ScenarioConfig]) -> list[dict[str, object]]:
    """Return a compact table describing all named scenarios."""
    return [
        {
            "scenario": item.name,
            "description": item.description,
            "initial_mass_kg": item.initial_mass_kg,
            "initial_position_m": item.initial_position_m,
            "initial_velocity_mps": item.initial_velocity_mps,
            "wind_mps": item.wind_mps,
            "gust_mps": item.gust_mps,
            "misalignment_deg": item.misalignment_deg,
        }
        for item in scenarios
    ]


def failure_taxonomy_rows(metrics_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Summarize observed failure modes across the configured evaluation suite."""
    counter: Counter[str] = Counter()
    for row in metrics_rows:
        modes = str(row["failure_modes"]).split(";")
        for mode in modes:
            counter[mode] += 1
    if "none" not in counter:
        counter["none"] = 0
    return [
        {
            "failure_mode": mode,
            "count": count,
            "definition": {
                "none": "Scenario met all configured touchdown criteria.",
                "no_touchdown": "Vehicle did not reach the ground by the run horizon.",
                "hard_touchdown": "Vertical touchdown speed exceeded threshold.",
                "horizontal_speed": "Horizontal touchdown speed exceeded threshold.",
                "lateral_miss": "Landing position error exceeded threshold.",
                "tilt_over": "Touchdown tilt angle exceeded threshold.",
                "excess_angular_rate": "Touchdown angular-rate norm exceeded threshold.",
                "fuel_exhaustion": "Final mass was at or too close to dry mass.",
                "divergence": "Position or speed exceeded numerical divergence limits.",
                "saturation_driven_instability": "Throttle or gimbal saturation persisted too often.",
                "oscillatory_descent": "Late-stage altitude history showed repeated oscillations.",
            }.get(mode, "Observed compound or unclassified failure mode."),
        }
        for mode, count in sorted(counter.items())
    ]


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scenario_set = named_scenarios()
    metrics_rows: list[dict[str, object]] = []
    for scenario in mass_sweep_scenarios():
        metrics_rows.append(run_one(scenario, "mass_sweep", args.seed, args.output_dir))
    for scenario in disturbance_sweep_scenarios():
        metrics_rows.append(run_one(scenario, "disturbance_sweep", args.seed, args.output_dir))
    for scenario in scenario_set:
        metrics_rows.append(run_one(scenario, "scenario_table", args.seed, args.output_dir))

    write_csv(args.output_dir / "metrics.csv", metrics_rows)
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(metrics_rows, stream, indent=2, sort_keys=True)
    write_csv(args.output_dir / "scenario_table.csv", scenario_table_rows(scenario_set))
    taxonomy = failure_taxonomy_rows(metrics_rows)
    write_csv(args.output_dir / "failure_taxonomy.csv", taxonomy)
    summary = {
        "seed": args.seed,
        "run_count": len(metrics_rows),
        "success_count": int(sum(bool(row["success"]) for row in metrics_rows)),
        "failure_count": int(sum(not bool(row["success"]) for row in metrics_rows)),
        "success_criteria": SuccessCriteria().to_dict(),
        "groups": sorted(set(str(row["group"]) for row in metrics_rows)),
        "representative_success_run_id": next(
            (str(row["run_id"]) for row in metrics_rows if bool(row["success"])),
            "",
        ),
        "representative_failure_run_id": next(
            (str(row["run_id"]) for row in metrics_rows if not bool(row["success"])),
            "",
        ),
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

