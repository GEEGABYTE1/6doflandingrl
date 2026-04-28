#script for comparing multiple rl evaluations runs and rank the candidates
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from phase1_dynamics.metrics import SuccessCriteria

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-dirs",
        nargs="+",
        type=Path,
        required=True,
        help="Evaluation directories containing metrics.json and config.json.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for comparison artifacts.")
    return parser.parse_args()


def normalized_violation_score(metrics: dict[str, object], criteria: SuccessCriteria) -> float:
    #lower-is-better violation score
    return float(
        abs(float(metrics["vertical_touchdown_velocity_mps"])) / criteria.max_vertical_speed_mps
        + float(metrics["horizontal_touchdown_velocity_mps"]) / criteria.max_horizontal_speed_mps
        + float(metrics["landing_position_error_m"]) / criteria.max_lateral_error_m
        + float(metrics["tilt_angle_deg"]) / criteria.max_tilt_deg
    )

def main() -> None:
    args = parse_args()
    criteria = SuccessCriteria()
    rows: list[dict[str, object]] = []
    for eval_dir in args.eval_dirs:
        metrics = json.loads((eval_dir / "metrics.json").read_text(encoding="utf-8"))
        config = json.loads((eval_dir / "config.json").read_text(encoding="utf-8"))
        row = {
            "run_id": eval_dir.name,
            "reward_profile": config.get("reward_profile", ""),
            "ppo_profile": config.get("ppo_profile", ""),
            "vertical_touchdown_velocity_mps": metrics["vertical_touchdown_velocity_mps"],
            "horizontal_touchdown_velocity_mps": metrics["horizontal_touchdown_velocity_mps"],
            "landing_position_error_m": metrics["landing_position_error_m"],
            "tilt_angle_deg": metrics["tilt_angle_deg"],
            "success": metrics["success"],
            "termination_reason": metrics["termination_reason"],
            "violation_score": normalized_violation_score(metrics, criteria),
        }
        rows.append(row)

    rows.sort(key=lambda item: (not bool(item["success"]), float(item["violation_score"])))
    best_run = rows[0]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "comparison.json").open("w", encoding="utf-8") as stream:
        json.dump(rows, stream, indent=2, sort_keys=True)
    with (args.output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "best_run_id": best_run["run_id"],
                "best_reward_profile": best_run["reward_profile"],
                "best_ppo_profile": best_run["ppo_profile"],
                "best_violation_score": best_run["violation_score"],
                "criteria": criteria.to_dict(),
            },
            stream,
            indent=2,
            sort_keys=True,
        )

if __name__ == "__main__":
    main()
