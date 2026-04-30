#iteratively searching the phase 2c brake-prior family with guardrails

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
import os
from pathlib import Path
import subprocess
import sys

@dataclass(frozen=True)
class PriorSettings:
    gate_altitude_m: float
    trigger_mps: float
    max_delta: float
    late_stage_altitude_m: float
    late_stage_extra_delta: float
    brake_floor_altitude_m: float
    brake_floor_base_throttle: float
    brake_floor_max_throttle: float


@dataclass(frozen=True)
class SearchSteps:
    gate_altitude_m: float = 1.0
    trigger_mps: float = 0.05
    max_delta: float = 0.02
    late_stage_altitude_m: float = 0.5
    late_stage_extra_delta: float = 0.02
    brake_floor_altitude_m: float = 0.5
    brake_floor_base_throttle: float = 0.02
    brake_floor_max_throttle: float = 0.01

@dataclass(frozen=True)
class SearchBounds:
    gate_altitude_min: float = 10.0
    gate_altitude_max: float = 30.0
    trigger_min: float = 0.0
    trigger_max: float = 1.0
    max_delta_min: float = 0.04
    max_delta_max: float = 0.30
    late_stage_altitude_min: float = 0.5
    late_stage_altitude_max: float = 6.0
    late_stage_extra_delta_min: float = 0.0
    late_stage_extra_delta_max: float = 0.20
    brake_floor_altitude_min: float = 2.0
    brake_floor_altitude_max: float = 8.0
    brake_floor_base_min: float = 0.70
    brake_floor_base_max: float = 0.98
    brake_floor_max_min: float = 0.85
    brake_floor_max_max: float = 1.0

@dataclass(frozen=True)
class Guardrails:
    max_horizontal_touchdown_velocity_mps: float
    max_landing_error_m: float
    max_tilt_angle_deg: float
    max_angular_rate_norm_radps: float


@dataclass(frozen=True)
class SearchCandidate:
    label: str
    settings: PriorSettings

PARAMETER_ORDER: tuple[str, ...] = (
    "late_stage_altitude_m",
    "late_stage_extra_delta",
    "brake_floor_base_throttle",
    "brake_floor_max_throttle",
    "max_delta",
    "gate_altitude_m",
    "trigger_mps",
    "brake_floor_altitude_m",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--total-timesteps", type=int, default=50_000)
    parser.add_argument("--scenario", type=str, default="nominal")
    parser.add_argument("--throttle-model", type=Path, required=True)
    parser.add_argument("--seed-residual-dir", type=Path, required=True)
    parser.add_argument("--seed-metrics", type=Path, required=True)
    parser.add_argument("--reward-profile", type=str, default="near_ground_touchdown_v1")
    parser.add_argument("--curriculum-profile", type=str, default="baseline_v1")
    parser.add_argument("--ppo-profile", type=str, default="baseline_default")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--plateau-patience", type=int, default=2)
    parser.add_argument("--improvement-threshold-mps", type=float, default=0.05)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--horizontal-guardrail-mps", type=float, default=None)
    parser.add_argument("--landing-error-guardrail-m", type=float, default=None)
    parser.add_argument("--tilt-guardrail-deg", type=float, default=None)
    parser.add_argument("--angular-rate-guardrail-radps", type=float, default=None)
    parser.add_argument("--gate-altitude-step", type=float, default=1.0)
    parser.add_argument("--trigger-step", type=float, default=0.05)
    parser.add_argument("--max-delta-step", type=float, default=0.02)
    parser.add_argument("--late-stage-altitude-step", type=float, default=0.5)
    parser.add_argument("--late-stage-extra-step", type=float, default=0.02)
    parser.add_argument("--brake-floor-altitude-step", type=float, default=0.5)
    parser.add_argument("--brake-floor-base-step", type=float, default=0.02)
    parser.add_argument("--brake-floor-max-step", type=float, default=0.01)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--matplotlib-cache-dir",
        type=Path,
        default=Path("outputs/.cache/matplotlib"),
        help="MPLCONFIGDIR passed to child scripts for deterministic cache writes.",
    )
    return parser.parse_args()


def run_command(cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_seed_settings(seed_residual_dir: Path) -> PriorSettings:
    config = read_json(seed_residual_dir / "config.json")
    env_cfg = config["residual_env_config"]
    return PriorSettings(
        gate_altitude_m=float(env_cfg["terminal_throttle_gate_altitude_m"]),
        trigger_mps=float(env_cfg["overspeed_brake_assist_trigger_mps"]),
        max_delta=float(env_cfg["overspeed_brake_assist_max_delta"]),
        late_stage_altitude_m=float(env_cfg["overspeed_brake_assist_late_stage_altitude_m"]),
        late_stage_extra_delta=float(env_cfg["overspeed_brake_assist_late_stage_extra_delta"]),
        brake_floor_altitude_m=float(env_cfg["brake_floor_altitude_m"]),
        brake_floor_base_throttle=float(env_cfg["brake_floor_base_throttle"]),
        brake_floor_max_throttle=float(env_cfg["brake_floor_max_throttle"]),
    )


def derive_guardrails(metrics: dict[str, object], args: argparse.Namespace) -> Guardrails:
    horizontal = (
        float(args.horizontal_guardrail_mps)
        if args.horizontal_guardrail_mps is not None
        else max(0.10, 1.25 * float(metrics["horizontal_touchdown_velocity_mps"]))
    )
    landing_error = (
        float(args.landing_error_guardrail_m)
        if args.landing_error_guardrail_m is not None
        else max(0.20, 1.25 * float(metrics["landing_position_error_m"]))
    )
    tilt = (
        float(args.tilt_guardrail_deg)
        if args.tilt_guardrail_deg is not None
        else max(0.20, 1.25 * float(metrics["tilt_angle_deg"]))
    )
    angular_rate = (
        float(args.angular_rate_guardrail_radps)
        if args.angular_rate_guardrail_radps is not None
        else max(0.01, 1.25 * float(metrics["angular_rate_norm_radps"]))
    )
    return Guardrails(
        max_horizontal_touchdown_velocity_mps=horizontal,
        max_landing_error_m=landing_error,
        max_tilt_angle_deg=tilt,
        max_angular_rate_norm_radps=angular_rate,
    )


def build_steps(args: argparse.Namespace) -> SearchSteps:
    return SearchSteps(
        gate_altitude_m=float(args.gate_altitude_step),
        trigger_mps=float(args.trigger_step),
        max_delta=float(args.max_delta_step),
        late_stage_altitude_m=float(args.late_stage_altitude_step),
        late_stage_extra_delta=float(args.late_stage_extra_step),
        brake_floor_altitude_m=float(args.brake_floor_altitude_step),
        brake_floor_base_throttle=float(args.brake_floor_base_step),
        brake_floor_max_throttle=float(args.brake_floor_max_step),
    )


def clamp_settings(settings: PriorSettings, bounds: SearchBounds) -> PriorSettings:
    clamped = PriorSettings(
        gate_altitude_m=min(max(settings.gate_altitude_m, bounds.gate_altitude_min), bounds.gate_altitude_max),
        trigger_mps=min(max(settings.trigger_mps, bounds.trigger_min), bounds.trigger_max),
        max_delta=min(max(settings.max_delta, bounds.max_delta_min), bounds.max_delta_max),
        late_stage_altitude_m=min(
            max(settings.late_stage_altitude_m, bounds.late_stage_altitude_min), bounds.late_stage_altitude_max
        ),
        late_stage_extra_delta=min(
            max(settings.late_stage_extra_delta, bounds.late_stage_extra_delta_min), bounds.late_stage_extra_delta_max
        ),
        brake_floor_altitude_m=min(
            max(settings.brake_floor_altitude_m, bounds.brake_floor_altitude_min), bounds.brake_floor_altitude_max
        ),
        brake_floor_base_throttle=min(
            max(settings.brake_floor_base_throttle, bounds.brake_floor_base_min), bounds.brake_floor_base_max
        ),
        brake_floor_max_throttle=min(
            max(settings.brake_floor_max_throttle, bounds.brake_floor_max_min), bounds.brake_floor_max_max
        ),
    )
    if clamped.brake_floor_max_throttle < clamped.brake_floor_base_throttle:
        clamped = PriorSettings(
            **{**asdict(clamped), "brake_floor_max_throttle": clamped.brake_floor_base_throttle}
        )
    return clamped


def generate_candidate_neighborhood(
    center: PriorSettings,
    steps: SearchSteps,
    bounds: SearchBounds,
    max_candidates: int | None,
) -> list[SearchCandidate]:
    raw: list[SearchCandidate] = []
    for param_name in PARAMETER_ORDER:
        step = getattr(steps, param_name)
        for suffix, sign in (("minus", -1.0), ("plus", 1.0)):
            values = asdict(center)
            values[param_name] = float(values[param_name]) + sign * float(step)
            candidate = clamp_settings(PriorSettings(**values), bounds)
            raw.append(SearchCandidate(label=f"{param_name}_{suffix}", settings=candidate))
    unique: list[SearchCandidate] = []
    seen: set[tuple[float, ...]] = set()
    for candidate in raw:
        key = tuple(round(value, 8) for value in asdict(candidate.settings).values())
        if key in seen or candidate.settings == center:
            continue
        seen.add(key)
        unique.append(candidate)
    if max_candidates is not None:
        return unique[: max(0, max_candidates)]
    return unique


def candidate_is_eligible(metrics: dict[str, object], guardrails: Guardrails) -> bool:
    if str(metrics.get("termination_reason", "")) != "touchdown":
        return False
    failure_modes = str(metrics.get("failure_modes", ""))
    if any(tag in failure_modes for tag in ("timeout", "diverg", "fuel", "oscillation")):
        return False
    return (
        float(metrics["horizontal_touchdown_velocity_mps"]) <= guardrails.max_horizontal_touchdown_velocity_mps
        and float(metrics["landing_position_error_m"]) <= guardrails.max_landing_error_m
        and float(metrics["tilt_angle_deg"]) <= guardrails.max_tilt_angle_deg
        and float(metrics["angular_rate_norm_radps"]) <= guardrails.max_angular_rate_norm_radps
    )


def vertical_improvement_mps(current_metrics: dict[str, object], candidate_metrics: dict[str, object]) -> float:
    return abs(float(current_metrics["vertical_touchdown_velocity_mps"])) - abs(
        float(candidate_metrics["vertical_touchdown_velocity_mps"])
    )


def choose_best_eligible(rows: list[dict[str, object]]) -> dict[str, object] | None:
    eligible = [row for row in rows if bool(row["eligible"])]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda row: (
            abs(float(row["vertical_touchdown_velocity_mps"])),
            float(row["horizontal_touchdown_velocity_mps"]),
            float(row["landing_position_error_m"]),
            float(row["tilt_angle_deg"]),
            float(row["angular_rate_norm_radps"]),
        ),
    )


def write_rows(output_dir: Path, rows: list[dict[str, object]]) -> None:
    with (output_dir / "phase2c_comparison.json").open("w", encoding="utf-8") as stream:
        json.dump(rows, stream, indent=2, sort_keys=True)
    with (output_dir / "phase2c_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
    env = {"MPLCONFIGDIR": str(args.matplotlib_cache_dir), **os.environ}
    train_script = repo_root / "phase2c_hybrid_rl" / "train_residual.py"
    eval_script = repo_root / "phase2c_hybrid_rl" / "eval_hybrid.py"
    current_settings = load_seed_settings(args.seed_residual_dir)
    current_metrics = read_json(args.seed_metrics)
    guardrails = derive_guardrails(current_metrics, args)
    steps = build_steps(args)
    bounds = SearchBounds()

    summary: dict[str, object] = {
        "seed": args.seed,
        "scenario": args.scenario,
        "total_timesteps": args.total_timesteps,
        "reward_profile": args.reward_profile,
        "curriculum_profile": args.curriculum_profile,
        "ppo_profile": args.ppo_profile,
        "throttle_model": str(args.throttle_model),
        "seed_residual_dir": str(args.seed_residual_dir),
        "seed_metrics": str(args.seed_metrics),
        "guardrails": asdict(guardrails),
        "search_steps": asdict(steps),
        "search_bounds": asdict(bounds),
        "starting_settings": asdict(current_settings),
        "starting_metrics": current_metrics,
        "rounds": [],
    }

    plateau_streak = 0
    stop_reason = "max_rounds_reached"
    final_best_dir = args.seed_residual_dir
    final_best_metrics = current_metrics
    final_best_settings = current_settings

    for round_idx in range(1, args.max_rounds + 1):
        round_dir = args.output_dir / f"round_{round_idx:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        candidates = generate_candidate_neighborhood(current_settings, steps, bounds, args.max_candidates)
        rows: list[dict[str, object]] = []

        for candidate in candidates:
            candidate_root = round_dir / candidate.label
            train_dir = candidate_root / "train"
            eval_dir = candidate_root / "eval"
            train_cmd = [
                str(args.python),
                str(train_script),
                "--seed",
                str(args.seed),
                "--total-timesteps",
                str(args.total_timesteps),
                "--reward-profile",
                args.reward_profile,
                "--curriculum-profile",
                args.curriculum_profile,
                "--ppo-profile",
                args.ppo_profile,
                "--action-mode",
                "tvc_only",
                "--enable-overspeed-brake-assist",
                "--enable-brake-floor",
                "--terminal-throttle-gate-altitude",
                str(candidate.settings.gate_altitude_m),
                "--terminal-throttle-gate-power",
                "1.5",
                "--overspeed-brake-assist-trigger",
                str(candidate.settings.trigger_mps),
                "--overspeed-brake-assist-full-scale",
                "3.0",
                "--overspeed-brake-assist-max-delta",
                str(candidate.settings.max_delta),
                "--overspeed-brake-assist-late-stage-altitude",
                str(candidate.settings.late_stage_altitude_m),
                "--overspeed-brake-assist-late-stage-extra-delta",
                str(candidate.settings.late_stage_extra_delta),
                "--brake-floor-altitude",
                str(candidate.settings.brake_floor_altitude_m),
                "--brake-floor-power",
                "2.0",
                "--brake-floor-safe-margin",
                "0.5",
                "--brake-floor-trigger",
                "0.25",
                "--brake-floor-full-scale",
                "3.0",
                "--brake-floor-base-throttle",
                str(candidate.settings.brake_floor_base_throttle),
                "--brake-floor-max-throttle",
                str(candidate.settings.brake_floor_max_throttle),
                "--throttle-model",
                str(args.throttle_model),
                "--output-dir",
                str(train_dir),
            ]
            run_command(train_cmd, cwd=repo_root, env=env)

            eval_cmd = [
                str(args.python),
                str(eval_script),
                "--seed",
                str(args.seed),
                "--scenario",
                args.scenario,
                "--throttle-model",
                str(args.throttle_model),
                "--residual-model",
                str(train_dir / "model.zip"),
                "--output-dir",
                str(eval_dir),
            ]
            run_command(eval_cmd, cwd=repo_root, env=env)
            metrics = read_json(eval_dir / "metrics.json")
            eligible = candidate_is_eligible(metrics, guardrails)
            rows.append(
                {
                    "label": candidate.label,
                    **asdict(candidate.settings),
                    "eligible": eligible,
                    "success": bool(metrics["success"]),
                    "failure_modes": metrics["failure_modes"],
                    "termination_reason": metrics["termination_reason"],
                    "vertical_touchdown_velocity_mps": float(metrics["vertical_touchdown_velocity_mps"]),
                    "horizontal_touchdown_velocity_mps": float(metrics["horizontal_touchdown_velocity_mps"]),
                    "landing_position_error_m": float(metrics["landing_position_error_m"]),
                    "tilt_angle_deg": float(metrics["tilt_angle_deg"]),
                    "angular_rate_norm_radps": float(metrics["angular_rate_norm_radps"]),
                    "train_dir": str(train_dir),
                    "eval_dir": str(eval_dir),
                }
            )

        write_rows(round_dir, rows)
        best_row = choose_best_eligible(rows)
        round_summary: dict[str, object] = {
            "round_index": round_idx,
            "center_settings": asdict(current_settings),
            "input_metrics": current_metrics,
            "candidate_count": len(rows),
            "best_candidate": best_row,
        }
        if best_row is None:
            plateau_streak += 1
            round_summary["promotion"] = None
            round_summary["plateau_streak"] = plateau_streak
        else:
            candidate_metrics = {
                "vertical_touchdown_velocity_mps": best_row["vertical_touchdown_velocity_mps"],
                "horizontal_touchdown_velocity_mps": best_row["horizontal_touchdown_velocity_mps"],
                "landing_position_error_m": best_row["landing_position_error_m"],
                "tilt_angle_deg": best_row["tilt_angle_deg"],
                "angular_rate_norm_radps": best_row["angular_rate_norm_radps"],
                "success": best_row["success"],
                "failure_modes": best_row["failure_modes"],
                "termination_reason": best_row["termination_reason"],
            }
            improvement = vertical_improvement_mps(current_metrics, candidate_metrics)
            round_summary["vertical_improvement_mps"] = improvement
            if bool(best_row["success"]):
                current_settings = PriorSettings(
                    **{key: float(best_row[key]) for key in asdict(current_settings).keys()}
                )
                current_metrics = candidate_metrics
                final_best_settings = current_settings
                final_best_metrics = candidate_metrics
                final_best_dir = Path(str(best_row["train_dir"]))
                round_summary["promotion"] = {"reason": "success", "label": best_row["label"]}
                summary["rounds"].append(round_summary)
                stop_reason = "success_reached"
                break
            if improvement >= args.improvement_threshold_mps:
                plateau_streak = 0
                current_settings = PriorSettings(
                    **{key: float(best_row[key]) for key in asdict(current_settings).keys()}
                )
                current_metrics = candidate_metrics
                final_best_settings = current_settings
                final_best_metrics = candidate_metrics
                final_best_dir = Path(str(best_row["train_dir"]))
                round_summary["promotion"] = {"reason": "improved", "label": best_row["label"]}
            else:
                plateau_streak += 1
                round_summary["promotion"] = None
            round_summary["plateau_streak"] = plateau_streak

        summary["rounds"].append(round_summary)
        if plateau_streak >= args.plateau_patience:
            stop_reason = "plateau"
            break

    summary["final_best_train_dir"] = str(final_best_dir)
    summary["final_best_settings"] = asdict(final_best_settings)
    summary["final_best_metrics"] = final_best_metrics
    summary["stop_reason"] = stop_reason
    (args.output_dir / "search_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
