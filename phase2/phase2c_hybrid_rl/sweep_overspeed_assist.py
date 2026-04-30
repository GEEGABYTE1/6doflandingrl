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
class AssistCandidate:
    label: str
    gate_altitude_m: float
    gate_power: float
    trigger_mps: float
    full_scale_mps: float
    max_delta: float
    late_stage_altitude_m: float
    late_stage_extra_delta: float

DEFAULT_CANDIDATES: tuple[AssistCandidate, ...] = (
    AssistCandidate(
        label="overspeed_assist_v3_baseline",
        gate_altitude_m=22.0,
        gate_power=1.5,
        trigger_mps=0.25,
        full_scale_mps=3.0,
        max_delta=0.16,
        late_stage_altitude_m=2.5,
        late_stage_extra_delta=0.06,
    ),
    AssistCandidate(
        label="overspeed_assist_sweep_late_extra",
        gate_altitude_m=22.0,
        gate_power=1.5,
        trigger_mps=0.25,
        full_scale_mps=3.0,
        max_delta=0.16,
        late_stage_altitude_m=2.5,
        late_stage_extra_delta=0.09,
    ),
    AssistCandidate(
        label="overspeed_assist_sweep_earlier_subzone",
        gate_altitude_m=22.0,
        gate_power=1.5,
        trigger_mps=0.25,
        full_scale_mps=3.0,
        max_delta=0.16,
        late_stage_altitude_m=3.5,
        late_stage_extra_delta=0.08,
    ),
    AssistCandidate(
        label="overspeed_assist_sweep_aggressive",
        gate_altitude_m=22.0,
        gate_power=1.5,
        trigger_mps=0.15,
        full_scale_mps=2.5,
        max_delta=0.18,
        late_stage_altitude_m=3.0,
        late_stage_extra_delta=0.08,
    ),
)

FOCUSED_CANDIDATES: tuple[AssistCandidate, ...] = (
    AssistCandidate(
        label="overspeed_assist_focus_baseline",
        gate_altitude_m=22.0,
        gate_power=1.5,
        trigger_mps=0.25,
        full_scale_mps=3.0,
        max_delta=0.16,
        late_stage_altitude_m=3.5,
        late_stage_extra_delta=0.08,
    ),
    AssistCandidate(
        label="overspeed_assist_focus_higher_extra",
        gate_altitude_m=22.0,
        gate_power=1.5,
        trigger_mps=0.25,
        full_scale_mps=3.0,
        max_delta=0.16,
        late_stage_altitude_m=3.5,
        late_stage_extra_delta=0.10,
    ),
    AssistCandidate(
        label="overspeed_assist_focus_higher_subzone",
        gate_altitude_m=22.0,
        gate_power=1.5,
        trigger_mps=0.25,
        full_scale_mps=3.0,
        max_delta=0.16,
        late_stage_altitude_m=4.0,
        late_stage_extra_delta=0.08,
    ),
    AssistCandidate(
        label="overspeed_assist_focus_blended",
        gate_altitude_m=22.0,
        gate_power=1.5,
        trigger_mps=0.25,
        full_scale_mps=3.0,
        max_delta=0.16,
        late_stage_altitude_m=4.0,
        late_stage_extra_delta=0.10,
    ),
)

SWEEP_PROFILES: dict[str, tuple[AssistCandidate, ...]] = {
    "v1": DEFAULT_CANDIDATES,
    "focused_v2": FOCUSED_CANDIDATES,
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--total-timesteps", type=int, default=50_000)
    parser.add_argument("--scenario", type=str, default="nominal")
    parser.add_argument("--throttle-model", type=Path, required=True)
    parser.add_argument(
        "--reward-profile",
        type=str,
        default="vertical_focus_v1",
        help="Residual reward profile to keep fixed during the sweep.",
    )
    parser.add_argument(
        "--curriculum-profile",
        type=str,
        default="baseline_v1",
        help="Residual curriculum profile to keep fixed during the sweep.",
    )
    parser.add_argument(
        "--ppo-profile",
        type=str,
        default="baseline_default",
        help="Residual PPO profile to keep fixed during the sweep.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--sweep-profile",
        type=str,
        default="v1",
        choices=sorted(SWEEP_PROFILES.keys()),
        help="Named assist-candidate set to evaluate.",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="Python interpreter used to launch the child training/eval scripts.",
    )
    parser.add_argument(
        "--matplotlib-cache-dir",
        type=Path,
        default=Path("outputs/.cache/matplotlib"),
        help="MPLCONFIGDIR passed to child scripts for deterministic cache writes.",
    )
    return parser.parse_args()


def run_command(cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def read_metrics(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_rows(candidates: tuple[AssistCandidate, ...], eval_dirs: dict[str, Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        metrics_path = eval_dirs[candidate.label] / "metrics.json"
        metrics = read_metrics(metrics_path)
        rows.append(
            {
                "label": candidate.label,
                "success": bool(metrics["success"]),
                "failure_modes": metrics["failure_modes"],
                "vertical_touchdown_velocity_mps": float(metrics["vertical_touchdown_velocity_mps"]),
                "horizontal_touchdown_velocity_mps": float(metrics["horizontal_touchdown_velocity_mps"]),
                "landing_position_error_m": float(metrics["landing_position_error_m"]),
                "tilt_angle_deg": float(metrics["tilt_angle_deg"]),
                "angular_rate_norm_radps": float(metrics["angular_rate_norm_radps"]),
            }
        )
    return rows


def write_rows(output_dir: Path, rows: list[dict[str, object]]) -> None:
    with (output_dir / "phase2c_comparison.json").open("w", encoding="utf-8") as stream:
        json.dump(rows, stream, indent=2, sort_keys=True)
    with (output_dir / "phase2c_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def choose_best(rows: list[dict[str, object]]) -> dict[str, object]:
    return min(
        rows,
        key=lambda row: (
            abs(float(row["vertical_touchdown_velocity_mps"])),
            float(row["horizontal_touchdown_velocity_mps"]),
            float(row["tilt_angle_deg"]),
            float(row["landing_position_error_m"]),
        ),
    )


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    args.matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
    env = {"MPLCONFIGDIR": str(args.matplotlib_cache_dir), **os.environ}
    train_script = repo_root / "phase2c_hybrid_rl" / "train_residual.py"
    eval_script = repo_root / "phase2c_hybrid_rl" / "eval_hybrid.py"
    candidates = SWEEP_PROFILES[args.sweep_profile]
    train_dirs: dict[str, Path] = {}
    eval_dirs: dict[str, Path] = {}
    for candidate in candidates:
        train_dir = output_dir / candidate.label / "train"
        eval_dir = output_dir / candidate.label / "eval"
        train_dirs[candidate.label] = train_dir
        eval_dirs[candidate.label] = eval_dir
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
            "--terminal-throttle-gate-altitude",
            str(candidate.gate_altitude_m),
            "--terminal-throttle-gate-power",
            str(candidate.gate_power),
            "--overspeed-brake-assist-trigger",
            str(candidate.trigger_mps),
            "--overspeed-brake-assist-full-scale",
            str(candidate.full_scale_mps),
            "--overspeed-brake-assist-max-delta",
            str(candidate.max_delta),
            "--overspeed-brake-assist-late-stage-altitude",
            str(candidate.late_stage_altitude_m),
            "--overspeed-brake-assist-late-stage-extra-delta",
            str(candidate.late_stage_extra_delta),
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
    rows = build_rows(candidates, eval_dirs)
    write_rows(output_dir, rows)
    best_row = choose_best(rows)
    best_candidate = next(candidate for candidate in candidates if candidate.label == best_row["label"])
    summary = {
        "seed": args.seed,
        "sweep_profile": args.sweep_profile,
        "total_timesteps": args.total_timesteps,
        "scenario": args.scenario,
        "reward_profile": args.reward_profile,
        "curriculum_profile": args.curriculum_profile,
        "ppo_profile": args.ppo_profile,
        "throttle_model": str(args.throttle_model),
        "best_candidate": {
            **asdict(best_candidate),
            **best_row,
        },
    }
    (output_dir / "best_candidate.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

if __name__ == "__main__":
    main()
