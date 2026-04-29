from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

from phase2_rl.landing_env import Phase2RLConfig, RocketLandingEnv

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7, help="Deterministic evaluation seed.")
    parser.add_argument("--model", type=Path, required=True, help="Path to trained PPO model zip.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Evaluation artifact directory.")
    parser.add_argument("--scenario", type=str, default="nominal", help="Evaluation scenario name.")
    return parser.parse_args()

def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        raise ValueError("Cannot write empty evaluation trajectory.")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def main() -> None:
    try:
        from stable_baselines3 import PPO
    except ModuleNotFoundError as exc:  
        raise ModuleNotFoundError(
            "Phase 2A evaluation requires stable-baselines3 and gymnasium. "
            "Install ../requirements.txt before running evaluate_ppo.py."
        ) from exc
    args = parse_args()
    config = Phase2RLConfig(scenario_name=args.scenario)
    env = RocketLandingEnv(config=config, seed=args.seed)
    model = PPO.load(str(args.model))
    observation, info = env.reset(seed=args.seed)
    terminated = False
    truncated = False
    final_info = info
    while not (terminated or truncated):
        action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, final_info = env.step(action)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "trajectory.csv", env.trajectory_rows)
    metrics = final_info.get("metrics", {})
    training_config_path = args.model.resolve().parent / "config.json"
    training_config: dict[str, object] = {}
    if training_config_path.exists():
        training_config = json.loads(training_config_path.read_text(encoding="utf-8"))
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2, sort_keys=True)
    with (args.output_dir / "config.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "seed": args.seed,
                "scenario": args.scenario,
                "model_path": str(args.model),
                "phase2_config": config.to_dict(),
                "reward_profile": training_config.get("reward_profile", ""),
                "ppo_profile": training_config.get("ppo_profile", ""),
                "training_config": training_config,
            },
            stream,
            indent=2,
            sort_keys=True,
        )

if __name__ == "__main__":
    main()
