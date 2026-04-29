#training of full 6dof 2b tvc with frozen throttle

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from phase2_rl.train_ppo import build_curriculum_callback

from phase2b_hierarchical_rl.experiment_configs import (
    phase2b_ppo_profile,
    ppo_profile_names,
    tvc_curriculum_profile,
    tvc_curriculum_profile_names,
    tvc_reward_profile,
    tvc_reward_profile_names,
)
from phase2b_hierarchical_rl.hierarchical_controller import FrozenThrottlePolicy
from phase2b_hierarchical_rl.tvc_env import TVCEnvConfig, TVCPolicyEnv

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7, help="Deterministic training seed.")
    parser.add_argument("--total-timesteps", type=int, default=150_000, help="TVC PPO horizon.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Training artifact directory.")
    parser.add_argument("--scenario", type=str, default="nominal", help="Training scenario name.")
    parser.add_argument("--throttle-model", type=Path, required=True, help="Frozen throttle PPO model.")
    parser.add_argument(
        "--reward-profile",
        type=str,
        default="baseline_v1",
        choices=tvc_reward_profile_names(),
        help="Named TVC reward profile.",
    )
    parser.add_argument(
        "--curriculum-profile",
        type=str,
        default="baseline_v1",
        choices=tvc_curriculum_profile_names(),
        help="Named TVC curriculum profile.",
    )
    parser.add_argument(
        "--ppo-profile",
        type=str,
        default="baseline_default",
        choices=ppo_profile_names(),
        help="Named PPO hyperparameter profile.",
    )
    parser.add_argument("--disable-curriculum", action="store_true", help="Disable TVC curriculum.")
    parser.add_argument("--disable-staged-wind", action="store_true", help="Disable staged wind in TVC curriculum.")
    return parser.parse_args()

def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_tvc_model(model: object, env: TVCPolicyEnv, seed: int) -> tuple[list[dict[str, float]], dict[str, object]]:
    observation, info = env.reset(seed=seed)
    terminated = False
    truncated = False
    final_info = info
    while not (terminated or truncated):
        action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, final_info = env.step(action)
    return env.trajectory_rows, final_info.get("metrics", {})


def main() -> None:
    try:
        from stable_baselines3 import PPO
    except ModuleNotFoundError as exc: 
        raise ModuleNotFoundError(
            "Phase 2B TVC training requires stable-baselines3 and gymnasium. "
            "Install ../requirements.txt before running train_tvc.py."
        ) from exc

    args = parse_args()
    throttle_policy = FrozenThrottlePolicy.from_path(args.throttle_model)
    config = TVCEnvConfig(
        scenario_name=args.scenario,
        reward_weights=tvc_reward_profile(args.reward_profile),
        curriculum_enabled=not args.disable_curriculum,
        staged_wind_enabled=not args.disable_staged_wind,
        curriculum_bounds=tvc_curriculum_profile(args.curriculum_profile),
    )
    ppo_cfg = phase2b_ppo_profile(args.ppo_profile)
    env = TVCPolicyEnv(throttle_policy=throttle_policy, config=config, seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "config.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "seed": args.seed,
                "total_timesteps": args.total_timesteps,
                "scenario": args.scenario,
                "throttle_model": str(args.throttle_model),
                "reward_profile": args.reward_profile,
                "curriculum_profile": args.curriculum_profile,
                "ppo_profile": args.ppo_profile,
                "tvc_env_config": config.to_dict(),
                "ppo_profile_config": ppo_cfg.to_dict(),
            },
            stream,
            indent=2,
            sort_keys=True,
        )
    model = PPO(
        "MlpPolicy",
        env,
        seed=args.seed,
        verbose=1,
        learning_rate=ppo_cfg.make_learning_rate(),
        n_steps=ppo_cfg.n_steps,
        batch_size=ppo_cfg.batch_size,
        gamma=ppo_cfg.gamma,
        gae_lambda=ppo_cfg.gae_lambda,
        ent_coef=ppo_cfg.ent_coef,
    )
    callback = build_curriculum_callback(args.total_timesteps) if config.curriculum_enabled else None
    model.learn(total_timesteps=args.total_timesteps, progress_bar=False, callback=callback)
    model.save(str(args.output_dir / "model"))

    eval_env = TVCPolicyEnv(
        throttle_policy=throttle_policy,
        config=TVCEnvConfig(
            scenario_name=config.scenario_name,
            dt_s=config.dt_s,
            max_duration_s=config.max_duration_s,
            success_criteria=config.success_criteria,
            reward_weights=config.reward_weights,
            curriculum_enabled=False,
            staged_wind_enabled=config.staged_wind_enabled,
            curriculum_bounds=config.curriculum_bounds,
            coordination_features=config.coordination_features,
        ),
        seed=args.seed,
    )
    rollout_rows, metrics = evaluate_tvc_model(model, eval_env, args.seed)
    write_csv(args.output_dir / "eval_trajectory.csv", rollout_rows)
    with (args.output_dir / "eval_metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2, sort_keys=True)
    with (args.output_dir / "training_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "seed": args.seed,
                "total_timesteps": args.total_timesteps,
                "model_path": str(args.output_dir / "model.zip"),
                "reward_profile": args.reward_profile,
                "curriculum_profile": args.curriculum_profile,
                "ppo_profile": args.ppo_profile,
                "throttle_model": str(args.throttle_model),
                "curriculum_enabled": config.curriculum_enabled,
                "staged_wind_enabled": config.staged_wind_enabled,
            },
            stream,
            indent=2,
            sort_keys=True,
        )

if __name__ == "__main__":
    main()
