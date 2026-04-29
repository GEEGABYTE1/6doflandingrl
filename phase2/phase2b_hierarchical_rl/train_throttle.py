#training the throttle

from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

from phase2b_hierarchical_rl.experiment_configs import (
    phase2b_ppo_profile,
    ppo_profile_names,
    throttle_env_overrides,
    throttle_reward_profile,
    throttle_reward_profile_names,
)
from phase2b_hierarchical_rl.throttle_env import ThrottleEnvConfig, VerticalThrottleEnv

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7, help="Deterministic training seed.")
    parser.add_argument("--total-timesteps", type=int, default=100_000, help="Throttle PPO horizon.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Training artifact directory.")
    parser.add_argument("--scenario", type=str, default="nominal", help="Training scenario name.")
    parser.add_argument(
        "--reward-profile",
        type=str,
        default="baseline_v1",
        choices=throttle_reward_profile_names(),
        help="Named throttle reward profile.",
    )
    parser.add_argument(
        "--ppo-profile",
        type=str,
        default="baseline_default",
        choices=ppo_profile_names(),
        help="Named PPO hyperparameter profile.",
    )
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_throttle_model(model: object, config: ThrottleEnvConfig, seed: int) -> tuple[list[dict[str, float]], dict[str, object]]:
    env = VerticalThrottleEnv(
        ThrottleEnvConfig(
            scenario_name=config.scenario_name,
            dt_s=config.dt_s,
            max_duration_s=config.max_duration_s,
            success_criteria=config.success_criteria,
            reward_weights=config.reward_weights,
            altitude_scale_m=config.altitude_scale_m,
            vertical_speed_scale_mps=config.vertical_speed_scale_mps,
            throttle_delta_limit=config.throttle_delta_limit,
            terminal_descent_rate_mps=config.terminal_descent_rate_mps,
            braking_accel_mps2=config.braking_accel_mps2,
            touchdown_zone_altitude_m=config.touchdown_zone_altitude_m,
            touchdown_gate_power=config.touchdown_gate_power,
            terminal_envelope_altitude_m=config.terminal_envelope_altitude_m,
            terminal_envelope_wide_margin_mps=config.terminal_envelope_wide_margin_mps,
            terminal_envelope_tight_margin_mps=config.terminal_envelope_tight_margin_mps,
            timeout_above_ground_penalty=config.timeout_above_ground_penalty,
            timeout_underdescent_penalty=config.timeout_underdescent_penalty,
            observation_mode=config.observation_mode,
            potential_mode=config.potential_mode,
            potential_gamma=config.potential_gamma,
            feature_eps=config.feature_eps,
            randomize_reset=False,
        ),
        seed=seed,
    )
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
            "Phase 2B throttle training requires stable-baselines3 and gymnasium. "
            "Install ../requirements.txt before running train_throttle.py."
        ) from exc

    args = parse_args()
    env_overrides = throttle_env_overrides(args.reward_profile)
    config = ThrottleEnvConfig(
        scenario_name=args.scenario,
        reward_weights=throttle_reward_profile(args.reward_profile),
        **env_overrides,
    )
    ppo_cfg = phase2b_ppo_profile(args.ppo_profile)
    env = VerticalThrottleEnv(config=config, seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "config.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "seed": args.seed,
                "total_timesteps": args.total_timesteps,
                "scenario": args.scenario,
                "reward_profile": args.reward_profile,
                "ppo_profile": args.ppo_profile,
                "throttle_env_config": config.to_dict(),
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
    model.learn(total_timesteps=args.total_timesteps, progress_bar=False)
    model.save(str(args.output_dir / "model"))

    rollout_rows, metrics = evaluate_throttle_model(model, config, args.seed)
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
                "ppo_profile": args.ppo_profile,
            },
            stream,
            indent=2,
            sort_keys=True,
        )

if __name__ == "__main__":
    main()
