
from __future__ import annotations
import argparse
import json
from pathlib import Path
from phase2_rl.experiment_configs import ppo_profile, ppo_profile_names, reward_profile, reward_profile_names
from phase2_rl.landing_env import Phase2RLConfig, RocketLandingEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7, help="Deterministic training seed.")
    parser.add_argument("--total-timesteps", type=int, default=200_000, help="PPO training horizon.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Training artifact directory.")
    parser.add_argument("--scenario", type=str, default="nominal", help="Training scenario name.")
    parser.add_argument(
        "--reward-profile",
        type=str,
        default="baseline_v2",
        choices=reward_profile_names(),
        help="Named reward profile.",
    )
    parser.add_argument(
        "--ppo-profile",
        type=str,
        default="baseline_default",
        choices=ppo_profile_names(),
        help="Named PPO hyperparameter profile.",
    )
    parser.add_argument(
        "--disable-curriculum",
        action="store_true",
        help="Disable the easy-to-hard training curriculum.",
    )
    parser.add_argument(
        "--disable-staged-wind",
        action="store_true",
        help="Disable staged wind inside the training curriculum.",
    )
    return parser.parse_args()


def build_curriculum_callback(total_timesteps: int):
    from stable_baselines3.common.callbacks import BaseCallback

    class CurriculumCallback(BaseCallback):
        def _on_step(self) -> bool:
            progress = min(float(self.num_timesteps) / max(float(total_timesteps), 1.0), 1.0)
            self.training_env.env_method("set_curriculum_progress", progress)
            return True

    return CurriculumCallback()


def main() -> None:
    try:
        from stable_baselines3 import PPO
    except ModuleNotFoundError as exc:  
        raise ModuleNotFoundError(
            "Phase 2A training requires stable-baselines3 and gymnasium. "
            "Install ../requirements.txt before running train_ppo.py."
        ) from exc

    args = parse_args()
    reward_cfg = reward_profile(args.reward_profile)
    ppo_cfg = ppo_profile(args.ppo_profile)
    config = Phase2RLConfig(
        scenario_name=args.scenario,
        reward_weights=reward_cfg,
        curriculum_enabled=not args.disable_curriculum,
        staged_wind_enabled=not args.disable_staged_wind,
    )
    env = RocketLandingEnv(config=config, seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "config.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "seed": args.seed,
                "total_timesteps": args.total_timesteps,
                "reward_profile": args.reward_profile,
                "ppo_profile": args.ppo_profile,
                "phase2_config": config.to_dict(),
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
    with (args.output_dir / "training_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "seed": args.seed,
                "total_timesteps": args.total_timesteps,
                "model_path": str(args.output_dir / "model.zip"),
                "reward_profile": args.reward_profile,
                "ppo_profile": args.ppo_profile,
                "curriculum_enabled": config.curriculum_enabled,
                "staged_wind_enabled": config.staged_wind_enabled,
            },
            stream,
            indent=2,
            sort_keys=True,
        )


if __name__ == "__main__":
    main()
