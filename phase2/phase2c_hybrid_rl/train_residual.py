from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

from phase1_dynamics.lqr_controller import GainScheduledLQRController
from phase2_rl.train_ppo import build_curriculum_callback
from phase2b_hierarchical_rl.hierarchical_controller import FrozenThrottlePolicy

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from phase2c_hybrid_rl.experiment_configs import (
        phase2c_ppo_profile,
        ppo_profile_names,
        residual_curriculum_profile,
        residual_curriculum_profile_names,
        residual_reward_profile,
        residual_reward_profile_names,
    )
    from phase2c_hybrid_rl.hybrid_env import HybridResidualEnv, ResidualEnvConfig
else:
    from .experiment_configs import (
        phase2c_ppo_profile,
        ppo_profile_names,
        residual_curriculum_profile,
        residual_curriculum_profile_names,
        residual_reward_profile,
        residual_reward_profile_names,
    )
    from .hybrid_env import HybridResidualEnv, ResidualEnvConfig

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--total-timesteps", type=int, default=50_000)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scenario", type=str, default="nominal")
    parser.add_argument("--throttle-model", type=Path, required=True)
    parser.add_argument(
        "--reward-profile",
        type=str,
        default="baseline_v1",
        choices=residual_reward_profile_names(),
    )
    parser.add_argument(
        "--curriculum-profile",
        type=str,
        default="baseline_v1",
        choices=residual_curriculum_profile_names(),
    )
    parser.add_argument(
        "--ppo-profile",
        type=str,
        default="baseline_default",
        choices=ppo_profile_names(),
    )
    parser.add_argument("--disable-curriculum", action="store_true")
    parser.add_argument("--disable-staged-wind", action="store_true")
    parser.add_argument(
        "--action-mode",
        type=str,
        default="tvc_only",
        choices=["tvc_only", "tvc_throttle"],
        help="Residual action mode: TVC-only or throttle-plus-TVC residual.",
    )
    parser.add_argument(
        "--include-prior-throttle-feature",
        action="store_true",
        help="Append normalized prior throttle to the residual observation.",
    )
    parser.add_argument(
        "--throttle-residual-positive-only",
        action="store_true",
        help="Restrict throttle residuals to terminal brake assistance only.",
    )
    parser.add_argument(
        "--enable-overspeed-brake-assist",
        action="store_true",
        help="Add a structured overspeed-triggered throttle assist to the Phase 2C prior.",
    )
    parser.add_argument(
        "--enable-energy-assist",
        action="store_true",
        help="Add an energy-based throttle assist to the Phase 2C prior.",
    )
    parser.add_argument(
        "--terminal-throttle-gate-altitude",
        type=float,
        default=None,
        help="Override the altitude where terminal throttle gating reaches zero above the pad.",
    )
    parser.add_argument(
        "--terminal-throttle-gate-power",
        type=float,
        default=None,
        help="Override the terminal throttle gate exponent.",
    )
    parser.add_argument(
        "--overspeed-brake-assist-trigger",
        type=float,
        default=None,
        help="Override the overspeed margin above vz_ref where brake assist begins.",
    )
    parser.add_argument(
        "--overspeed-brake-assist-full-scale",
        type=float,
        default=None,
        help="Override the overspeed margin where brake assist reaches full strength.",
    )
    parser.add_argument(
        "--overspeed-brake-assist-max-delta",
        type=float,
        default=None,
        help="Override the maximum additive throttle delta from the structured brake assist prior.",
    )
    parser.add_argument(
        "--overspeed-brake-assist-late-stage-altitude",
        type=float,
        default=None,
        help="Override the late-stage subzone altitude for extra terminal brake assist.",
    )
    parser.add_argument(
        "--overspeed-brake-assist-late-stage-extra-delta",
        type=float,
        default=None,
        help="Override the extra additive throttle delta used only in the late-stage subzone.",
    )
    parser.add_argument("--energy-gate-altitude", type=float, default=None)
    parser.add_argument("--energy-gate-power", type=float, default=None)
    parser.add_argument("--energy-full-scale", type=float, default=None)
    parser.add_argument("--energy-shape-power", type=float, default=None)
    parser.add_argument("--energy-max-delta", type=float, default=None)
    parser.add_argument("--energy-touchdown-speed", type=float, default=None)
    parser.add_argument("--energy-braking-accel", type=float, default=None)
    parser.add_argument(
        "--enable-stopping-floor",
        action="store_true",
        help="Enable a stopping-distance-based throttle floor on top of the prior.",
    )
    parser.add_argument("--stopping-floor-gate-altitude", type=float, default=None)
    parser.add_argument("--stopping-floor-gate-power", type=float, default=None)
    parser.add_argument("--stopping-floor-altitude-floor", type=float, default=None)
    parser.add_argument("--stopping-floor-touchdown-speed", type=float, default=None)
    parser.add_argument("--stopping-floor-min-downward-speed", type=float, default=None)
    parser.add_argument(
        "--enable-brake-floor",
        action="store_true",
        help="Enable an overspeed-armed near-ground throttle floor on top of the prior.",
    )
    parser.add_argument("--brake-floor-altitude", type=float, default=None)
    parser.add_argument("--brake-floor-power", type=float, default=None)
    parser.add_argument("--brake-floor-safe-margin", type=float, default=None)
    parser.add_argument("--brake-floor-trigger", type=float, default=None)
    parser.add_argument("--brake-floor-full-scale", type=float, default=None)
    parser.add_argument("--brake-floor-base-throttle", type=float, default=None)
    parser.add_argument("--brake-floor-max-throttle", type=float, default=None)
    parser.add_argument("--brake-floor-shape-power", type=float, default=None)
    parser.add_argument("--brake-floor-late-stage-altitude", type=float, default=None)
    parser.add_argument("--brake-floor-late-stage-extra-throttle", type=float, default=None)
    parser.add_argument(
        "--enable-guidance-prior",
        action="store_true",
        help="Enable a physics-guided terminal braking throttle floor on top of the learned prior.",
    )
    parser.add_argument("--guidance-gate-altitude", type=float, default=None)
    parser.add_argument("--guidance-gate-power", type=float, default=None)
    parser.add_argument("--guidance-target-touchdown-speed", type=float, default=None)
    parser.add_argument("--guidance-altitude-floor", type=float, default=None)
    parser.add_argument("--guidance-base-throttle", type=float, default=None)
    parser.add_argument("--guidance-late-stage-altitude", type=float, default=None)
    parser.add_argument("--guidance-late-stage-extra-throttle", type=float, default=None)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def evaluate_residual_model(model: object, env: HybridResidualEnv, seed: int) -> tuple[list[dict[str, float]], dict[str, object]]:
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
            "Phase 2C training requires stable-baselines3 and gymnasium."
        ) from exc
    args = parse_args()
    throttle_policy = FrozenThrottlePolicy.from_path(args.throttle_model)
    config = ResidualEnvConfig(
        scenario_name=args.scenario,
        reward_weights=residual_reward_profile(args.reward_profile),
        curriculum_enabled=not args.disable_curriculum,
        staged_wind_enabled=not args.disable_staged_wind,
        curriculum_bounds=residual_curriculum_profile(args.curriculum_profile),
        action_mode=args.action_mode,
        include_prior_throttle_feature=args.include_prior_throttle_feature,
        throttle_residual_positive_only=args.throttle_residual_positive_only,
        overspeed_brake_assist_enabled=args.enable_overspeed_brake_assist,
        energy_assist_enabled=args.enable_energy_assist,
        stopping_floor_enabled=args.enable_stopping_floor,
        brake_floor_enabled=args.enable_brake_floor,
        guidance_enabled=args.enable_guidance_prior,
    )
    if args.terminal_throttle_gate_altitude is not None:
        config = type(config)(**{**config.__dict__, "terminal_throttle_gate_altitude_m": float(args.terminal_throttle_gate_altitude)})
    if args.terminal_throttle_gate_power is not None:
        config = type(config)(**{**config.__dict__, "terminal_throttle_gate_power": float(args.terminal_throttle_gate_power)})
    if args.overspeed_brake_assist_trigger is not None:
        config = type(config)(**{**config.__dict__, "overspeed_brake_assist_trigger_mps": float(args.overspeed_brake_assist_trigger)})
    if args.overspeed_brake_assist_full_scale is not None:
        config = type(config)(**{**config.__dict__, "overspeed_brake_assist_full_scale_mps": float(args.overspeed_brake_assist_full_scale)})
    if args.overspeed_brake_assist_max_delta is not None:
        config = type(config)(**{**config.__dict__, "overspeed_brake_assist_max_delta": float(args.overspeed_brake_assist_max_delta)})
    if args.overspeed_brake_assist_late_stage_altitude is not None:
        config = type(config)(**{**config.__dict__, "overspeed_brake_assist_late_stage_altitude_m": float(args.overspeed_brake_assist_late_stage_altitude)})
    if args.overspeed_brake_assist_late_stage_extra_delta is not None:
        config = type(config)(**{**config.__dict__, "overspeed_brake_assist_late_stage_extra_delta": float(args.overspeed_brake_assist_late_stage_extra_delta)})
    if args.energy_gate_altitude is not None:
        config = type(config)(**{**config.__dict__, "energy_gate_altitude_m": float(args.energy_gate_altitude)})
    if args.energy_gate_power is not None:
        config = type(config)(**{**config.__dict__, "energy_gate_power": float(args.energy_gate_power)})
    if args.energy_full_scale is not None:
        config = type(config)(**{**config.__dict__, "energy_full_scale": float(args.energy_full_scale)})
    if args.energy_shape_power is not None:
        config = type(config)(**{**config.__dict__, "energy_shape_power": float(args.energy_shape_power)})
    if args.energy_max_delta is not None:
        config = type(config)(**{**config.__dict__, "energy_max_delta": float(args.energy_max_delta)})
    if args.energy_touchdown_speed is not None:
        config = type(config)(
            **{**config.__dict__, "energy_touchdown_speed_mps": float(args.energy_touchdown_speed)}
        )
    if args.energy_braking_accel is not None:
        config = type(config)(**{**config.__dict__, "energy_braking_accel_mps2": float(args.energy_braking_accel)})
    if args.stopping_floor_gate_altitude is not None:
        config = type(config)(
            **{**config.__dict__, "stopping_floor_gate_altitude_m": float(args.stopping_floor_gate_altitude)}
        )
    if args.stopping_floor_gate_power is not None:
        config = type(config)(
            **{**config.__dict__, "stopping_floor_gate_power": float(args.stopping_floor_gate_power)}
        )
    if args.stopping_floor_altitude_floor is not None:
        config = type(config)(
            **{**config.__dict__, "stopping_floor_altitude_floor_m": float(args.stopping_floor_altitude_floor)}
        )
    if args.stopping_floor_touchdown_speed is not None:
        config = type(config)(
            **{**config.__dict__, "stopping_floor_touchdown_speed_mps": float(args.stopping_floor_touchdown_speed)}
        )
    if args.stopping_floor_min_downward_speed is not None:
        config = type(config)(
            **{
                **config.__dict__,
                "stopping_floor_min_downward_speed_mps": float(args.stopping_floor_min_downward_speed),
            }
        )
    if args.brake_floor_altitude is not None:
        config = type(config)(**{**config.__dict__, "brake_floor_altitude_m": float(args.brake_floor_altitude)})
    if args.brake_floor_power is not None:
        config = type(config)(**{**config.__dict__, "brake_floor_power": float(args.brake_floor_power)})
    if args.brake_floor_safe_margin is not None:
        config = type(config)(**{**config.__dict__, "brake_floor_safe_margin_mps": float(args.brake_floor_safe_margin)})
    if args.brake_floor_trigger is not None:
        config = type(config)(**{**config.__dict__, "brake_floor_trigger_mps": float(args.brake_floor_trigger)})
    if args.brake_floor_full_scale is not None:
        config = type(config)(**{**config.__dict__, "brake_floor_full_scale_mps": float(args.brake_floor_full_scale)})
    if args.brake_floor_base_throttle is not None:
        config = type(config)(**{**config.__dict__, "brake_floor_base_throttle": float(args.brake_floor_base_throttle)})
    if args.brake_floor_max_throttle is not None:
        config = type(config)(**{**config.__dict__, "brake_floor_max_throttle": float(args.brake_floor_max_throttle)})
    if args.brake_floor_shape_power is not None:
        config = type(config)(**{**config.__dict__, "brake_floor_shape_power": float(args.brake_floor_shape_power)})
    if args.brake_floor_late_stage_altitude is not None:
        config = type(config)(
            **{**config.__dict__, "brake_floor_late_stage_altitude_m": float(args.brake_floor_late_stage_altitude)}
        )
    if args.brake_floor_late_stage_extra_throttle is not None:
        config = type(config)(
            **{
                **config.__dict__,
                "brake_floor_late_stage_extra_throttle": float(args.brake_floor_late_stage_extra_throttle),
            }
        )
    if args.guidance_gate_altitude is not None:
        config = type(config)(**{**config.__dict__, "guidance_gate_altitude_m": float(args.guidance_gate_altitude)})
    if args.guidance_gate_power is not None:
        config = type(config)(**{**config.__dict__, "guidance_gate_power": float(args.guidance_gate_power)})
    if args.guidance_target_touchdown_speed is not None:
        config = type(config)(
            **{**config.__dict__, "guidance_target_touchdown_speed_mps": float(args.guidance_target_touchdown_speed)}
        )
    if args.guidance_altitude_floor is not None:
        config = type(config)(**{**config.__dict__, "guidance_altitude_floor_m": float(args.guidance_altitude_floor)})
    if args.guidance_base_throttle is not None:
        config = type(config)(**{**config.__dict__, "guidance_base_throttle": float(args.guidance_base_throttle)})
    if args.guidance_late_stage_altitude is not None:
        config = type(config)(
            **{**config.__dict__, "guidance_late_stage_altitude_m": float(args.guidance_late_stage_altitude)}
        )
    if args.guidance_late_stage_extra_throttle is not None:
        config = type(config)(
            **{
                **config.__dict__,
                "guidance_late_stage_extra_throttle": float(args.guidance_late_stage_extra_throttle),
            }
        )
    ppo_cfg = phase2c_ppo_profile(args.ppo_profile)
    env = HybridResidualEnv(
        throttle_policy=throttle_policy,
        lqr_controller=GainScheduledLQRController(),
        config=config,
        seed=args.seed,
    )
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
                "action_mode": args.action_mode,
                "include_prior_throttle_feature": args.include_prior_throttle_feature,
                "throttle_residual_positive_only": args.throttle_residual_positive_only,
                "enable_overspeed_brake_assist": args.enable_overspeed_brake_assist,
                "enable_energy_assist": args.enable_energy_assist,
                "enable_stopping_floor": args.enable_stopping_floor,
                "terminal_throttle_gate_altitude": args.terminal_throttle_gate_altitude,
                "terminal_throttle_gate_power": args.terminal_throttle_gate_power,
                "overspeed_brake_assist_trigger": args.overspeed_brake_assist_trigger,
                "overspeed_brake_assist_full_scale": args.overspeed_brake_assist_full_scale,
                "overspeed_brake_assist_max_delta": args.overspeed_brake_assist_max_delta,
                "overspeed_brake_assist_late_stage_altitude": args.overspeed_brake_assist_late_stage_altitude,
                "overspeed_brake_assist_late_stage_extra_delta": args.overspeed_brake_assist_late_stage_extra_delta,
                "energy_gate_altitude": args.energy_gate_altitude,
                "energy_gate_power": args.energy_gate_power,
                "energy_full_scale": args.energy_full_scale,
                "energy_shape_power": args.energy_shape_power,
                "energy_max_delta": args.energy_max_delta,
                "energy_touchdown_speed": args.energy_touchdown_speed,
                "energy_braking_accel": args.energy_braking_accel,
                "stopping_floor_gate_altitude": args.stopping_floor_gate_altitude,
                "stopping_floor_gate_power": args.stopping_floor_gate_power,
                "stopping_floor_altitude_floor": args.stopping_floor_altitude_floor,
                "stopping_floor_touchdown_speed": args.stopping_floor_touchdown_speed,
                "stopping_floor_min_downward_speed": args.stopping_floor_min_downward_speed,
                "enable_brake_floor": args.enable_brake_floor,
                "brake_floor_altitude": args.brake_floor_altitude,
                "brake_floor_power": args.brake_floor_power,
                "brake_floor_safe_margin": args.brake_floor_safe_margin,
                "brake_floor_trigger": args.brake_floor_trigger,
                "brake_floor_full_scale": args.brake_floor_full_scale,
                "brake_floor_base_throttle": args.brake_floor_base_throttle,
                "brake_floor_max_throttle": args.brake_floor_max_throttle,
                "brake_floor_shape_power": args.brake_floor_shape_power,
                "brake_floor_late_stage_altitude": args.brake_floor_late_stage_altitude,
                "brake_floor_late_stage_extra_throttle": args.brake_floor_late_stage_extra_throttle,
                "enable_guidance_prior": args.enable_guidance_prior,
                "guidance_gate_altitude": args.guidance_gate_altitude,
                "guidance_gate_power": args.guidance_gate_power,
                "guidance_target_touchdown_speed": args.guidance_target_touchdown_speed,
                "guidance_altitude_floor": args.guidance_altitude_floor,
                "guidance_base_throttle": args.guidance_base_throttle,
                "guidance_late_stage_altitude": args.guidance_late_stage_altitude,
                "guidance_late_stage_extra_throttle": args.guidance_late_stage_extra_throttle,
                "residual_env_config": config.to_dict(),
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

    eval_env = HybridResidualEnv(
        throttle_policy=throttle_policy,
        lqr_controller=GainScheduledLQRController(),
        config=ResidualEnvConfig(
            scenario_name=config.scenario_name,
            dt_s=config.dt_s,
            max_duration_s=config.max_duration_s,
            success_criteria=config.success_criteria,
            reward_weights=config.reward_weights,
            curriculum_enabled=False,
            staged_wind_enabled=config.staged_wind_enabled,
            curriculum_bounds=config.curriculum_bounds,
            coordination_features=config.coordination_features,
            residual_gimbal_limit_rad=config.residual_gimbal_limit_rad,
            residual_throttle_delta_limit=config.residual_throttle_delta_limit,
            action_mode=config.action_mode,
            include_prior_throttle_feature=config.include_prior_throttle_feature,
            terminal_throttle_gate_altitude_m=config.terminal_throttle_gate_altitude_m,
            terminal_throttle_gate_power=config.terminal_throttle_gate_power,
            near_ground_gate_altitude_m=config.near_ground_gate_altitude_m,
            near_ground_gate_power=config.near_ground_gate_power,
            near_ground_safe_margin_mps=config.near_ground_safe_margin_mps,
            brake_floor_enabled=config.brake_floor_enabled,
            brake_floor_altitude_m=config.brake_floor_altitude_m,
            brake_floor_power=config.brake_floor_power,
            brake_floor_safe_margin_mps=config.brake_floor_safe_margin_mps,
            brake_floor_trigger_mps=config.brake_floor_trigger_mps,
            brake_floor_full_scale_mps=config.brake_floor_full_scale_mps,
            brake_floor_base_throttle=config.brake_floor_base_throttle,
            brake_floor_max_throttle=config.brake_floor_max_throttle,
            throttle_residual_positive_only=config.throttle_residual_positive_only,
        ),
        seed=args.seed,
    )
    rollout_rows, metrics = evaluate_residual_model(model, eval_env, args.seed)
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
                "action_mode": args.action_mode,
                "include_prior_throttle_feature": args.include_prior_throttle_feature,
                "throttle_residual_positive_only": args.throttle_residual_positive_only,
                "enable_overspeed_brake_assist": args.enable_overspeed_brake_assist,
                "terminal_throttle_gate_altitude": args.terminal_throttle_gate_altitude,
                "terminal_throttle_gate_power": args.terminal_throttle_gate_power,
                "overspeed_brake_assist_trigger": args.overspeed_brake_assist_trigger,
                "overspeed_brake_assist_full_scale": args.overspeed_brake_assist_full_scale,
                "overspeed_brake_assist_max_delta": args.overspeed_brake_assist_max_delta,
                "overspeed_brake_assist_late_stage_altitude": args.overspeed_brake_assist_late_stage_altitude,
                "overspeed_brake_assist_late_stage_extra_delta": args.overspeed_brake_assist_late_stage_extra_delta,
                "enable_brake_floor": args.enable_brake_floor,
                "brake_floor_altitude": args.brake_floor_altitude,
                "brake_floor_power": args.brake_floor_power,
                "brake_floor_safe_margin": args.brake_floor_safe_margin,
                "brake_floor_trigger": args.brake_floor_trigger,
                "brake_floor_full_scale": args.brake_floor_full_scale,
                "brake_floor_base_throttle": args.brake_floor_base_throttle,
                "brake_floor_max_throttle": args.brake_floor_max_throttle,
                "curriculum_enabled": config.curriculum_enabled,
                "staged_wind_enabled": config.staged_wind_enabled,
            },
            stream,
            indent=2,
            sort_keys=True,
        )

if __name__ == "__main__":
    main()
