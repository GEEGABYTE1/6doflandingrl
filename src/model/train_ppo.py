"""
ppo training for 6DOF rocket landing (Phase 2).

Usage:
    python train_ppo.py                          # default 5M steps
    python train_ppo.py --timesteps 2000000      # quick test run
    python train_ppo.py --timesteps 5000000 --n-envs 16 --wind-final 15

Outputs (all under runs/<run_id>/):
    checkpoint_<step>.zip   — policy checkpoints every 100k steps
    best_model.zip          — best mean reward seen during training
    training_log.csv        — reward curve for Figure P6
    training_log.png        — reward curve plot
    ppo_config.json         — full hyperparameter record for the paper

"""

import os, sys, json, argparse, time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback, EvalCallback
)
from stable_baselines3.common.monitor import Monitor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.environment.landing_env import RocketLandingEnv


PPO_HPARAMS = dict(

    policy          = "MlpPolicy",
    policy_kwargs   = dict(
        net_arch        = [256, 256],
        activation_fn   = __import__("torch").nn.Tanh,
    ),

    learning_rate   = 3e-4,
    n_steps         = 2048,     
    batch_size      = 512,
    n_epochs        = 10,
    gamma           = 0.995,     
    gae_lambda      = 0.95,
    clip_range      = 0.2,
    ent_coef        = 0.005,     
    vf_coef         = 0.5,
    max_grad_norm   = 0.5,
    normalize_advantage = True,
    verbose         = 1,
)


SENSOR_NOISE = np.array([
    2.0, 2.0, 2.0,         
    0.1, 0.1, 0.1,          
    0.002, 0.002, 0.002, 0.002,  
    0.005, 0.005, 0.005,    
    500.0,                 
    0., 0., 0., 0.          
], dtype=np.float32)


#wind curriculum
class WindCurriculumCallback(BaseCallback):
    """
    Gradually increase wind disturbance as training progresses.
    """
    STAGES = [0.0, 5.0, 15.0] 

    def __init__(self, total_timesteps: int, wind_final: float, n_envs: int,
                 log_path: Path, verbose: int = 1):
        super().__init__(verbose)
        self.total_ts   = total_timesteps
        self.wind_final = wind_final
        self.n_envs     = n_envs
        self.log_path   = log_path
        self.stages     = self.STAGES + [wind_final]
        self.breakpoints= [int(total_timesteps * i / len(self.stages))
                           for i in range(1, len(self.stages) + 1)]
        self.current_stage = 0
        self._ep_rewards  : list = []
        self._ep_successes: list = []
        self._log_rows    : list = []

    def _on_step(self) -> bool:
      
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._ep_rewards.append(info["episode"]["r"])
            if info.get("terminated_reason") in ("success", "crashed", "landed", "timeout"):
                self._ep_successes.append(1 if info.get("success") else 0)

        
        ts = self.num_timesteps
        stage_idx = sum(1 for bp in self.breakpoints if ts >= bp)
        stage_idx = min(stage_idx, len(self.stages) - 1)

        if stage_idx != self.current_stage:
            new_wind = self.stages[stage_idx]
            self.current_stage = stage_idx
          
            for env_idx in range(self.n_envs):
                try:
                    self.training_env.env_method(
                        "set_wind_level", new_wind, indices=[env_idx]
                    )
                except Exception:
                    pass
            if self.verbose:
                print(f"\n[Curriculum] step={ts:,}  →  wind={new_wind} m/s")

     
        if ts % 10_000 < self.n_envs:
            mean_rew = np.mean(self._ep_rewards[-200:]) if self._ep_rewards else 0.
            suc_rate = np.mean(self._ep_successes[-200:]) if self._ep_successes else 0.
            wind_now = self.stages[self.current_stage]
            self._log_rows.append(dict(
                timestep    = ts,
                mean_reward = round(mean_rew, 2),
                success_rate= round(suc_rate, 4),
                wind_level  = wind_now,
            ))
            if self.verbose >= 2:
                print(f"  ts={ts:>8,}  rew={mean_rew:>8.1f}  "
                      f"suc={suc_rate:.2%}  wind={wind_now:.0f} m/s")

        return True  

    def _on_training_end(self):
        if not self._log_rows:
            return
        df = pd.DataFrame(self._log_rows)
        csv_path = self.log_path / "training_log.csv"
        df.to_csv(csv_path, index=False)
        print(f"[Log] Training log saved → {csv_path}")
        _plot_reward_curve(df, self.log_path / "training_log.png")



def _plot_reward_curve(df: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("PPO Training Progress", fontsize=14, fontweight="bold")

    
    ax = axes[0]
    ax.plot(df["timestep"] / 1e6, df["mean_reward"], color="#2196F3", lw=1.5, label="Mean reward (200-ep)")
    
    if len(df) > 20:
        smooth = pd.Series(df["mean_reward"]).rolling(20, min_periods=1).mean()
        ax.plot(df["timestep"] / 1e6, smooth, color="#F44336", lw=2.5, label="Smoothed")
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_ylabel("Mean Episode Reward", fontsize=11)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    
    ax2 = axes[1]
    ax2.plot(df["timestep"] / 1e6, df["success_rate"] * 100,
             color="#4CAF50", lw=1.5, label="Success rate %")
    if len(df) > 20:
        smooth_suc = pd.Series(df["success_rate"] * 100).rolling(20, min_periods=1).mean()
        ax2.plot(df["timestep"] / 1e6, smooth_suc, color="#388E3C", lw=2.5, label="Smoothed")

    ax2b = ax2.twinx()
    ax2b.plot(df["timestep"] / 1e6, df["wind_level"], color="#FF9800",
              lw=1.5, ls="--", label="Wind (m/s)")
    ax2b.set_ylabel("Wind V_ref (m/s)", color="#FF9800", fontsize=10)
    ax2b.tick_params(axis="y", labelcolor="#FF9800")

    ax2.set_xlabel("Timesteps (M)", fontsize=11)
    ax2.set_ylabel("Success Rate (%)", fontsize=11)
    ax2.set_ylim(-5, 105)
    ax2.legend(loc="upper left")
    ax2b.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    # Shade curriculum stages
    stage_colors = ["#E3F2FD", "#FFF3E0", "#FCE4EC", "#E8F5E9"]
    breakpoints  = [0] + list(df[df["wind_level"].diff() != 0]["timestep"] / 1e6) + [df["timestep"].max() / 1e6]
    wind_labels  = df.groupby("wind_level")["timestep"].min().sort_values()
    for i in range(len(breakpoints) - 1):
        for ax_s in axes:
            ax_s.axvspan(breakpoints[i], breakpoints[i + 1],
                         alpha=0.12, color=stage_colors[i % len(stage_colors)])

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Reward curve saved → {out_path}")



def train(
    total_timesteps : int   = 5_000_000,
    n_envs          : int   = 8,
    wind_final      : float = 15.0,
    seed            : int   = 42,
    run_name        : str   = None,
    checkpoint_freq : int   = 100_000,
    add_noise       : bool  = True,
):
    run_id   = run_name or f"ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir  = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Phase 2 PPO Training — 6DOF Rocket Landing")
    print(f"  Run  : {run_id}")
    print(f"  Steps: {total_timesteps:,}   Envs: {n_envs}")
    print(f"  Wind curriculum: 0 → 5 → 15 → {wind_final} m/s")
    print(f"  Noise: {'ON' if add_noise else 'OFF'}")
    print(f"{'='*60}\n")


    noise = SENSOR_NOISE if add_noise else None

    def make_env(rank):
        def _init():
            env = RocketLandingEnv(
                wind_level    = 0.0,     
                randomise_ics = True,
                noise_std     = noise,
                seed          = seed + rank,
            )
            env = Monitor(env)
            return env
        return _init

    vec_env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    vec_env = VecMonitor(vec_env)

    eval_env = Monitor(RocketLandingEnv(
        wind_level    = 5.0,
        randomise_ics = False,
        noise_std     = None,
        seed          = 9999,
    ))

    
    checkpoint_cb = CheckpointCallback(
        save_freq   = max(checkpoint_freq // n_envs, 1),
        save_path   = str(ckpt_dir),
        name_prefix = "checkpoint",
        verbose     = 1,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = str(run_dir),
        log_path             = str(run_dir),
        eval_freq            = max(50_000 // n_envs, 1),
        n_eval_episodes      = 20,
        deterministic        = True,
        verbose              = 1,
    )
    curriculum_cb = WindCurriculumCallback(
        total_timesteps = total_timesteps,
        wind_final      = wind_final,
        n_envs          = n_envs,
        log_path        = run_dir,
        verbose         = 1,
    )

   
    model = PPO(
        env  = vec_env,
        seed = seed,
        **PPO_HPARAMS,
    )

    
    config = dict(
        run_id          = run_id,
        total_timesteps = total_timesteps,
        n_envs          = n_envs,
        wind_final      = wind_final,
        seed            = seed,
        add_noise       = add_noise,
        **{k: str(v) for k, v in PPO_HPARAMS.items()},
    )
    with open(run_dir / "ppo_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"[Config] Saved → {run_dir / 'ppo_config.json'}")


    t0 = time.time()
    model.learn(
        total_timesteps = total_timesteps,
        callback        = [checkpoint_cb, eval_cb, curriculum_cb],
        progress_bar    = True,
    )
    elapsed = time.time() - t0


    final_path = run_dir / "final_model"
    model.save(str(final_path))
    print(f"\n[Done] Training complete in {elapsed/3600:.1f}h")
    print(f"  Final model → {final_path}.zip")
    print(f"  Best model  → {run_dir / 'best_model.zip'}")

    vec_env.close()
    eval_env.close()
    return str(run_dir)


#cli 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO for 6DOF rocket landing")
    parser.add_argument("--timesteps",      type=int,   default=5_000_000)
    parser.add_argument("--n-envs",         type=int,   default=8)
    parser.add_argument("--wind-final",     type=float, default=15.0)
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--run-name",       type=str,   default=None)
    parser.add_argument("--checkpoint-freq",type=int,   default=100_000)
    parser.add_argument("--no-noise",       action="store_true")
    args = parser.parse_args()

    train(
        total_timesteps = args.timesteps,
        n_envs          = args.n_envs,
        wind_final      = args.wind_final,
        seed            = args.seed,
        run_name        = args.run_name,
        checkpoint_freq = args.checkpoint_freq,
        add_noise       = not args.no_noise,
    )