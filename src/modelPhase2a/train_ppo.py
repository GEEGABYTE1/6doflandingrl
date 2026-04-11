"""
ppo training with IC curriculum (path a). no bc.
    # default: start stage=2, no noise, no randomise (as configured)
    python src/training/train_ppo.py --n-envs 8

    # full scenario from stage 0 with randomisation
    python src/training/train_ppo.py --start-stage 0 --randomise --add-noise

    # quick smoke test
    python src/training/train_ppo.py --timesteps 200000 --n-envs 4

curriculum advances IC stage when rolling success rate >= 60% over 200 episodes.

outputs (runs/<run_id>/):
    best_model.zip, final_model.zip
    training_log.csv, training_log.png
    ppo_config.json

update: model has been evaluated with updated script.
verdict: move to hierarchical learning 

"""
import sys, json, argparse, time
from datetime import datetime
from pathlib import Path
from collections import deque
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env   import SubprocVecEnv, VecMonitor
from stable_baselines3.common.monitor   import Monitor
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback, EvalCallback
)
import torch.nn as nn

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from landing_env import RocketLandingEnv, IC_STAGES, N_STAGES

PPO_KWARGS = dict(
    policy        = "MlpPolicy",
    policy_kwargs = dict(net_arch=[256, 256], activation_fn=nn.Tanh),
    learning_rate = 3e-4,
    n_steps       = 2048,
    batch_size    = 512,
    n_epochs      = 10,
    gamma         = 0.995,
    gae_lambda    = 0.95,
    clip_range    = 0.2,
    ent_coef      = 0.005,
    vf_coef       = 0.5,
    max_grad_norm = 0.5,
    normalize_advantage = True,
    verbose       = 1,
)

SENSOR_NOISE = np.array([
    2., 2., 2., 0.1, 0.1, 0.1,
    0.002, 0.002, 0.002, 0.002,
    0.005, 0.005, 0.005, 500.,
    0., 0., 0., 0.
], dtype=np.float32)

ADVANCE_THRESHOLD = 0.25   # lowered from 0.60 — 40% stochastic ≈ 70%+ deterministic --> switched to 25% to allow more time for learning each stage. we can play around with this.
WINDOW_SIZE       = 100    
MIN_STEPS_STAGE   = 30_000 

class CurriculumCallback(BaseCallback):
    def __init__(self, n_envs, run_dir, start_stage=2, verbose=1):
        super().__init__(verbose)
        self.n_envs          = n_envs
        self.run_dir         = run_dir
        self.stage           = start_stage
        self.stage_start_ts  = 0
        self._suc_window     = deque(maxlen=WINDOW_SIZE)
        self._rew_window     = deque(maxlen=200)
        self._log            = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._rew_window.append(info["episode"]["r"])
            if info.get("terminated_reason") in ("success","crashed","landed","timeout"):
                self._suc_window.append(1 if info.get("success") else 0)
        ts = self.num_timesteps
        if (self.stage < N_STAGES - 1
                and len(self._suc_window) >= WINDOW_SIZE
                and ts - self.stage_start_ts >= MIN_STEPS_STAGE
                and np.mean(self._suc_window) >= ADVANCE_THRESHOLD):
            self.stage += 1
            self.stage_start_ts = ts
            self._suc_window.clear()
            alt, vz, pos, vel, wind = IC_STAGES[self.stage]
            print(f"\n{'='*55}")
            print(f"  [Curriculum] → STAGE {self.stage}  (step {ts:,})")
            print(f"  alt={alt:.0f}m  vz={vz:.0f}m/s  "
                  f"offset=±{pos:.0f}m  wind={wind:.0f}m/s")
            print(f"{'='*55}\n")
            for i in range(self.n_envs):
                try:
                    self.training_env.env_method(
                        "set_ic_stage", self.stage, indices=[i])
                except Exception:
                    pass
        if ts % 10_000 < self.n_envs:
            mr = float(np.mean(self._rew_window)) if self._rew_window else 0.
            sr = float(np.mean(self._suc_window)) if self._suc_window else 0.
            self._log.append(dict(
                timestep     = ts,
                mean_reward  = round(mr, 2),
                success_rate = round(sr, 4),
                ic_stage     = self.stage,
                stage_alt    = IC_STAGES[self.stage][0],
            ))
            if ts % 50_000 < self.n_envs:
                print(f"  ts={ts:>8,}  rew={mr:>8.1f}  "
                      f"suc={sr:.1%}  stage={self.stage}  "
                      f"alt={IC_STAGES[self.stage][0]:.0f}m")
        return True

    def _on_training_end(self):
        if not self._log:
            return
        df = pd.DataFrame(self._log)
        csv = self.run_dir / "training_log.csv"
        df.to_csv(csv, index=False)
        _plot(df, self.run_dir / "training_log.png")
        print(f"[Log] → {csv}")

STAGE_COLORS = ["#E3F2FD","#FFF3E0","#F3E5F5","#E8F5E9","#FCE4EC"]

def _plot(df, path):
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    fig.suptitle("PPO Training — IC Curriculum (Path A)",
                 fontsize=13, fontweight="bold")
    ts_m = df.timestep / 1e6
    for i in range(N_STAGES):
        mask = df.ic_stage == i
        if mask.any():
            t0 = ts_m[mask].iloc[0]
            t1 = ts_m[mask].iloc[-1]
            for ax in axes:
                ax.axvspan(t0, t1, alpha=0.10,
                           color=STAGE_COLORS[i % len(STAGE_COLORS)])
    ax = axes[0]
    ax.plot(ts_m, df.mean_reward, color="#2196F3", lw=1.0, alpha=0.5)
    if len(df) > 20:
        ax.plot(ts_m, df.mean_reward.rolling(20, min_periods=1).mean(),
                color="#E53935", lw=2.5, label="Smoothed")
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_ylabel("Mean Episode Reward")
    ax.grid(True, alpha=0.3); ax.legend()

    ax = axes[1]
    ax.plot(ts_m, df.success_rate*100, color="#4CAF50", lw=1.0, alpha=0.5)
    if len(df) > 20:
        ax.plot(ts_m, (df.success_rate*100).rolling(20, min_periods=1).mean(),
                color="#1B5E20", lw=2.5, label="Smoothed")
    ax.axhline(60, color="orange", lw=1.2, ls="--", label="Advance threshold (60%)")
    ax.set_ylabel("Success Rate (%)"); ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3); ax.legend(fontsize=9)

    ax = axes[2]
    ax.step(ts_m, df.ic_stage, color="#9C27B0", lw=2.5, where="post")
    ax.set_yticks(range(N_STAGES))
    ax.set_yticklabels([f"S{i}: {IC_STAGES[i][0]:.0f}m"
                        for i in range(N_STAGES)], fontsize=8)
    ax.set_ylabel("IC Stage"); ax.set_xlabel("Timesteps (M)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[Plot] → {path}")

def train(total_ts     : int  = 5_000_000,
          n_envs       : int  = 8,
          seed         : int  = 42,
          run_name     : str  = None,
          ckpt_freq    : int  = 100_000,
          add_noise    : bool = False,
          start_stage  : int  = 0,
          randomise_ics: bool = False):

    run_id  = run_name or f"ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    noise = SENSOR_NOISE if add_noise else None
    print(f"\n{'='*60}")
    print(f"  Phase 2 PPO — IC Curriculum (Path A, no BC)")
    print(f"  Run        : {run_id}")
    print(f"  Timesteps  : {total_ts:,}   Envs: {n_envs}")
    print(f"  Start stage: {start_stage}  ({IC_STAGES[start_stage][0]:.0f}m / "
          f"{IC_STAGES[start_stage][1]:.0f} m/s)")
    print(f"  Randomise  : {randomise_ics}")
    print(f"  Noise      : {add_noise}")
    print(f"  Advance at : {ADVANCE_THRESHOLD:.0%} success over {WINDOW_SIZE} eps")
    print(f"{'='*60}\n")

    #training env 
    def make_env(rank):
        def _init():
            return Monitor(RocketLandingEnv(
                ic_stage      = start_stage,
                randomise_ics = randomise_ics,
                noise_std     = noise,
                seed          = seed + rank,
            ))
        return _init
    vec_env = VecMonitor(SubprocVecEnv([make_env(i) for i in range(n_envs)]))

    # eval env — stage 2 (300m, no wind, deterministic) 
    eval_env = Monitor(RocketLandingEnv(
        ic_stage      = 2,
        randomise_ics = False,
        noise_std     = None,
        seed          = 9999,
    ))
    model = PPO(env=vec_env, seed=seed, **PPO_KWARGS)
    callbacks = [
        CheckpointCallback(
            save_freq   = max(ckpt_freq // n_envs, 1),
            save_path   = str(run_dir / "checkpoints"),
            name_prefix = "ckpt",
            verbose     = 0,
        ),
        EvalCallback(
            eval_env,
            best_model_save_path = str(run_dir),
            log_path             = str(run_dir),
            eval_freq            = max(50_000 // n_envs, 1),
            n_eval_episodes      = 20,
            deterministic        = True,
            verbose              = 1,
        ),
        CurriculumCallback(
            n_envs       = n_envs,
            run_dir      = run_dir,
            start_stage  = start_stage,
            verbose      = 1,
        ),
    ]


    json.dump(dict(
        run_id=run_id, total_ts=total_ts, n_envs=n_envs,
        seed=seed, add_noise=add_noise,
        start_stage=start_stage, randomise_ics=randomise_ics,
        advance_threshold=ADVANCE_THRESHOLD,
        window_size=WINDOW_SIZE,
        min_steps_stage=MIN_STEPS_STAGE,
        stages=[dict(alt=a, vz=vz, pos=p, vel=v, wind=w)
                for a, vz, p, v, w in IC_STAGES],
        **{k: str(v) for k, v in PPO_KWARGS.items()},
    ), open(run_dir / "ppo_config.json", "w"), indent=2)

    t0 = time.time()
    model.learn(total_timesteps=total_ts, callback=callbacks, progress_bar=True)

    model.save(str(run_dir / "final_model"))
    vec_env.close(); eval_env.close()

    print(f"\n[Done] {(time.time()-t0)/3600:.1f}h")
    print(f"  Best model  → {run_dir / 'best_model.zip'}")
    print(f"  Final model → {run_dir / 'final_model.zip'}")
    print(f"  Training log→ {run_dir / 'training_log.png'}")
    return str(run_dir)



if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train PPO with IC curriculum")
    p.add_argument("--timesteps",    type=int,   default=5_000_000)
    p.add_argument("--n-envs",       type=int,   default=8)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--run-name",     type=str,   default=None)
    p.add_argument("--ckpt-freq",    type=int,   default=100_000)
    p.add_argument("--start-stage",  type=int,   default=0,
                   help="IC stage to start training at (0=easiest, 4=full scenario)")
    p.add_argument("--randomise",    action="store_true",
                   help="Randomise initial conditions each episode")
    p.add_argument("--add-noise",    action="store_true",
                   help="Add sensor noise during training")
    args = p.parse_args()

    train(
        total_ts      = args.timesteps,
        n_envs        = args.n_envs,
        seed          = args.seed,
        run_name      = args.run_name,
        ckpt_freq     = args.ckpt_freq,
        start_stage   = args.start_stage,
        randomise_ics = args.randomise,
        add_noise     = args.add_noise,
    )