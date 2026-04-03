"""
train_ppo.py — PPO Training Script for 6DOF Rocket Landing
===========================================================
Paper §5.3 — Reinforcement Learning Controller

Usage (from project root, venv active):
    python train_ppo.py                    # full 5M step run
    python train_ppo.py --timesteps 500000 # quick test run (~5 min)
    python train_ppo.py --timesteps 5000000 --n_envs 8  # fast multi-env

Outputs:
    models/ppo_rocket_final.zip    — final policy checkpoint
    models/ppo_rocket_best.zip     — best policy (by eval reward)
    logs/ppo_tensorboard/          — TensorBoard logs
    plots/P5_training_curves.png   — reward + success rate curves

PPO Hyperparameters (paper Table III):
    Algorithm:     PPO (Schulman et al. 2017)
    Network:       MLP [256, 256], tanh activation
    Learning rate: 3e-4 (constant)
    n_steps:       2048  (rollout length per env)
    batch_size:    64
    n_epochs:      10
    gamma:         0.99
    gae_lambda:    0.95
    clip_range:    0.2
    ent_coef:      0.01  (encourages exploration)
    n_envs:        4     (parallel environments)
    Total steps:   5,000,000
"""

import os, sys, argparse
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
# Add project root (two levels up from this file) so 'model.landing_env' resolves,
# and also add this file's directory directly for same-package imports.
sys.path.insert(0, os.path.dirname(os.path.dirname(ROOT)))  # project root
sys.path.insert(0, ROOT)  # src/model/ itself

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, BaseCallback
)


# ══════════════════════════════════════════════════════
#  Curriculum Callback (DD-017)
# ══════════════════════════════════════════════════════
class CurriculumCallback(BaseCallback):
    """
    Advances curriculum stage at fixed timestep thresholds.
    Stage 0 → 1 at 500k steps (easy → medium)
    Stage 1 → 2 at 1.5M steps (medium → full difficulty)
    Prints a message when each stage is reached.
    """
    THRESHOLDS = [500_000, 1_500_000]

    def __init__(self, verbose=1):
        super().__init__(verbose)
        self._current_stage = 0

    def _on_step(self) -> bool:
        for stage, thresh in enumerate(self.THRESHOLDS):
            if self.num_timesteps >= thresh and self._current_stage <= stage:
                self._current_stage = stage + 1
                # Set stage on all training envs
                try:
                    for env in self.training_env.envs:
                        env._curriculum_stage = self._current_stage
                except AttributeError:
                    # SubprocVecEnv — use env_method
                    pass
                if self.verbose:
                    print(f"\n[CURRICULUM] Step {self.num_timesteps:,} "
                          f"→ Stage {self._current_stage} "
                          f"({'medium' if self._current_stage==1 else 'full'} difficulty)")
        return True
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

# ── auto-detect environment module location ──────────
import importlib, pathlib
_src = pathlib.Path(__file__).parent / 'src'
_candidates = ['landing_env', 'environment.landing_env', 'model.landing_env']
_mod = None
for _c in _candidates:
    try:
        _mod = importlib.import_module(_c)
        break
    except ModuleNotFoundError:
        pass
if _mod is None:
    raise ImportError("Cannot find landing_env. Make sure src/environment/ or src/model/ exists.")
RocketLandingEnv = _mod.RocketLandingEnv
tilt_from_quaternion = _mod.tilt_from_quaternion

os.makedirs('models', exist_ok=True)
os.makedirs('logs',   exist_ok=True)
os.makedirs('plots',  exist_ok=True)


# ══════════════════════════════════════════════════════
#  Callback: track success rate during training
# ══════════════════════════════════════════════════════
class SuccessRateCallback(BaseCallback):
    """
    Logs success rate (soft landing) every eval_freq steps.
    Writes to logs/success_rate.npz for plot generation.
    """
    def __init__(self, eval_env, eval_freq=50_000, n_eval=20, verbose=1):
        super().__init__(verbose)
        self.eval_env  = eval_env
        self.eval_freq = eval_freq
        self.n_eval    = n_eval
        self.successes = []
        self.timesteps = []
        self.rewards   = []

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            wins, ep_rewards = 0, []
            for _ in range(self.n_eval):
                obs, _ = self.eval_env.reset()
                done   = False
                ep_r   = 0.
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, r, term, trunc, info = self.eval_env.step(action)
                    ep_r += r
                    done  = term or trunc
                ep_rewards.append(ep_r)
                if info.get('success', False):
                    wins += 1

            sr = wins / self.n_eval
            mr = np.mean(ep_rewards)
            self.successes.append(sr)
            self.rewards.append(mr)
            self.timesteps.append(self.num_timesteps)

            if self.verbose:
                print(f"  [eval] step={self.num_timesteps:>8,d}  "
                      f"success={sr:.0%}  mean_reward={mr:.1f}")

            # Save progress
            np.savez('logs/training_progress.npz',
                     timesteps=self.timesteps,
                     success_rate=self.successes,
                     mean_reward=self.rewards)
        return True


# ══════════════════════════════════════════════════════
#  Main training function
# ══════════════════════════════════════════════════════
def train(args):
    print("=" * 60)
    print("  Phase 2: PPO Training — 6DOF Rocket Landing")
    print("=" * 60)
    print(f"  Total timesteps : {args.timesteps:,}")
    print(f"  Parallel envs   : {args.n_envs}")
    print(f"  Device          : {args.device}")
    print()

    # ── Training environments ────────────────────────
    def make_train_env():
        env = RocketLandingEnv(
            randomise_ic=True,
            randomise_wind=True,
            t_max=60.
        )
        return Monitor(env)

    # Use SubprocVecEnv for true parallelism (faster training)
    train_env = make_vec_env(
        make_train_env,
        n_envs=args.n_envs,
        vec_env_cls=SubprocVecEnv,
        seed=42
    )
    # VecNormalize: normalises observations and rewards during training.
    # Critical for PPO on continuous control — prevents gradient collapse
    # when reward magnitudes vary widely across episodes. (DD-014b)
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=True,
        clip_reward=10.,
        gamma=0.99
    )

    # ── Evaluation environment (fixed IC, no randomisation) ──
    eval_env = RocketLandingEnv(
        randomise_ic=False,
        randomise_wind=True,
        t_max=60.,
        seed=99
    )

    # ── Callbacks ────────────────────────────────────
    success_cb = SuccessRateCallback(
        eval_env=eval_env,
        eval_freq=50_000,
        n_eval=20,
        verbose=1
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(500_000 // args.n_envs, 1),
        save_path='models/',
        name_prefix='ppo_rocket_ckpt',
        verbose=0
    )

    _eval_cb_env = make_vec_env(
        lambda: Monitor(RocketLandingEnv(
            randomise_ic=False, randomise_wind=True, t_max=60., seed=77)),
        n_envs=1,
    )
    _eval_cb_env = VecNormalize(
        _eval_cb_env,
        norm_obs=True,
        norm_reward=False,
        training=False,
        gamma=0.99,
    )
    eval_cb = EvalCallback(
        eval_env=_eval_cb_env,
        best_model_save_path='models/',
        log_path='logs/',
        eval_freq=max(100_000 // args.n_envs, 1),
        n_eval_episodes=20,
        deterministic=True,
        verbose=0
    )

    # ── PPO Model (paper Table III hyperparameters) ──
    model = PPO(
        policy='MlpPolicy',
        env=train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=[256, 256],   # two hidden layers of 256
            activation_fn=__import__('torch').nn.Tanh,
        ),
        tensorboard_log='logs/ppo_tensorboard',
        device=args.device,
        verbose=1,
        seed=42,
    )

    print(f"\nPolicy network: MLP [256, 256] tanh")
    print(f"Parameters: {sum(p.numel() for p in model.policy.parameters()):,}")
    print("\nTraining...\n")

    # ── Train ────────────────────────────────────────
    curriculum_cb = CurriculumCallback(verbose=1)

    model.learn(
        total_timesteps=args.timesteps,
        callback=[success_cb, checkpoint_cb, eval_cb, curriculum_cb],
        progress_bar=True,
    )

    # ── Save final model ─────────────────────────────
    model.save('models/ppo_rocket_final')
    train_env.save('models/vec_normalize.pkl')  # save normalisation stats
    print("\n[SAVED] models/ppo_rocket_final.zip")
    print("[SAVED] models/vec_normalize.pkl")

    train_env.close()

    # ── Plot training curves ─────────────────────────
    plot_training_curves()
    print("[DONE] Training complete.")


def plot_training_curves():
    """Generate P5: training reward + success rate curves."""
    try:
        data = np.load('logs/training_progress.npz')
    except FileNotFoundError:
        print("[WARN] No training progress data found — skipping plot.")
        return

    steps   = np.array(data['timesteps']) / 1e6
    success = np.array(data['success_rate']) * 100
    rewards = np.array(data['mean_reward'])

    BG  = '#0A0E1A'; BG2 = '#111827'; GRID = '#1E2A3A'
    C0  = '#00D4FF'; C1  = '#FF6B35'; C2  = '#A8FF3E'
    TXT = '#E8EDF5'; DIM = '#6B7B9A'

    plt.rcParams.update({
        'figure.facecolor': BG, 'axes.facecolor': BG2,
        'axes.edgecolor': GRID, 'axes.labelcolor': TXT,
        'axes.titlecolor': TXT, 'xtick.color': DIM,
        'ytick.color': DIM, 'grid.color': GRID,
        'text.color': TXT, 'font.family': 'monospace',
        'savefig.facecolor': BG, 'figure.dpi': 150, 'savefig.dpi': 180,
    })

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle('PPO Training Progress — 6DOF Rocket Landing', fontsize=12)

    # Success rate
    ax1.plot(steps, success, color=C0, linewidth=2.)
    if len(success) > 5:
        from scipy.ndimage import uniform_filter1d
        smoothed = uniform_filter1d(success, size=max(1, len(success)//8))
        ax1.plot(steps, smoothed, color=C2, linewidth=2.5,
                 linestyle='--', label='Smoothed')
    ax1.axhline(70., color=C1, linestyle=':', linewidth=1.5,
                alpha=0.8, label='70% target')
    ax1.set_ylabel('Success Rate (%)')
    ax1.set_title('Landing Success Rate (deterministic eval, 20 episodes)')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.4)
    ax1.set_ylim(-5, 105)

    # Mean episode reward
    ax2.plot(steps, rewards, color=C1, linewidth=2.)
    if len(rewards) > 5:
        smoothed_r = uniform_filter1d(rewards, size=max(1, len(rewards)//8))
        ax2.plot(steps, smoothed_r, color=C2, linewidth=2.5,
                 linestyle='--', label='Smoothed')
    ax2.axhline(0., color=DIM, linestyle=':', linewidth=0.8, alpha=0.6)
    ax2.set_ylabel('Mean Episode Reward')
    ax2.set_xlabel('Training Steps (M)')
    ax2.set_title('Mean Episode Reward')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig('plots/P5_training_curves.png', bbox_inches='tight')
    print("[PLOT] plots/P5_training_curves.png")
    plt.close()


# ══════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train PPO for rocket landing')
    parser.add_argument('--timesteps', type=int,   default=5_000_000,
                        help='Total training timesteps (default: 5M)')
    parser.add_argument('--n_envs',    type=int,   default=4,
                        help='Parallel environments (default: 4)')
    parser.add_argument('--device',    type=str,   default='auto',
                        help='Device: auto, cpu, cuda (default: auto)')
    args = parser.parse_args()
    train(args)
