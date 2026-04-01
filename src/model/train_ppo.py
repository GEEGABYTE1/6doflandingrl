"""
training script
===========================================================
Paper 5.3
"""

import os, sys, argparse
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(ROOT))  

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, BaseCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from model.landing_env import RocketLandingEnv

os.makedirs('models', exist_ok=True)
os.makedirs('logs',   exist_ok=True)
os.makedirs('plots',  exist_ok=True)


class SuccessRateCallback(BaseCallback):
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

    
            np.savez('logs/training_progress.npz',
                     timesteps=self.timesteps,
                     success_rate=self.successes,
                     mean_reward=self.rewards)
        return True

def train(args):
    print("=" * 60)
    print("  Phase 2: PPO Training — 6DOF Rocket Landing")
    print("=" * 60)
    print(f"  Total timesteps : {args.timesteps:,}")
    print(f"  Parallel envs   : {args.n_envs}")
    print(f"  Device          : {args.device}")
    print()

   
    def make_train_env():
        env = RocketLandingEnv(
            randomise_ic=True,
            randomise_wind=True,
            t_max=60.
        )
        return Monitor(env)

    train_env = make_vec_env(
        make_train_env,
        n_envs=args.n_envs,
        vec_env_cls=SubprocVecEnv,
        seed=42
    )

    eval_env = RocketLandingEnv(
        randomise_ic=False,
        randomise_wind=True,
        t_max=60.,
        seed=99
    )

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

    eval_cb = EvalCallback(
        eval_env=Monitor(RocketLandingEnv(
            randomise_ic=False, randomise_wind=True, t_max=60., seed=77)),
        best_model_save_path='models/',
        log_path='logs/',
        eval_freq=max(100_000 // args.n_envs, 1),
        n_eval_episodes=20,
        deterministic=True,
        verbose=0
    )

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
            net_arch=[256, 256],   
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

    model.learn(
        total_timesteps=args.timesteps,
        callback=[success_cb, checkpoint_cb, eval_cb],
        progress_bar=True,
    )

    model.save('models/ppo_rocket_final')
    print("\n[SAVED] models/ppo_rocket_final.zip")

    train_env.close()

    plot_training_curves()
    print("[DONE] Training complete.")


def plot_training_curves():
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
