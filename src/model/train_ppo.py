'''
ppo training 

python train_ppo.py                          # default 5M steps
python train_ppo.py --timesteps 2000000      # quick test run
python train_ppo.py --timesteps 5000000 --n-envs 16 --wind-final 15

outputs (all under runs/<run_id>/):
    checkpoint_<step>.zip   — policy checkpoints every 100k steps
    best_model.zip          — best mean reward seen during training
    training_log.csv        — reward curve for Figure P6
    training_log.png        — reward curve plot
    ppo_config.json         — full hyperparameter record for the paper

wind curriculum:
    0–25%   of training  →  V_ref = 0     m/s   (learn basic landing)
    25–50%  of training  →  V_ref = 5     m/s   (light wind)
    50–75%  of training  →  V_ref = 15    m/s   (moderate wind)
    75–100% of training  →  V_ref = final m/s   (target evaluation level)
'''


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

#from paper 5.2
PPO_HPARAMS = dict(
    # Network
    policy          = "MlpPolicy",
    policy_kwargs   = dict(
        net_arch        = [256, 256],
        activation_fn   = __import__("torch").nn.Tanh,
    ),
    # PPO core
    learning_rate   = 3e-4,
    n_steps         = 2048,      
    batch_size      = 512,
    n_epochs        = 10,
    gamma           = 0.995,     # high gamma — landing reward is sparse/terminal
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


#wind curriculum callback 

class WindCurriculumCallback(BaseCallback):
    '''
    gradually increase wind disturbance as the model learns
    '''
    STAGES = [0.0, 5.0, 15.0] 
    def __init__(self, total_timesteps:int, wind_final: float, n_envs:int, log_path: Path, verbose:int=1):
        super().__init__(verbose)
        self.total_ts = total_timesteps
        self.wind_final = wind_final
        self.n_envs = n_envs 
        self.log_path = log_path 
        self.stages = self.STAGES + [wind_final]
        self.breakpoints = [int(total_timesteps * i / len(self.stages)) for i in range(1, len(self.stages) + 1)]
        self.current_stage = 0 
        self._ep_rewards: list = [] 
        self._ep_successes: list = [] 
        self._log_rows: list = [] 

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
            if self.verboes:
                print(f"\n[Curriculum] step={ts:,}  →  wind={new_wind} m/s")
        
        if ts % 10_000 < self.n_envs:
            mean_rew = np.mean(self._ep_rewards[-200:]) if self._ep_rewards else 0.
            suc_rate = np.mean(self._ep_successes[-200:]) if self._ep_successes else 0.
            wind_now = self.stages[self.current_stage] 
            self._log_rows.append(dict(
                timestep = ts. 
                mean_reward = round(mean_rew, 2),
                success_rate = round(suc_rate, 4),
                wind
            ))