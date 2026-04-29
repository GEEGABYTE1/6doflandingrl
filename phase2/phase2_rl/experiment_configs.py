# experiment profiles for 2A

from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Callable
from .reward import RewardWeights

@dataclass(frozen=True)
class PPOProfile:
    # basline PPO example
    name: str
    learning_rate: float = 3.0e-4
    linear_schedule: bool = False
    n_steps: int = 2048
    batch_size: int = 256
    gamma: float = 0.995
    gae_lambda: float = 0.98
    ent_coef: float = 0.005

    def to_dict(self) -> dict[str, float | int | bool | str]:
        return asdict(self)

    def make_learning_rate(self) -> float | Callable[[float], float]:
        if not self.linear_schedule:
            return float(self.learning_rate)
        initial_lr = float(self.learning_rate)
        return lambda progress_remaining: initial_lr * float(progress_remaining)


def reward_profile(name: str) -> RewardWeights:
    profiles: dict[str, RewardWeights] = {
        "baseline_v2": RewardWeights(
            altitude_progress=0.05,
            lateral_progress=0.0,
            lateral_error=1.10,
            outward_velocity=0.0,
            vertical_speed=0.80,
            horizontal_speed=1.00,
            tilt=0.20,
            angular_rate=0.12,
            control_effort=0.02,
            success_bonus=450.0,
            touchdown_failure_penalty=180.0,
            timeout_penalty=220.0,
            divergence_penalty=320.0,
            terminal_vertical_speed=8.0,
            terminal_horizontal_speed=12.0,
            terminal_lateral_error=10.0,
            terminal_tilt=2.0,
        ),
        "lateral_progress_experimental": RewardWeights(
            altitude_progress=0.05,
            lateral_progress=2.50,
            lateral_error=0.85,
            outward_velocity=0.60,
            vertical_speed=0.80,
            horizontal_speed=1.00,
            tilt=0.20,
            angular_rate=0.12,
            control_effort=0.02,
            success_bonus=450.0,
            touchdown_failure_penalty=180.0,
            timeout_penalty=220.0,
            divergence_penalty=320.0,
            terminal_vertical_speed=8.0,
            terminal_horizontal_speed=12.0,
            terminal_lateral_error=10.0,
            terminal_tilt=2.0,
        ),
    }
    if name not in profiles:
        raise KeyError(f"Unknown Phase 2A reward profile: {name}")
    return profiles[name]

def ppo_profile(name: str) -> PPOProfile:
    profiles: dict[str, PPOProfile] = {
        "baseline_default": PPOProfile(name="baseline_default"),
        "short_rollout_low_entropy": PPOProfile(
            name="short_rollout_low_entropy",
            n_steps=1024,
            batch_size=128,
            ent_coef=0.002,
        ),
        "short_rollout_low_entropy_linear_lr": PPOProfile(
            name="short_rollout_low_entropy_linear_lr",
            n_steps=1024,
            batch_size=128,
            ent_coef=0.001,
            linear_schedule=True,
        ),
    }
    if name not in profiles:
        raise KeyError(f"Unknown Phase 2A PPO profile: {name}")
    return profiles[name]

def reward_profile_names() -> list[str]:
    return ["baseline_v2", "lateral_progress_experimental"]

def ppo_profile_names() -> list[str]:
    return ["baseline_default", "short_rollout_low_entropy", "short_rollout_low_entropy_linear_lr"]
