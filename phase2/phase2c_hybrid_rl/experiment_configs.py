from __future__ import annotations

from dataclasses import dataclass
from phase2_rl.curriculum import CurriculumBounds
from phase2_rl.experiment_configs import PPOProfile, ppo_profile, ppo_profile_names

@dataclass(frozen=True)
class ResidualRewardWeights:
    """Reward weights for the Phase 2C residual TVC task."""

    lateral_error: float = 1.35
    horizontal_speed: float = 1.20
    tilt: float = 0.45
    angular_rate: float = 0.22
    vertical_speed_coupling: float = 0.18
    residual_effort: float = 0.04
    residual_rate: float = 0.02
    success_bonus: float = 520.0
    touchdown_failure_penalty: float = 180.0
    timeout_penalty: float = 220.0
    divergence_penalty: float = 320.0
    terminal_horizontal_speed: float = 12.0
    terminal_lateral_error: float = 12.0
    terminal_tilt: float = 5.0
    terminal_vertical_speed: float = 2.0
    terminal_vertical_tracking: float = 0.0
    near_ground_overspeed: float = 0.0


def residual_reward_profile(name: str) -> ResidualRewardWeights:
    profiles = {
        "baseline_v1": ResidualRewardWeights(),
        "vertical_focus_v1": ResidualRewardWeights(
            lateral_error=1.20,
            horizontal_speed=1.10,
            tilt=0.40,
            angular_rate=0.20,
            vertical_speed_coupling=0.45,
            residual_effort=0.04,
            residual_rate=0.02,
            success_bonus=540.0,
            touchdown_failure_penalty=200.0,
            timeout_penalty=220.0,
            divergence_penalty=320.0,
            terminal_horizontal_speed=10.0,
            terminal_lateral_error=10.0,
            terminal_tilt=4.0,
            terminal_vertical_speed=6.0,
        ),
        "terminal_brake_focus_v1": ResidualRewardWeights(
            lateral_error=1.05,
            horizontal_speed=1.00,
            tilt=0.36,
            angular_rate=0.18,
            vertical_speed_coupling=0.60,
            residual_effort=0.04,
            residual_rate=0.02,
            success_bonus=540.0,
            touchdown_failure_penalty=210.0,
            timeout_penalty=220.0,
            divergence_penalty=320.0,
            terminal_horizontal_speed=9.0,
            terminal_lateral_error=9.0,
            terminal_tilt=3.5,
            terminal_vertical_speed=8.5,
            terminal_vertical_tracking=2.8,
        ),
        "terminal_brake_one_sided_v1": ResidualRewardWeights(
            lateral_error=1.05,
            horizontal_speed=1.00,
            tilt=0.36,
            angular_rate=0.18,
            vertical_speed_coupling=0.65,
            residual_effort=0.03,
            residual_rate=0.015,
            success_bonus=540.0,
            touchdown_failure_penalty=220.0,
            timeout_penalty=220.0,
            divergence_penalty=320.0,
            terminal_horizontal_speed=9.0,
            terminal_lateral_error=9.0,
            terminal_tilt=3.5,
            terminal_vertical_speed=10.0,
            terminal_vertical_tracking=3.5,
        ),
        "near_ground_touchdown_v1": ResidualRewardWeights(
            lateral_error=1.20,
            horizontal_speed=1.10,
            tilt=0.40,
            angular_rate=0.20,
            vertical_speed_coupling=0.45,
            residual_effort=0.04,
            residual_rate=0.02,
            success_bonus=540.0,
            touchdown_failure_penalty=200.0,
            timeout_penalty=220.0,
            divergence_penalty=320.0,
            terminal_horizontal_speed=10.0,
            terminal_lateral_error=10.0,
            terminal_tilt=4.0,
            terminal_vertical_speed=6.0,
            near_ground_overspeed=4.0,
        ),
    }
    if name not in profiles:
        raise KeyError(f"Unknown Phase 2C residual reward profile: {name}")
    return profiles[name]


def residual_curriculum_profile(name: str) -> CurriculumBounds:
    profiles = {
        "baseline_v1": CurriculumBounds(),
    }
    if name not in profiles:
        raise KeyError(f"Unknown Phase 2C residual curriculum profile: {name}")
    return profiles[name]


def phase2c_ppo_profile(name: str) -> PPOProfile:
    return ppo_profile(name)


def residual_reward_profile_names() -> list[str]:
    return [
        "baseline_v1",
        "vertical_focus_v1",
        "terminal_brake_focus_v1",
        "terminal_brake_one_sided_v1",
        "near_ground_touchdown_v1",
    ]


def residual_curriculum_profile_names() -> list[str]:
    return ["baseline_v1"]


__all__ = [
    "ResidualRewardWeights",
    "phase2c_ppo_profile",
    "ppo_profile_names",
    "residual_curriculum_profile",
    "residual_curriculum_profile_names",
    "residual_reward_profile",
    "residual_reward_profile_names",
]
