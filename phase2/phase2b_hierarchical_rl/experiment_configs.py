#named experiment profiles

from __future__ import annotations
from phase2_rl.curriculum import CurriculumBounds
from phase2_rl.experiment_configs import PPOProfile, ppo_profile, ppo_profile_names
from .throttle_env import ThrottleRewardWeights
from .tvc_env import TVCRewardWeights

def throttle_reward_profile(name: str) -> ThrottleRewardWeights:
    profiles: dict[str, ThrottleRewardWeights] = {
        "baseline_v1": ThrottleRewardWeights(),
        "soft_landing_v1": ThrottleRewardWeights(
            altitude_progress=0.06,
            vertical_speed=0.95,
            low_altitude_vertical_speed=2.40,
            throttle_effort=0.02,
            success_bonus=320.0,
            touchdown_failure_penalty=210.0,
            timeout_penalty=160.0,
            divergence_penalty=260.0,
            terminal_vertical_speed=14.0,
        ),
        "flare_tracking_v1": ThrottleRewardWeights(
            altitude_progress=0.04,
            vertical_speed=0.70,
            vertical_tracking=2.20,
            low_altitude_vertical_speed=2.80,
            throttle_effort=0.01,
            throttle_rate=0.08,
            success_bonus=340.0,
            touchdown_failure_penalty=220.0,
            timeout_penalty=160.0,
            divergence_penalty=260.0,
            terminal_vertical_speed=16.0,
        ),
        "touchdown_refine_v1": ThrottleRewardWeights(
            altitude_progress=0.03,
            vertical_speed=0.55,
            vertical_tracking=3.40,
            low_altitude_vertical_speed=4.20,
            throttle_effort=0.01,
            throttle_rate=0.10,
            success_bonus=360.0,
            touchdown_failure_penalty=260.0,
            timeout_penalty=170.0,
            divergence_penalty=260.0,
            terminal_vertical_speed=24.0,
        ),
        "touchdown_gate_v1": ThrottleRewardWeights(
            altitude_progress=0.03,
            vertical_speed=0.50,
            vertical_tracking=2.60,
            low_altitude_vertical_speed=3.20,
            touchdown_vertical_speed=7.50,
            touchdown_tracking=5.00,
            throttle_effort=0.01,
            throttle_rate=0.10,
            success_bonus=380.0,
            touchdown_failure_penalty=280.0,
            timeout_penalty=170.0,
            divergence_penalty=260.0,
            terminal_vertical_speed=28.0,
        ),
        "touchdown_gate_commit_v1": ThrottleRewardWeights(
            altitude_progress=0.04,
            vertical_speed=0.55,
            vertical_tracking=2.20,
            low_altitude_vertical_speed=2.80,
            touchdown_vertical_speed=5.50,
            touchdown_tracking=3.00,
            touchdown_underdescent=6.00,
            throttle_effort=0.01,
            throttle_rate=0.08,
            success_bonus=380.0,
            touchdown_failure_penalty=260.0,
            timeout_penalty=170.0,
            divergence_penalty=260.0,
            terminal_vertical_speed=24.0,
        ),
        "touchdown_envelope_v1": ThrottleRewardWeights(
            altitude_progress=0.04,
            vertical_speed=0.55,
            vertical_tracking=2.20,
            low_altitude_vertical_speed=2.80,
            touchdown_vertical_speed=5.00,
            touchdown_tracking=2.80,
            touchdown_underdescent=6.50,
            terminal_envelope_overspeed=7.00,
            terminal_envelope_underspeed=4.50,
            throttle_effort=0.01,
            throttle_rate=0.08,
            success_bonus=400.0,
            touchdown_failure_penalty=270.0,
            timeout_penalty=170.0,
            divergence_penalty=260.0,
            terminal_vertical_speed=26.0,
        ),
        "touchdown_envelope_soft_v1": ThrottleRewardWeights(
            altitude_progress=0.04,
            vertical_speed=0.55,
            vertical_tracking=2.20,
            low_altitude_vertical_speed=2.80,
            touchdown_vertical_speed=5.20,
            touchdown_tracking=2.80,
            touchdown_underdescent=5.50,
            terminal_envelope_overspeed=8.50,
            terminal_envelope_underspeed=2.50,
            throttle_effort=0.01,
            throttle_rate=0.08,
            success_bonus=405.0,
            touchdown_failure_penalty=270.0,
            timeout_penalty=170.0,
            divergence_penalty=260.0,
            terminal_vertical_speed=26.0,
        ),
        "braking_discoverability_v1": ThrottleRewardWeights(
            altitude_progress=0.04,
            vertical_speed=0.55,
            vertical_tracking=2.20,
            low_altitude_vertical_speed=2.80,
            touchdown_vertical_speed=5.50,
            touchdown_tracking=3.00,
            touchdown_underdescent=6.00,
            throttle_effort=0.01,
            throttle_rate=0.08,
            success_bonus=380.0,
            touchdown_failure_penalty=260.0,
            timeout_penalty=170.0,
            divergence_penalty=260.0,
            terminal_vertical_speed=24.0,
            potential_shaping=2.50,
        ),
    }
    if name not in profiles:
        raise KeyError(f"Unknown Phase 2B throttle reward profile: {name}")
    return profiles[name]


def throttle_env_overrides(name: str) -> dict[str, float]:
    profiles: dict[str, dict[str, float]] = {
        "baseline_v1": {},
        "soft_landing_v1": {},
        "flare_tracking_v1": {},
        "touchdown_refine_v1": {},
        "touchdown_gate_v1": {
            "touchdown_zone_altitude_m": 12.0,
            "touchdown_gate_power": 2.0,
        },
        "touchdown_gate_commit_v1": {
            "touchdown_zone_altitude_m": 12.0,
            "touchdown_gate_power": 2.0,
            "terminal_envelope_altitude_m": 20.0,
            "terminal_envelope_wide_margin_mps": 2.0,
            "terminal_envelope_tight_margin_mps": 0.4,
            "timeout_above_ground_penalty": 2.0,
            "timeout_underdescent_penalty": 14.0,
        },
        "touchdown_envelope_v1": {
            "touchdown_zone_altitude_m": 12.0,
            "touchdown_gate_power": 2.0,
            "terminal_envelope_altitude_m": 20.0,
            "terminal_envelope_wide_margin_mps": 1.8,
            "terminal_envelope_tight_margin_mps": 0.3,
            "timeout_above_ground_penalty": 2.0,
            "timeout_underdescent_penalty": 14.0,
        },
        "touchdown_envelope_soft_v1": {
            "touchdown_zone_altitude_m": 12.0,
            "touchdown_gate_power": 2.0,
            "terminal_envelope_altitude_m": 22.0,
            "terminal_envelope_wide_margin_mps": 2.6,
            "terminal_envelope_tight_margin_mps": 0.6,
            "timeout_above_ground_penalty": 2.0,
            "timeout_underdescent_penalty": 12.0,
        },
        "braking_discoverability_v1": {
            "touchdown_zone_altitude_m": 12.0,
            "touchdown_gate_power": 2.0,
            "terminal_envelope_altitude_m": 20.0,
            "terminal_envelope_wide_margin_mps": 2.0,
            "terminal_envelope_tight_margin_mps": 0.4,
            "timeout_above_ground_penalty": 2.0,
            "timeout_underdescent_penalty": 14.0,
            "observation_mode": "braking_awareness_v1",
            "potential_mode": "stopping_distance_ratio_v1",
            "potential_gamma": 0.99,
            "feature_eps": 1.0e-3,
        },
    }
    if name not in profiles:
        raise KeyError(f"Unknown Phase 2B throttle env override profile: {name}")
    return dict(profiles[name])


def tvc_reward_profile(name: str) -> TVCRewardWeights:
    profiles: dict[str, TVCRewardWeights] = {
        "baseline_v1": TVCRewardWeights(),
        "lateral_recovery_v1": TVCRewardWeights(
            lateral_progress=1.40,
            lateral_error=1.80,
            outward_velocity=0.90,
            horizontal_speed=1.40,
            tilt=0.36,
            angular_rate=0.18,
            control_effort=0.02,
            vertical_speed_coupling=0.10,
            success_bonus=520.0,
            touchdown_failure_penalty=220.0,
            timeout_penalty=240.0,
            divergence_penalty=340.0,
            terminal_horizontal_speed=18.0,
            terminal_lateral_error=20.0,
            terminal_tilt=6.0,
            terminal_vertical_speed=1.5,
        ),
    }
    if name not in profiles:
        raise KeyError(f"Unknown Phase 2B TVC reward profile: {name}")
    return profiles[name]


def phase2b_ppo_profile(name: str) -> PPOProfile:
    return ppo_profile(name)


def tvc_curriculum_profile(name: str) -> CurriculumBounds:
    profiles: dict[str, CurriculumBounds] = {
        "baseline_v1": CurriculumBounds(),
        "near_pad_stabilization_v1": CurriculumBounds(
            min_altitude_m=25.0,
            max_altitude_m=120.0,
            min_lateral_offset_m=0.15,
            max_lateral_offset_m=6.0,
            min_attitude_error_deg=0.1,
            max_attitude_error_deg=1.0,
            min_wind_mps=0.0,
            max_wind_mps=0.6,
            min_gust_mps=0.0,
            max_gust_mps=0.1,
            min_misalignment_deg=0.0,
            max_misalignment_deg=0.06,
        ),
    }
    if name not in profiles:
        raise KeyError(f"Unknown Phase 2B TVC curriculum profile: {name}")
    return profiles[name]


def throttle_reward_profile_names() -> list[str]:
    return [
        "baseline_v1",
        "soft_landing_v1",
        "flare_tracking_v1",
        "touchdown_refine_v1",
        "touchdown_gate_v1",
        "touchdown_gate_commit_v1",
        "touchdown_envelope_v1",
        "touchdown_envelope_soft_v1",
        "braking_discoverability_v1",
    ]


def tvc_reward_profile_names() -> list[str]:
    return ["baseline_v1", "lateral_recovery_v1"]


def tvc_curriculum_profile_names() -> list[str]:
    return ["baseline_v1", "near_pad_stabilization_v1"]

__all__ = [
    "phase2b_ppo_profile",
    "ppo_profile_names",
    "throttle_env_overrides",
    "throttle_reward_profile",
    "throttle_reward_profile_names",
    "tvc_curriculum_profile",
    "tvc_curriculum_profile_names",
    "tvc_reward_profile",
    "tvc_reward_profile_names",
]
