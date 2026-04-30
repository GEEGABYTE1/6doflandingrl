from __future__ import annotations

import numpy as np

from phase2b_hierarchical_rl.coordination_features import (
    CoordinationFeatureConfig,
    flare_reference_vertical_speed,
)

def terminal_throttle_residual_gate(
    altitude_m: float,
    gate_altitude_m: float,
    gate_power: float,
) -> float:
    gate_altitude_m = max(float(gate_altitude_m), 1.0e-6)
    gate_power = max(float(gate_power), 1.0)
    progress = np.clip((gate_altitude_m - float(altitude_m)) / gate_altitude_m, 0.0, 1.0)
    return float(progress**gate_power)


__all__ = ["terminal_throttle_residual_gate"]

def overspeed_severity(
    altitude_m: float,
    vertical_speed_mps: float,
    config: CoordinationFeatureConfig,
    trigger_mps: float,
    full_scale_mps: float,
) -> float:
    vz_ref = flare_reference_vertical_speed(altitude_m, config)
    overspeed_error = max(vz_ref - float(vertical_speed_mps), 0.0)
    if overspeed_error <= trigger_mps:
        return 0.0
    span = max(float(full_scale_mps) - float(trigger_mps), 1.0e-6)
    return float(np.clip((overspeed_error - float(trigger_mps)) / span, 0.0, 1.0))

def stopping_required_deceleration(
    altitude_m: float,
    vertical_speed_mps: float,
    altitude_floor_m: float,
    touchdown_speed_mps: float,
) -> float:
    downward_speed_mps = max(-float(vertical_speed_mps), 0.0)
    touchdown_speed_mps = max(float(touchdown_speed_mps), 0.0)
    altitude_for_braking_m = max(float(altitude_m), float(altitude_floor_m), 1.0e-6)
    return max(
        (downward_speed_mps**2 - touchdown_speed_mps**2) / (2.0 * altitude_for_braking_m),
        0.0,
    )

def vertical_specific_energy_excess(
    altitude_m: float,
    vertical_speed_mps: float,
    touchdown_speed_mps: float,
    braking_accel_mps2: float,
    standard_gravity_mps2: float,
) -> float:
    altitude_m = max(float(altitude_m), 0.0)
    downward_speed_mps = max(-float(vertical_speed_mps), 0.0)
    touchdown_speed_mps = max(float(touchdown_speed_mps), 0.0)
    braking_accel_mps2 = max(float(braking_accel_mps2), 1.0e-6)
    desired_speed_mps = float(np.sqrt(touchdown_speed_mps**2 + 2.0 * braking_accel_mps2 * altitude_m))
    specific_energy_actual = float(standard_gravity_mps2) * altitude_m + 0.5 * downward_speed_mps**2
    specific_energy_desired = float(standard_gravity_mps2) * altitude_m + 0.5 * desired_speed_mps**2
    return max(specific_energy_actual - specific_energy_desired, 0.0)


def overspeed_brake_assist_delta(
    altitude_m: float,
    vertical_speed_mps: float,
    config: CoordinationFeatureConfig,
    gate_altitude_m: float,
    gate_power: float,
    trigger_mps: float,
    full_scale_mps: float,
    max_delta: float,
    late_stage_altitude_m: float = 0.0,
    late_stage_extra_delta: float = 0.0,
) -> float:
    # assist activates only inside the terminal braking window
    gate = terminal_throttle_residual_gate(
        altitude_m=altitude_m,
        gate_altitude_m=gate_altitude_m,
        gate_power=gate_power,
    )
    if gate <= 0.0 or max_delta <= 0.0:
        return 0.0
    strength = overspeed_severity(
        altitude_m=altitude_m,
        vertical_speed_mps=vertical_speed_mps,
        config=config,
        trigger_mps=trigger_mps,
        full_scale_mps=full_scale_mps,
    )
    if strength <= 0.0:
        return 0.0
    assist = float(max_delta * gate * strength)
    if late_stage_altitude_m > 0.0 and late_stage_extra_delta > 0.0:
        late_gate = terminal_throttle_residual_gate(
            altitude_m=altitude_m,
            gate_altitude_m=late_stage_altitude_m,
            gate_power=1.0,
        )
        assist += float(late_stage_extra_delta * late_gate * strength)
    return assist

def energy_assist_delta(
    altitude_m: float,
    vertical_speed_mps: float,
    gate_altitude_m: float,
    gate_power: float,
    full_scale: float,
    shape_power: float,
    max_delta: float,
    touchdown_speed_mps: float,
    braking_accel_mps2: float,
    standard_gravity_mps2: float,
) -> float:
    gate = terminal_throttle_residual_gate(
        altitude_m=altitude_m,
        gate_altitude_m=gate_altitude_m,
        gate_power=gate_power,
    )
    if gate <= 0.0 or max_delta <= 0.0:
        return 0.0
    excess = vertical_specific_energy_excess(
        altitude_m=altitude_m,
        vertical_speed_mps=vertical_speed_mps,
        touchdown_speed_mps=touchdown_speed_mps,
        braking_accel_mps2=braking_accel_mps2,
        standard_gravity_mps2=standard_gravity_mps2,
    )
    if excess <= 0.0:
        return 0.0
    full_scale = max(float(full_scale), 1.0e-6)
    shape_power = max(float(shape_power), 1.0e-6)
    strength = float(np.clip(excess / full_scale, 0.0, 1.0) ** shape_power)
    return float(max_delta * gate * strength)

def stopping_distance_floor_throttle(
    altitude_m: float,
    vertical_speed_mps: float,
    mass_kg: float,
    min_throttle: float,
    max_throttle: float,
    max_thrust_n: float,
    standard_gravity_mps2: float,
    gate_altitude_m: float,
    gate_power: float,
    altitude_floor_m: float,
    touchdown_speed_mps: float,
    min_downward_speed_mps: float,
) -> float:
    gate = terminal_throttle_residual_gate(
        altitude_m=altitude_m,
        gate_altitude_m=gate_altitude_m,
        gate_power=gate_power,
    )
    downward_speed_mps = max(-float(vertical_speed_mps), 0.0)
    if gate <= 0.0 or downward_speed_mps < max(float(min_downward_speed_mps), 0.0):
        return 0.0
    required_accel_mps2 = stopping_required_deceleration(
        altitude_m=altitude_m,
        vertical_speed_mps=vertical_speed_mps,
        altitude_floor_m=altitude_floor_m,
        touchdown_speed_mps=touchdown_speed_mps,
    )
    thrust_required_n = float(mass_kg) * (float(standard_gravity_mps2) + required_accel_mps2)
    raw_throttle = gate * (thrust_required_n / max(float(max_thrust_n), 1.0e-6))
    return float(np.clip(raw_throttle, min_throttle, max_throttle))

def overspeed_brake_floor_throttle(
    altitude_m: float,
    vertical_speed_mps: float,
    config: CoordinationFeatureConfig,
    gate_altitude_m: float,
    gate_power: float,
    safe_margin_mps: float,
    trigger_mps: float,
    full_scale_mps: float,
    base_throttle: float,
    max_throttle: float,
    shape_power: float = 1.0,
    late_stage_altitude_m: float = 0.0,
    late_stage_extra_throttle: float = 0.0,
) -> float:
    gate = terminal_throttle_residual_gate(
        altitude_m=altitude_m,
        gate_altitude_m=gate_altitude_m,
        gate_power=gate_power,
    )
    if gate <= 0.0:
        return 0.0
    strength = overspeed_severity(
        altitude_m=altitude_m,
        vertical_speed_mps=vertical_speed_mps,
        config=config,
        trigger_mps=max(float(trigger_mps), float(safe_margin_mps)),
        full_scale_mps=full_scale_mps,
    )
    if strength <= 0.0:
        return 0.0
    shape_power = max(float(shape_power), 1.0e-6)
    shaped_strength = float(np.clip(strength, 0.0, 1.0) ** shape_power)
    base_throttle = float(base_throttle)
    max_throttle = float(max_throttle)
    floor = base_throttle + (max_throttle - base_throttle) * gate * shaped_strength
    if late_stage_altitude_m > 0.0 and late_stage_extra_throttle > 0.0:
        late_gate = terminal_throttle_residual_gate(
            altitude_m=altitude_m,
            gate_altitude_m=late_stage_altitude_m,
            gate_power=1.0,
        )
        floor += float(late_stage_extra_throttle * late_gate * shaped_strength)
    return float(np.clip(floor, 0.0, max_throttle))

def guidance_brake_throttle(
    altitude_m: float,
    vertical_speed_mps: float,
    mass_kg: float,
    min_throttle: float,
    max_throttle: float,
    max_thrust_n: float,
    standard_gravity_mps2: float,
    gate_altitude_m: float,
    gate_power: float,
    target_touchdown_speed_mps: float,
    altitude_floor_m: float,
    base_throttle: float = 0.0,
    late_stage_altitude_m: float = 0.0,
    late_stage_extra_throttle: float = 0.0,
) -> float:

    gate = terminal_throttle_residual_gate(
        altitude_m=altitude_m,
        gate_altitude_m=gate_altitude_m,
        gate_power=gate_power,
    )
    if gate <= 0.0:
        return 0.0

    downward_speed_mps = max(-float(vertical_speed_mps), 0.0)
    target_touchdown_speed_mps = max(float(target_touchdown_speed_mps), 0.0)
    altitude_for_guidance_m = max(float(altitude_m), float(altitude_floor_m), 1.0e-6)
    required_accel_mps2 = max(
        (downward_speed_mps**2 - target_touchdown_speed_mps**2) / (2.0 * altitude_for_guidance_m),
        0.0,
    )
    thrust_required_n = float(mass_kg) * (float(standard_gravity_mps2) + required_accel_mps2)
    raw_throttle = float(np.clip(thrust_required_n / max(float(max_thrust_n), 1.0e-6), min_throttle, max_throttle))
    base_throttle = float(np.clip(base_throttle, min_throttle, max_throttle))
    if base_throttle > 0.0:
        guided_target = max(raw_throttle, base_throttle)
        scheduled = base_throttle + gate * (guided_target - base_throttle)
    else:
        scheduled = gate * raw_throttle
    if late_stage_altitude_m > 0.0 and late_stage_extra_throttle > 0.0:
        late_gate = terminal_throttle_residual_gate(
            altitude_m=altitude_m,
            gate_altitude_m=late_stage_altitude_m,
            gate_power=1.0,
        )
        scheduled += float(late_stage_extra_throttle * late_gate)
    return float(np.clip(scheduled, min_throttle, max_throttle))

__all__ = [
    "energy_assist_delta",
    "guidance_brake_throttle",
    "overspeed_brake_assist_delta",
    "overspeed_brake_floor_throttle",
    "overspeed_severity",
    "stopping_distance_floor_throttle",
    "stopping_required_deceleration",
    "terminal_throttle_residual_gate",
    "vertical_specific_energy_excess",
]
