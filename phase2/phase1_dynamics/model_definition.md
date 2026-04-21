# Phase 1 Model Definition

## State, Control, Outputs

The simulator state is

`x = [r_I, v_I, q_BI, omega_B, m]`

with expanded vector

`[x_I, y_I, z_I, vx_I, vy_I, vz_I, qw, qx, qy, qz, p_B, q_B, r_B, mass]`.

The control is

`u = [throttle, pitch_gimbal, yaw_gimbal]`,

where throttle is clipped to `[0, 1]`, pitch gimbal tilts thrust toward body `+x_B`, yaw gimbal tilts thrust toward body `+y_B`, and both gimbal angles are clipped to the engine limit.

Reported outputs include trajectory state histories, throttle and gimbal histories, dynamic pressure, touchdown time, vertical and horizontal touchdown velocity, tilt, angular-rate norm, landing position error, fuel use, saturation fraction, and failure classification.

## Frames and Quaternion Convention

The inertial frame `I` is locally flat with `+z_I` upward. Gravity is `[0, 0, -g]`. The body frame `B` has `+z_B` along the vehicle longitudinal axis from engine toward nose. Upright hover aligns `+z_B` with `+z_I`.

Quaternions are scalar-first, `q_BI = [qw, qx, qy, qz]`, and map body-frame vectors to inertial-frame vectors. Body angular velocity `omega_B = [p, q, r]` is expressed in body axes. Quaternion propagation uses

`q_dot = 0.5 q_BI otimes [0, omega_B]`.

The RK4 integrator renormalizes the quaternion after every step.

## Dynamics

Translational dynamics:

`r_dot_I = v_I`

`v_dot_I = (F_thrust,I + F_aero,I + [0, 0, -m g]) / m`.

Rotational dynamics:

`I_B omega_dot_B = M_B - omega_B x (I_B omega_B)`.

Mass depletion:

`m_dot = -T / (Isp g0)` when `m > m_dry`; otherwise thrust and mass flow are set to zero.

## Propulsion and TVC Mapping

Nominal thrust points along body `+z_B`. For pitch gimbal `delta_p` and yaw gimbal `delta_y`, the body thrust direction is normalized from

`[sin(delta_p), sin(delta_y), cos(delta_p) cos(delta_y)]`.

The engine position relative to the center of mass is `[0, 0, -L]`, so the thrust moment is

`M_TVC,B = r_engine,B x F_thrust,B`.

For hover-trim LQR design, the small-angle control model uses

`ax ~= g(theta + delta_p)`

`ay ~= g(delta_y - phi)`

`az ~= (Tmax / m) delta_throttle`

`pdot ~= L T_hover delta_y / Ixx`

`qdot ~= -L T_hover delta_p / Iyy`.

The vertical reference uses an energy-based braking profile:

`v_z,ref = -sqrt(v_touch^2 + 2 a_brake h)`,

clipped by the configured maximum descent rate. This gives a fast coast/brake descent rather than a hover-like glide slope. The default values are `v_touch = 0.6 m/s` and `a_brake = 2.5 m/s^2`.

## Atmosphere, Aerodynamics, Disturbances

Atmosphere uses a simplified ISA troposphere to 11 km with exponential continuation above that altitude.

Aerodynamic drag is computed from air-relative inertial velocity:

`F_D,I = -0.5 rho Cd A ||v_rel|| v_rel`.

Aerodynamic moments include a center-of-pressure offset and dynamic-pressure-scaled rotational damping:

`M_aero,B = r_cp,B x F_D,B - C_rot q_dyn A L_ref omega_B`.

Disturbances are separable:

- Wind: steady inertial wind plus optional sinusoidal gust.
- Thrust misalignment: fixed pitch/yaw gimbal bias.
- Sensor noise: Gaussian state perturbations reserved for future estimated-state controllers.

## Success Criteria and Failure Taxonomy

Default Phase 1 success thresholds:

- `|vertical touchdown velocity| <= 2.0 m/s`
- `horizontal touchdown velocity <= 1.0 m/s`
- `landing position error <= 2.0 m`
- `touchdown tilt <= 10 deg`
- `touchdown angular-rate norm <= 0.5 rad/s`
- no fuel exhaustion, no numerical divergence, and no persistent saturation.

Failure modes are classified as hard touchdown, horizontal speed, lateral miss, tilt-over, excess angular rate, fuel exhaustion, divergence, saturation-driven instability, oscillatory descent, or no touchdown.

## Yaw Underactuation

A single centered TVC engine cannot create moment about body `+z_B`. Yaw angle and yaw rate are therefore measured, reported, and included in the public 12-state LQR error vector, but excluded from the Riccati solve with zero feedback gains. Yaw-related behavior is not hidden in evaluation tables; it is reported through attitude, angular-rate, and failure metrics.
