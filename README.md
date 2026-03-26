# Reinforcement Learning for 6DOF Rocket Powered Descent and Landing Under Uncertainty
## Living Research Log — ArXiv Paper Draft

> **Instructions:** This document is updated after every build phase.
> Sections: Design Decisions · Equations · Plots · Results · References

---

## Tentative Title

**"Comparative Analysis of Classical and Reinforcement Learning Controllers for
6-DOF Rocket Powered Descent and Landing Under Stochastic Disturbances"**

*Candidate shorter title:* "RL vs. LQR for Autonomous Rocket Vertical Landing"

---

## Authors / Affiliation
- TBD

---

## Abstract (Draft — update after results)

We present a full 6-degree-of-freedom (6DOF) rigid-body simulation environment
for rocket powered descent and landing, incorporating variable mass dynamics,
aerodynamic forces and moments, thrust vector control (TVC), sensor noise, and
stochastic disturbances including wind shear and thrust misalignment. We
implement a Linear Quadratic Regulator (LQR) as a classical baseline and train a
Proximal Policy Optimization (PPO) agent using the Stable Baselines 3 library.
Monte Carlo rollouts under varied initial conditions and disturbance magnitudes
are used to characterize success rates, landing precision, and failure modes.
Results demonstrate [TBD after experiments].

---

## 1. Introduction

### Motivation
- SpaceX Falcon 9 / Starship booster catch: vertical propulsive landing is the
  key enabling technology for reusable launch vehicles.
- Classical GNC (PID/LQR) requires careful linearization and gain scheduling;
  brittle under large disturbances.
- RL offers a data-driven policy that can adapt to nonlinear regimes, but
  stability guarantees are weaker.

### Contributions (planned)
1. Open-source 6DOF simulation environment with full aero (drag + pitch/yaw moments).
2. LQR baseline with gain-scheduled linearization about descent trajectory.
3. PPO agent trained with shaped reward on soft-landing objective.
4. Monte Carlo comparison: success rate, landing error, fuel use, failure modes.
5. Failure mode analysis under parametric variation (wind, thrust misalignment,
   sensor noise).

---

## 2. System Model

### 2.1 Coordinate Frames

| Frame | Symbol | Description |
|-------|--------|-------------|
| Inertial (ENU) | `I` | Earth-fixed, origin at landing pad |
| Body | `B` | Fixed to rocket CoM, z-axis along thrust axis |
| Wind | `W` | Aligned with velocity vector |

**Design decision:** Use East-North-Up (ENU) inertial frame. Positive z is up.
Rocket body z-axis points upward (thrust direction). This is consistent with
aerospace convention for vertical landing problems.

---

### 2.2 State Vector

Full 13-state rigid-body representation:

```
x = [r_x, r_y, r_z,        # position (m) in inertial frame
     v_x, v_y, v_z,        # velocity (m/s) in inertial frame
     q_w, q_x, q_y, q_z,  # unit quaternion (body->inertial)
     ω_x, ω_y, ω_z]        # angular velocity (rad/s) in body frame
```

**Design decision:** Quaternion attitude representation (vs. Euler angles) to
avoid gimbal lock and ensure smooth dynamics integration. Euler angles recovered
for output/plotting only.

---

### 2.3 Translational Dynamics

Newton's second law in the inertial frame:

```
m(t) * r̈ = F_thrust + F_aero + F_gravity
```

Expanded:

```
r̈ = (1/m(t)) * [R_BI * f_thrust_body + f_aero_inertial] + g_vec
```

Where:
- `m(t)` = current mass (kg), decreasing with fuel burn
- `R_BI` = rotation matrix from body to inertial (derived from quaternion q)
- `f_thrust_body` = thrust force vector in body frame
- `f_aero_inertial` = aerodynamic force in inertial frame
- `g_vec = [0, 0, -g]` with `g = 9.80665 m/s²`

**Quaternion to rotation matrix:**
```
R_BI = I + 2*q_w*[q_vec]× + 2*[q_vec]×²
```
where `[q_vec]×` is the skew-symmetric matrix of the vector part `[q_x, q_y, q_z]`.

---

### 2.4 Variable Mass / Fuel Burn

Tsiolkovsky rocket equation for mass flow:

```
ṁ = -T / (Isp * g0)
```

Where:
- `T` = thrust magnitude (N)
- `Isp` = specific impulse (s)  — **design decision: Isp = 311 s** (Merlin-class LOX/RP-1)
- `g0 = 9.80665 m/s²`

Mass bounds:
- `m_dry` = 22,000 kg  (Falcon 9 first stage dry mass approximation)
- `m_prop_max` = 400,000 kg  (full tank; we start with a partial load for landing burn)
- At landing burn initiation: `m0 ≈ 25,000–30,000 kg` (dry + reserve fuel)

**Design decision:** Landing burn scenario only (not full ascent). Initial
altitude ~1500 m, initial downward velocity ~-80 m/s. This matches the
terminal descent phase after boostback and reentry burns.

---

### 2.5 Rotational Dynamics

Euler's equation in body frame:

```
I_body * ω̇ = τ_total - ω × (I_body * ω)
```

Where:
- `I_body` = inertia tensor (diagonal approximation, kg·m²)
- `τ_total` = total torque in body frame (from TVC + aero moments)
- `ω × (I_body * ω)` = gyroscopic term

**Inertia tensor (design decision — cylinder approximation):**
```
I_xx = I_yy = (1/12) * m * (3*r² + h²)   # transverse
I_zz = (1/2) * m * r²                     # axial (spin axis)
```
With `r = 1.85 m` (radius), `h = 42.6 m` (height). Updated each timestep as
mass decreases (assumes uniform mass distribution along z-axis).

**Quaternion kinematics:**
```
q̇ = (1/2) * q ⊗ [0, ω_x, ω_y, ω_z]
```
Quaternion re-normalized every integration step to prevent drift.

---

### 2.6 Thrust Vector Control (TVC)

Thrust is gimballed; the nozzle deflects by angles `(δ_y, δ_z)` (gimbal angles
about body y and z axes respectively).

Thrust force in body frame:
```
f_thrust_body = T * [sin(δ_z), -sin(δ_y), cos(δ_y)*cos(δ_z)]ᵀ
           ≈ T * [δ_z, -δ_y, 1]ᵀ   (small-angle linearization)
```

TVC torque in body frame (moment arm `l_tvc` from nozzle to CoM):
```
τ_TVC = l_tvc × f_thrust_body
```

**Design decisions:**
- `T_min = 0.4 * T_max` (throttle range: 40–100% of max thrust)
- `T_max = 934,000 N` (one Merlin engine at full thrust)
- `δ_max = ±7°` (gimbal angle limit)
- `δ̇_max = 15°/s` (gimbal rate limit — actuator dynamics)
- `l_tvc = -h/2` (nozzle at bottom of vehicle; CoM assumed at center)

---

### 2.7 Aerodynamic Model

Forces and moments computed in the wind/velocity frame, then rotated to inertial.

**Dynamic pressure:**
```
q_dyn = 0.5 * ρ(h) * |v_rel|²
```
Where `v_rel = v_inertial - v_wind` and ρ(h) is ISA atmospheric density.

**ISA density (troposphere, h < 11,000 m):**
```
ρ(h) = ρ0 * (1 - L*h/T0)^(g0*M / (R*L))
```
With `ρ0=1.225 kg/m³`, `L=0.0065 K/m`, `T0=288.15 K`, `M=0.029 kg/mol`,
`R=8.314 J/(mol·K)`.

**Aerodynamic force (drag-dominant with normal force):**
```
F_drag = -q_dyn * S_ref * CD(α) * v̂_rel
F_normal = q_dyn * S_ref * CN(α) * n̂
```

**Aerodynamic pitching moment:**
```
M_pitch = q_dyn * S_ref * D_ref * Cm(α)
```

**Angle of attack:**
```
α = arccos(v̂_rel · ẑ_body)   (angle between velocity and body z-axis)
```

**Aerodynamic coefficients (design decision — polynomial fit, small-α regime):**
```
CD(α) = CD0 + k * α²        # CD0=0.3, k=0.5
CN(α) = CN_α * α            # CN_α=0.1 /rad (slender body approx)
Cm(α) = Cm_α * α            # Cm_α=-0.05 /rad  (static stability margin)
```

**Design decision:** Negative `Cm_α` = statically stable configuration
(aerodynamic center aft of CoM during descent). This is realistic for a
fin-stabilized or heavy-nose-down descent.

Reference area: `S_ref = π * r² = 10.75 m²`
Reference length: `D_ref = 2*r = 3.7 m`

---

### 2.8 Disturbances

**Wind model (Dryden turbulence approximation — discrete):**
```
v_wind(t) = v_wind_mean + v_wind_gust(t)
v_wind_gust update: first-order Gauss-Markov
  dv_gust = -(1/τ_wind)*v_gust*dt + σ_wind * sqrt(2/τ_wind) * dW
```
Parameters: `τ_wind = 5.0 s`, `σ_wind = 3.0 m/s` (moderate turbulence)

**Thrust misalignment:**
```
Δf_thrust = T * [ε_x, ε_y, 0]ᵀ  where ε ~ N(0, σ_thrust²)
```
`σ_thrust = 0.002` (0.2% of thrust magnitude per axis)

**Sensor noise (added to observations):**
```
r_meas = r_true + N(0, σ_pos²)       σ_pos = 0.1 m
v_meas = v_true + N(0, σ_vel²)       σ_vel = 0.05 m/s
q_meas = q_true + N(0, σ_att²)       σ_att = 0.001 rad (equiv.)
ω_meas = ω_true + N(0, σ_gyro²)      σ_gyro = 0.01 rad/s
```

---

## 3. Integration Scheme

**Design decision:** 4th-order Runge-Kutta (RK4) with fixed timestep `dt = 0.02 s` (50 Hz).

This matches typical flight computer update rates and provides sufficient
accuracy for the dynamics time scales involved (aero forces change on ~0.1 s
timescales).

State derivative function `f(t, x, u)` returns `[ṙ, v̇, q̇, ω̇, ṁ]`.

---

## 4. Classical Controller — LQR

### 4.1 Linearization

Linearize about a nominal hover/descent trajectory point:
```
x* = [0,0,h*, 0,0,vz*, 1,0,0,0, 0,0,0]  (upright, descending)
u* = [T*, 0, 0]                           (throttle for hover + descent rate)
```

The full 13-state system is linearized numerically (finite differences on `f`):
```
A = ∂f/∂x |_(x*,u*)
B = ∂f/∂u |_(x*,u*)
```

**Design decision:** Separate horizontal (x,y) and vertical (z) channels for
LQR design, exploiting approximate decoupling at small angles. Full coupled LQR
also implemented for comparison.

### 4.2 LQR Cost Matrices

```
Q = diag([q_r, q_r, q_z,          # position weights
          q_v, q_v, q_vz,         # velocity weights
          q_q, q_q, q_q, q_q,     # attitude weights
          q_w, q_w, q_w])         # angular rate weights

R = diag([r_T, r_δy, r_δz])       # control effort weights
```

**Tuned values (design decision — to be updated after testing):**
```
q_r=10, q_z=10, q_v=1, q_vz=5, q_q=100, q_w=10
r_T=0.01, r_δy=1.0, r_δz=1.0
```

Solved via `scipy.linalg.solve_continuous_are` (CARE).

---

## 5. RL Controller — PPO

### 5.1 Observation Space
```
obs = [r_x, r_y, r_z,             # position (normalized by h0)
       v_x, v_y, v_z,             # velocity (normalized by v_max)
       q_w, q_x, q_y, q_z,        # quaternion (unit, no norm needed)
       ω_x, ω_y, ω_z,             # angular rate (normalized by ω_max)
       m_frac,                    # fuel fraction remaining [0,1]
       v_wind_x, v_wind_y]        # wind estimate (if available)
```
Total: 18-dimensional continuous observation.

### 5.2 Action Space
```
act = [δ_T, δ_y, δ_z]   ∈ [-1, 1]³  (normalized)
```
Mapped to physical controls:
```
T = T_min + (δ_T + 1)/2 * (T_max - T_min)
gimbal_y = δ_y * δ_max
gimbal_z = δ_z * δ_max
```

### 5.3 Reward Function

```
r(t) = r_alive + r_upright + r_velocity + r_fuel + r_terminal
```

Components:
```
r_alive    = +0.1                              # per-step survival bonus
r_upright  = +0.5 * cos(θ_tilt)              # tilt penalty (0 when upright)
r_velocity = -0.01 * |v|²                    # penalize high speed
r_fuel     = -0.001 * (T/T_max)              # fuel efficiency
r_terminal:
  SUCCESS  = +1000 - 10*|r_xy| - 100*|v_z|  # soft landing bonus
  CRASH    = -500                             # hard landing / tip-over
  TIMEOUT  = -200                            # episode too long
```

**Design decision:** Dense reward shaping to guide early training. Terminal bonus
heavily weighted toward landing precision and soft touchdown. Tilt limit for
success: `θ_tilt < 15°` at touchdown. Velocity limit: `|v_z| < 2 m/s`,
`|v_xy| < 1 m/s`.

### 5.4 PPO Hyperparameters (initial — to be tuned)

```
policy:           MlpPolicy
learning_rate:    3e-4
n_steps:          2048
batch_size:       64
n_epochs:         10
gamma:            0.99
gae_lambda:       0.95
clip_range:       0.2
ent_coef:         0.01
vf_coef:          0.5
max_grad_norm:    0.5
network:          [256, 256] (actor + critic shared trunk)
total_timesteps:  5,000,000
```

---

## 6. Monte Carlo Evaluation Protocol

- N = 500 episodes per controller
- Initial conditions sampled uniformly:
  ```
  h0     ~ U(800, 1500) m
  vz0    ~ U(-100, -60) m/s
  vxy0   ~ U(-5, 5) m/s each axis
  θ0     ~ U(-5°, 5°) (initial tilt)
  ω0     ~ U(-0.1, 0.1) rad/s each axis
  ```
- Wind: σ_wind ∈ {0, 3, 6, 9} m/s (four disturbance levels)
- Thrust misalignment: included in all runs

**Metrics:**
- Landing success rate (%)
- Landing position error |r_xy| (m)
- Landing velocity |v_z| at touchdown (m/s)
- Tilt angle at touchdown (°)
- Fuel consumed (kg)
- Episode length (s)

---

## 7. Plots Registry

All plots saved to `plots/` directory. Update this table as plots are generated.

| ID | Filename | Description | Phase | Status |
|----|----------|-------------|-------|--------|
| P1 | `trajectory_3d.png` | 3D trajectory (position vs time) | Phase 1 | TODO |
| P2 | `state_history.png` | Full state history (12 panels) | Phase 1 | TODO |
| P3 | `lqr_landing_sequence.png` | LQR landing top-view + altitude | Phase 1 | TODO |
| P4 | `reward_curve.png` | PPO training reward vs timesteps | Phase 2 | TODO |
| P5 | `success_rate_vs_wind.png` | Success rate (LQR vs PPO vs wind σ) | Phase 2 | TODO |
| P6 | `landing_error_scatter.png` | Landing position scatter (500 runs) | Phase 2 | TODO |
| P7 | `fuel_comparison.png` | Fuel use CDF (LQR vs PPO) | Phase 2 | TODO |
| P8 | `failure_mode_analysis.png` | Failure classification breakdown | Phase 2 | TODO |
| P9 | `tilt_velocity_envelope.png` | Tilt vs touchdown velocity scatter | Phase 2 | TODO |

---

## 8. Code Structure

```
rocket_landing_sim/
├── dynamics/
│   ├── rigid_body.py       # 6DOF EOM, RK4 integrator
│   ├── aerodynamics.py     # Drag, normal force, pitching moment
│   ├── atmosphere.py       # ISA density model
│   └── disturbances.py     # Wind, thrust misalignment, sensor noise
├── controllers/
│   ├── lqr.py              # Linearization + LQR gain solve
│   └── baseline_pd.py      # Simple PD (sanity check, optional)
├── rl/
│   ├── env.py              # Gymnasium environment wrapper
│   ├── train.py            # PPO training script (SB3)
│   └── evaluate.py         # Monte Carlo rollout runner
├── analysis/
│   ├── monte_carlo.py      # Batch evaluation + stats
│   └── plots.py            # All figure generation
├── tests/
│   └── test_dynamics.py    # Unit tests (energy conservation, etc.)
├── paper_log.md            # This file
└── README.md
```

---

## 9. Design Decisions Log

| # | Decision | Rationale | Alternative considered |
|---|----------|-----------|----------------------|
| D1 | Quaternion attitude rep. | No gimbal lock, smooth interpolation | Euler angles (simpler but singular) |
| D2 | RK4, dt=0.02s | Accuracy vs speed; matches flight computer rate | RK45 adaptive (slower, overkill) |
| D3 | ENU inertial frame | Standard aerospace convention | NED (common avionics, less intuitive for "up") |
| D4 | Terminal descent phase only | Tractable problem scope; directly relevant to landing | Full ascent+descent (too long for RL episode) |
| D5 | Isp=311s, Merlin-class engine | Realistic, well-documented parameters | Generic hypothetical engine |
| D6 | Polynomial aero coefficients | Simple, differentiable, tunable | Lookup tables (more accurate, harder to implement) |
| D7 | Negative Cm_α (static stability) | Realistic for descent; simplifies control | Unstable vehicle (harder, more interesting for future work) |
| D8 | LQR as classical baseline | Optimal linear control; cleaner comparison than PID | PID cascade (more common in industry, less principled) |
| D9 | PPO as RL algorithm | On-policy, stable, well-tuned in SB3 | DDPG/SAC (off-policy, potentially more sample-efficient) |
| D10 | Dense reward shaping | Essential for PPO convergence on long horizon | Sparse reward (elegant but very hard to learn) |

---

## 10. References (to be completed)

1. Acikmese, B. & Ploen, S.R. (2007). Convex programming approach to powered descent guidance for Mars landing. *AIAA Journal of Guidance, Control, and Dynamics*.
2. Malyuta, D. et al. (2021). Advances in trajectory optimization for space vehicle control. *Annual Reviews in Control*.
3. Gaudet, B., Linares, R. & Furfaro, R. (2020). Deep reinforcement learning for six degree-of-freedom planetary landing. *Advances in Space Research*.
4. Schulman, J. et al. (2017). Proximal Policy Optimization Algorithms. *arXiv:1707.06347*.
5. Raffin, A. et al. (2021). Stable-Baselines3: Reliable RL Implementations. *JMLR*.
6. Stevens, B.L., Lewis, F.L. & Johnson, E.N. (2015). *Aircraft Control and Simulation*. Wiley.

---

*Last updated: Phase 0 — Pre-build design log complete*
*Next update: After Phase 1 (dynamics engine + LQR) implementation*