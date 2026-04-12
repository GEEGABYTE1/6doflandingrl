# Phase 2: Hierarchical PPO Training

## Why Hierarchical RL?

The original Phase 2 that was proposed used a monolithic PPO agent controlling all three outputs
`[τ, δp, δy]` simultaneously. After extensive training (30M+ steps, IC curriculum
through 6 stages), the policy consistently failed at the full 1000m/-100 m/s scenario. Even changed it to a slightly 
weaker wind model (IC = 4) instead of (IC = 5). The RL agent did learn, but not on all variables (the descent variable vz was 
still an issue).

**Root cause: two fundamentally different control problems in one network.**

| Sub-problem | What it needs | Timescale |
|---|---|---|
| Vertical deceleration (throttle) | altitude, vz, mass | Slow — energy budget over full 10-15s descent |
| Lateral/attitude (TVC) | quaternion, angular rates, position | Fast — continuous disturbance rejection |

When a single MLP trains on both simultaneously, gradients from each problem
interfere. Improving vz tracking degrades attitude control and vice versa.
The network never settles — it's solving two different optimal control problems
with one set of weights and one reward signal that only becomes informative
near episode termination.

This is a known failure mode. Gaudet et al. (2020) avoid it by using a
pre-defined velocity field as the tracking target, effectively hardcoding
the guidance law and only learning the tracking controller. Our approach
instead learns both — but decomposes them.

**What the curriculum showed us:**
At intermediate stages (300-600m), PPO demonstrated it *can* solve the lateral
and attitude problems well:
- Lateral error: **7.2m vs LQR 29.6m** ✅
- TVC chatter RMS: **0.010 vs LQR 9.078** ✅
- Fuel: **4592 vs LQR 5000 kg** ✅

But touchdown vz was -69 m/s — it never learned to decelerate. The lateral
policy was working. The throttle policy was not. So we split them.

---

## Architecture: Two-Layer Hierarchical Controller

```
Full action space [τ, δp, δy]
         ↓
┌──────────────────────┐    ┌──────────────────────────┐
│   Layer 1            │    │   Layer 2                 │
│   Throttle Policy    │    │   TVC Policy              │
│                      │    │                           │
│   Input  (3-dim):    │    │   Input  (19-dim):        │
│     altitude         │    │     full 18-dim obs       │
│     vz               │    │     + τ from Layer 1      │
│     mass             │    │                           │
│                      │    │   Output (2-dim):         │
│   Output (1-dim): τ  │    │     δp, δy                │
└──────────────────────┘    └──────────────────────────┘
         ↓                            ↓
         └──────────── [τ, δp, δy] ───┘
                            ↓
                     dynamics engine
                            ↓
                        next state
```

**Layer 1 — Throttle Policy** is trained first, in isolation, in a simplified
1D environment. The lateral and attitude states are zeroed out. The only
variables that matter for vertical deceleration are altitude, vz, and mass.
This is a 3-dim state, 1-dim action problem — trivially easy for PPO.
Converges in ~500k steps.

**Layer 2 — TVC Policy** is trained second, in the full 6DOF environment,
with the throttle policy frozen. PPO only controls `[δp, δy]`. Since vertical
deceleration is now handled correctly by Layer 1, the reward signal for
the TVC policy is clean and immediate — attitude and lateral errors are
visible from step 1, not just at episode termination.

This is the key insight: by the time we train Layer 2, the environment
looks like the intermediate stages where PPO was already succeeding. The
TVC policy is essentially being asked to do what it already demonstrated
it could do.

---

## Project Structure

```
src/
├── dynamics/
│   └── dynamics.py              ← Phase 1: 6DOF EOM, RK4, ISA, wind (unchanged)
├── controllers/
│   └── lqr.py                   ← Phase 1: gain-scheduled LQR + PD (unchanged)
├── environment/
│   ├── landing_env.py           ← IC curriculum environment (6 stages)
│   ├── throttle_env.py          ← NEW: 1D throttle-only environment
│   └── tvc_env.py               ← NEW: Full 6DOF env with frozen throttle
├── training/
│   ├── train_throttle.py        ← NEW: Layer 1 training
│   └── train_ppo.py             ← Updated: trains Layer 2 (TVC only)
└── evaluation/
    └── evaluate.py              ← Updated: HierarchicalController combining both
```

---

## Run Order

```bash
pip install gymnasium stable-baselines3 torch pandas matplotlib

# Step 1 — Train throttle policy (~500k steps, ~20 min)
python src/training/train_throttle.py

# Step 2 — Train TVC policy with frozen throttle (~5M steps, ~2-4h)
python src/training/train_ppo.py \
    --throttle-model models/throttle_policy.zip \
    --n-envs 8

# Step 3 — Evaluate hierarchical controller vs LQR
python src/evaluation/evaluate.py \
    --throttle-model models/throttle_policy.zip \
    --tvc-model runs/<run_id>/best_model.zip
```

---

## Environment Details

### `throttle_env.py` — Layer 1 Training Environment

| Property | Value |
|---|---|
| Observation space | `Box(3,)` — [alt/1000, vz/150, mass/125000] |
| Action space | `Box(1,)` ∈ [-1,1] → τ ∈ [0.4, 1.0] |
| Physics | Same RK4 engine, lateral/attitude zeroed |
| Reward | `-0.002*(vz-vz_ref)²` + soft landing terminal |
| IC | Full scenario: 1000m, -100 m/s (no curriculum needed — 1D is easy) |

### `tvc_env.py` — Layer 2 Training Environment

| Property | Value |
|---|---|
| Observation space | `Box(19,)` — full 18-dim normalised obs + τ |
| Action space | `Box(2,)` ∈ [-1,1] → [δp, δy] ∈ [±6°] |
| Throttle | Provided by frozen Layer 1 policy at each step |
| Reward | All v4 terms **except** vz tracking (throttle handles that) |
| IC curriculum | Same 6-stage curriculum as Phase 2A |

### Reward Function

**Layer 1 throttle reward** (pure vertical):
```
r = -0.002 * (vz - vz_ref(alt))²          ← track reference
  + terminal: +2000 success, -500 crash, -1000 if |vz|>10 at TD
```

**Layer 2 TVC reward** (attitude + lateral, no vz term):
```
r = +1.0                                    survival
  - 0.08 * speed          (alt < 200m)      near-ground braking signal
  - 0.05 * tilt - 0.003 * tilt²             quadratic tilt penalty
  - 0.5  * |omega|                           angular rate
  - 0.0003 * horiz_err                       lateral position
  - 0.02  * horiz_vel                        lateral velocity
  - 0.0002 * fuel_step                       fuel
  - 5.0  * (Δτ)²                            throttle smoothness
  - 20.0 * (Δδp² + Δδy²)                   TVC smoothness ← chatter fix
  - 0.5  * (|δp| + |δy|)                    TVC magnitude
  + terminal: +2000 success, -500 crash
```

---

## Training Configuration

### Layer 1 — Throttle Policy

| Hyperparameter | Value |
|---|---|
| Algorithm | PPO (SB3) |
| Policy network | MLP [64, 64] + tanh (small — 3-dim input) |
| Timesteps | 500k |
| Parallel envs | 4 |
| γ (discount) | 0.995 |

### Layer 2 — TVC Policy

| Hyperparameter | Value |
|---|---|
| Algorithm | PPO (SB3) |
| Policy network | MLP [256, 256] + tanh |
| Timesteps | 5M+ |
| Parallel envs | 8 (SubprocVecEnv) |
| γ (discount) | 0.995 |
| IC curriculum | 6 stages, advance at 25% success over 100 eps |

---

## Outputs

```
models/
└── throttle_policy.zip          ← trained Layer 1 (frozen during Layer 2 training)

runs/<run_id>/
├── ppo_config.json
├── training_log.csv / .png      ← Figure P6
├── best_model.zip               ← best TVC policy
└── checkpoints/

results/
├── P1b_trajectory_compare.png
├── P2b_states_compare.png
├── P3b_control_comparison.png
└── metrics_table.csv
```

---

## LQR Weaknesses → Hierarchical PPO Fixes

| LQR Problem | Hierarchical PPO Fix |
|---|---|
| Lateral drift (97m error) | TVC policy: dense lateral penalty, learned disturbance rejection |
| TVC chattering ±6° | TVC policy: smoothness penalty `-20·(Δδ)²` |
| Full propellant consumption | Throttle policy: learns minimum-fuel deceleration profile |
| vz=-69 m/s at touchdown (monolithic PPO failure) | Throttle policy: dedicated 1D training, solves it in 500k steps |

---

## Why This Works for the Paper

The decomposition maps onto a well-established control theory concept:
**timescale separation**. In classical GNC (Guidance, Navigation & Control),
guidance (where to go) and control (how to get there) are always designed
separately. The throttle policy is the guidance layer; the TVC policy is
the control layer. We are doing the same thing but learning both from data
rather than deriving them analytically.

Paper framing: *"We propose a hierarchical decomposition of the 6DOF landing
problem into a guidance layer (vertical deceleration via throttle) and a
control layer (attitude and lateral correction via TVC). This decomposition
reflects the natural timescale separation between vertical and lateral
dynamics and enables each sub-policy to converge independently — addressing
the gradient interference that prevents monolithic PPO from solving the
full terminal descent scenario."*

---

## Phase 3 (next)

Monte Carlo: 500 episodes × {LQR, Hierarchical PPO} × wind ∈ {0, 5, 15, 30} m/s.
Produces paper Figures P7 (success rates), P8 (error boxplots), P9 (failure clusters).
Requires hierarchical PPO success rate > 70% on full 1000m scenario before running.