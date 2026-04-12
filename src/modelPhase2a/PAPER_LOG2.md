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
- Lateral error: **7.2m vs LQR 29.6m** 
- TVC chatter RMS: **0.010 vs LQR 9.078** 
- Fuel: **4592 vs LQR 5000 kg** 

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


## LQR Weaknesses → Hierarchical PPO Fixes

| LQR Problem | Hierarchical PPO Fix |
|---|---|
| Lateral drift (97m error) | TVC policy: dense lateral penalty, learned disturbance rejection |
| TVC chattering ±6° | TVC policy: smoothness penalty `-20·(Δδ)²` |
| Full propellant consumption | Throttle policy: learns minimum-fuel deceleration profile |
| vz=-69 m/s at touchdown (monolithic PPO failure) | Throttle policy: dedicated 1D training, solves it in 500k steps |

---

## Phase 3 (next)

Monte Carlo: 500 episodes × {LQR, Hierarchical PPO} × wind ∈ {0, 5, 15, 30} m/s.
Produces paper Figures P7 (success rates), P8 (error boxplots), P9 (failure clusters).
Requires hierarchical PPO success rate > 70% on full 1000m scenario before running.