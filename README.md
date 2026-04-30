# AI-Enhanced 6DOF Rocket Landing Simulator

This repository is a research-grade rocket/vehicle descent benchmark for classical control and reinforcement learning. It builds a 6DOF propulsive landing simulator with variable mass, quaternion attitude, thrust vector control, simplified atmosphere/aerodynamics, modular disturbances, and reproducible controller comparisons.

The project is intentionally a benchmark, not an exact reconstruction of any real launch vehicle, mission, guidance system, or flight-software stack. The main technical result is that a classical gain-scheduled LQR baseline remains strongest overall, flat PPO and pure hierarchical PPO expose useful failure modes, and a hybrid residual-PPO controller with structured terminal-braking priors becomes the first learned-controller path in this repo to satisfy the locked nominal touchdown criteria.

## Overview

The research asks whether reinforcement learning can become credible for a 6DOF rocket/vehicle descent benchmark when it is evaluated against a serious classical controller instead of an easy toy environment. The project’s answer is deliberately nuanced: plain PPO is not enough, hierarchy helps only partially, and the successful learned-controller path requires classical structure plus targeted terminal-braking priors.

The work contributes a reproducible propulsive descent simulator, a locked gain-scheduled LQR baseline, two PPO-only baselines, a hybrid residual-PPO controller, and a Monte Carlo robustness study. The simulator uses a 14-state rigid-body model with inertial position and velocity, scalar-first quaternion attitude, body angular rates, and variable mass. The modeled world includes gravity, thrust, thrust-vector control, simplified atmosphere and aerodynamics, wind/gust disturbances, and thrust misalignment.

All controllers are judged with the same touchdown criteria: vertical speed, horizontal speed, landing error, tilt, angular-rate norm, fuel exhaustion/divergence, and actuator saturation. This shared metric layer is important because the paper treats failed controllers as evidence, not clutter. Flat PPO reveals that a monolithic policy can learn pieces of the task while still crashing vertically. Hierarchical PPO shows that separating throttle and TVC improves vertical behavior but can lose coupled lateral-attitude coordination. Hybrid residual PPO then uses a learned throttle prior, LQR TVC prior, and bounded residual policy to recover lateral and attitude stability.

The central technical bottleneck is terminal vertical energy management. Several reward-only and residual-throttle attempts did not close the hard-touchdown gap. The final successful nominal controller adds structured prior-side braking, especially a stopping-distance floor that raises throttle when the remaining altitude and downward speed imply insufficient braking authority. In the 500-episode-per-level Monte Carlo study, this hybrid controller becomes the only RL-based family with substantial robustness, while its remaining failures concentrate mostly under the strongest disturbance setting.

### Controller Performance

**Phase 1 LQR:** the classical baseline. It is a gain-scheduled hover-trim LQR running in the full 6DOF simulator, and it succeeds cleanly in the nominal case.

![Phase 1 LQR landing](/phase2/outputs/phase4_blog/landing_lqr.gif)

**Phase 2A flat PPO:** the monolithic RL baseline. One PPO policy commands throttle and TVC from the full normalized observation. It reaches the ground, but lands far too fast and misses the touchdown criteria.

![Phase 2A flat PPO landing](/phase2/outputs/phase4_blog/landing_flat_ppo.gif)

**Phase 2B hierarchical PPO:** the pure hierarchy baseline. A throttle PPO handles vertical motion while a TVC PPO handles lateral/attitude control. The split helps vertical speed relative to flat PPO, but the composed controller still loses lateral and tilt quality.

![Phase 2B hierarchical PPO landing](/phase2/outputs/phase4_blog/landing_hierarchical_ppo.gif)

**Phase 2C hybrid PPO:** the promoted learned controller. It uses a learned throttle prior, LQR TVC prior, bounded residual PPO, and structured terminal braking. This is the first learned-controller path in the repo that satisfies the locked nominal touchdown criteria.

![Phase 2C hybrid PPO landing](/phase2/outputs/phase4_blog/landing_hybrid_ppo.gif)

The generated visuals also include the Monte Carlo success-rate animation and the final hybrid comparison panel:

![Monte Carlo success rates](/phase2/outputs/phase4_blog/mc_success_rates.gif)

![Phase 2C hybrid comparison](/phase2/outputs/phase4_blog/phase2c_stopping_floor_panel.png)

## Methodology Summary

The benchmark is organized as five connected layers:

- **Phase 1, dynamics and classical baseline:** full 6DOF simulator, RK4 integration, variable mass, scalar-first quaternion attitude, TVC propulsion, simplified ISA-style atmosphere, drag/aero moments, wind/gust/thrust-misalignment disturbances, and a gain-scheduled hover-trim LQR controller.
- **Phase 2A, flat PPO:** one PPO policy observes the normalized full landing state and commands throttle plus pitch/yaw gimbal. It is the required monolithic RL baseline.
- **Phase 2B, hierarchical PPO:** a reduced-order throttle policy handles vertical energy management, while a full-6DOF TVC policy handles attitude/lateral correction with the frozen throttle policy in the loop.
- **Phase 2C, hybrid residual PPO:** a frozen learned throttle prior is combined with LQR TVC commands and a residual PPO policy. Later passes add structured terminal braking priors, including the promoted stopping-distance floor.
- **Phase 3, evaluation:** all promoted controllers are evaluated in the same locked full-6DOF simulator under composite wind/gust/thrust-misalignment disturbance levels.

Success criteria are shared across phases: touchdown vertical speed <= `2.0 m/s`, horizontal speed <= `1.0 m/s`, lateral error <= `2.0 m`, tilt <= `10 deg`, angular-rate norm <= `0.5 rad/s`, no fuel exhaustion/divergence, and acceptable actuator saturation. Yaw is measured, but excluded from pass/fail because the current single centered TVC model has no body-z yaw torque authority.

## Key Results

Nominal saved-controller results on seed `7`:

| Controller | Success | Vertical touchdown | Horizontal touchdown | Landing error | Tilt | Source |
|---|---:|---:|---:|---:|---:|---|
| Phase 1 gain-scheduled LQR | yes | `-0.65 m/s` | `0.04 m/s` | `0.047 m` | `0.11 deg` | `outputs/phase1_evaluation/runs/scenario_table__nominal/metrics.json` |
| Phase 2A flat PPO | no | `-26.12 m/s` | `1.61 m/s` | `12.19 m` | `4.55 deg` | `outputs/phase2_rl/eval_go_no_go_baseline_v2_default_seed7_50k/metrics.json` |
| Phase 2B hierarchical PPO | no | `-10.12 m/s` | `8.26 m/s` | `12.53 m` | `21.28 deg` | `outputs/phase2b_hierarchical_rl/eval_nominal_seed7_40k_50k_flare_tracking_v1/metrics.json` |
| Phase 2C hybrid PPO + stopping floor | yes | `-1.68 m/s` | `0.04 m/s` | `0.048 m` | `0.07 deg` | `outputs/phase2c_hybrid_rl/eval_nominal_seed7_50k_stopping_floor_v1/metrics.json` |

Full Phase 3 Monte Carlo result, `500` episodes per disturbance level and four composite levels (`2000` episodes per controller):

| Controller | Success rate | Mean landing error | Mean vertical touchdown | Mean tilt |
|---|---:|---:|---:|---:|
| Phase 1 LQR | `100.0%` | `0.045 m` | `-0.64 m/s` | `0.09 deg` |
| Phase 2A flat PPO | `0.0%` | `14.11 m` | `-28.81 m/s` | `25.26 deg` |
| Phase 2B hierarchical PPO | `0.0%` | `43.20 m` | `-17.07 m/s` | `28.40 deg` |
| Phase 2C hybrid PPO | `72.9%` | `0.041 m` | `-2.01 m/s` | `0.07 deg` |

The Phase 3 tables live in `outputs/phase3_evaluation/mc_seed7_500ep_composite_levels_v1/`. The headline interpretation is that hybridization closes the broad lateral/attitude instability seen in PPO-only controllers; the remaining failures concentrate mostly in hard touchdown under the strongest disturbance level.

## Repository Structure

The main entry point for the current project is this directory:

```text
/Users/jaivalpatel/Desktop/6dof/phase2
```

Important files and directories:

- `README.md` - this overview and reproduction guide.
- `agents.md` - working instructions and scope boundaries for coding agents.
- `design_decisions.md` - chronological record of modeling, control, reward, and evaluation decisions.
- `paper_log.md` - experiment log with commands, artifacts, and interpretations.
- `plots_manifest.md` - figure provenance, generator commands, captions, and source artifacts.
- `phase1_dynamics/` - simulator core, quaternion utilities, RK4 integration, propulsion/TVC, atmosphere, aerodynamics, disturbances, scenarios, metrics, LQR controller, validation, and Phase 1 plot scripts.
- `phase2_rl/` - flat PPO Gymnasium environment, observation normalization, reward decomposition, curriculum, PPO training/evaluation scripts, config profiles, and comparison plots.
- `phase2b_hierarchical_rl/` - reduced-order throttle environment, TVC environment, frozen policy composition, hierarchical controller, training/evaluation scripts, coordination features, and hierarchy plots.
- `phase2c_hybrid_rl/` - hybrid residual environment/controller, terminal-braking priors, residual PPO training/evaluation, Phase 2C comparisons, and prior-search utilities.
- `phase3_evaluation/` - Monte Carlo controller adapters, metrics aggregation, controller comparison, success-rate plots, dispersion plots, and failure-mode plots.
- `phase4_blog/` - static research article and reproducible article-asset generator.
- `tests/` - smoke and regression tests for Phase 1 physics, Phase 2A, Phase 2B, Phase 2C, and Phase 3 evaluation.
- `outputs/` - saved trajectories, metrics, trained checkpoints, plots, comparison tables, and blog assets.

## Main Entry Points

Common scripts are run from `/Users/jaivalpatel/Desktop/6dof/phase2`:

```bash
# Phase 1 validation and classical baseline artifacts
python phase1_dynamics/validate_simulator.py --output-dir outputs/phase1_validation
python phase1_dynamics/evaluate_phase1.py --output-dir outputs/phase1_evaluation
python phase1_dynamics/plot_phase1_suite.py \
  --input-dir outputs/phase1_evaluation \
  --validation-dir outputs/phase1_validation \
  --output-dir outputs/phase1_figures

# Phase 2A flat PPO
python phase2_rl/check_env.py --seed 7 --steps 5
python phase2_rl/train_ppo.py --seed 7 --total-timesteps 50000 \
  --reward-profile baseline_v2 \
  --ppo-profile baseline_default \
  --output-dir outputs/phase2_rl/go_no_go_baseline_v2_default_seed7_50k
python phase2_rl/evaluate_ppo.py --seed 7 \
  --model outputs/phase2_rl/go_no_go_baseline_v2_default_seed7_50k/model.zip \
  --output-dir outputs/phase2_rl/eval_go_no_go_baseline_v2_default_seed7_50k

# Phase 2B hierarchical PPO
python phase2b_hierarchical_rl/train_throttle.py --seed 7 --total-timesteps 40000 \
  --reward-profile flare_tracking_v1 \
  --ppo-profile baseline_default \
  --output-dir outputs/phase2b_hierarchical_rl/throttle_seed7_40k_flare_tracking_v1
python phase2b_hierarchical_rl/train_tvc.py --seed 7 --total-timesteps 50000 \
  --throttle-model outputs/phase2b_hierarchical_rl/throttle_seed7_40k_flare_tracking_v1/model.zip \
  --reward-profile baseline_v1 \
  --ppo-profile baseline_default \
  --output-dir outputs/phase2b_hierarchical_rl/tvc_seed7_50k_flare_tracking_v1
python phase2b_hierarchical_rl/eval_hierarchical.py --seed 7 --scenario nominal \
  --throttle-model outputs/phase2b_hierarchical_rl/throttle_seed7_40k_flare_tracking_v1/model.zip \
  --tvc-model outputs/phase2b_hierarchical_rl/tvc_seed7_50k_flare_tracking_v1/model.zip \
  --output-dir outputs/phase2b_hierarchical_rl/eval_nominal_seed7_40k_50k_flare_tracking_v1

# Phase 2C promoted hybrid controller
python phase2c_hybrid_rl/eval_hybrid.py --seed 7 --scenario nominal \
  --throttle-model outputs/phase2b_hierarchical_rl/throttle_seed7_40k_touchdown_gate_commit_v1/model.zip \
  --residual-model outputs/phase2c_hybrid_rl/residual_seed7_50k_stopping_floor_v1/model.zip \
  --output-dir outputs/phase2c_hybrid_rl/eval_nominal_seed7_50k_stopping_floor_v1

# Phase 3 Monte Carlo and plots
python phase3_evaluation/monte_carlo.py --seed 7 --episodes-per-level 500 \
  --output-dir outputs/phase3_evaluation/mc_seed7_500ep_composite_levels_v1
python phase3_evaluation/plot_success_rates.py \
  --input-dir outputs/phase3_evaluation/mc_seed7_500ep_composite_levels_v1 \
  --output-dir outputs/phase3_evaluation/mc_seed7_500ep_composite_levels_v1/figures
python phase3_evaluation/plot_dispersion.py \
  --input-dir outputs/phase3_evaluation/mc_seed7_500ep_composite_levels_v1 \
  --output-dir outputs/phase3_evaluation/mc_seed7_500ep_composite_levels_v1/figures
python phase3_evaluation/plot_failure_modes.py \
  --input-dir outputs/phase3_evaluation/mc_seed7_500ep_composite_levels_v1 \
  --output-dir outputs/phase3_evaluation/mc_seed7_500ep_composite_levels_v1/figures
```

For Matplotlib in headless environments, prefix plotting commands with:

```bash
MPLCONFIGDIR=outputs/.cache/matplotlib
```

## Reproducing Results

1. Create and activate a Python environment from the parent requirements file:

```bash
cd /Users/jaivalpatel/Desktop/6dof/phase2
python -m venv .venv
. .venv/bin/activate
pip install -r ../requirements.txt
```

2. Run the regression tests:

```bash
python -m compileall phase1_dynamics phase2_rl phase2b_hierarchical_rl phase2c_hybrid_rl phase3_evaluation tests
python -m pytest tests
```

3. Rebuild deterministic Phase 1 artifacts:

```bash
python phase1_dynamics/validate_simulator.py --output-dir outputs/phase1_validation
python phase1_dynamics/evaluate_phase1.py --output-dir outputs/phase1_evaluation
python phase1_dynamics/plot_phase1_suite.py \
  --input-dir outputs/phase1_evaluation \
  --validation-dir outputs/phase1_validation \
  --output-dir outputs/phase1_figures
```

4. Reproduce the saved controller evaluations from existing checkpoints:

```bash
python phase2_rl/evaluate_ppo.py --seed 7 \
  --model outputs/phase2_rl/go_no_go_baseline_v2_default_seed7_50k/model.zip \
  --output-dir outputs/phase2_rl/eval_go_no_go_baseline_v2_default_seed7_50k

python phase2b_hierarchical_rl/eval_hierarchical.py --seed 7 --scenario nominal \
  --throttle-model outputs/phase2b_hierarchical_rl/throttle_seed7_40k_flare_tracking_v1/model.zip \
  --tvc-model outputs/phase2b_hierarchical_rl/tvc_seed7_50k_flare_tracking_v1/model.zip \
  --output-dir outputs/phase2b_hierarchical_rl/eval_nominal_seed7_40k_50k_flare_tracking_v1

python phase2c_hybrid_rl/eval_hybrid.py --seed 7 --scenario nominal \
  --throttle-model outputs/phase2b_hierarchical_rl/throttle_seed7_40k_touchdown_gate_commit_v1/model.zip \
  --residual-model outputs/phase2c_hybrid_rl/residual_seed7_50k_stopping_floor_v1/model.zip \
  --output-dir outputs/phase2c_hybrid_rl/eval_nominal_seed7_50k_stopping_floor_v1
```

5. Rebuild the full Monte Carlo comparison and article visuals:

```bash
python phase3_evaluation/monte_carlo.py --seed 7 --episodes-per-level 500 \
  --output-dir outputs/phase3_evaluation/mc_seed7_500ep_composite_levels_v1
python phase3_evaluation/compare_controllers.py \
  --input-dir outputs/phase3_evaluation/mc_seed7_500ep_composite_levels_v1 \
  --output-dir outputs/phase3_evaluation/mc_seed7_500ep_composite_levels_v1/comparison
python phase3_evaluation/plot_success_rates.py \
  --input-dir outputs/phase3_evaluation/mc_seed7_500ep_composite_levels_v1 \
  --output-dir outputs/phase3_evaluation/mc_seed7_500ep_composite_levels_v1/figures
python phase3_evaluation/plot_dispersion.py \
  --input-dir outputs/phase3_evaluation/mc_seed7_500ep_composite_levels_v1 \
  --output-dir outputs/phase3_evaluation/mc_seed7_500ep_composite_levels_v1/figures
python phase3_evaluation/plot_failure_modes.py \
  --input-dir outputs/phase3_evaluation/mc_seed7_500ep_composite_levels_v1 \
  --output-dir outputs/phase3_evaluation/mc_seed7_500ep_composite_levels_v1/figures
python phase4_blog/generate_blog_assets.py
```

Training all PPO checkpoints from scratch is slower and may vary slightly with hardware/library versions even with fixed seeds. The repository therefore treats saved `config.json`, `metrics.json`, `trajectory.csv`, comparison CSV/JSON files, and plot manifests as the primary reproducibility layer.

## Output Conventions

Every major run writes structured artifacts under `outputs/<phase>/<run_name>/`:

- `config.json` records seed, scenario, profiles, checkpoint paths, and controller settings.
- `trajectory.csv` or `eval_trajectory.csv` records time histories.
- `metrics.json` or `eval_metrics.json` records touchdown metrics and failure modes.
- comparison scripts write CSV and JSON tables.
- plotting scripts consume saved CSV/JSON artifacts rather than rerunning training or controller rollouts.

Keep `design_decisions.md`, `paper_log.md`, and `plots_manifest.md` updated whenever modeling assumptions, experiment results, or figures change.
