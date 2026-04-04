"""
evaluate_ppo.py — Generate Phase 2 Comparison Plots (LQR vs PPO)
=================================================================
Paper §6 — Results

Usage (from project root, venv active):
    python evaluate_ppo.py                         # uses models/ppo_rocket_final.zip
    python evaluate_ppo.py --model models/ppo_rocket_best.zip
    python evaluate_ppo.py --wind 10               # test at specific wind speed

Outputs:
    plots/P1b_trajectory_compare.png  — 3D trajectories side by side
    plots/P2b_states_compare.png      — state history comparison (8 panels each)
    plots/P3b_control_compare.png     — throttle + TVC comparison
    plots/P6_metrics_table.png        — summary metrics table (paper Table II)
    results/metrics_table.csv         — numerical results for LaTeX
"""

import os, sys, argparse
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(ROOT)  # src/
sys.path.insert(0, os.path.dirname(SRC))  # project root
sys.path.insert(0, SRC)   # src/ — so 'dynamics.dynamics' and 'controllers.lqr' resolve
sys.path.insert(0, ROOT)  # src/model/ — so 'landing_env' resolves directly

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
import matplotlib.cm as cm

from dynamics.dynamics import (
    VehicleParams, RocketSimulator, WindModel,
    quat_normalize, quat_to_rotmat, rk4_step
)
from controllers.lqr import LQRController
# ── auto-detect environment module location ──────────
import importlib, pathlib
_src = pathlib.Path(__file__).parent / 'src'
_candidates = ['landing_env', 'environment.landing_env', 'model.landing_env']
_mod = None
for _c in _candidates:
    try:
        _mod = importlib.import_module(_c)
        break
    except ModuleNotFoundError:
        pass
if _mod is None:
    raise ImportError("Cannot find landing_env. Make sure src/environment/ or src/model/ exists.")
RocketLandingEnv = _mod.RocketLandingEnv
tilt_from_quaternion = _mod.tilt_from_quaternion

os.makedirs('plots',   exist_ok=True)
os.makedirs('results', exist_ok=True)

# ── Plot style ────────────────────────────────────────
BG   = '#0A0E1A'; BG2  = '#111827'; GRID = '#1E2A3A'
CLQR = '#FF6B35'; CPPO = '#00D4FF'; C2   = '#A8FF3E'
C3   = '#FF3366'; DIM  = '#6B7B9A'; TXT  = '#E8EDF5'

plt.rcParams.update({
    'figure.facecolor': BG,   'axes.facecolor':  BG2,
    'axes.edgecolor':   GRID, 'axes.labelcolor': TXT,
    'axes.titlecolor':  TXT,  'xtick.color':     DIM,
    'ytick.color':      DIM,  'grid.color':       GRID,
    'grid.linewidth':   0.5,  'text.color':       TXT,
    'font.family':      'monospace', 'font.size': 8,
    'axes.titlesize':   9,    'axes.labelsize':   8,
    'legend.facecolor': BG2,  'legend.edgecolor': GRID,
    'legend.fontsize':  7,    'figure.dpi':       150,
    'savefig.dpi':      180,  'savefig.facecolor': BG,
    'lines.linewidth':  1.6,
})


# ══════════════════════════════════════════════════════
#  Run a single episode and return full history
# ══════════════════════════════════════════════════════
def run_lqr(wind_speed=5., seed=42) -> dict:
    params = VehicleParams()
    wind   = WindModel(
        V_ref=wind_speed, direction_deg=45.,
        turbulence_intensity=0.10,
        rng=np.random.default_rng(seed)
    )
    mis = np.array([np.deg2rad(0.1), 0.])
    sim = RocketSimulator(params, wind_model=wind, dt=0.05, misalignment=mis)
    ctrl = LQRController(params)
    ctrl.precompute_gains(verbose=False)

    noise = np.zeros(18)
    noise[0:3]=2.; noise[3:6]=0.1; noise[6:10]=0.002; noise[10:13]=0.005

    s = sim.make_initial_state(altitude=1000., vz=-100., vx=0., vy=0., pitch_deg=1.)
    return sim.run(s, ctrl, t_max=120., noise_std=noise)


def run_ppo(model_path: str, wind_speed=5., seed=42, curriculum_stage=2) -> dict:
    """
    Evaluate a saved PPO policy on one episode.
    Uses raw RocketLandingEnv (no VecNormalize wrapper in eval).
    Applies saved obs normalisation stats manually if available.
    """
    from stable_baselines3 import PPO
    import pickle

    model = PPO.load(model_path)

    # Load obs normalisation stats if available
    vec_stats = os.path.join(os.path.dirname(model_path), 'vec_normalize.pkl')
    obs_mean = obs_var = None
    if os.path.exists(vec_stats):
        try:
            from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
            _dummy = DummyVecEnv([lambda: RocketLandingEnv(
                randomise_ic=False, fixed_wind_speed=wind_speed,
                t_max=60., seed=seed)])
            _vn = VecNormalize.load(vec_stats, _dummy)
            obs_mean = _vn.obs_rms.mean.copy()
            obs_var  = _vn.obs_rms.var.copy()
            _dummy.close()
        except Exception as e:
            print(f"  [WARN] Could not load VecNormalize stats: {e}")

    def normalise_obs(obs_raw):
        if obs_mean is not None:
            return np.clip(
                (obs_raw - obs_mean) / np.sqrt(obs_var + 1e-8), -10., 10.
            ).astype(np.float32)
        return obs_raw

    # Run raw environment
    # Build eval env at the requested curriculum stage
    # Stage 0/1: use appropriate alt/vz for that stage
    _stage_alt = {0: 150., 1: 400., 2: 1000.}
    _stage_vz  = {0: -15., 1: -40., 2: -100.}
    eval_alt = _stage_alt.get(curriculum_stage, 1000.)
    eval_vz  = _stage_vz.get(curriculum_stage, -100.)

    env  = RocketLandingEnv(
        randomise_ic=False, fixed_wind_speed=wind_speed,
        t_max=60., seed=seed)
    env._curriculum_stage = curriculum_stage
    # Override make_initial_state defaults by patching the call
    _orig_make = env.make_initial_state if hasattr(env, 'make_initial_state') else None
    raw_obs, _ = env.reset(seed=seed)
    # Manually set state to stage-appropriate IC
    import sys as _sys; _sys.path.insert(0, 'src')
    env._state = env._sim.make_initial_state(
        altitude=eval_alt, vz=eval_vz, vx=0., vy=0., pitch_deg=1.)
    raw_obs = env._get_obs()
    obs = normalise_obs(raw_obs)

    times   = [0.]
    states  = [env._state.copy()]
    actions = []
    done    = False
    info    = {}

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        raw_obs, _, term, trunc, info = env.step(action)
        obs  = normalise_obs(raw_obs)
        done = term or trunc

        times.append(env._t)
        states.append(env._state.copy())

        # Convert normalised action → physical for plotting
        p   = env.params
        tau = p.throttle_min + 0.5*(action[0]+1.)*(p.throttle_max - p.throttle_min)
        dp  = action[1] * p.tvc_max
        dy  = action[2] * p.tvc_max
        actions.append(np.array([tau, dp, dy]))

    env.close()
    actions.append(actions[-1] if actions else np.zeros(3))

    S = np.array(states); T = np.array(times); A = np.array(actions)
    f   = S[-1]
    spd = float(np.linalg.norm(f[3:6]))
    tlt = float(np.rad2deg(tilt_from_quaternion(f[6:10])))
    err = float(np.linalg.norm(f[0:2]))
    success = info.get('success', spd < 5. and tlt < 15. and err < 150.)

    return {
        'times': T, 'states': S, 'actions': A,
        'reason': info.get('reason', 'unknown'),
        'metrics': {
            'success':          success,
            'touchdown_vz':     float(f[5]),
            'touchdown_vel':    spd,
            'tilt_deg':         tlt,
            'landing_pos_err':  err,
            'fuel_consumed':    float(S[0,13] - f[13]),
            'flight_time':      float(T[-1]),
        }
    }


def tilt_history(states):
    return np.array([
        np.rad2deg(tilt_from_quaternion(states[i,6:10]))
        for i in range(len(states))
    ])


# ══════════════════════════════════════════════════════
#  P1b — Trajectory comparison
# ══════════════════════════════════════════════════════
def plot_P1b(lqr: dict, ppo: dict, wind_speed: float):
    Sl, Sp = lqr['states'], ppo['states']

    fig = plt.figure(figsize=(16, 7))
    fig.suptitle(f'Trajectory Comparison: LQR vs PPO  (wind = {wind_speed} m/s)',
                 fontsize=11)
    gs = gridspec.GridSpec(1, 2, wspace=0.3)

    # ── 3D trajectories ───────────────────────────────
    ax = fig.add_subplot(gs[0], projection='3d')
    ax.set_facecolor(BG2)
    ax.plot(Sl[:,0], Sl[:,1], Sl[:,2], color=CLQR, linewidth=2., label='LQR')
    ax.plot(Sp[:,0], Sp[:,1], Sp[:,2], color=CPPO, linewidth=2., label='PPO')

    th = np.linspace(0, 2*np.pi, 60)
    ax.plot(50*np.cos(th), 50*np.sin(th), np.zeros(60),
            color=C2, linewidth=1.5, linestyle='--', label='50m pad')
    ax.scatter([0],[0],[0], s=80, c=C2, marker='^', zorder=5, label='Target')
    ax.scatter([Sl[-1,0]], [Sl[-1,1]], [0], s=80, c=CLQR, marker='*', zorder=5)
    ax.scatter([Sp[-1,0]], [Sp[-1,1]], [0], s=80, c=CPPO, marker='*', zorder=5)

    ax.set_xlabel('East (m)'); ax.set_ylabel('North (m)'); ax.set_zlabel('Alt (m)')
    ax.set_title('3D Trajectory')
    ax.legend(fontsize=7)
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
    for p in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        p.set_edgecolor(GRID)

    # ── Lateral error vs altitude ─────────────────────
    ax2 = fig.add_subplot(gs[1])
    lqr_err = np.linalg.norm(Sl[:,0:2], axis=1)
    ppo_err = np.linalg.norm(Sp[:,0:2], axis=1)
    ax2.plot(Sl[:,2], lqr_err, color=CLQR, linewidth=2., label='LQR')
    ax2.plot(Sp[:,2], ppo_err, color=CPPO, linewidth=2., label='PPO')
    ax2.axhline(150., color='#FFD700', linestyle='--', linewidth=1.2,
                alpha=0.8, label='150m success threshold')
    ax2.scatter([0.], [lqr_err[-1]], s=80, c=CLQR, marker='*', zorder=5)
    ax2.scatter([0.], [ppo_err[-1]], s=80, c=CPPO, marker='*', zorder=5)
    ax2.set_xlabel('Altitude (m)'); ax2.set_ylabel('Lateral Error (m)')
    ax2.set_title('Lateral Error vs Altitude')
    ax2.legend(); ax2.grid(True, alpha=0.4)
    ax2.invert_xaxis()

    plt.savefig('plots/P1b_trajectory_compare.png', bbox_inches='tight')
    print('[PLOT] P1b_trajectory_compare.png')
    plt.close()


# ══════════════════════════════════════════════════════
#  P2b — State history comparison
# ══════════════════════════════════════════════════════
def plot_P2b(lqr: dict, ppo: dict, wind_speed: float):
    Tl, Sl = lqr['times'], lqr['states']
    Tp, Sp = ppo['times'], ppo['states']

    tilts_l = tilt_history(Sl)
    tilts_p = tilt_history(Sp)

    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(f'State History Comparison: LQR vs PPO  (wind = {wind_speed} m/s)',
                 fontsize=11)
    gs = gridspec.GridSpec(4, 4, hspace=0.5, wspace=0.35)

    pairs = [
        # (row, col_lqr, col_ppo, title, ylabel, lqr_data, ppo_data)
        (0,0,1,'Altitude (m)',       'Alt (m)',        Sl[:,2],   Sp[:,2]),
        (0,2,3,'Total Speed (m/s)',  'Speed (m/s)',
            np.linalg.norm(Sl[:,3:6],axis=1), np.linalg.norm(Sp[:,3:6],axis=1)),
        (1,0,1,'Lateral Pos Error (m)','Err (m)',
            np.linalg.norm(Sl[:,0:2],axis=1), np.linalg.norm(Sp[:,0:2],axis=1)),
        (1,2,3,'|Horiz Vel| (m/s)',  'Vel (m/s)',
            np.linalg.norm(Sl[:,3:5],axis=1), np.linalg.norm(Sp[:,3:5],axis=1)),
        (2,0,1,'Tilt (deg)',         'Tilt (°)',       tilts_l,   tilts_p),
        (2,2,3,'|Angular Rate| (rad/s)','Rate (r/s)',
            np.linalg.norm(Sl[:,10:13],axis=1), np.linalg.norm(Sp[:,10:13],axis=1)),
        (3,0,1,'Propellant Used (kg)','Fuel (kg)',
            Sl[0,13]-Sl[:,13],  Sp[0,13]-Sp[:,13]),
        (3,2,3,'Vz (m/s)',           'vz (m/s)',       Sl[:,5],   Sp[:,5]),
    ]

    for row, cl, cp, title, ylabel, dl, dp in pairs:
        ax_l = fig.add_subplot(gs[row, cl])
        ax_l.plot(Tl, dl, color=CLQR)
        ax_l.set_title(f'LQR — {title}', color=CLQR, fontsize=8)
        ax_l.set_ylabel(ylabel, fontsize=7)
        ax_l.set_xlabel('Time (s)', fontsize=7)
        ax_l.grid(True, alpha=0.3)

        ax_p = fig.add_subplot(gs[row, cp])
        ax_p.plot(Tp, dp, color=CPPO)
        ax_p.set_title(f'PPO — {title}', color=CPPO, fontsize=8)
        ax_p.set_ylabel(ylabel, fontsize=7)
        ax_p.set_xlabel('Time (s)', fontsize=7)
        ax_p.grid(True, alpha=0.3)

    plt.savefig('plots/P2b_states_compare.png', bbox_inches='tight')
    print('[PLOT] P2b_states_compare.png')
    plt.close()


# ══════════════════════════════════════════════════════
#  P3b — Control history comparison
# ══════════════════════════════════════════════════════
def plot_P3b(lqr: dict, ppo: dict, wind_speed: float):
    Tl, Al = lqr['times'], lqr['actions']
    Tp, Ap = ppo['times'], ppo['actions']
    nl, np_ = min(len(Tl), len(Al)), min(len(Tp), len(Ap))

    # Compute RMS action change (chatter metric)
    def rms_delta(A, n):
        d = np.diff(np.rad2deg(A[:n, 1:3]), axis=0)
        return np.sqrt(np.mean(d**2, axis=0))

    rms_l = rms_delta(Al, nl)
    rms_p = rms_delta(Ap, np_)

    fig, axes = plt.subplots(3, 2, figsize=(16, 9), sharex='col')
    fig.suptitle(f'Control History: LQR vs PPO  (wind = {wind_speed} m/s)',
                 fontsize=11)

    labels_l = ['LQR'] * 3
    labels_p = ['PPO'] * 3
    cols_l   = [CLQR] * 3
    cols_p   = [CPPO] * 3

    # Throttle
    axes[0,0].plot(Tl[:nl], Al[:nl,0], color=CLQR)
    axes[0,0].set_ylabel('Throttle τ'); axes[0,0].set_title('LQR', color=CLQR)
    axes[0,0].set_ylim(0.35, 1.05); axes[0,0].grid(True, alpha=0.3)

    axes[0,1].plot(Tp[:np_], Ap[:np_,0], color=CPPO)
    axes[0,1].set_ylabel('Throttle τ'); axes[0,1].set_title('PPO', color=CPPO)
    axes[0,1].set_ylim(0.35, 1.05); axes[0,1].grid(True, alpha=0.3)

    # TVC Pitch
    axes[1,0].plot(Tl[:nl], np.rad2deg(Al[:nl,1]), color=CLQR)
    axes[1,0].axhline(6., color=C3, linestyle='--', linewidth=1., alpha=0.7)
    axes[1,0].axhline(-6., color=C3, linestyle='--', linewidth=1., alpha=0.7)
    axes[1,0].set_ylabel('TVC Pitch δp (deg)')
    axes[1,0].text(0.98, 0.02, f'RMS Δ = {rms_l[0]:.2f}°/step',
                   transform=axes[1,0].transAxes, ha='right', va='bottom',
                   color=CLQR, fontsize=7,
                   bbox=dict(boxstyle='round', facecolor=BG2, alpha=0.8))
    axes[1,0].grid(True, alpha=0.3)

    axes[1,1].plot(Tp[:np_], np.rad2deg(Ap[:np_,1]), color=CPPO)
    axes[1,1].axhline(6., color=C3, linestyle='--', linewidth=1., alpha=0.7)
    axes[1,1].axhline(-6., color=C3, linestyle='--', linewidth=1., alpha=0.7)
    axes[1,1].set_ylabel('TVC Pitch δp (deg)')
    axes[1,1].text(0.98, 0.02, f'RMS Δ = {rms_p[0]:.2f}°/step',
                   transform=axes[1,1].transAxes, ha='right', va='bottom',
                   color=CPPO, fontsize=7,
                   bbox=dict(boxstyle='round', facecolor=BG2, alpha=0.8))
    axes[1,1].grid(True, alpha=0.3)

    # TVC Yaw
    axes[2,0].plot(Tl[:nl], np.rad2deg(Al[:nl,2]), color=CLQR)
    axes[2,0].axhline(6., color=C3, linestyle='--', linewidth=1., alpha=0.7)
    axes[2,0].axhline(-6., color=C3, linestyle='--', linewidth=1., alpha=0.7)
    axes[2,0].set_ylabel('TVC Yaw δy (deg)'); axes[2,0].set_xlabel('Time (s)')
    axes[2,0].text(0.98, 0.02, f'RMS Δ = {rms_l[1]:.2f}°/step',
                   transform=axes[2,0].transAxes, ha='right', va='bottom',
                   color=CLQR, fontsize=7,
                   bbox=dict(boxstyle='round', facecolor=BG2, alpha=0.8))
    axes[2,0].grid(True, alpha=0.3)

    axes[2,1].plot(Tp[:np_], np.rad2deg(Ap[:np_,2]), color=CPPO)
    axes[2,1].axhline(6., color=C3, linestyle='--', linewidth=1., alpha=0.7)
    axes[2,1].axhline(-6., color=C3, linestyle='--', linewidth=1., alpha=0.7)
    axes[2,1].set_ylabel('TVC Yaw δy (deg)'); axes[2,1].set_xlabel('Time (s)')
    axes[2,1].text(0.98, 0.02, f'RMS Δ = {rms_p[1]:.2f}°/step',
                   transform=axes[2,1].transAxes, ha='right', va='bottom',
                   color=CPPO, fontsize=7,
                   bbox=dict(boxstyle='round', facecolor=BG2, alpha=0.8))
    axes[2,1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('plots/P3b_control_comparison.png', bbox_inches='tight')
    print('[PLOT] P3b_control_comparison.png')
    plt.close()


# ══════════════════════════════════════════════════════
#  P6 — Metrics table
# ══════════════════════════════════════════════════════
def save_metrics(lqr: dict, ppo: dict, wind_speed: float):
    ml, mp = lqr['metrics'], ppo['metrics']

    rows = [
        ('Success',            str(ml['success']),                str(mp['success'])),
        ('Touchdown vz (m/s)', f"{ml['touchdown_vz']:.2f}",       f"{mp['touchdown_vz']:.2f}"),
        ('Touchdown speed (m/s)',f"{ml['touchdown_vel']:.2f}",    f"{mp['touchdown_vel']:.2f}"),
        ('Tilt at landing (°)',f"{ml['tilt_deg']:.2f}",           f"{mp['tilt_deg']:.2f}"),
        ('Position error (m)', f"{ml['landing_pos_err']:.1f}",    f"{mp['landing_pos_err']:.1f}"),
        ('Fuel used (kg)',      f"{ml['fuel_consumed']:.0f}",      f"{mp['fuel_consumed']:.0f}"),
        ('Flight time (s)',     f"{ml['flight_time']:.1f}",        f"{mp['flight_time']:.1f}"),
    ]

    # Save CSV
    import csv
    with open('results/metrics_table.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Metric', 'LQR', 'PPO'])
        w.writerows(rows)
    print('[SAVE] results/metrics_table.csv')

    # Plot table
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle(f'Results Summary: LQR vs PPO  (wind = {wind_speed} m/s)',
                 fontsize=11)
    ax.axis('off')

    col_labels = ['Metric', 'LQR', 'PPO']
    tbl = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.8)

    # Colour header
    for j in range(3):
        tbl[0,j].set_facecolor('#1E2A3A')
        tbl[0,j].set_text_props(color=TXT, fontweight='bold')

    # Colour LQR/PPO columns
    for i in range(1, len(rows)+1):
        tbl[i,0].set_facecolor(BG2)
        tbl[i,1].set_facecolor('#2A1A0A'); tbl[i,1].set_text_props(color=CLQR)
        tbl[i,2].set_facecolor('#0A1A2A'); tbl[i,2].set_text_props(color=CPPO)

    plt.savefig('plots/P6_metrics_table.png', bbox_inches='tight')
    print('[PLOT] plots/P6_metrics_table.png')
    plt.close()

    # Print to console
    print(f"\n{'Metric':<28}  {'LQR':>10}  {'PPO':>10}")
    print('-' * 52)
    for r in rows:
        print(f"  {r[0]:<26}  {r[1]:>10}  {r[2]:>10}")


# ══════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,   default='models/ppo_rocket_final.zip')
    parser.add_argument('--wind',  type=float, default=5.)
    parser.add_argument('--seed',  type=int,   default=42)
    parser.add_argument('--stage', type=int,   default=2,
                        help='Curriculum stage for eval: 0=easy, 1=medium, 2=full (default: 2)')
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"[ERROR] Model not found: {args.model}")
        print("        Run  python train_ppo.py  first.")
        sys.exit(1)

    stage_labels = {0: 'easy (80-200m, -8 to -20 m/s)',
                    1: 'medium (200-600m, -20 to -60 m/s)',
                    2: 'full (800-1200m, -80 to -120 m/s)'}
    print(f"[EVAL] Curriculum stage {args.stage}: {stage_labels.get(args.stage, 'full')}")
    print(f"[EVAL] Running LQR  (wind={args.wind} m/s, seed={args.seed})...")
    lqr = run_lqr(wind_speed=args.wind, seed=args.seed)
    ml  = lqr['metrics']
    print(f"       vz={ml['touchdown_vz']:.2f}  tilt={ml['tilt_deg']:.2f}°  "
          f"err={ml['landing_pos_err']:.1f}m  success={ml['success']}")

    print(f"[EVAL] Running PPO  (wind={args.wind} m/s, seed={args.seed})...")
    ppo = run_ppo(args.model, wind_speed=args.wind, seed=args.seed,
                  curriculum_stage=args.stage)
    mp  = ppo['metrics']
    print(f"       vz={mp['touchdown_vz']:.2f}  tilt={mp['tilt_deg']:.2f}°  "
          f"err={mp['landing_pos_err']:.1f}m  success={mp['success']}")

    print("\n[PLOTS] Generating comparison plots...")
    plot_P1b(lqr, ppo, args.wind)
    plot_P2b(lqr, ppo, args.wind)
    plot_P3b(lqr, ppo, args.wind)
    save_metrics(lqr, ppo, args.wind)
    print("\n[DONE] All Phase 2 plots saved to plots/")
