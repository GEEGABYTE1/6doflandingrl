"""
eval script

usage:
    python src/evaluation/evaluate.py --model runs/<id>/best_model.zip
    python src/evaluation/evaluate.py --model runs/<id>/best_model.zip --wind 5
"""

import sys, argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.dynamics.dynamics import (
    VehicleParams, RocketSimulator, WindModel,
    quat_normalize, quat_to_rotmat
)
from src.controllers.lqr      import LQRController
from landing_env import (
    normalise_obs, physical_action, tilt_deg,
    MAX_TOUCH_SPEED, MAX_TOUCH_TILT, MAX_POS_ERR
)

COLORS = {"LQR": "#E53935", "PPO": "#1E88E5"}
LW     = {"LQR": 1.8,       "PPO": 2.0}

NOISE = np.array([2.,2.,2., 0.1,0.1,0.1,
                  0.002,0.002,0.002,0.002,
                  0.005,0.005,0.005, 500.,
                  0.,0.,0.,0.], dtype=np.float32)

class PPOCtrl:
    def __init__(self, path):
        from stable_baselines3 import PPO
        self.model = PPO.load(path)
        print(f"[PPO] Loaded {path}")

    def get_action(self, state, t):
        obs = normalise_obs(state).reshape(1, -1)
        act, _ = self.model.predict(obs, deterministic=True)
        return physical_action(act[0])

    def reset(self): pass

def run_episode(ctrl, wind_speed=5., seed=42):
    params = VehicleParams()
    wind   = WindModel(V_ref=wind_speed, h_ref=10., direction_deg=45.,
                       turbulence_intensity=0.10,
                       rng=np.random.default_rng(seed+1)) if wind_speed > 0 else None
    misalign = np.array([np.deg2rad(0.1), np.deg2rad(0.1)])
    sim    = RocketSimulator(params=params, wind_model=wind, dt=0.05,
                             misalignment=misalign)
    s0     = sim.make_initial_state(altitude=1000., vz=-100.)
    return sim.run(s0, ctrl, t_max=120., noise_std=NOISE)

def plot_control(results, out):
    fig, axes = plt.subplots(3, 2, figsize=(16, 10), sharex="col")
    fig.suptitle("Control History: LQR vs PPO  (wind = 5 m/s)",
                 fontsize=14, fontweight="bold")
    labels = ["Throttle τ", "TVC Pitch δp (deg)", "TVC Yaw δy (deg)"]
    scales = [1., np.degrees(1.), np.degrees(1.)]
    ylims  = [(0.35, 1.05), (-7, 7), (-7, 7)]

    for col, (name, res) in enumerate(results.items()):
        T = res["times"][:-1]
        A = res["actions"][:len(T)]
        for row in range(3):
            ax   = axes[row][col]
            vals = A[:, row] * scales[row]
            ax.plot(T, vals, color=COLORS[name], lw=LW[name], alpha=0.85)
            if row in (1, 2): ax.axhline(0, color="gray", lw=0.8, ls="--")
            ax.set_ylim(*ylims[row]); ax.grid(True, alpha=0.3)
            if col == 0:  ax.set_ylabel(labels[row], fontsize=10)
            if row == 0:  ax.set_title(name, fontsize=12,
                                        color=COLORS[name], fontweight="bold")
            if row == 2:  ax.set_xlabel("Time (s)", fontsize=10)
            if row in (1, 2):
                rms = float(np.sqrt(np.mean(np.diff(vals)**2)))
                ax.text(0.97, 0.95, f"RMS Δ = {rms:.2f}°/step",
                        transform=ax.transAxes, ha="right", va="top",
                        fontsize=8, color=COLORS[name],
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[Plot] {out}")


def plot_trajectory(results, out):
    fig = plt.figure(figsize=(14, 7))
    ax3 = fig.add_subplot(121, projection="3d")
    for name, res in results.items():
        S = res["states"]
        ax3.plot(S[:,0], S[:,1], S[:,2], color=COLORS[name], lw=LW[name], label=name)
        ax3.scatter(*S[-1,:3], color=COLORS[name], s=80, marker="*")
    th = np.linspace(0, 2*np.pi, 100)
    ax3.plot(50*np.cos(th), 50*np.sin(th), np.zeros(100), "g--", lw=1., alpha=0.5)
    ax3.scatter(0,0,0, c="green", s=150, marker="^", label="Target")
    ax3.set_xlabel("East (m)"); ax3.set_ylabel("North (m)"); ax3.set_zlabel("Alt (m)")
    ax3.set_title("3D Trajectory"); ax3.legend(fontsize=9)

    ax2 = fig.add_subplot(122)
    for name, res in results.items():
        S = res["states"]
        lat = np.sqrt(S[:,0]**2 + S[:,1]**2)
        ax2.plot(S[:,2], lat, color=COLORS[name], lw=LW[name], label=name)
        ax2.scatter(S[-1,2], lat[-1], color=COLORS[name], s=80, marker="*")
    ax2.axhline(150, color="orange", ls="--", lw=1., label="150m threshold")
    ax2.set_xlabel("Altitude (m)"); ax2.set_ylabel("Lateral Error (m)")
    ax2.set_title("Lateral Error vs Altitude")
    ax2.invert_xaxis(); ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

    fig.suptitle("Trajectory Comparison: LQR vs PPO  (wind = 5 m/s)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[Plot] {out}")

def plot_states(results, out):
    panels = [
        ("Altitude (m)",          lambda S: S[:,2]),
        ("Total Speed (m/s)",     lambda S: np.linalg.norm(S[:,3:6], axis=1)),
        ("Lateral Pos Error (m)", lambda S: np.sqrt(S[:,0]**2+S[:,1]**2)),
        ("|Horiz Vel| (m/s)",     lambda S: np.linalg.norm(S[:,3:5], axis=1)),
        ("Tilt (deg)",            lambda S: np.array([tilt_deg(S[i,6:10]) for i in range(len(S))])),
        ("|Angular Rate| (rad/s)",lambda S: np.linalg.norm(S[:,10:13], axis=1)),
        ("Propellant Used (kg)",  lambda S: S[0,13] - S[:,13]),
        ("Vz (m/s)",              lambda S: S[:,5]),
    ]
    fig, axes = plt.subplots(4, 4, figsize=(20, 14))
    fig.suptitle("State History Comparison: LQR vs PPO  (wind = 5 m/s)",
                 fontsize=14, fontweight="bold")
    for pi, (title, fn) in enumerate(panels):
        for ci, (name, res) in enumerate(results.items()):
            ax   = axes[pi//2][pi%2*2+ci]
            T, S = res["times"], res["states"]
            ax.plot(T, fn(S), color=COLORS[name], lw=1.6)
            ax.set_title(f"{name} — {title}", fontsize=8, color=COLORS[name])
            ax.grid(True, alpha=0.3); ax.tick_params(labelsize=7)
            if pi//2 == 3: ax.set_xlabel("Time (s)", fontsize=8)
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[Plot] {out}")

def metrics_table(results, out_dir):
    rows = []
    for name, res in results.items():
        m = res["metrics"]
        A = res["actions"]
        dp_d = np.diff(np.degrees(A[:,1]))
        dy_d = np.diff(np.degrees(A[:,2]))
        rms  = float(np.sqrt(np.mean(dp_d**2 + dy_d**2)))
        rows.append(dict(
            Controller=name, Success="✓" if m["success"] else "✗",
            Reason=m["reason"],
            Landing_Err_m      = round(m["landing_pos_err"], 1),
            Touchdown_Speed_ms = round(m["touchdown_vel"], 2),
            Touchdown_Vz_ms    = round(m["touchdown_vz"], 2),
            Tilt_deg           = round(m["tilt_deg"], 2),
            Fuel_kg            = round(m["fuel_consumed"], 1),
            Flight_time_s      = round(m["flight_time"], 1),
            TVC_RMS_chatter    = round(rms, 3),
        ))
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "metrics_table.csv", index=False)
    print("\n" + "="*65)
    print("  RESULTS TABLE")
    print("="*65)
    print(df.to_string(index=False))
    print("="*65)
    return df


#main funcs
def evaluate(model_path, wind=5., seed=42, out_dir="results"):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    print("\n[LQR] Building controller...")
    lqr = LQRController(VehicleParams())
    lqr.precompute_gains(verbose=True)

    print(f"\n[PPO] Loading {model_path}")
    ppo = PPOCtrl(model_path)

    print(f"\n[Eval] Running episodes (wind={wind} m/s, seed={seed}) ...")
    results = {
        "LQR": run_episode(lqr, wind, seed),
        "PPO": run_episode(ppo, wind, seed),
    }

    plot_control   (results, out / "P3b_control_comparison.png")
    plot_trajectory(results, out / "P1b_trajectory_compare.png")
    plot_states    (results, out / "P2b_states_compare.png")
    metrics_table  (results, out)
    print(f"\n[Done] All outputs → {out}/")







if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model",   required=True)
    p.add_argument("--wind",    type=float, default=5.)
    p.add_argument("--seed",    type=int,   default=42)
    p.add_argument("--out-dir", type=str,   default="results")
    args = p.parse_args()
    evaluate(args.model, args.wind, args.seed, args.out_dir)