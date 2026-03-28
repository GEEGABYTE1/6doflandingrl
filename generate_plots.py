"""Generate all Phase 1 publication plots.

Run from the project root:
    python generate_plots.py

Outputs saved to plots/
"""
import sys, os

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'src'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
import matplotlib.cm as cm

from dynamics.dynamics import VehicleParams, RocketSimulator, WindModel, quat_normalize, quat_to_rotmat
from controllers.lqr import LQRController, get_tilt

os.makedirs(os.path.join(ROOT, 'plots'), exist_ok=True)
PLOTS_DIR = os.path.join(ROOT, 'plots')


BG   = '#0A0E1A'; BG2  = '#111827'; GRID = '#1E2A3A'
C0   = '#00D4FF'; C1   = '#FF6B35'; C2   = '#A8FF3E'
C3   = '#FF3366'; DIM  = '#6B7B9A'; TXT  = '#E8EDF5'

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': BG2,
    'axes.edgecolor':   GRID,'axes.labelcolor': TXT,
    'axes.titlecolor':  TXT, 'xtick.color':     DIM,
    'ytick.color':      DIM, 'grid.color':       GRID,
    'grid.linewidth':   0.6, 'text.color':       TXT,
    'font.family':      'monospace','font.size': 9,
    'axes.titlesize':   10,  'axes.labelsize':   9,
    'legend.facecolor': BG2, 'legend.edgecolor': GRID,
    'legend.fontsize':  8,   'figure.dpi':       150,
    'savefig.dpi':      180, 'savefig.facecolor': BG,
    'lines.linewidth':  1.6,
})


params = VehicleParams()
wind   = WindModel(V_ref=10., direction_deg=45., turbulence_intensity=0.10,
                   rng=np.random.default_rng(42))
sim    = RocketSimulator(params, wind_model=wind, dt=0.05,
                         misalignment=np.array([np.deg2rad(0.1), 0.]))
ctrl   = LQRController(params)
ctrl.precompute_gains(verbose=True)
noise  = np.zeros(18)
noise[0:3]=2.; noise[3:6]=0.1; noise[6:10]=0.002; noise[10:13]=0.005

state  = sim.make_initial_state(altitude=1000., vz=-100., vx=0., vy=0., pitch_deg=1.)
result = sim.run(state, ctrl, t_max=120., noise_std=noise)
m      = result['metrics']
T      = result['times']
S      = result['states']
A      = result['actions']

print(f'\nSUCCESS={m["success"]}  vz={m["touchdown_vz"]:.2f}  tilt={m["tilt_deg"]:.2f}°  err={m["landing_pos_err"]:.1f}m')

# Helper: tilt history
tilts_deg = np.array([
    np.rad2deg(np.arccos(np.clip(
        -(quat_to_rotmat(quat_normalize(S[i,6:10])) @ np.array([0,0,1]))[2], -1., 1.)))
    for i in range(len(T))
])


fig = plt.figure(figsize=(10, 8))
ax  = fig.add_subplot(111, projection='3d')
ax.set_facecolor(BG2)

speed = np.linalg.norm(S[:,3:6], axis=1)
norm  = Normalize(vmin=0, vmax=speed.max())
cmap  = cm.plasma

for i in range(len(T)-1):
    c = cmap(norm(speed[i]))
    ax.plot(S[i:i+2,0], S[i:i+2,1], S[i:i+2,2]/1000,
            color=c, linewidth=1.8, alpha=0.9)

th = np.linspace(0, 2*np.pi, 60)
ax.plot(50*np.cos(th), 50*np.sin(th), np.zeros(60),
        color=C2, linewidth=2., alpha=0.7)
ax.scatter([0],[0],[0], s=80, c=C2, marker='o', zorder=5, label='Landing pad')
ax.scatter([S[0,0]],[S[0,1]],[S[0,2]/1000], s=80, c=C0,
           marker='^', zorder=5, label=f'Start 1000m')
ax.scatter([S[-1,0]],[S[-1,1]],[S[-1,2]/1000], s=80, c=C1,
           marker='v', zorder=5, label=f'Touchdown vz={m["touchdown_vz"]:.1f}m/s')

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cb = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.1)
cb.set_label('Speed (m/s)', color=TXT)
plt.setp(cb.ax.yaxis.get_ticklabels(), color=DIM)

ax.set_xlabel('East (m)', labelpad=8)
ax.set_ylabel('North (m)', labelpad=8)
ax.set_zlabel('Altitude (km)', labelpad=8)
ax.set_title('6DOF Rocket Landing — LQR Controller\n'
             'Starship-class terminal descent | 1km entry | wind 10 m/s | colored by speed',
             pad=12, fontsize=10)
ax.legend(fontsize=8)
ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
    pane.set_edgecolor(GRID)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'P1_trajectory_3d.png'), bbox_inches='tight')
print('[PLOT] P1_trajectory_3d.png')
plt.close()


fig = plt.figure(figsize=(16, 11))
fig.suptitle('6DOF State History — LQR Controller (Phase 1 | Wind 10 m/s | Noise + Misalignment)',
             fontsize=11, y=0.98)
gs = gridspec.GridSpec(4, 2, hspace=0.45, wspace=0.3)

def styled_ax(fig, gs_pos, title, ylabel, xlabel='Time (s)'):
    ax = fig.add_subplot(gs_pos)
    ax.set_title(title, fontsize=9, pad=4)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.grid(True, alpha=0.4)
    return ax


ax = styled_ax(fig, gs[0,0], 'Altitude', 'Alt (m)')
ax.plot(T, S[:,2], color=C0, linewidth=2.)
ax.fill_between(T, S[:,2], alpha=0.1, color=C0)


ax = styled_ax(fig, gs[0,1], 'Velocity', 'Speed (m/s)')
ax.plot(T, np.linalg.norm(S[:,3:6],axis=1), color=C1, label='|v|')
ax.plot(T, np.abs(S[:,5]), color=C2, linestyle='--', linewidth=1.2, label='|vz|')
ax.legend()


ax = styled_ax(fig, gs[1,0], 'Horizontal Position', 'Position (m)')
ax.plot(T, S[:,0], color=C0, label='East (x)')
ax.plot(T, S[:,1], color=C1, linestyle='--', label='North (y)')
ax.axhline(0, color=C2, linewidth=0.8, alpha=0.5, linestyle=':')
ax.legend()


ax = styled_ax(fig, gs[1,1], 'Horizontal Velocity', 'Vel (m/s)')
ax.plot(T, S[:,3], color=C0, label='vx')
ax.plot(T, S[:,4], color=C1, linestyle='--', label='vy')
ax.axhline(0, color=C2, linewidth=0.8, alpha=0.5, linestyle=':')
ax.legend()


ax = styled_ax(fig, gs[2,0], 'Vehicle Tilt', 'Tilt (deg)')
ax.plot(T, tilts_deg, color=C0, linewidth=2.)
ax.axhline(15., color=C3, linestyle='--', linewidth=1., alpha=0.8, label='15° limit')
ax.fill_between(T, tilts_deg, alpha=0.15, color=C0)
ax.legend()


ax = styled_ax(fig, gs[2,1], 'Angular Rates', 'Rate (deg/s)')
ax.plot(T, np.rad2deg(S[:,10]), color=C0, label='ωx')
ax.plot(T, np.rad2deg(S[:,11]), color=C1, linestyle='--', label='ωy')
ax.plot(T, np.rad2deg(S[:,12]), color=C2, linestyle=':', label='ωz')
ax.legend()


ax = styled_ax(fig, gs[3,0], 'Propellant Consumed', 'Fuel used (kg)')
fuel_used = S[0,13] - S[:,13]
ax.plot(T, fuel_used, color=C2, linewidth=2.)
ax.fill_between(T, fuel_used, alpha=0.2, color=C2)


ax = styled_ax(fig, gs[3,1], 'Vertical Velocity', 'vz (m/s)')
ax.plot(T, S[:,5], color=C0, linewidth=2.)
ax.axhline(-5., color=C2, linestyle='--', linewidth=1., alpha=0.8, label='−5 m/s limit')
ax.axhline(0., color=DIM, linestyle=':', linewidth=0.8, alpha=0.5)
ax.legend()

plt.savefig(os.path.join(PLOTS_DIR, 'P2_states_history.png'), bbox_inches='tight')
print('[PLOT] P2_states_history.png')
plt.close()


#coontrol history plot
fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
fig.suptitle('Control History — LQR Controller (Phase 1)', fontsize=11)
n = min(len(T), len(A))

axes[0].plot(T[:n], A[:n,0]*100, color=C0)
axes[0].fill_between(T[:n], A[:n,0]*100, alpha=0.12, color=C0)
axes[0].axhline(40, color=C3, linestyle='--', linewidth=1., alpha=0.7, label='Min throttle 40%')
axes[0].axhline(100, color=C2, linestyle='--', linewidth=1., alpha=0.7, label='Max throttle 100%')
axes[0].set_ylabel('Throttle (%)'); axes[0].set_title('Engine Throttle')
axes[0].legend(); axes[0].set_ylim(0, 110); axes[0].grid(True, alpha=0.4)

axes[1].plot(T[:n], np.rad2deg(A[:n,1]), color=C1)
axes[1].fill_between(T[:n], np.rad2deg(A[:n,1]), alpha=0.12, color=C1)
axes[1].axhline(6,  color=C3, linestyle='--', linewidth=1., alpha=0.7, label='±6° limit')
axes[1].axhline(-6, color=C3, linestyle='--', linewidth=1., alpha=0.7)
axes[1].set_ylabel('Gimbal (deg)'); axes[1].set_title('TVC Pitch Deflection')
axes[1].legend(); axes[1].grid(True, alpha=0.4)

axes[2].plot(T[:n], np.rad2deg(A[:n,2]), color=C2)
axes[2].fill_between(T[:n], np.rad2deg(A[:n,2]), alpha=0.12, color=C2)
axes[2].axhline(6,  color=C3, linestyle='--', linewidth=1., alpha=0.7, label='±6° limit')
axes[2].axhline(-6, color=C3, linestyle='--', linewidth=1., alpha=0.7)
axes[2].set_ylabel('Gimbal (deg)'); axes[2].set_title('TVC Yaw Deflection')
axes[2].set_xlabel('Time (s)'); axes[2].legend(); axes[2].grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'P3_control_history.png'), bbox_inches='tight')
print('[PLOT] P3_control_history.png')
plt.close()


#phase portrait
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Phase Portrait — Altitude vs Velocity', fontsize=11)

g0 = 9.80665; T_thrust = params.thrust_max
m_avg = np.mean(S[:,13])
a_net = T_thrust*0.925/m_avg - g0
h_ref = np.linspace(0, 1100, 300)
v_par = 50.*np.sqrt(h_ref/1000.)   # parabolic reference

ax = axes[0]
ax.plot(np.abs(S[:,5]), S[:,2], color=C0, linewidth=2., label='LQR trajectory')
ax.plot(v_par, h_ref, color=C3, linestyle='--', linewidth=1.5, alpha=0.8,
        label='Parabolic reference')
ax.scatter([abs(m["touchdown_vz"])], [0.], s=120, c=C2, zorder=5,
           label=f'Touchdown vz={m["touchdown_vz"]:.1f} m/s')
ax.set_xlabel('|Vertical Speed| (m/s)'); ax.set_ylabel('Altitude (m)')
ax.set_title('Altitude vs Descent Speed (Phase Portrait)')
ax.legend(); ax.grid(True, alpha=0.4)

ax = axes[1]
sc = ax.scatter(np.linalg.norm(S[:,3:6], axis=1), S[:,2],
                c=T, cmap='plasma', s=4, alpha=0.8)
ax.set_xlabel('Total Speed (m/s)'); ax.set_ylabel('Altitude (m)')
ax.set_title('Total Speed vs Altitude (colored by time)')
cb = fig.colorbar(sc, ax=ax); cb.set_label('Time (s)', color=TXT)
plt.setp(cb.ax.yaxis.get_ticklabels(), color=DIM)
ax.grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'P4_phase_portrait.png'), bbox_inches='tight')
print('[PLOT] P4_phase_portrait.png')
plt.close()

print(f'\nAll plots saved to plots/')
print(f'SUCCESS={m["success"]}  vz={m["touchdown_vz"]:.2f} m/s  tilt={m["tilt_deg"]:.2f}°  err={m["landing_pos_err"]:.1f}m  fuel={m["fuel_consumed"]:.0f}kg')
