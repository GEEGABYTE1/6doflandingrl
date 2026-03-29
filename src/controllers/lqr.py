"""
6dof rocket landing option b
=============================================================
Paper §5.1 — classical Baseline Controller

body-frame convention (q=[0,1,0,0] = nozzle-down base):
  wy=+1 -> nozzle tips East  -> euler_pitch = -R[0,2]  -> use euler[1] = -R[0,2]
  wx=+1 -> nozzle tips North -> euler_roll  = -R[1,2]  -> use euler[0] = -R[1,2]
  wz=+1 -> spin about nozzle axis (yaw)

LQR reduced state (10-dim, DD-008):
  [x, y, z, vx, vy, vz, tilt_EW, tilt_NS, wy, wx]
  tilt_EW = R[0,2]  (East  component of nozzle in inertial; +ve = nozzle East)
  tilt_NS = R[1,2]  (North component of nozzle in inertial; +ve = nozzle North)
  wy = omega[1] drives tilt_EW
  wx = omega[0] drives tilt_NS

Control: u = [throttle, delta_pitch, delta_yaw]
  delta_pitch -> TVC East-West gimbal (M about body-y -> corrects tilt_EW)
  delta_yaw   -> TVC North-South gimbal (M about body-x -> corrects tilt_NS)
"""

import numpy as np
from scipy.linalg import solve_continuous_are
from typing import List
from dataclasses import dataclass

from dynamics.dynamics import (
    VehicleParams, quat_normalize, quat_to_rotmat,
    equations_of_motion, thrust_force_moment
)

G0 = 9.80665


#frame correction tilt
def get_tilt(q: np.ndarray):
    """
    returns [tilt_EW, tilt_NS] = [R[0,2], R[1,2]].
    these are the East and North components of the nozzle (+z_body)
    direction in the inertial frame.  Zero when upright.
   
    """
    R = quat_to_rotmat(quat_normalize(q))
    return np.array([R[0, 2], R[1, 2]])


#numerical jacobian
def numerical_jacobian(f, x0, u0, eps=1e-4):
    n, m = len(x0), len(u0)
    A, B = np.zeros((n, n)), np.zeros((n, m))
    for i in range(n):
        dx = np.zeros(n); dx[i] = eps
        A[:, i] = (f(x0 + dx, u0) - f(x0 - dx, u0)) / (2 * eps)
    for j in range(m):
        du = np.zeros(m); du[j] = eps
        B[:, j] = (f(x0, u0 + du) - f(x0, u0 - du)) / (2 * eps)
    return A, B


#lqr gain 
@dataclass
class LQRGains:
    K: np.ndarray
    altitude: float


def compute_lqr_gains(altitude, mass, params: VehicleParams, Q, R_lqr):
    """
    Linearise 10-dim reduced dynamics around hover.
    State: [x,y,z, vx,vy,vz, tilt_EW, tilt_NS, wy, wx]
    """
    tau_eq = np.clip(mass * G0 / params.thrust_max,
                     params.throttle_min, params.throttle_max)
    u_eq = np.array([tau_eq, 0., 0.])

    def rdyn(x10, u):
        
        tilt_ew, tilt_ns = x10[6], x10[7]
        wy, wx = x10[8], x10[9]
        a_ew = float(tilt_ew)
        a_ns = float(tilt_ns)
        q_approx = quat_normalize(np.array([0., 1., a_ew / 2., -a_ns / 2.]))

        full = np.zeros(18)
        full[0:3]  = x10[0:3]
        full[3:6]  = x10[3:6]
        full[6:10] = q_approx
        full[10]   = wx        # omega[0] = wx (drives tilt_NS)
        full[11]   = wy        # omega[1] = wy (drives tilt_EW)
        full[12]   = 0.        # wz (yaw — not in our 10-dim state)
        full[13]   = mass

        dxdt = equations_of_motion(0., full, u, params)

        #angular acceleration
        F_tb, M_tvc, _ = thrust_force_moment(u, mass, params)
        I_inv  = params.inertia_inv(mass)
        I_diag = np.diag(params.inertia_tensor(mass))
        oc     = np.array([wx, wy, 0.])
        alpha  = I_inv @ (M_tvc - np.cross(oc, I_diag * oc))

        # tilt_EW_dot ≈ wy  (from kinematics above)
        # tilt_NS_dot ≈ wx
        return np.array([
            x10[3], x10[4], x10[5],       
            dxdt[3], dxdt[4], dxdt[5],    
            wy,  wx,                       
            alpha[1], alpha[0]         
        ])

    x_eq = np.zeros(10)
    x_eq[2] = altitude

    try:
        A, B = numerical_jacobian(rdyn, x_eq, u_eq)
        if not (np.isfinite(A).all() and np.isfinite(B).all()):
            return None
        P = solve_continuous_are(A - 1e-4 * np.eye(10), B, Q, R_lqr)
        K = np.linalg.solve(R_lqr, B.T @ P)
        return LQRGains(K=K, altitude=altitude)
    except Exception as e:
        print(f"  [LQR] CARE failed h={altitude:.0f}m: {e}")
        return None


def default_Q_R():
    Q = np.diag([
        0.05, 0.05, 2.0,      
        0.5,  0.5,  500.0,    
        1.0,  1.0,            
        1.0,  1.0             
    ])
    R = np.diag([0.01, 100., 100.])
    return Q, R


#gain-scheduled LQR 
class LQRController:
    ALT_BPS = [1000., 700., 500., 300., 150., 50., 10.]

    def __init__(self, params: VehicleParams, Q=None, R_lqr=None):
        self.params = params
        self.Q, self.R = (Q, R_lqr) if Q is not None else default_Q_R()
        self.gains: List[LQRGains] = []

        self.v0 = 50.    # parabolic reference scale
        self.h0 = 1000.  # entry altitude
        self.kp = 0.08   # throttle / (m/s) vz error

    def precompute_gains(self, verbose=True):
        if verbose:
            print("[LQR] Precomputing gain schedule...")
        mass_sched = {1000.: 125000., 700.: 124500., 500.: 124000.,
                      300.:  123500., 150.: 123000.,  50.: 122500., 10.: 122000.}
        for alt in self.ALT_BPS:
            m = mass_sched[alt]
            if verbose:
                print(f"  h={alt:6.0f}m  m={m:.0f}kg ...", end=" ")
            g = compute_lqr_gains(alt, m, self.params, self.Q, self.R)
            if g:
                self.gains.append(g)
                if verbose:
                    print(f"OK  K_max={np.max(np.abs(g.K)):.3f}")
            else:
                if verbose:
                    print("FAILED")
        if verbose:
            print(f"[LQR] Ready: {len(self.gains)}/{len(self.ALT_BPS)} breakpoints")

    def _get_K(self, alt):
        if not self.gains:
            return np.zeros((3, 10))
        alts = np.array([g.altitude for g in self.gains])
        alt  = np.clip(alt, alts.min(), alts.max())
        idx  = np.clip(np.searchsorted(alts, alt), 1, len(alts) - 1)
        lo, hi = alts[idx - 1], alts[idx]
        if abs(hi - lo) < 1e-3:
            return self.gains[idx - 1].K
        a = (alt - lo) / (hi - lo)
        return (1 - a) * self.gains[idx - 1].K + a * self.gains[idx].K

    def get_action(self, state: np.ndarray, t: float) -> np.ndarray:
        pos   = state[0:3];  vel  = state[3:6]
        q     = quat_normalize(state[6:10])
        omega = state[10:13]; mass = state[13]
        alt   = pos[2]


        vz_ref = -self.v0 * np.sqrt(max(alt, 0.) / self.h0)
        vz_ref = np.clip(vz_ref, -100., -1.5)


        tau_ff   = mass * G0 / self.params.thrust_max
        vz_err   = vz_ref - vel[2]
        throttle = np.clip(tau_ff + self.kp * vz_err,
                           self.params.throttle_min, self.params.throttle_max)

        # Lateral correction expressed as desired tilt setpoints (rad).
        
        tilt   = get_tilt(q)    # [R[0,2], R[1,2]] = nozzle EW/NS components
        kp_att = 6.0;  kd_att = 20.0          # attitude PD (deg/deg, deg/(deg/s))
        kd_vel = 0.001; kp_pos = 0.00003       # lateral (rad per m/s, rad per m)

        # Desired tilt to null lateral velocity + position error
        tEW_des = np.clip(-kd_vel*vel[0] - kp_pos*pos[0], -np.deg2rad(1.5), np.deg2rad(1.5))
        tNS_des = np.clip(+kd_vel*vel[1] + kp_pos*pos[1], -np.deg2rad(1.5), np.deg2rad(1.5))

        # Tilt error 
        err_EW = tilt[0] - tEW_des
        err_NS = tilt[1] - tNS_des

        dp_cmd = -(kp_att * np.rad2deg(err_EW) + kd_att * np.rad2deg(omega[1]))
        dy_cmd = +(kp_att * np.rad2deg(err_NS) + kd_att * np.rad2deg(omega[0]))

        dp = np.clip(np.deg2rad(dp_cmd), -self.params.tvc_max, self.params.tvc_max)
        dy = np.clip(np.deg2rad(dy_cmd), -self.params.tvc_max, self.params.tvc_max)

        return np.array([throttle, dp, dy])

    def reset(self):
        pass
