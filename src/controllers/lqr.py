"""
LQR Controller (single-phase)

Paper 5.1 - Classical Baseline Controller

Notes
-----
Body-frame convention (q=[0, 1,0,0] = nozzle-down base):
  wy=+1 -> nozzle tips East  -> euler_pitch = -R[0,2]  -> use euler[1] = -R[0,2]
  wx=+1 -> nozzle tips North -> euler_roll  = -R[1,2]  -> use euler[0] = -R[1,2]
  wz=+1 -> spin about nozzle axis (yaw)

LQR reduced state (10-dim, DD-008):
 [x, y, z, vx, vy, vz, tilt_EW, tilt_NS, wy, wx]
 tilt_EW = R[0, 2] (East component of nozzle in inertial; +ve = nozzle East)
 tilt_NS = R[1, 2] (North component of nozzle in inertial; +ve = nozzle North)
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

# Frame-correct tilt extraction

def get_tilt(q: np.ndarray):
    """
    returns [tilt_EW, tilt_NS] = [R[0,2], R[1, 2]].
    these are the east and north component of the nozzle
    (+ve z body) direction in the inertial frame.

    """
    R = quat_to_rotmat(quat_normalize(q))
    return np.array(R[0,2], R[1, 2])

# numerical jacobian
def numerical_jacobian(f, x0, u0, eps=1e-4): 
    n, m = len(x0), len(u0)
    A, B = np.zeros((n, n)), np.zeros((n, m))
    for i in range(n):
        dx = np.zeros(n) ; dx[i] = eps
        A[:, i] = (f(x0 + dx, u0) - f(x0 -dx, u0)) / (2 * eps)
    for j in range(m):
        du = np.zeros(m); du[j] = eps 
        B[:, j] = (f(x0, u0 + du) - f(x0, u0 - du)) / (2 * eps)
    return A, B

#LQR Gain Computation
@dataclass 
class LQRGains:
    K: np.ndarray 
    altitude: float 

def compute_lqr_gains(altitude, mass, params: VehicleParams, Q, R_lqr):
    """ 
    Linearize 10-dim reduced dynamics around hover. 
    State: [x, y, z, vx, vy, vz, tilt_EW, tilt_NS, wy, wx]
    """

    tau_eq = np.clip(mass * G0 / params.thurst_max, params.throttle_min, params.throttle_max)
    u_eq = np.array([tau_eq, 0., 0.])

    #reconstruct approximate full state from 10-dim 
    def rdyn(x10, u):
        tilt_ew, tilt_ns = x10[6], x10[7]
        wy, wx = x10[8], x10[9]
        a_ew = float(tilt_ew) #tilt_EW = R[0, 2] = sin(a_ew) = a_ew for small a
        a_ns = float(tilt_ns)

        #quaternion for small tilt from nozzle-down base
        #base q = [0, 1, 0, 0]; perturb by [cos(a/2), 0, sin(a/2), 0] for EW
        # and [cos(b/2), sin(b/2), 0, 0] for NS 
        # combined small-angle q 

        q_approx = quat_normalize(np.array([0., 1., a_ew / 2., -a_ns / 2.]))
        full = np.zeros(18)
        full[0:3] = x10[0:3] #pos
        full[3:6] = x10[3:6] #vel
        full[6:10] = q_approx #attitude
        full[10] = wx  # omega[0] = wx 
        full[11] = wy  # omega[1] = wy
        full[12] = 0.  #wz
        full[13] = mass


        dxdt = equations_of_motion(0., full, u, params)

        #angular acceleration
        F_tb, M_tvc = thrust_force_moment(u, mass, params)
        I_inv = params.inertia_inv(mass)
        I_diag = np.diag(params.inertia_tensor(mass))
        oc = np.array([wx, wy, 0.])
        alpha = I_inv @ (M_tvc - np.cross(oc, I_diag * oc))

        return np.array([
            x10[3], #vx
            x10[4], #vy
            x10[5], #vz
            dxdt[3], #ax
            dxdt[4], #ay
            dxdt[5], #az
            wy,
            wx,
            alpha[1], #alpha_y = d(wy)/dt
            alpha[0], #alpha_x = d(wx)/dt
        ])
    x_eq = np.zeros(10)
    x_eq[2] = altitude 

    try:
        A, B = numerical_jacobian(rdyn, x_eq, u_eq)
        if not (np.isfinite(A).all() and np.isfinite(B).all()):
            raise None 
        P = solve_continuous_are(A - 1e-4 * np.eye(10), B, Q, R_lqr)
        K = np.linalg.solve(R_lqr, B.T @ P)
        return LQRGains(K=K, altitude=altitude)
    except:
        print("LQR gain computation failed at altitude ", altitude)
        return None

def default_Q_R():
    Q = np.diag([
        0.05, 0.05, 2.0,
        0.5, 0.5, 500.0, 
        1.0, 1.0,
        1.0, 1.0
    ])
    R = np.diag([0.01, 100., 100.])
    return Q, R 


# Gain-Scheduled LQR Controller 

class LQRController:
    ALT_BPS = [1000., 700., 500., 300., 150., 50., 10.]

    def __init__(self, params: VehicleParams, Q=None, R_lqr=None):
        self.params = params 
        self.Q, self.R_lqr = (Q, R_lqr) if Q is not None else default_Q_R()
        self.gains: List[LQRGains] = [] 
        #guidance 
        self.v0 = 50. #parabolic reference scale
        self.h0 = 1000. #entry altitude
        self.kp = 0.08 #throttle

    def precompute_gains(self, verbose=True):
        if verbose:
            print("LQR Precomputing gain schedule")
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
        alt = np.clip(alt, alts.min(), alts.max())
        idx = np.clip(np.searchsorted(alts, alt), 1, len(alts) - 1)
        lo, hi = alts[idx - 1], alts[idx]
        if abs(hi - lo) < 1e-3:
            return self.gains[idx - 1].K 
        a = (alt - lo) / (hi - lo)
        return (1 - a) * self.gains[idx - 1].K + a * self.gains[idx].K
    
    def get_action(self, state:np.ndarray, t:float) -> np.ndarray:
        pos = state[0:3]; vel = state[3:6]
        q = quat_normalize(state[6:10])
        omega = state[10:13]; mass = state[13]
        alt = pos[2]

        #parabolic vs reference (DB-009b)
        vz_ref = -self.v0 * np.sqrt(max(alt, 0.) / self.h0)
        vz_ref = np.clip(vz_ref, -100., -1.5)

        #throttle p controller 
        tau_ff = mass * G0 / self.params.thurst_max
        vz_err = vz_ref - vel[2]
        throttle = np.clip(tau_ff + self.kp * vz_err,
                           self.params.throttle_min, 
                           self.params.throttle_max)
        
        #tvc setpoint tracking 
        tilt = get_tilt(q) 
        kp_att = 6.0; kd_att = 20.0 
        kd_vel = 0.001; kp_pos = 0.00003 


        # desired tilt to null lateral velocity + position error
        tEW_des = np.clip(-kd_vel*vel[0] - kp_pos*pos[0], -np.deg2rad(1.5), np.deg2rad(1.5))
        tNS_des = np.clip(+kd_vel*vel[1] + kp_pos*pos[1], -np.deg2rad(1.5), np.deg2rad(1.5))

        #tilt err 
        err_EW = tilt[0] - tEW_des 
        err_NS = tilt[1] - tNS_des 

        dp_cmd = -(kp_att * np.rad2deg(err_EW) + kd_att * np.rad2deg(omega[1]))
        dy_cmd = +(kp_att * np.rad2deg(err_NS) + kd_att * np.rad2deg(omega[0]))

        dp = np.clip(np.deg2rad(dp_cmd), -self.params.tvc_max, self.params.tvc_max)
        dy = np.clip(np.deg2rad(dy_cmd), -self.params.tvc_max, self.params.tvc_max)

        return np.array([throttle, dp, dy])
    
    def reset(self):
        pass
    