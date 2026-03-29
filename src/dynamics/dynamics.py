"""
rigid-body dynamics engine 

Paper to take notes from: "AI-Enhanced 6DOF Rocket Landing: LQR vs PPO Under Uncertainty"

Scenario (Option B, DD-001c):
  Terminal descent from 1,000 m altitude, -100 m/s vertical velocity.
  Single Raptor engine (2 MN), post-entry-burn vehicle mass 125,000 kg.
  Framing: "We model the terminal descent phase following booster entry
  burn completion, consistent with Blackmore (2012) and Szmuk (2018)."
  Required avg throttle: 92.5% — fully within [40%, 100%] actuator range.

Design Decisions → PAPER_LOG.md DD-001 through DD-010.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional

#vehicle params
@dataclass
class VehicleParams:
    """
    starship-class booster, terminal descent phase (DD-001c).

    single center Raptor (2 MN sea-level).
    At m=125,000 kg: tau_ff=61.3% > tau_min=40% → fully controllable.
    Propellant budget: 5,000 kg for ~13s at 80% throttle.
    """
    # Mass
    mass_dry:   float = 120_000.0   
    mass_prop:  float =   5_000.0   
    mass_total: float = field(init=False)

    # Geometry
    length:     float = 70.0        
    radius:     float =  4.5        
    ref_area:   float = field(init=False)   
    ref_length: float = field(init=False)   

    # Inertia 
    Ixx_full: float = 1.5e8;  Iyy_full: float = 4.5e9;  Izz_full: float = 4.5e9
    Ixx_dry:  float = 1.2e8;  Iyy_dry:  float = 3.8e9;  Izz_dry:  float = 3.8e9

    # Propulsion — single Raptor
    thrust_max:   float = 2_000_000.0  # N
    Isp:          float =       363.0  # s  
    throttle_min: float =         0.4
    throttle_max: float =         1.0

    # TVC
    tvc_max: float = np.deg2rad(6.0)   # pm 6 deg

    # Aerodynamics
    Cd:       float = 0.5    
    Cm_alpha: float = -0.2  
                             # much weaker aero stability than nose-first (DD-006b)
    Cn_beta:  float = -0.2   
    xcop:     float = 0.55   
    xcog:     float = 0.45   

    def __post_init__(self):
        self.mass_total = self.mass_dry + self.mass_prop
        self.ref_area   = np.pi * self.radius**2
        self.ref_length = 2.0 * self.radius

    def inertia_tensor(self, m: float) -> np.ndarray:
        f = np.clip((m - self.mass_dry) / self.mass_prop, 0., 1.)
        Ixx = self.Ixx_dry + f*(self.Ixx_full - self.Ixx_dry)
        Iyy = self.Iyy_dry + f*(self.Iyy_full - self.Iyy_dry)
        Izz = self.Izz_dry + f*(self.Izz_full - self.Izz_dry)
        return np.diag([Ixx, Iyy, Izz])

    def inertia_inv(self, m: float) -> np.ndarray:
        return np.diag(1.0 / np.diag(self.inertia_tensor(m)))


#ISA Atmosphere
def isa_atmosphere(alt_m: float) -> Tuple[float, float, float]:
    T0=288.15; P0=101325.; L=0.0065; g0=9.80665; R=287.05
    h = float(np.clip(alt_m, 0., 80_000.))
    if h <= 11_000.:
        T = T0 - L*h
        P = P0*(T/T0)**(g0/(L*R))
    else:
        T11 = T0 - L*11_000.; P11 = P0*(T11/T0)**(g0/(L*R))
        T = T11;  P = P11*np.exp(-g0*(h-11_000.)/(R*T11))
    rho = P/(R*T)
    return rho, P, T


#quaternion
def quat_normalize(q):
    return q / np.linalg.norm(q)

def quat_enforce_convention(q):
    """Enforce q0 >= 0 (resolve double-cover ambiguity, DD-002)."""
    return q if q[0] >= 0 else -q

def quat_to_rotmat(q) -> np.ndarray:
    q0,q1,q2,q3 = q/np.linalg.norm(q)
    return np.array([
        [1-2*(q2**2+q3**2),  2*(q1*q2-q0*q3),   2*(q1*q3+q0*q2)],
        [2*(q1*q2+q0*q3),    1-2*(q1**2+q3**2),  2*(q2*q3-q0*q1)],
        [2*(q1*q3-q0*q2),    2*(q2*q3+q0*q1),    1-2*(q1**2+q2**2)]
    ])

def quat_kinematics(q, omega):
    q0,q1,q2,q3 = q
    Xi = np.array([[-q1,-q2,-q3],[q0,-q3,q2],[q3,q0,-q1],[-q2,q1,q0]])
    return 0.5 * Xi @ omega


#wind model
@dataclass
class WindModel:
    """
    Power-law wind shear + Gaussian turbulence (DD-007).
    V(h) = V_ref · (h/h_ref)^α,  α=0.143 (open terrain).
    """
    V_ref: float = 0.0
    h_ref: float = 10.0
    alpha_shear: float = 0.143
    direction_deg: float = 0.0
    turbulence_intensity:float = 0.10
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng())

    def wind_enu(self, alt: float) -> np.ndarray:
        h  = max(float(alt), 0.1)
        speed = self.V_ref * (h/self.h_ref)**self.alpha_shear
        turb  = self.rng.normal(0, speed*self.turbulence_intensity, 3)
        turb[2] = 0.
        ang = np.deg2rad(self.direction_deg)
        return np.array([speed*np.cos(ang), speed*np.sin(ang), 0.]) + turb


#aero
def aero_forces_moments(v_enu, q_att, alt, p: VehicleParams,
                         wind: Optional['WindModel']=None):
    rho,_,_ = isa_atmosphere(alt)
    v_wind  = wind.wind_enu(alt) if wind else np.zeros(3)
    v_rel   = v_enu - v_wind
    V       = np.linalg.norm(v_rel)
    if V < 1e-3:
        return np.zeros(3), np.zeros(3)

    q_dyn = 0.5*rho*V**2
    F_drag = -q_dyn * p.Cd * p.ref_area * (v_rel/V)   # inertial

    R_T    = quat_to_rotmat(q_att).T                   # inertial→body
    v_body = R_T @ v_rel
    Vx,Vy,Vz = v_body
    # AoA/sideslip from nozzle axis (+z_body)
    alpha = np.arctan2(Vx,  abs(Vz) + 1e-6)
    beta  = np.arctan2(Vy,  abs(Vz) + 1e-6)
    M_pitch = q_dyn * p.ref_area * p.ref_length * p.Cm_alpha * alpha
    M_yaw   = q_dyn * p.ref_area * p.ref_length * p.Cn_beta  * beta
    return F_drag, np.array([0., M_pitch, M_yaw])


#thrust force
def thrust_force_moment(action, mass, p: VehicleParams,
                         misalign: Optional[np.ndarray]=None):
    """
    TVC model (DD-005):
      T_body = T·[sin δp cos δy,  sin δy,  −cos δp cos δy]
      M_tvc  = r_nozzle × T_body,   r_nozzle = [0,0,L/2]
    """
    g0      = 9.80665
    tau     = np.clip(action[0], p.throttle_min, p.throttle_max)
    dp      = np.clip(action[1], -p.tvc_max, p.tvc_max)
    dy      = np.clip(action[2], -p.tvc_max, p.tvc_max)
    if misalign is not None:
        dp += misalign[0]; dy += misalign[1]

    T_mag = tau * p.thrust_max
    F_body = T_mag * np.array([
         np.sin(dp)*np.cos(dy),
         np.sin(dy),
        -np.cos(dp)*np.cos(dy)
    ])
    r_noz  = np.array([0., 0., p.length/2.])
    M_tvc  = np.cross(r_noz, F_body)
    mdot   = -T_mag / (p.Isp * g0)
    return F_body, M_tvc, mdot


#equations of motion
def equations_of_motion(t, state, action, p: VehicleParams,
                          wind=None, misalign=None) -> np.ndarray:
    g0    = 9.80665
    r     = state[0:3];  v = state[3:6]
    q     = quat_normalize(state[6:10]);  omega = state[10:13]
    mass  = max(state[13], p.mass_dry+1.)
    alt   = r[2]
    R = quat_to_rotmat(q)
    F_grav = np.array([0., 0., -mass*g0])
    F_tb, M_tvc, mdot = thrust_force_moment(action, mass, p, misalign)
    F_thrust = R @ F_tb
    F_drag, M_aero = aero_forces_moments(v, q, alt, p, wind)
    rdot   = v
    vdot   = (F_thrust + F_drag + F_grav) / mass
    I      = p.inertia_tensor(mass)
    I_inv  = p.inertia_inv(mass)
    M_tot  = M_tvc + M_aero
    wdot   = I_inv @ (M_tot - np.cross(omega, I@omega))
    qdot   = quat_kinematics(q, omega)
    if mass <= p.mass_dry + 1.:
        mdot = 0.
    dxdt        = np.zeros_like(state)
    dxdt[0:3]   = rdot;  dxdt[3:6]  = vdot
    dxdt[6:10]  = qdot;  dxdt[10:13]= wdot
    dxdt[13]    = mdot
    return dxdt


#RK4 integrator
def rk4_step(t, state, action, dt, p, wind=None, misalign=None):
    def f(t_, s_): return equations_of_motion(t_, s_, action, p, wind, misalign)
    k1 = f(t,       state)
    k2 = f(t+dt/2,  state+dt/2*k1)
    k3 = f(t+dt/2,  state+dt/2*k2)
    k4 = f(t+dt,    state+dt   *k3)
    ns = state + (dt/6.)*(k1+2*k2+2*k3+k4)
    ns[6:10] = quat_enforce_convention(quat_normalize(ns[6:10]))
    ns[13]   = max(ns[13], p.mass_dry)
    return ns, t+dt


#simulator
class RocketSimulator:
    def __init__(self, params=None, wind_model=None, dt=0.05, misalignment=None):
        self.params  = params or VehicleParams()
        self.wind    = wind_model
        self.dt      = dt
        self.misalign= misalignment

    def make_initial_state(self, altitude=1000., vz=-100., vx=0., vy=0.,
                            pitch_deg=0., omega_pitch=0.,
                            random_offset=False, rng=None) -> np.ndarray:
        """
        Default: 1,000 m altitude, -100 m/s vertical (Option B, DD-001c).
        """
        if rng is None: rng = np.random.default_rng()
        x_off = rng.uniform(-200,200) if random_offset else 0.
        y_off = rng.uniform(-200,200) if random_offset else 0.
        vx_r  = rng.uniform(-15,15)   if random_offset else vx
        vy_r  = rng.uniform(-15,15)   if random_offset else vy

        pr = np.deg2rad(pitch_deg)
        q_base = np.array([0.,1.,0.,0.])   # nozzle-down
        if abs(pr) > 1e-6:
            dq = np.array([np.cos(pr/2),0.,np.sin(pr/2),0.])
            b0,b1,b2,b3 = q_base; p0,p1,p2,p3 = dq
            q_init = np.array([p0*b0-p1*b1-p2*b2-p3*b3,
                                p0*b1+p1*b0+p2*b3-p3*b2,
                                p0*b2-p1*b3+p2*b0+p3*b1,
                                p0*b3+p1*b2-p2*b1+p3*b0])
        else:
            q_init = q_base.copy()
        q_init = quat_enforce_convention(quat_normalize(q_init))

        s = np.zeros(18)
        s[0],s[1],s[2] = x_off, y_off, altitude
        s[3],s[4],s[5] = vx_r, vy_r, vz
        s[6:10]         = q_init
        s[11]           = omega_pitch
        s[13]           = self.params.mass_total
        return s

    def run(self, initial_state, controller, t_max=120., noise_std=None) -> dict:
        state = initial_state.copy()
        t     = 0.
        if hasattr(controller, 'reset'): controller.reset()

        times=[t]; states=[state.copy()]; actions=[]
        reason = "timeout"

        while t < t_max:
            obs = state.copy()
            if noise_std is not None:
                obs += np.random.randn(18) * noise_std

            action = controller.get_action(obs, t)
            actions.append(action.copy())

            state, t = rk4_step(t, state, action, self.dt,
                                 self.params, self.wind, self.misalign)
            times.append(t); states.append(state.copy())

            alt   = state[2]
            speed = np.linalg.norm(state[3:6])
            if alt  <= 0.:     reason="landed";           break
            if alt  > 5_000.:  reason="altitude_exceeded";break
            if speed> 1_000.:  reason="speed_exceeded";   break

        actions.append(actions[-1] if actions else np.zeros(3))

        T = np.array(times); S = np.array(states); A = np.array(actions)
        return {"times":T,"states":S,"actions":A,
                "reason":reason,"metrics":self._metrics(S,A,T,reason),
                "params":self.params}

    def _metrics(self, S, A, T, reason) -> dict:
        f   = S[-1]
        pos_err = np.linalg.norm(f[0:2])
        spd     = np.linalg.norm(f[3:6])
        vz      = f[5]
        R_f     = quat_to_rotmat(quat_normalize(f[6:10]))
        nozzle  = R_f @ np.array([0.,0.,1.])
        tilt    = np.rad2deg(np.arccos(np.clip(-nozzle[2],-1.,1.)))
        fuel    = S[0,13] - f[13]
        max_w   = np.max(np.linalg.norm(S[:,10:13],axis=1))
        # Success criteria (paper §4.2, DD-010):
        #   - touchdown speed < 5 m/s
        #   - tilt < 15 deg
        #   - horizontal error < 150 m  (Starship drone ship ~80×50m, 150m generous)
        success = (
            reason == "landed"
            and spd  < 5.
            and tilt < 15.
            and pos_err < 150.
        )
        return dict(success=success, reason=reason,
                    landing_pos_err=pos_err, touchdown_vel=spd,
                    touchdown_vz=vz, tilt_deg=tilt,
                    fuel_consumed=fuel, max_omega_deg_s=np.rad2deg(max_w),
                    flight_time=T[-1])
