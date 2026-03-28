'''
dynamics engine 

paper inspired from: AI-enhanced 6dof rocket landing: lqr vs ppo under unc.


scenario (DD-001c): terminal descent from 1000m alttude, -100 m/s vertical velocity.
single raptor engine (2MN), post-entry-burn vehicle mass 125,000 kg. 
Framing: modelling the terminal descent phase following booster entry 
burn completion, consistent with Blackmore (2012) and Szmuk (2018). 

Required avg throttle: 92.5% - fully within [40%, 100%] actuator range.
'''

#vehicle params 
import numpy as np 
from dataclasses import dataclass, field 
from typing import Tuple, Optional


@dataclass 
class VehicleParams:
    ''' 
    starship-class booster, terminal descent phase (DD-001c)
    single center raptor (2MN sea-level). 
    At m=125,000 kh: tau_ff = 61.3% > tau_min=40% --> fully controllable.
    Propellant budget: 5000 kg for approx 13s at 80% throttle.
    
    '''

    # mass
    mass_dry:   float = 120_000.0   # kg
    mass_prop:  float =   5_000.0   # kg  (terminal descent budget)
    mass_total: float = field(init=False)

    # geometry
    length:     float = 70.0        # m
    radius:     float =  4.5        # m
    ref_area:   float = field(init=False)   
    ref_length: float = field(init=False)   
    Ixx_full: float = 1.5e8;  Iyy_full: float = 4.5e9;  Izz_full: float = 4.5e9
    Ixx_dry:  float = 1.2e8;  Iyy_dry:  float = 3.8e9;  Izz_dry:  float = 3.8e9

    # Propulsion — single raptor
    thrust_max:   float = 2_000_000.0  # N
    Isp:          float =       363.0  # s  (sea-level)
    throttle_min: float =         0.4
    throttle_max: float =         1.0

    # TVC
    tvc_max: float = np.deg2rad(6.0)   # pm 6°

    # Aerodynamics
    Cd:       float = 0.5    # drag coeff (subsonic avg)
    Cm_alpha: float = -0.2   
                             # much weaker aero stability than nose-first (DD-006b)
    Cn_beta:  float = -0.2   # yaw moment  / rad sideslip
    xcop:     float = 0.55   # CoP fraction from nose
    xcog:     float = 0.45   # CoM fraction from nose

    def __post_init__(self):
        self.mass_total = self.mass_dry + self.mass_prop
        self.ref_area = np.pi * self.radius ** 2 
        self.ref_length = 2.0 * self.radius 

    def inertia_tensor(self, m: float) -> np.ndarray:
        f= np.clip((m - self.mass_dry) / self.mass_prop, 0., 1.)
        Ixx = self.Ixx_dry + f*(self.Ixx_full - self.Ixx_dry)
        Iyy = self.Iyy_dry + f*(self.Iyy_full - self.Iyy_dry)
        Izz = self.Izz_dry + f*(self.Izz_full - self.Izz_dry)
        return np.diag([Ixx, Iyy, Izz])

    def inertia_inv(self, m: float) -> np.ndarray:
        return np.diag(1.0 / np.diag(self.inertia_tensor(m)))