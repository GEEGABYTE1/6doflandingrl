'''
gymnasium env for 6dof starship class rocket landing

phase 2: ppo training wrapper around phase 1 dynamics engine

Observation : 18-dim  [r(3), v(3), q(4), ω(3), m(1), reserved(4)] — normalised
Action      : 3-dim   normalised [-1, 1], rescaled internally to physical limits
Reward      : r_landing + r_attitude + r_fuel + r_smoothness + r_survival

'''

import numpy as np 
import gymnasium as gym 
from gymnasium import spaces 
from typing import Optional, Tuple, Dict, Any 

import sys, os 
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.dynamics.dynamics import (
    VehicleParams, RocketSimulator, WindModel,
    quat_normalize, quat_to_rotmat, rk4_step
)


#constants

G0            = 9.80665
MASS_DRY      = 120_000.0
MASS_PROP     =   5_000.0
MASS_TOTAL    = 125_000.0
THRUST_MAX    = 2_000_000.0
THROTTLE_MIN  = 0.4
THROTTLE_MAX  = 1.0
TVC_MAX_RAD   = np.deg2rad(6.0)
ISP           = 363.0

INIT_ALT      = 1_000.0    
INIT_VZ       = -100.0      
DT            = 0.05        
MAX_STEPS     = 2_000       

# Success thresholds (DD-010)
MAX_TOUCH_SPEED = 5.0       
MAX_TOUCH_TILT  = 15.0      
MAX_POS_ERR     = 150.0     

OBS_SCALE = np.array([
    500., 500., 1000.,        
    50.,  50.,  150.,         
    1.,   1.,   1.,   1.,      
    1.,   1.,   1.,            
    MASS_TOTAL,                 
    1.,   1.,   1.,   1.       
    
], dtype=np.float32)

def normalise_obs(state:np.ndarray) -> np.ndarray:
    obs = state.astype(np.float32) / OBS_SCALE
    return np.clip(obs, -5.0, 5.0)

def _tilt_deg(q:np.ndarray) -> float:
    "0 deg is upright"
    R = quat_to_rotmat(quat_normalize(q))
    nozzle_inertial = R @ np.array([0., 0., 1.]) #z_body -> inertial
    cos_tilt = np.clip(-nozzle_inertial[2], -1., 1.)
    return float(np.degrees(np.arccos(cos_tilt)))


#reward shaping weights 

class RewardWeights:

    SUCCESS_BONUS    =  500.0   # clean landing bonus
    CRASH_PENALTY    = -200.0   # hit ground too fast / too tilted
    TIMEOUT_PENALTY  = -100.0   # didn't land
    W_HORIZ_ERR      =  -0.002  # per metre of horizontal error
    W_VZ_TRACK       =  -0.5    # per (m/s) deviation from reference vz profile
    W_HORIZ_VEL      =  -0.05   # per (m/s) lateral speed
    W_TILT           =  -0.3    # per degree of tilt
    W_OMEGA          =  -0.2    # per (rad/s) angular rate magnitude
    W_FUEL           =  -0.005  # per kg consumed per step
    W_SMOOTH_THROTTLE = -2.0    # per (Δτ)²
    W_SMOOTH_TVC      = -50.0   # per (Δδ)²  (TVC in rad → tight penalty)
    SURVIVAL          =  0.05   # small reward just for being alive & descending


#gym env

class RocketLandingEnv(gym.Env):
    '''
    Observation space : Box(18,)  normalised
    Action space      : Box(3,) in [-1, 1]
    '''

    metadata = {"render_modes": []}

    def __init__(
            self,
            wind_level: float=0.0, 
            randomise_ics:bool = True,
            noise_std: Optional[np.ndarray] = None,
            misalignment: Optional[np.ndarray] = None,
            reward_weights: Optional[RewardWeights] = None,
            seed: Optional[int] = None
    ): 
        super().__init__() 

        self.params = VehicleParams()
        self.rng = np.random.default_rng(seed)
        self.wind_level = wind_level 
        self.randomise = randomise_ics 
        self.noise_std = noise_std 
        self.misalign = misalignment 
        self.W = reward_weights or RewardWeights() 


        self.observation_space = spaces.Box(
            low=-5.0, high=5.0, shape=(18,), dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        #episode state 
        self._state: np.ndarray = np.zeros(18)
        self._prev_action: np.ndarray = np.zeros(3) 
        self._prev_mass: float = MASS_TOTAL 
        self._step_count: int = 0 
        self._wind: Optional[WindModel] = None
        self._make_wind() 

    def set_wind_level(self, v_ref:float):
        self.wind_level = v_ref 
        self._make_wind()

    def reset (
            self, 
            *, 
            seed: Optional[int] = None,
            options: Optional[dict] = None,

    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        self._make_wind() 

        if self.randomise:
            x_off = self.rng.uniform(-200., 200.)
            y_off = self.rng.uniform(-200., 200.)
            vx0 = self.rng.uniform(-15., 15.)
            vy0 = self.rng.uniform(-15., -15.)
            vz0 = self.rng.uniform(-110., -90.)
            pitch0 = self.rng.uniform(-5., 5.)

        else:
            x_off = y_off = vx0 = vy0 = 0. 
            vz0 = INIT_VZ
            pitch0= 0. 
        sim = RocketSimulator(params=self.params, dt=DT) 
        self._state = sim.make_initial_state(
            altitude=INIT_ALT,
            vz=vz0, vx=vx0, vy=vy0,
            pitch_deg = pitch0,
            random_offset = False,
        )

        self._state[0] = x_off 
        self._state[1] = y_off 

        self._prev_action = self._physical_action(np.zeros(3, dtype=np.float32))
        self._prev_mass = float(self.state[13])
        self._step_count = 0 

        return self._observe(), {} 
    

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        phys = self._physical_action(action)

        self._state, _ = rk4_step(
            t= self._step_count * DT,
            state = self._state, 
            action = phys, 
            dt = DT,
            p = self.params, 
            wind = self._wind,
            misalign = self.misalign
        )

        #otional sensor noise 
        obs_state = self._state.copy() 
        if self.noise_std is not None:
            obs_state += self.rng.standard_normal(18) * self.noise_std 

        self._step_count += 1

        #termination logic 
        alt = float(self._state[2])
        speed = float(np.linalg.norm(self._state[3:6]))
        tilt = _tilt_deg(self._state[6:10])
        landed = alt <= 0.0 
        crashed = ( 
            (alt <= 0.0 and (speed > MAX_TOUCH_SPEED * 3 or tilt > MAX_TOUCH_TILT * 2))
            or alt > 5_000.
            or speed > 800.
            or tilt > 80.
        )
        timeout = self._step_count >= MAX_STEPS 
        terminated = landed or crashed 
        truncated = timeout and not terminated 

        reward = self._compute_reward(
            state = self.state,
            obs_state = obs_state,
            phys_action = phys,
            landed = landed,
            crashed = crashed,
            timeout = timeout, 
            tilt_deg = tilt,
            speed = speed

        )

        self._prev_action = phys.copy() 
        self._prev_mass = float(self._state[13])
        pos_err = float(np.linalg.norm(self._state[0:2]))
        success = ( 
            landed and not crashed
            and speed   < MAX_TOUCH_SPEED
            and tilt    < MAX_TOUCH_TILT
            and pos_err < MAX_POS_ERR
        )
        info = dict(
            altitude    = alt,
            speed       = speed,
            tilt_deg    = tilt,
            pos_err_m   = pos_err,
            vz          = float(self._state[5]),
            mass        = float(self._state[13]),
            fuel_used   = MASS_TOTAL - float(self._state[13]),
            step        = self._step_count,
            success     = success,
            terminated_reason = (
                "success"  if success  else
                "crashed"  if crashed  else
                "landed"   if landed   else
                "timeout"  if timeout  else
                "flying"
            ),
        )
        return self._observe(obs_state), float(reward), terminated, truncated, info
    
    def _make_wind(self):
        if self.wind_level > 0.:
            direction = self.rng.uniform(0., 360.)
            self._wind = WindModel(
                V_ref = self.wind_level,
                h_ref = 10., 
                direction_deg=direction,
                turbulence_intensity=0.10, 
                rng = np.random.default_rng(int(self.rng.integers(1e9)))
            )
        else:
            self._wind = None

    def _observe(self, state:Optional[np.ndarray] = None) -> np.ndarray:
        s = self._state if state is None else state 
        return normalise_obs(s)
    
    def _physical_action(self, norm_action: np.ndarray) -> np.ndarray: 
        '''
        norm[0] in [-1, 1] -> [0.4, 1.0]
        norm [1] in [-1, 1] -> [-6 deg, + 6 deg]
        norm [2] in [-1, 1] -> [-6 deg, + 6deg]
        
        '''
        a = np.clip(norm_action, -1., 1.)
        tau = THROTTLE_MIN + (a[0] + 1.) * 0.5 * (THROTTLE_MAX - THROTTLE_MIN)
        dp = a[1] * TVC_MAX_RAD
        dy = a[2] * TVC_MAX_RAD
        return np.array([tau, dp, dy], dtype=np.float64)
    
    def _vz_reference(self, alt:float) -> float:
        '''
        parabolic vz ref - same as lqr guidance law
        '''
        v0 = 50.; h0=INIT_ALT 
        vz_ref = -v0 * np.sqrt(max(alt, 0.) / h0)
        return float(np.clip(vz_ref, -110., -1.5))
    
    def _compute_reward(
            self, state, obs_state, phys_action, landed, crashed, timeout, tilt_deg, speed 
    ) -> float:
        W = self.W 
        r = 0.0 
        alt = float(state[2])
        pos = state[0:2]
        vel = state[3:6]
        omega = state[10:13]
        mass = float(state[13])


        if not (landed or crashed or timeout):
            r += W.SURVIVAL 

        horiz_err = float(np.linalg.norm(pos))
        vz_err = abs(vel[2] - self._vz_reference(alt))
        horiz_err = float(np.linalg.norm(vel[0:2]))

        proximity = 1.0 + 2.0 * (1.0 - alt / INIT_ALT)
        r += W.W_HORIZ_ERR  * horiz_err  * proximity
        r += W.W_VZ_TRACK   * vz_err
        r += W.W_HORIZ_VEL  * horiz_vel  * proximity

        r += W.W_TILT  * tilt_deg
        r += W.W_OMEGA * float(np.linalg.norm(omega))
        fuel_step = self._prev_mass - mass
        r += W.W_FUEL * max(fuel_step, 0.)
        delta_throttle = phys_action[0] - self._prev_action[0]
        delta_tvc_p    = phys_action[1] - self._prev_action[1]
        delta_tvc_y    = phys_action[2] - self._prev_action[2]

        r += W.W_SMOOTH_THROTTLE * delta_throttle**2
        r += W.W_SMOOTH_TVC      * (delta_tvc_p**2 + delta_tvc_y**2)


        if landed or crashed:
            pos_err = float(np.linalg.norm(state[0:2]))
            success = (
                not crashed 
                and speed < MAX_TOUCH_SPEED 
                and tilt_deg < MAX_TOUCH_TILT
                and pos_err < MAX_POS_ERR
            )

            if success:
                precision_bonus = max(0., 1. - pos_err / MAX_POS_ERR)
                speed_bonus = max(0., 1. - speed / MAX_TOUCH_SPEED) 
                tilt_bonus = max(0., 1. - tilt_deg / MAX_TOUCH_TILT)
                r += W.SUCCESS_BONUS * (0.5 + 0.5 * (precision_bonus + speed_bonus + tilt_bonus)/3.)
            elif crashed:
                r += W.CRASH_PENALTY 
            else:
                r += W.CRASH_PENALTY * 0.5 
            
        elif timeout:
            r += W.TIMEOUT_PENALTY
        return r


        
