"""
landing_env.py  —  Gymnasium environment, Phase 2 / Path A (IC curriculum).

Reward function v4 — unchanged from clean restart:
  [A] Dense quadratic vz tracking  r = -0.002*(vz - vz_ref)²
  [B] Hard vz gate at touchdown    |vz|>10 m/s → -1000, no success credit
  [C] Smoothness 10×               bang-bang ±6° costs -0.88/step
  [D] Quadratic tilt penalty       30° tilt costs -4.2/step
  [E] TVC magnitude penalty        prevents actuator-limit freeze

IC Curriculum (Path A) — 5 stages gated on SUCCESS RATE, not timesteps:

  Stage | Alt   | vz      | Pos offset | Lat vel | Wind
  ------+-------+---------+------------+---------+------
    0   |  50m  | -10 m/s |    0m      |  0 m/s  |  0
    1   | 150m  | -30 m/s |   ±30m     | ±5 m/s  |  0
    2   | 300m  | -55 m/s |   ±75m     | ±10 m/s |  5 m/s
    3   | 600m  | -75 m/s |  ±150m     | ±12 m/s | 10 m/s
    4   | 1000m | -100m/s |  ±200m     | ±15 m/s | 15 m/s  ← full scenario

Stage 0 is easy enough that even a near-random policy will occasionally
land softly, giving PPO immediate gradient signal from the first episodes.
The callback advances stages when 60%+ success is sustained over 200 episodes.
"""

from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.dynamics.dynamics import (
    VehicleParams, RocketSimulator, WindModel,
    quat_normalize, quat_to_rotmat, rk4_step,
)

# ── Constants ─────────────────────────────────────────────────────────────────
MASS_TOTAL   = 125_000.0
THROTTLE_MIN = 0.4
THROTTLE_MAX = 1.0
TVC_MAX_RAD  = np.deg2rad(6.0)
DT           = 0.05
MAX_STEPS    = 2_000
VZ_REF_ALT   = 1_000.     # reference altitude for parabolic vz profile

# Success thresholds (DD-010)
MAX_TOUCH_SPEED = 5.0
MAX_TOUCH_TILT  = 15.0
MAX_POS_ERR     = 150.0

# ── IC curriculum stages: (alt, vz_nom, pos_off, vel_off, wind) ──────────────
IC_STAGES = [
    (  50.,  -10.,   0.,  0.,  0.),   # 0 trivial — was giving 30% success
    ( 150.,  -30.,  30.,  5.,  0.),   # 1 short burn, no wind
    ( 300.,  -55.,  75., 10.,  5.),   # 2 medium + light wind
    ( 600.,  -75., 150., 12., 10.),   # 3 long + moderate wind
    ( 800.,  -88., 175., 13., 12.),   # 4 NEW intermediate — bridges gap to full scenario
    (1000., -100., 200., 15., 15.),   # 5 full scenario
]
N_STAGES = len(IC_STAGES)

# ── Normalisation ─────────────────────────────────────────────────────────────
OBS_SCALE = np.array([
    500., 500., 1000.,
    50.,  50.,  150.,
    1., 1., 1., 1.,
    1., 1., 1.,
    MASS_TOTAL,
    1., 1., 1., 1.,
], dtype=np.float32)


def normalise_obs(s: np.ndarray) -> np.ndarray:
    return np.clip(s.astype(np.float32) / OBS_SCALE, -5., 5.)


def tilt_deg(q: np.ndarray) -> float:
    R = quat_to_rotmat(quat_normalize(q))
    return float(np.degrees(np.arccos(np.clip(-(R @ np.array([0.,0.,1.]))[2], -1., 1.))))


def vz_reference(alt: float) -> float:
    return float(np.clip(-50. * np.sqrt(max(alt, 0.) / VZ_REF_ALT), -110., -1.5))


def physical_action(norm: np.ndarray) -> np.ndarray:
    a = np.clip(norm, -1., 1.)
    return np.array([
        THROTTLE_MIN + (a[0] + 1.) * 0.5 * (THROTTLE_MAX - THROTTLE_MIN),
        a[1] * TVC_MAX_RAD,
        a[2] * TVC_MAX_RAD,
    ], dtype=np.float64)


# ── Reward weights (v4) ───────────────────────────────────────────────────────
class RW:
    SUCCESS   =  2000.0;  CRASH   = -500.0;  TIMEOUT  = -150.0
    VZ_GATE   = -1000.0;  VZ_THRESH = 10.0
    SURVIVAL  =     1.0
    VZ_QUAD   =   -0.002   # [A]
    NEAR_SPD  =   -0.08    # below 200m
    TILT_LIN  =   -0.05    # [D]
    TILT_QUAD =   -0.003   # [D]
    OMEGA     =   -0.5
    HORIZ_ERR =   -0.0003
    HORIZ_VEL =   -0.02
    FUEL      =   -0.0002
    SM_TAU    =   -5.0     # [C]
    SM_TVC    =  -20.0     # [C]
    TVC_MAG   =   -0.5     # [E]


# ── Environment ───────────────────────────────────────────────────────────────
class RocketLandingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        ic_stage     : int   = 0,
        randomise_ics: bool  = True,
        noise_std    : Optional[np.ndarray] = None,
        misalignment : Optional[np.ndarray] = None,
        seed         : Optional[int] = None,
    ):
        super().__init__()
        self.params   = VehicleParams()
        self.rng      = np.random.default_rng(seed)
        self.noise    = noise_std
        self.misalign = misalignment
        self.rand_ics = randomise_ics
        self.ic_stage = int(np.clip(ic_stage, 0, N_STAGES - 1))

        self.observation_space = spaces.Box(-5., 5., (18,), dtype=np.float32)
        self.action_space      = spaces.Box(-1., 1., (3,),  dtype=np.float32)

        self._state    = np.zeros(18)
        self._prev_act = physical_action(np.zeros(3))
        self._prev_m   = MASS_TOTAL
        self._steps    = 0
        self._wind     = None
        self._make_wind()

    # ── Curriculum API ────────────────────────────────────────────────────────
    def set_ic_stage(self, stage: int):
        """Called by TrainingCallback when success rate threshold is met."""
        old = self.ic_stage
        self.ic_stage = int(np.clip(stage, 0, N_STAGES - 1))
        if self.ic_stage != old:
            self._make_wind()

    def get_stage_info(self) -> dict:
        a, vz, p, v, w = IC_STAGES[self.ic_stage]
        return dict(stage=self.ic_stage, alt=a, vz=vz, pos_off=p, vel_off=v, wind=w)

    # ── Gymnasium API ─────────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._make_wind()

        alt, vz_nom, pos_off, vel_off, _ = IC_STAGES[self.ic_stage]

        if self.rand_ics:
            x0  = self.rng.uniform(-pos_off, pos_off) if pos_off > 0 else 0.
            y0  = self.rng.uniform(-pos_off, pos_off) if pos_off > 0 else 0.
            vx0 = self.rng.uniform(-vel_off, vel_off) if vel_off > 0 else 0.
            vy0 = self.rng.uniform(-vel_off, vel_off) if vel_off > 0 else 0.
            vz0 = self.rng.uniform(vz_nom * 1.1, vz_nom * 0.9)
            plim = min(2. + self.ic_stage, 5.)
            p0  = self.rng.uniform(-plim, plim)
        else:
            x0 = y0 = vx0 = vy0 = 0.
            vz0 = vz_nom; p0 = 0.

        sim = RocketSimulator(params=self.params, dt=DT)
        self._state = sim.make_initial_state(
            altitude=alt, vz=vz0, vx=vx0, vy=vy0, pitch_deg=p0)
        self._state[0] = x0
        self._state[1] = y0
        self._prev_act = physical_action(np.zeros(3, dtype=np.float32))
        self._prev_m   = float(self._state[13])
        self._steps    = 0
        return self._obs(), {}

    def step(self, action: np.ndarray):
        phys = physical_action(action)
        self._state, _ = rk4_step(
            self._steps * DT, self._state, phys, DT,
            self.params, self._wind, self.misalign)
        self._steps += 1

        obs_s = self._state.copy()
        if self.noise is not None:
            obs_s += self.rng.standard_normal(18) * self.noise

        alt  = float(self._state[2])
        spd  = float(np.linalg.norm(self._state[3:6]))
        tilt = tilt_deg(self._state[6:10])

        landed  = alt <= 0.
        crashed = ((alt <= 0. and (spd > MAX_TOUCH_SPEED*4 or tilt > MAX_TOUCH_TILT*2))
                   or alt > 6000. or spd > 800. or tilt > 85.)
        timeout    = self._steps >= MAX_STEPS
        terminated = landed or crashed
        truncated  = timeout and not terminated

        rew = self._reward(self._state, phys, landed, crashed, timeout, tilt, spd)
        self._prev_act = phys.copy()
        self._prev_m   = float(self._state[13])

        pos_err = float(np.linalg.norm(self._state[0:2]))
        vz_now  = float(self._state[5])
        success = (landed and not crashed
                   and spd < MAX_TOUCH_SPEED and tilt < MAX_TOUCH_TILT
                   and pos_err < MAX_POS_ERR and abs(vz_now) <= RW.VZ_THRESH)

        info = dict(
            altitude=alt, speed=spd, tilt_deg=tilt, pos_err_m=pos_err,
            vz=vz_now, mass=float(self._state[13]),
            fuel_used=MASS_TOTAL - float(self._state[13]),
            step=self._steps, success=success, ic_stage=self.ic_stage,
            terminated_reason=(
                "success" if success else "crashed" if crashed else
                "landed"  if landed  else "timeout" if timeout else "flying"),
        )
        return self._obs(obs_s), float(rew), terminated, truncated, info

    # ── Internal ──────────────────────────────────────────────────────────────
    def _make_wind(self):
        _, _, _, _, wind_v = IC_STAGES[self.ic_stage]
        if wind_v > 0.:
            self._wind = WindModel(
                V_ref=wind_v, h_ref=10.,
                direction_deg=self.rng.uniform(0., 360.),
                turbulence_intensity=0.10,
                rng=np.random.default_rng(int(self.rng.integers(1_000_000_000))),
            )
        else:
            self._wind = None

    def _obs(self, s=None):
        return normalise_obs(self._state if s is None else s)

    def _reward(self, s, phys, landed, crashed, timeout, tilt, spd) -> float:
        alt = float(s[2]); vel = s[3:6]; pos = s[0:2]
        w = s[10:13]; mass = float(s[13]); r = 0.

        if not (landed or crashed or timeout):
            r += RW.SURVIVAL

        vz_err = float(vel[2]) - vz_reference(alt)
        r += RW.VZ_QUAD * (vz_err ** 2)

        if alt < 200.:
            r += RW.NEAR_SPD * spd

        r += RW.TILT_LIN  * tilt
        r += RW.TILT_QUAD * (tilt ** 2)
        r += RW.OMEGA     * float(np.linalg.norm(w))
        r += RW.HORIZ_ERR * float(np.linalg.norm(pos))
        r += RW.HORIZ_VEL * float(np.linalg.norm(vel[0:2]))
        r += RW.FUEL      * max(self._prev_m - mass, 0.)

        r += RW.SM_TAU  * (phys[0] - self._prev_act[0]) ** 2
        r += RW.SM_TVC  * ((phys[1]-self._prev_act[1])**2 + (phys[2]-self._prev_act[2])**2)
        r += RW.TVC_MAG * (abs(phys[1]) + abs(phys[2]))

        if landed or crashed:
            if abs(float(vel[2])) > RW.VZ_THRESH:
                return r + RW.VZ_GATE + RW.CRASH   # hard gate, no partial credit

            pos_err = float(np.linalg.norm(s[0:2]))
            success = (not crashed and spd < MAX_TOUCH_SPEED
                       and tilt < MAX_TOUCH_TILT and pos_err < MAX_POS_ERR)
            if success:
                prec = max(0., 1. - pos_err/MAX_POS_ERR)
                spdb = max(0., 1. - spd/MAX_TOUCH_SPEED)
                tltb = max(0., 1. - tilt/MAX_TOUCH_TILT)
                r += RW.SUCCESS * (0.5 + 0.5*(prec+spdb+tltb)/3.)
            else:
                r += RW.CRASH * (1.0 if crashed else 0.4)
        elif timeout:
            r += RW.TIMEOUT

        return r