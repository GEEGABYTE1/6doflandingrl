"""
RocketLandingEnv — Gymnasium Environment for 6DOF Rocket Landing
=================================================================
Paper §5.2 — RL Environment Design

Wraps the 6DOF RocketSimulator in a Gymnasium-compatible interface
for PPO training via Stable Baselines 3.

Design Decisions (DD-012 through DD-016):
------------------------------------------
DD-012 | Observation space (18-dim, normalised to ~[-1, 1]):
    [0:3]   position       / [1000, 1000, 1000] m
    [3:6]   velocity       / [150, 150, 150]    m/s
    [6:10]  quaternion     (already unit norm, no scaling)
    [10:13] angular rates  / [1.0, 1.0, 1.0]   rad/s
    [13]    mass fraction  (m - m_dry) / m_prop  ∈ [0,1]
    [14:17] prev_action    (3-dim, for smoothness)
    [17]    reserved → normalised altitude ∈ [0,1]
  Total: 18-dim. Includes previous action so policy can learn smoothness.

DD-013 | Action space (3-dim, normalised to [-1, 1]):
    [0]  throttle  → mapped to [0.4, 1.0]
    [1]  TVC pitch → mapped to [-6°, +6°]
    [2]  TVC yaw   → mapped to [-6°, +6°]
  SB3 PPO uses tanh squashing by default — normalised space is critical.

DD-014 | Reward function (fixed from Phase 2 reward-hacking lesson):
    Dense (every step):
      r_survive   = +1.0                           (stay alive)
      r_vz        = -0.05 * (vz - vz_ref(h))²     (track decel curve)
      r_tilt      = -50.0 * tilt²                  (penalise tilt)
      r_smooth    = -2.0  * ||Δu||²               (penalise action jerk)
      r_pos       = -0.001 * horiz_error²          (soft position nudge)
    Terminal:
      r_land      = +500                           if success
      r_crash_vz  = -1000 * clip(|vz|-5, 0, ∞)   if landed too fast
      r_crash_tilt= -500  * tilt_deg              if landed tilted
      r_timeout   = -200                           if timeout

    Key fix: r_crash_vz makes high-speed landing catastrophically
    penalised — the agent CANNOT earn reward by racing to the pad.
    r_vz dense reward forces it to track the parabolic decel curve
    throughout the episode, not just at the end.

DD-015 | Episode termination:
    - altitude ≤ 0          (landed — may be success or crash)
    - altitude > 1500 m     (escaped upward)
    - |speed| > 200 m/s     (structural failure)
    - t > t_max = 60 s      (timeout — generous to allow slow approaches)

DD-016 | Randomised initial conditions for training diversity:
    altitude:    Uniform[800, 1200] m
    vz:          Uniform[-120, -80] m/s
    vx, vy:      Normal(0, 5) m/s
    pitch, yaw:  Normal(0, 5) deg
    wind speed:  Uniform[0, 15] m/s  (randomised per episode)
    wind dir:    Uniform[0, 360] deg
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from dynamics.dynamics import (
    VehicleParams, RocketSimulator, WindModel,
    quat_normalize, quat_to_rotmat
)


# ══════════════════════════════════════════════════════
#  Normalisation constants
# ══════════════════════════════════════════════════════
OBS_SCALE = np.array([
    1000., 1000., 1000.,   # position      (3)
    150.,  150.,  150.,    # velocity      (3)
    1.,    1.,    1., 1.,  # quaternion    (4)
    1.0,   1.0,   1.0,     # angular rates (3)
    1.,                    # mass fraction (1)
    1.,    1.,    1.,      # prev_action   (3)
], dtype=np.float32)  # total = 17


def tilt_from_quaternion(q: np.ndarray) -> float:
    """Returns tilt angle in radians from nozzle-down upright."""
    R      = quat_to_rotmat(quat_normalize(q))
    nozzle = R @ np.array([0., 0., 1.])
    return float(np.arccos(np.clip(-nozzle[2], -1., 1.)))


def vz_reference(altitude: float, v0: float = 50., h0: float = 1000.) -> float:
    """Parabolic deceleration reference (same as LQR guidance, DD-009b)."""
    return -v0 * np.sqrt(max(altitude, 0.) / h0)


# ══════════════════════════════════════════════════════
#  Environment
# ══════════════════════════════════════════════════════
class RocketLandingEnv(gym.Env):
    """
    6DOF Rocket Landing — Gymnasium Environment.

    Observation: 18-dim normalised vector (see DD-012)
    Action:      3-dim normalised to [-1, 1]   (see DD-013)
    Reward:      shaped dense + terminal        (see DD-014)
    """

    metadata = {'render_modes': []}

    def __init__(
        self,
        dt:           float = 0.05,
        t_max:        float = 60.0,
        randomise_ic: bool  = True,
        randomise_wind:bool = True,
        fixed_wind_speed:float = None,   # override for eval
        seed:         int   = None,
    ):
        super().__init__()

        self.dt            = dt
        self.t_max         = t_max
        self.randomise_ic  = randomise_ic
        self.randomise_wind= randomise_wind
        self.fixed_wind    = fixed_wind_speed
        self.params        = VehicleParams()

        # Gymnasium spaces
        self.observation_space = spaces.Box(
            low=-np.ones(17, dtype=np.float32),
            high=np.ones(17, dtype=np.float32),
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-np.ones(3, dtype=np.float32),
            high=np.ones(3, dtype=np.float32),
            dtype=np.float32
        )

        # Internal state
        self._rng      = np.random.default_rng(seed)
        self._sim      = None
        self._state    = None
        self._t        = 0.
        self._prev_act = np.zeros(3, dtype=np.float32)
        self._step_n   = 0

    # ── Helpers ───────────────────────────────────────
    def _make_wind(self) -> WindModel:
        if self.fixed_wind is not None:
            speed = float(self.fixed_wind)
        elif self.randomise_wind:
            speed = float(self._rng.uniform(0., 15.))
        else:
            speed = 0.
        direction = float(self._rng.uniform(0., 360.))
        return WindModel(
            V_ref=speed,
            direction_deg=direction,
            turbulence_intensity=0.10,
            rng=np.random.default_rng(int(self._rng.integers(0, 2**31)))
        )

    def _action_to_physical(self, action: np.ndarray) -> np.ndarray:
        """Map [-1,1]³ → [τ, δp_rad, δy_rad]."""
        p = self.params
        tau = p.throttle_min + 0.5*(action[0]+1.)*(p.throttle_max - p.throttle_min)
        dp  = action[1] * p.tvc_max
        dy  = action[2] * p.tvc_max
        return np.array([tau, dp, dy], dtype=np.float64)

    def _get_obs(self) -> np.ndarray:
        s   = self._state
        alt = max(s[2], 0.)
        mf  = np.clip(
            (s[13] - self.params.mass_dry) / self.params.mass_prop, 0., 1.
        )
        raw = np.concatenate([
            s[0:3],                      # position
            s[3:6],                      # velocity
            quat_normalize(s[6:10]),     # quaternion
            s[10:13],                    # angular rates
            [mf],                        # mass fraction
            self._prev_act,              # prev action
        ]).astype(np.float32)
        return np.clip(raw / OBS_SCALE, -1., 1.)

    def _compute_reward(
        self,
        state:     np.ndarray,
        action:    np.ndarray,
        prev_act:  np.ndarray,
        done:      bool,
        reason:    str,
    ) -> float:

        alt   = state[2]
        vz    = state[5]
        horiz = np.linalg.norm(state[0:2])
        tilt  = tilt_from_quaternion(state[6:10])
        tilt_deg = np.rad2deg(tilt)

        # ── Dense rewards (every step) ─────────────────
        # DD-014b: Rescaled ~100x from v1. VecNormalize handles further scaling.
        # 1. Survival bonus
        r_survive = 0.1

        # 2. vz tracking — follow parabolic decel reference
        vz_ref = vz_reference(alt)
        r_vz   = -0.0005 * (vz - vz_ref) ** 2

        # 3. Tilt penalty
        r_tilt = -5.0 * tilt ** 2   # increased 10x — tilt must dominate smoothness

        # 4. Action smoothness
        delta_u  = action - prev_act
        r_smooth = -0.002 * float(np.dot(delta_u, delta_u))  # reduced 10x — stop gimbal lock

        # 5. Soft horizontal position nudge
        r_pos = -0.00001 * horiz ** 2

        r_dense = r_survive + r_vz + r_tilt + r_smooth + r_pos

        if not done:
            return float(r_dense)

        # ── Terminal rewards ───────────────────────────
        spd = np.linalg.norm(state[3:6])

        if reason == 'landed':
            success = spd < 5. and tilt_deg < 15. and horiz < 150.

            if success:
                precision_bonus = max(0., 150. - horiz) * 0.1
                softness_bonus  = max(0., 5. - spd) * 4.
                r_land = 100. + precision_bonus + softness_bonus
            else:
                r_land = 0.

            # Crash penalties — scaled to be painful but not overwhelming
            vz_excess    = max(0., abs(vz) - 5.)
            r_crash_vz   = -10. * vz_excess

            tilt_excess  = max(0., tilt_deg - 15.)
            r_crash_tilt = -5.  * tilt_excess

            # Penalise horizontal velocity at landing (was 32 m/s in v4)
            horiz_spd    = float(np.linalg.norm(state[3:5]))
            r_crash_horiz = -5. * max(0., horiz_spd - 3.)

            return float(r_dense + r_land + r_crash_vz + r_crash_tilt + r_crash_horiz)

        elif reason == 'timeout':
            return float(r_dense - 20.)

        else:
            return float(r_dense - 50.)

    # ── Gymnasium API ──────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        wind = self._make_wind()
        mis  = self._rng.uniform(-1., 1., 2) * np.deg2rad(0.2)

        self._sim = RocketSimulator(
            params=VehicleParams(),
            wind_model=wind,
            dt=self.dt,
            misalignment=mis
        )

        if self.randomise_ic:
            # ── Curriculum learning (DD-017) ──────────────────────
            # Stage 0 (0–500k steps):   easy   — low alt, low speed
            # Stage 1 (500k–1.5M):      medium — mid alt, mid speed
            # Stage 2 (1.5M+):          full   — full randomisation
            # CurriculumCallback in train_ppo.py advances the stage.
            stage = getattr(self, '_curriculum_stage', 0)

            if stage == 0:
                # Easy: agent can accidentally land and get reward signal
                alt   = float(self._rng.uniform(80.,   200.))
                vz    = float(self._rng.uniform(-20.,  -8.))
                vx    = float(self._rng.normal(0.,     0.5))
                vy    = float(self._rng.normal(0.,     0.5))
                pitch = float(self._rng.normal(0.,     1.0))
                ox    = float(self._rng.normal(0.,     0.005))
            elif stage == 1:
                # Medium
                alt   = float(self._rng.uniform(200.,  600.))
                vz    = float(self._rng.uniform(-60.,  -20.))
                vx    = float(self._rng.normal(0.,     2.))
                vy    = float(self._rng.normal(0.,     2.))
                pitch = float(self._rng.normal(0.,     3.))
                ox    = float(self._rng.normal(0.,     0.01))
            else:
                # Full difficulty
                alt   = float(self._rng.uniform(800.,  1200.))
                vz    = float(self._rng.uniform(-120., -80.))
                vx    = float(self._rng.normal(0.,     5.))
                vy    = float(self._rng.normal(0.,     5.))
                pitch = float(self._rng.normal(0.,     5.))
                ox    = float(self._rng.normal(0.,     0.02))
        else:
            alt, vz, vx, vy, pitch, ox = 1000., -100., 0., 0., 0., 0.

        self._state    = self._sim.make_initial_state(
            altitude=alt, vz=vz, vx=vx, vy=vy,
            pitch_deg=pitch, omega_pitch=ox
        )
        self._t        = 0.
        self._prev_act = np.zeros(3, dtype=np.float32)
        self._step_n   = 0

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, -1., 1.).astype(np.float32)
        phys   = self._action_to_physical(action)

        # Apply sensor noise to observation (not true state)
        noise       = np.zeros(18)
        noise[0:3]  = 2.;    noise[3:6]  = 0.1
        noise[6:10] = 0.002; noise[10:13]= 0.005
        obs_noisy   = self._state + np.random.randn(18) * noise

        # Integrate one step
        from dynamics.dynamics import rk4_step
        new_state, self._t = rk4_step(
            self._t, self._state, phys, self.dt,
            self._sim.params, self._sim.wind, self._sim.misalign
        )
        self._state  = new_state
        self._step_n += 1

        # Termination
        alt   = self._state[2]
        speed = np.linalg.norm(self._state[3:6])

        terminated = False
        reason     = 'running'

        if alt <= 0.:
            terminated = True; reason = 'landed'
        elif alt > 1500.:
            terminated = True; reason = 'escaped'
        elif speed > 200.:
            terminated = True; reason = 'speed_limit'

        truncated = (self._t >= self.t_max) and not terminated
        if truncated:
            reason = 'timeout'

        done = terminated or truncated

        reward = self._compute_reward(
            self._state, action, self._prev_act, done, reason
        )

        self._prev_act = action.copy()

        obs  = self._get_obs()
        info = {
            'altitude':   alt,
            'vz':         self._state[5],
            'speed':      speed,
            'tilt_deg':   np.rad2deg(tilt_from_quaternion(self._state[6:10])),
            'horiz_err':  np.linalg.norm(self._state[0:2]),
            'reason':     reason,
            'success':    (reason == 'landed'
                           and speed < 5.
                           and np.rad2deg(tilt_from_quaternion(self._state[6:10])) < 15.
                           and np.linalg.norm(self._state[0:2]) < 150.),
        }

        return obs, reward, terminated, truncated, info
