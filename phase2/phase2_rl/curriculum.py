#training for flat PPO baseline
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from phase1_dynamics.scenarios import ScenarioConfig

@dataclass(frozen=True)
class CurriculumBounds:
    # we decided to do easy-to-hard curriculum bounds
    min_altitude_m: float = 20.0
    max_altitude_m: float = 120.0
    min_lateral_offset_m: float = 0.5
    max_lateral_offset_m: float = 12.0
    min_attitude_error_deg: float = 0.2
    max_attitude_error_deg: float = 1.5
    min_wind_mps: float = 0.0
    max_wind_mps: float = 1.0
    min_gust_mps: float = 0.0
    max_gust_mps: float = 0.2
    min_misalignment_deg: float = 0.0
    max_misalignment_deg: float = 0.1


def _lerp(low: float, high: float, alpha: float) -> float:
    return float((1.0 - alpha) * low + alpha * high)

def _staged_wind_alpha(progress: float) -> tuple[float, float]:
    #early: no wind and no gust
    # middle: steady wind only
    # late: steady wind plus small gust
    if progress < 1.0 / 3.0:
        return 0.0, 0.0
    if progress < 2.0 / 3.0:
        steady_alpha = (progress - 1.0 / 3.0) / (1.0 / 3.0)
        return float(np.clip(steady_alpha, 0.0, 1.0)), 0.0
    gust_alpha = (progress - 2.0 / 3.0) / (1.0 / 3.0)
    return 1.0, float(np.clip(gust_alpha, 0.0, 1.0))

@dataclass
class TrainingCurriculum:
    bounds: CurriculumBounds = CurriculumBounds()
    progress: float = 0.0
    staged_wind_enabled: bool = True

    def set_progress(self, progress: float) -> None:
        self.progress = float(np.clip(progress, 0.0, 1.0))

    def sample(self, base: ScenarioConfig, rng: np.random.Generator) -> ScenarioConfig:
        alpha = self.progress
        difficulty = float(alpha * alpha * alpha)
        if self.staged_wind_enabled:
            steady_wind_alpha, gust_alpha = _staged_wind_alpha(alpha)
        else:
            steady_wind_alpha, gust_alpha = 1.0, 1.0
        altitude = rng.uniform(
            self.bounds.min_altitude_m,
            _lerp(self.bounds.min_altitude_m, self.bounds.max_altitude_m, difficulty),
        )
        lateral = rng.uniform(
            self.bounds.min_lateral_offset_m,
            _lerp(self.bounds.min_lateral_offset_m, self.bounds.max_lateral_offset_m, difficulty),
        )
        heading = rng.uniform(-np.pi, np.pi)
        x0 = lateral * np.cos(heading)
        y0 = lateral * np.sin(heading)
        attitude_mag = rng.uniform(
            self.bounds.min_attitude_error_deg,
            _lerp(self.bounds.min_attitude_error_deg, self.bounds.max_attitude_error_deg, difficulty),
        )
        roll = rng.uniform(-attitude_mag, attitude_mag)
        pitch = rng.uniform(-attitude_mag, attitude_mag)
        yaw = rng.uniform(-0.25 * attitude_mag, 0.25 * attitude_mag)
        wind_mag = rng.uniform(
            self.bounds.min_wind_mps,
            _lerp(self.bounds.min_wind_mps, self.bounds.max_wind_mps, difficulty * steady_wind_alpha),
        )
        wind_heading = rng.uniform(-np.pi, np.pi)
        gust_mag = rng.uniform(
            self.bounds.min_gust_mps,
            _lerp(self.bounds.min_gust_mps, self.bounds.max_gust_mps, difficulty * gust_alpha),
        )
        gust_heading = rng.uniform(-np.pi, np.pi)
        misalignment = rng.uniform(
            self.bounds.min_misalignment_deg,
            _lerp(self.bounds.min_misalignment_deg, self.bounds.max_misalignment_deg, difficulty),
        )
        vertical_speed_mag = rng.uniform(3.0, _lerp(4.5, 14.0, difficulty))
        vertical_speed = -float(vertical_speed_mag)
        horizontal_speed = rng.uniform(0.0, _lerp(0.2, 0.6, difficulty))
        vx0 = -0.5 * x0 / max(altitude / 12.0, 1.0) + horizontal_speed * np.cos(wind_heading + np.pi / 2.0)
        vy0 = -0.5 * y0 / max(altitude / 12.0, 1.0) + horizontal_speed * np.sin(wind_heading + np.pi / 2.0)
        return ScenarioConfig(
            name=f"curriculum_p{alpha:.2f}",
            description=f"Training curriculum sample at progress {alpha:.2f}.",
            initial_position_m=(float(x0), float(y0), float(altitude)),
            initial_velocity_mps=(float(vx0), float(vy0), float(vertical_speed)),
            initial_attitude_error_deg=(float(roll), float(pitch), float(yaw)),
            initial_mass_kg=base.initial_mass_kg,
            wind_mps=(float(wind_mag * np.cos(wind_heading)), float(wind_mag * np.sin(wind_heading)), 0.0),
            gust_mps=(float(gust_mag * np.cos(gust_heading)), float(gust_mag * np.sin(gust_heading)), 0.0),
            gust_frequency_hz=base.gust_frequency_hz,
            misalignment_deg=(float(misalignment), float(-misalignment)),
            duration_s=base.duration_s,
            dt_s=base.dt_s,
        )
