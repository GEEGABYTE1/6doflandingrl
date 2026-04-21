"""Atmospheric density model for Phase 1 landing simulations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AtmosphereSample:
    """Local atmospheric properties at a given altitude."""

    density: float
    pressure: float
    temperature: float
    speed_of_sound: float


@dataclass(frozen=True)
class ISAAtmosphere:
    """Simplified ISA troposphere model with exponential continuation.

    The model uses the standard lapse-rate equations up to 11 km and an
    exponential density continuation above that altitude. Phase 1 landing
    trajectories stay close to the ground, but this keeps the interface usable
    for later higher-altitude experiments.
    """

    sea_level_density: float = 1.225
    sea_level_pressure: float = 101_325.0
    sea_level_temperature: float = 288.15
    lapse_rate: float = 0.0065
    gas_constant: float = 287.05287
    gamma: float = 1.4
    gravity: float = 9.80665
    tropopause_altitude: float = 11_000.0
    high_altitude_scale_height: float = 7_200.0

    def sample(self, altitude_m: float) -> AtmosphereSample:
        """Return atmospheric properties at geometric altitude in meters."""
        altitude = max(0.0, float(altitude_m))
        if altitude <= self.tropopause_altitude:
            temperature = self.sea_level_temperature - self.lapse_rate * altitude
            pressure = self.sea_level_pressure * (
                temperature / self.sea_level_temperature
            ) ** (self.gravity / (self.gas_constant * self.lapse_rate))
            density = pressure / (self.gas_constant * temperature)
        else:
            base = self.sample(self.tropopause_altitude)
            temperature = base.temperature
            pressure = base.pressure * np.exp(
                -(altitude - self.tropopause_altitude) / self.high_altitude_scale_height
            )
            density = base.density * np.exp(
                -(altitude - self.tropopause_altitude) / self.high_altitude_scale_height
            )
        speed_of_sound = float(np.sqrt(self.gamma * self.gas_constant * temperature))
        return AtmosphereSample(
            density=float(density),
            pressure=float(pressure),
            temperature=float(temperature),
            speed_of_sound=speed_of_sound,
        )

