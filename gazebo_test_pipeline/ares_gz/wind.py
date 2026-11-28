"""Seeded wind: steady wind + discrete 1-cosine gusts + Dryden-style turbulence.

* Gusts use the 1-cosine shape from aviation gust standards (e.g. 14 CFR
  25.341): ``U(t) = U_peak/2 * (1 - cos(2*pi*(t - t0)/D))`` for one gust of
  duration D, each blowing in its own random horizontal direction.
* Turbulence is a first-order (Dryden-type) Gauss-Markov process per axis
  with the MIL-HDBK-1797 low-altitude length scales at 7 m altitude
  (L_u = L_v ~ 50 m, L_w = altitude), vertical sigma = horizontal / 1.92.
* The air velocity becomes a force on the drone through quadratic drag,
  ``F = 0.5 * rho * CdA * |v_air - v_drone| * (v_air - v_drone)``, which is
  applied to the body in Gazebo. Drag also acts in calm air.

Everything is NED (metres, seconds) and reproducible from (profile, seed).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

AIR_DENSITY = 1.225  # kg/m^3, sea level


@dataclass(frozen=True)
class WindProfile:
    name: str = "calm"
    mean_speed_mps: float = 0.0
    mean_direction_deg: float | None = None  # direction the wind blows FROM, clockwise from north; None = random
    gust_peak_mps: float = 0.0
    gust_count: int = 0
    gust_duration_s: tuple = (1.5, 3.0)
    gust_window_s: tuple = (2.0, 22.0)  # gust start times are drawn inside this window
    turbulence_sigma_mps: float = 0.0  # horizontal standard deviation
    altitude_m: float = 7.0
    reference_airspeed_mps: float = 2.5  # used for turbulence time constants (cruise speed)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def is_calm(self) -> bool:
        return self.mean_speed_mps == 0 and self.gust_peak_mps == 0 and self.turbulence_sigma_mps == 0


@dataclass(frozen=True)
class DragModel:
    """Quadratic drag of the 1.5 kg test quadcopter (frontal ~0.06 m^2, top ~0.12 m^2, Cd ~1)."""

    horizontal_cda_m2: float = 0.06
    vertical_cda_m2: float = 0.12

    def force_ned(self, air_velocity_ned, drone_velocity_ned) -> np.ndarray:
        relative = np.asarray(air_velocity_ned, dtype=float) - np.asarray(drone_velocity_ned, dtype=float)
        speed = float(np.linalg.norm(relative))
        cda = np.array([self.horizontal_cda_m2, self.horizontal_cda_m2, self.vertical_cda_m2])
        return 0.5 * AIR_DENSITY * cda * speed * relative


CALM = WindProfile()
PRESETS = {
    "calm": CALM,
    # Agreed levels: light gusts peak at 3 m/s, medium at 6 m/s; strong (8 m/s) probes the limit:
    # beyond ~6 m/s relative wind the drag exceeds the controller's 1 m/s^2 horizontal authority.
    "light": WindProfile(name="light", mean_speed_mps=1.0, gust_peak_mps=3.0, gust_count=4,
                         turbulence_sigma_mps=0.3),
    "medium": WindProfile(name="medium", mean_speed_mps=2.0, gust_peak_mps=6.0, gust_count=4,
                          turbulence_sigma_mps=0.6),
    "strong": WindProfile(name="strong", mean_speed_mps=3.0, gust_peak_mps=8.0, gust_count=4,
                          turbulence_sigma_mps=0.9),
}


@dataclass(frozen=True)
class Gust:
    start_s: float
    duration_s: float
    peak_mps: float
    heading_deg: float  # direction the gust blows TOWARD, clockwise from north

    def velocity_ned(self, time_s: float) -> np.ndarray:
        phase = (time_s - self.start_s) / self.duration_s
        if phase < 0.0 or phase > 1.0:
            return np.zeros(3)
        magnitude = 0.5 * self.peak_mps * (1.0 - math.cos(2.0 * math.pi * phase))
        heading = math.radians(self.heading_deg)
        return magnitude * np.array([math.cos(heading), math.sin(heading), 0.0])


class WindField:
    """Precomputed wind on a fixed time grid (the 20 Hz control ticks)."""

    def __init__(self, profile: WindProfile, seed: int, stop_time_s: float, dt_s: float = 0.05):
        self.profile = profile
        self.seed = int(seed)
        self.dt_s = dt_s
        rng = np.random.default_rng([self.seed, 0x57494E44])  # "WIND"; independent of the lidar stream
        count = int(round(stop_time_s / dt_s)) + 1
        self.times = np.arange(count) * dt_s

        if profile.mean_direction_deg is None:
            from_deg = float(rng.uniform(0.0, 360.0))
        else:
            from_deg = float(profile.mean_direction_deg)
        toward = math.radians(from_deg + 180.0)
        self.mean_from_deg = from_deg
        mean = profile.mean_speed_mps * np.array([math.cos(toward), math.sin(toward), 0.0])

        self.gusts: list[Gust] = []
        if profile.gust_count > 0 and profile.gust_peak_mps > 0:
            low, high = profile.gust_window_s
            starts = np.sort(rng.uniform(low, high, profile.gust_count))
            for start in starts:
                self.gusts.append(Gust(
                    start_s=float(start),
                    duration_s=float(rng.uniform(*profile.gust_duration_s)),
                    peak_mps=float(profile.gust_peak_mps),
                    heading_deg=float(rng.uniform(0.0, 360.0)),
                ))

        turbulence = np.zeros((count, 3))
        sigma_h = profile.turbulence_sigma_mps
        if sigma_h > 0:
            altitude_ft = profile.altitude_m / 0.3048
            scale = 0.177 + 0.000823 * altitude_ft
            length_h = profile.altitude_m / scale ** 1.2
            length_v = profile.altitude_m
            sigma_v = sigma_h * scale ** 0.4
            airspeed = max(profile.reference_airspeed_mps, profile.mean_speed_mps, 0.5)
            sigmas = np.array([sigma_h, sigma_h, sigma_v])
            decay = np.exp(-airspeed * dt_s / np.array([length_h, length_h, length_v]))
            state = rng.normal(0.0, 1.0, 3) * sigmas
            for k in range(count):
                turbulence[k] = state
                state = decay * state + sigmas * np.sqrt(1.0 - decay ** 2) * rng.normal(0.0, 1.0, 3)
        self.length_scales_m = None if sigma_h <= 0 else (float(length_h), float(length_v))

        self.velocity = np.tile(mean, (count, 1)) + turbulence
        for k, time_s in enumerate(self.times):
            for gust in self.gusts:
                self.velocity[k] += gust.velocity_ned(float(time_s))

    def at(self, time_s: float) -> np.ndarray:
        index = int(round(time_s / self.dt_s))
        index = min(max(index, 0), self.velocity.shape[0] - 1)
        return self.velocity[index]

    def describe(self) -> dict:
        return {
            "profile": self.profile.to_dict(),
            "seed": self.seed,
            "mean_from_deg": self.mean_from_deg,
            "turbulence_length_scales_m": self.length_scales_m,
            "gusts": [asdict(gust) for gust in self.gusts],
            "max_speed_mps": float(np.max(np.linalg.norm(self.velocity, axis=1))) if self.velocity.size else 0.0,
        }
