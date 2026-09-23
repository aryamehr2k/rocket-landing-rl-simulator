"""Air density, aerodynamic loads and wind.

Signs and frames follow docs/conventions.md. Loads are returned in the body frame.
"""

import math
from dataclasses import dataclass

import numpy as np

from rocketsim.config import AeroConfig
from rocketsim.simconfig import EnvironmentConfig, WindConfig

LATERAL = np.array([1.0, 1.0, 0.0])


def air_density(altitude: float, environment: EnvironmentConfig) -> float:
    """Exponential atmosphere. Altitude is above the pad; site elevation is added here."""
    height_above_sea = altitude + environment.site_elevation
    return environment.sea_level_density * math.exp(-height_above_sea / environment.scale_height)


@dataclass(frozen=True)
class AeroLoads:
    """Drag plus normal force, and the normal force moment about the CG, in body axes."""

    force: np.ndarray
    moment: np.ndarray


def aerodynamic_loads(
    aero: AeroConfig, reference_area: float, density: float, airspeed_body: np.ndarray, cg: float
) -> AeroLoads:
    """Drag along the relative wind and the normal force on the lateral airspeed at the CP."""
    speed = float(np.linalg.norm(airspeed_body))
    factor = 0.5 * density * reference_area * speed
    drag = -factor * aero.drag_coefficient * airspeed_body
    normal = -factor * aero.normal_force_slope * airspeed_body * LATERAL
    cp_position = np.array([0.0, 0.0, cg - aero.cp])
    return AeroLoads(force=drag + normal, moment=np.cross(cp_position, normal))


class Wind:
    """Horizontal wind vector: a steady part plus first order filtered gusts.

    Each horizontal component of the gust is an Ornstein-Uhlenbeck process with the
    configured standard deviation and time constant, updated once per physics step.
    """

    def __init__(self, config: WindConfig, rng: np.random.Generator | None = None) -> None:
        self.config = config
        self.rng = rng if rng is not None else np.random.default_rng()
        direction = config.steady_direction
        self.steady = config.steady * np.array([math.cos(direction), math.sin(direction)])
        self.gust = np.zeros(2)

    def reset(self) -> None:
        self.gust = np.zeros(2)

    @property
    def vector(self) -> np.ndarray:
        return self.steady + self.gust

    def step(self, dt: float) -> np.ndarray:
        """Advance the gust filter by one physics step and return the new wind vector."""
        if self.config.gust_std > 0.0:
            tau = self.config.gust_time_constant
            noise = self.config.gust_std * math.sqrt(2.0 * dt / tau) * self.rng.standard_normal(2)
            self.gust = self.gust - self.gust / tau * dt + noise
        return self.vector
