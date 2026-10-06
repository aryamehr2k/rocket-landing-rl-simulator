"""Per-flight errors the flight computer does not know: motor strength, dry mass, device drag area.

Only the physics sees these draws. Thrust scales multiply thrust but not mass flow.
"""

from dataclasses import dataclass, replace

import numpy as np

from rocketsim.config import RocketConfig
from rocketsim.simconfig import RandomizeConfig


@dataclass(frozen=True)
class EpisodeDraw:
    landing_thrust_scale: float = 1.0
    ascent_thrust_scale: float = 1.0
    dry_mass_offset: float = 0.0
    device_drag_area_scale: float = 1.0


def draw_episode(config: RandomizeConfig | None, rng: np.random.Generator) -> EpisodeDraw:
    """Draw one flight's errors. Without a randomize section nothing is drawn and the rng is untouched."""
    if config is None:
        return EpisodeDraw()
    return EpisodeDraw(
        landing_thrust_scale=float(rng.uniform(*config.landing_thrust_scale)),
        ascent_thrust_scale=float(rng.uniform(*config.ascent_thrust_scale)),
        dry_mass_offset=float(rng.uniform(*config.dry_mass_offset)),
        device_drag_area_scale=float(rng.uniform(*config.device_drag_area_scale)),
    )


def true_rocket(rocket: RocketConfig, draw: EpisodeDraw) -> RocketConfig:
    """The rocket the physics flies: the nominal file with the drawn mass and drag area errors applied."""
    airframe = replace(rocket.airframe, dry_mass=rocket.airframe.dry_mass + draw.dry_mass_offset)
    device = rocket.drag_device
    if device is not None:
        device = replace(device, drag_area=device.drag_area * draw.device_drag_area_scale)
    return replace(rocket, airframe=airframe, drag_device=device)
