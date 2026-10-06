"""A deployable drag device: petals or flaps with a drag area Cd*A at one station, opened from 0 (shut) to 1.

Placed ahead of the CG it steadies a tail-first fall like shuttlecock feathers; behind the CG it tips the rocket over.
"""

import math
from dataclasses import dataclass

import numpy as np

from rocketsim.quaternion import cross
from rocketsim.yaml_section import Section

HALF = 0.5
FLAT_PLATE_CD = 1.2
FULLY_OPEN = 1.0
SHUT = 0.0


@dataclass(frozen=True)
class DragDeviceConfig:
    drag_area: float
    station: float
    deploy_time: float
    retract_time: float


def load_drag_device(section: Section, length_mm: float) -> DragDeviceConfig:
    """Read the `drag_device` section of a rocket YAML."""
    section.only_keys("drag_area_cm2", "station_from_nose_mm", "deploy_time_s", "retract_time_s")
    return DragDeviceConfig(
        drag_area=section.number("drag_area_cm2", above=0.0),
        station=section.number("station_from_nose_mm", minimum=0.0, maximum=length_mm),
        deploy_time=section.number("deploy_time_s", above=0.0),
        retract_time=section.number("retract_time_s", above=0.0),
    )


def device_loads(
    device: DragDeviceConfig,
    fraction: float,
    density: float,
    airspeed_body: np.ndarray,
    omega_body: np.ndarray,
    cg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Body frame force and moment about the CG of the device opened to `fraction`.

    The relative wind at the device is the body airspeed plus omega cross r, with r the body
    position of the device relative to the centre of gravity, so a turning rocket is damped.
    """
    arm = np.array([0.0, 0.0, cg - device.station])
    local_wind = airspeed_body + cross(omega_body, arm)
    speed = float(np.linalg.norm(local_wind))
    force = -HALF * density * device.drag_area * fraction * speed * local_wind
    return force, cross(arm, force)


def drag_factor(device: DragDeviceConfig | None, fraction: float, density: float) -> float:
    """Half rho Cd A of the device at this opening, the coefficient of speed squared."""
    if device is None:
        return 0.0
    return HALF * density * device.drag_area * fraction


def terminal_speed(mass: float, gravity: float, total_drag_factor: float) -> float:
    """Speed at which drag equals weight: sqrt(m g / k) for drag k v^2."""
    if total_drag_factor <= 0.0:
        return math.inf
    return math.sqrt(mass * gravity / total_drag_factor)


def drag_area_for_terminal_speed(mass: float, gravity: float, density: float, speed: float) -> float:
    """Total Cd*A that makes drag equal weight at `speed`: 2 m g / (rho v^2)."""
    return 2.0 * mass * gravity / (density * speed ** 2)


def plate_area(drag_area: float) -> float:
    """Flat plate area that gives a drag area, at the flat plate drag coefficient."""
    return drag_area / FLAT_PLATE_CD
