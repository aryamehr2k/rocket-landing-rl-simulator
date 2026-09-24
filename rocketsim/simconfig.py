"""Simulation settings: integration rate, atmosphere, site and wind, from a training YAML."""

from dataclasses import dataclass
from pathlib import Path

from rocketsim.config import ControlConfig
from rocketsim.yaml_section import ConfigError, Section, read_yaml_mapping

RATE_RATIO_TOLERANCE = 1e-9
NO_SCALE = (1.0, 1.0)
NO_OFFSET = (0.0, 0.0)


@dataclass(frozen=True)
class SimulationConfig:
    physics_rate_hz: float
    max_flight_time: float
    pad_hold_time: float

    @property
    def dt(self) -> float:
        return 1.0 / self.physics_rate_hz


@dataclass(frozen=True)
class EnvironmentConfig:
    site_elevation: float
    gravity: float
    sea_level_density: float
    scale_height: float


@dataclass(frozen=True)
class WindConfig:
    steady: float
    steady_direction: float
    gust_std: float
    gust_time_constant: float


@dataclass(frozen=True)
class RandomizeConfig:
    """Per-flight errors the flight computer does not know about, each drawn uniformly from [low, high]."""

    landing_thrust_scale: tuple[float, float] = NO_SCALE
    ascent_thrust_scale: tuple[float, float] = NO_SCALE
    dry_mass_offset: tuple[float, float] = NO_OFFSET
    device_drag_area_scale: tuple[float, float] = NO_SCALE


@dataclass(frozen=True)
class SimConfig:
    simulation: SimulationConfig
    environment: EnvironmentConfig
    wind: WindConfig
    source: str
    randomize: RandomizeConfig | None = None


def load_sim_config(path: str | Path) -> SimConfig:
    """Load the simulation, environment and wind sections of a training YAML."""
    data, source = read_yaml_mapping(path)
    root = Section(data, source)
    sim = root.sub("simulation")
    sim.only_keys("physics_rate_hz", "max_flight_time_s", "pad_hold_time_s")
    env = root.sub("environment")
    env.only_keys("site_elevation_m", "gravity_mps2", "sea_level_density_kgpm3", "scale_height_m")
    wind = root.sub("wind")
    wind.only_keys("steady_mps", "steady_direction_deg", "gust_std_mps", "gust_time_constant_s")
    randomize = _randomize(root.sub("randomize")) if root.has("randomize") else None
    return SimConfig(
        simulation=SimulationConfig(
            physics_rate_hz=sim.number("physics_rate_hz", above=0.0),
            max_flight_time=sim.number("max_flight_time_s", above=0.0),
            pad_hold_time=sim.number("pad_hold_time_s", minimum=0.0),
        ),
        environment=EnvironmentConfig(
            site_elevation=env.number("site_elevation_m", minimum=0.0),
            gravity=env.number("gravity_mps2", above=0.0),
            sea_level_density=env.number("sea_level_density_kgpm3", above=0.0),
            scale_height=env.number("scale_height_m", above=0.0),
        ),
        wind=WindConfig(
            steady=wind.number("steady_mps", minimum=0.0),
            steady_direction=wind.number("steady_direction_deg"),
            gust_std=wind.number("gust_std_mps", minimum=0.0),
            gust_time_constant=wind.number("gust_time_constant_s", above=0.0),
        ),
        source=source,
        randomize=randomize,
    )


def _randomize(section: Section) -> RandomizeConfig:
    section.only_keys("landing_thrust_scale", "ascent_thrust_scale", "dry_mass_g", "device_drag_area_scale")
    config = RandomizeConfig(
        landing_thrust_scale=section.pair("landing_thrust_scale") if section.has("landing_thrust_scale") else NO_SCALE,
        ascent_thrust_scale=section.pair("ascent_thrust_scale") if section.has("ascent_thrust_scale") else NO_SCALE,
        dry_mass_offset=section.pair("dry_mass_g") if section.has("dry_mass_g") else NO_OFFSET,
        device_drag_area_scale=section.pair("device_drag_area_scale") if section.has("device_drag_area_scale") else NO_SCALE,
    )
    for name in ("landing_thrust_scale", "ascent_thrust_scale", "device_drag_area_scale"):
        if getattr(config, name)[0] <= 0.0:
            raise ConfigError(f"{section.where(name)} must stay above zero")
    return config


def physics_steps_per_control_step(simulation: SimulationConfig, control: ControlConfig) -> int:
    """Physics steps in one control period. Raises if the rates do not divide evenly."""
    ratio = simulation.physics_rate_hz / control.control_rate_hz
    if abs(ratio - round(ratio)) > RATE_RATIO_TOLERANCE or ratio < 1.0:
        raise ConfigError(
            f"physics_rate_hz ({simulation.physics_rate_hz}) must be an integer multiple of "
            f"control_rate_hz ({control.control_rate_hz})"
        )
    return int(round(ratio))
