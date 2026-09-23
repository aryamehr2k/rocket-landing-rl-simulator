"""Simulation settings: integration rate, atmosphere, site and wind, from a training YAML."""

from dataclasses import dataclass
from pathlib import Path

from rocketsim.config import ControlConfig
from rocketsim.yaml_section import ConfigError, Section, read_yaml_mapping

RATE_RATIO_TOLERANCE = 1e-9


@dataclass(frozen=True)
class SimulationConfig:
    physics_rate_hz: float
    max_flight_time: float

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
class SimConfig:
    simulation: SimulationConfig
    environment: EnvironmentConfig
    wind: WindConfig
    source: str


def load_sim_config(path: str | Path) -> SimConfig:
    """Load the simulation, environment and wind sections of a training YAML."""
    data, source = read_yaml_mapping(path)
    root = Section(data, source)
    sim = root.sub("simulation")
    sim.only_keys("physics_rate_hz", "max_flight_time_s")
    env = root.sub("environment")
    env.only_keys("site_elevation_m", "gravity_mps2", "sea_level_density_kgpm3", "scale_height_m")
    wind = root.sub("wind")
    wind.only_keys("steady_mps", "steady_direction_deg", "gust_std_mps", "gust_time_constant_s")
    return SimConfig(
        simulation=SimulationConfig(
            physics_rate_hz=sim.number("physics_rate_hz", above=0.0),
            max_flight_time=sim.number("max_flight_time_s", above=0.0),
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
    )


def physics_steps_per_control_step(simulation: SimulationConfig, control: ControlConfig) -> int:
    """Physics steps in one control period. Raises if the rates do not divide evenly."""
    ratio = simulation.physics_rate_hz / control.control_rate_hz
    if abs(ratio - round(ratio)) > RATE_RATIO_TOLERANCE or ratio < 1.0:
        raise ConfigError(
            f"physics_rate_hz ({simulation.physics_rate_hz}) must be an integer multiple of "
            f"control_rate_hz ({control.control_rate_hz})"
        )
    return int(round(ratio))
