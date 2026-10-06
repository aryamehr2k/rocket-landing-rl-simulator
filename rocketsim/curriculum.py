"""Curriculum: how much wind, sensor noise and hidden error the policy trains with as training goes on."""

from dataclasses import dataclass, replace

from rocketsim.config import RocketConfig
from rocketsim.sensors import BarometerConfig, ImuChannelConfig, ImuConfig, SensorsConfig
from rocketsim.simconfig import RandomizeConfig, SimConfig, WindConfig

HALF = 0.5


@dataclass(frozen=True)
class CurriculumLevel:
    """Training conditions: wind values in m/s (None keeps the YAML's wind), the rest scale factors."""

    wind_max: float | None = None
    gust_std: float | None = None
    sensor_noise: float = 1.0
    hidden_errors: float = 1.0


@dataclass(frozen=True)
class CurriculumStage:
    """A level that applies up to `until`, a fraction of training (0 at the start, 1 at the end)."""

    until: float
    level: CurriculumLevel


FULL_LEVEL = CurriculumLevel()


def level_for(stages: tuple[CurriculumStage, ...], progress: float) -> CurriculumLevel:
    """The level of the first stage whose `until` is at or above the progress, else full scale."""
    for stage in stages:
        if progress <= stage.until:
            return stage.level
    return FULL_LEVEL


def scale_channel(channel: ImuChannelConfig, factor: float) -> ImuChannelConfig:
    return replace(channel, noise_std=channel.noise_std * factor, bias_std=channel.bias_std * factor)


def scale_sensors(sensors: SensorsConfig, factor: float) -> SensorsConfig:
    imu = replace(
        sensors.imu,
        accelerometer=scale_channel(sensors.imu.accelerometer, factor),
        gyroscope=scale_channel(sensors.imu.gyroscope, factor),
    )
    baro: BarometerConfig = replace(
        sensors.barometer, noise_std=sensors.barometer.noise_std * factor, bias_std=sensors.barometer.bias_std * factor
    )
    gps = sensors.gps
    if gps is not None:
        gps = replace(
            gps, position_noise_std=gps.position_noise_std * factor, position_drift_std=gps.position_drift_std * factor,
            velocity_noise_std=gps.velocity_noise_std * factor,
        )
    return replace(sensors, imu=imu, barometer=baro, gps=gps)


def level_wind(wind: WindConfig, level: CurriculumLevel) -> WindConfig:
    """The YAML's wind with the level's steady maximum and gust standard deviation, where given."""
    steady = wind.steady if level.wind_max is None else level.wind_max
    gust_std = wind.gust_std if level.gust_std is None else level.gust_std
    return replace(wind, steady=steady, gust_std=gust_std)


def scale_range(low_high: tuple[float, float], factor: float) -> tuple[float, float]:
    centre = HALF * (low_high[0] + low_high[1])
    half_width = HALF * (low_high[1] - low_high[0]) * factor
    return centre - half_width, centre + half_width


def scale_randomize(randomize: RandomizeConfig | None, factor: float) -> RandomizeConfig | None:
    if randomize is None:
        return None
    return RandomizeConfig(
        landing_thrust_scale=scale_range(randomize.landing_thrust_scale, factor),
        ascent_thrust_scale=scale_range(randomize.ascent_thrust_scale, factor),
        dry_mass_offset=scale_range(randomize.dry_mass_offset, factor),
        device_drag_area_scale=scale_range(randomize.device_drag_area_scale, factor),
    )


def apply_level(rocket: RocketConfig, sim: SimConfig, level: CurriculumLevel) -> tuple[RocketConfig, SimConfig]:
    """The rocket and simulation settings with the level's scaling applied."""
    computer = replace(rocket.computer, sensors=scale_sensors(rocket.computer.sensors, level.sensor_noise))
    scaled_rocket = replace(rocket, computer=computer)
    scaled_sim = replace(sim, wind=level_wind(sim.wind, level), randomize=scale_randomize(sim.randomize, level.hidden_errors))
    return scaled_rocket, scaled_sim
