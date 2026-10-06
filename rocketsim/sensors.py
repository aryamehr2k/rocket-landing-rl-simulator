"""Sensor models and the sensor YAML: a three axis IMU, a barometer and an optional GPS.

A sample is the true value from `lag` seconds earlier plus a per-flight bias and white noise, clipped to the range.
"""

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from rocketsim.yaml_section import Section, read_yaml_mapping

AXES = 3
HORIZONTAL = 2
GPS_DRIFT_TIME_CONSTANT = 60.0  # s, how slowly a GPS position error wanders
SAMPLE_TIME_TOLERANCE = 1e-9


@dataclass(frozen=True)
class ImuChannelConfig:
    noise_std: float
    bias_std: float
    range: float


@dataclass(frozen=True)
class ImuConfig:
    rate_hz: float
    lag: float
    accelerometer: ImuChannelConfig
    gyroscope: ImuChannelConfig

    @property
    def dt(self) -> float:
        return 1.0 / self.rate_hz


@dataclass(frozen=True)
class BarometerConfig:
    rate_hz: float
    lag: float
    noise_std: float
    bias_std: float


@dataclass(frozen=True)
class GpsConfig:
    rate_hz: float
    lag: float
    position_noise_std: float
    position_drift_std: float
    velocity_noise_std: float


@dataclass(frozen=True)
class SensorsConfig:
    name: str
    imu: ImuConfig
    barometer: BarometerConfig
    source: str
    gps: GpsConfig | None = None


def load_sensors_config(path: str | Path) -> SensorsConfig:
    """Load and validate a sensor YAML."""
    data, source = read_yaml_mapping(path)
    root = Section(data, source)
    root.only_keys("name", "imu", "barometer", "gps")
    imu = root.sub("imu")
    imu.only_keys("rate_hz", "lag_s", "accelerometer", "gyroscope")
    baro = root.sub("barometer")
    baro.only_keys("rate_hz", "lag_s", "noise_std_m", "bias_std_m")
    return SensorsConfig(
        name=root.string("name"),
        imu=ImuConfig(
            rate_hz=imu.number("rate_hz", above=0.0),
            lag=imu.number("lag_s", minimum=0.0),
            accelerometer=_channel(imu.sub("accelerometer"), "mps2"),
            gyroscope=_channel(imu.sub("gyroscope"), "deg_per_s"),
        ),
        barometer=BarometerConfig(
            rate_hz=baro.number("rate_hz", above=0.0),
            lag=baro.number("lag_s", minimum=0.0),
            noise_std=baro.number("noise_std_m", minimum=0.0),
            bias_std=baro.number("bias_std_m", minimum=0.0),
        ),
        source=source,
        gps=_gps(root.sub("gps")) if root.has("gps") else None,
    )


def _gps(section: Section) -> GpsConfig:
    section.only_keys("rate_hz", "lag_s", "position_noise_std_m", "position_drift_std_m", "velocity_noise_std_mps")
    return GpsConfig(
        rate_hz=section.number("rate_hz", above=0.0),
        lag=section.number("lag_s", minimum=0.0),
        position_noise_std=section.number("position_noise_std_m", minimum=0.0),
        position_drift_std=section.number("position_drift_std_m", minimum=0.0),
        velocity_noise_std=section.number("velocity_noise_std_mps", minimum=0.0),
    )


def _channel(section: Section, unit: str) -> ImuChannelConfig:
    section.only_keys(f"noise_std_{unit}", f"bias_std_{unit}", f"range_{unit}")
    return ImuChannelConfig(
        noise_std=section.number(f"noise_std_{unit}", minimum=0.0),
        bias_std=section.number(f"bias_std_{unit}", minimum=0.0),
        range=section.number(f"range_{unit}", above=0.0),
    )


@dataclass(frozen=True)
class ImuSample:
    """Specific force (m/s^2) and angular rate (rad/s), both in the body frame."""

    time: float
    specific_force: np.ndarray
    angular_rate: np.ndarray


@dataclass(frozen=True)
class BaroSample:
    time: float
    altitude: float


@dataclass(frozen=True)
class GpsSample:
    time: float
    position: np.ndarray  # horizontal x, y in metres, relative to the pad on average
    velocity: np.ndarray


@dataclass(frozen=True)
class Truth:
    """What the sensors would read without any error, recorded once per physics step."""

    specific_force: np.ndarray
    angular_rate: np.ndarray
    altitude: float
    horizontal_position: np.ndarray = field(default_factory=lambda: np.zeros(HORIZONTAL))
    horizontal_velocity: np.ndarray = field(default_factory=lambda: np.zeros(HORIZONTAL))


class SensorSuite:
    """Samples the IMU and barometer from a history of true values, adding lag, bias and noise."""

    def __init__(self, config: SensorsConfig, dt: float, rng: np.random.Generator) -> None:
        self.config = config
        self.dt = dt
        self.rng = rng
        self.imu_lag_steps = int(round(config.imu.lag / dt))
        self.baro_lag_steps = int(round(config.barometer.lag / dt))
        self.gps_lag_steps = int(round(config.gps.lag / dt)) if config.gps is not None else 0
        longest = max(self.imu_lag_steps, self.baro_lag_steps, self.gps_lag_steps)
        self.history: deque[Truth] = deque(maxlen=longest + 1)
        self.reset()

    def reset(self) -> None:
        """Draw new biases and forget the history. Call once per flight."""
        imu, baro = self.config.imu, self.config.barometer
        self.accel_bias = self.rng.normal(0.0, imu.accelerometer.bias_std, AXES)
        self.gyro_bias = self.rng.normal(0.0, imu.gyroscope.bias_std, AXES)
        self.baro_bias = float(self.rng.normal(0.0, baro.bias_std))
        self.history.clear()
        self.imu_samples = 0
        self.baro_samples = 0
        self.gps_samples = 0
        self.gps_drift = np.zeros(HORIZONTAL)
        if self.config.gps is not None:
            self.gps_drift = self.rng.normal(0.0, self.config.gps.position_drift_std, HORIZONTAL)
        self.last_imu: ImuSample | None = None
        self.last_baro: BaroSample | None = None
        self.last_gps: GpsSample | None = None

    def record(self, truth: Truth) -> None:
        self.history.append(truth)

    def sample(self, t: float) -> tuple[ImuSample | None, BaroSample | None]:
        """The samples due at time t; None for a sensor whose next sample is later."""
        imu = baro = None
        if t + SAMPLE_TIME_TOLERANCE >= self.imu_samples / self.config.imu.rate_hz:
            imu = self.last_imu = self._imu_sample(t, self._delayed(self.imu_lag_steps))
            self.imu_samples += 1
        if t + SAMPLE_TIME_TOLERANCE >= self.baro_samples / self.config.barometer.rate_hz:
            baro = self.last_baro = self._baro_sample(t, self._delayed(self.baro_lag_steps))
            self.baro_samples += 1
        return imu, baro

    def _delayed(self, steps: int) -> Truth:
        # The oldest recorded value stands in until the history is long enough.
        return self.history[max(0, len(self.history) - 1 - steps)]

    def _imu_sample(self, t: float, truth: Truth) -> ImuSample:
        accel, gyro = self.config.imu.accelerometer, self.config.imu.gyroscope
        force = truth.specific_force + self.accel_bias + self.rng.normal(0.0, accel.noise_std, AXES)
        rate = truth.angular_rate + self.gyro_bias + self.rng.normal(0.0, gyro.noise_std, AXES)
        return ImuSample(t, np.clip(force, -accel.range, accel.range), np.clip(rate, -gyro.range, gyro.range))

    def _baro_sample(self, t: float, truth: Truth) -> BaroSample:
        noise = float(self.rng.normal(0.0, self.config.barometer.noise_std))
        return BaroSample(t, truth.altitude + self.baro_bias + noise)

    def sample_gps(self, t: float) -> GpsSample | None:
        """The GPS sample due at time t, if the sensor file has a GPS and one is due."""
        gps = self.config.gps
        if gps is None or t + SAMPLE_TIME_TOLERANCE < self.gps_samples / gps.rate_hz:
            return None
        self.gps_samples += 1
        truth = self._delayed(self.gps_lag_steps)
        # The position error wanders slowly (first order Gauss-Markov) on top of white noise.
        sample_dt = 1.0 / gps.rate_hz
        decay = sample_dt / GPS_DRIFT_TIME_CONSTANT
        kick = gps.position_drift_std * np.sqrt(2.0 * decay) * self.rng.standard_normal(HORIZONTAL)
        self.gps_drift = self.gps_drift * (1.0 - decay) + kick
        position = truth.horizontal_position + self.gps_drift + self.rng.normal(0.0, gps.position_noise_std, HORIZONTAL)
        velocity = truth.horizontal_velocity + self.rng.normal(0.0, gps.velocity_noise_std, HORIZONTAL)
        self.last_gps = GpsSample(t, position, velocity)
        return self.last_gps
