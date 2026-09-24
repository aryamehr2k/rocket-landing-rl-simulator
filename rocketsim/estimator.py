"""Fixed gain state estimator for the flight computer.

Runs in float32 with the same operations in the same order as the C version so both give the
same numbers. On the pad it averages the gyro for its bias, the accelerometer for the initial
attitude and the barometer for its offset. In flight it integrates the gyro into the attitude,
the accelerometer into velocity and position, and corrects altitude and vertical speed with
each barometer sample. See docs/conventions.md.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np

from rocketsim import quaternion
from rocketsim.guidance_config import EstimatorConfig
from rocketsim.sensors import BaroSample, ImuSample

F32 = np.float32
ZERO, HALF, ONE, TWO = F32(0.0), F32(0.5), F32(1.0), F32(2.0)
UP = np.array([0.0, 0.0, 1.0], dtype=F32)
MIN_AXIS_NORM = F32(1e-6)


@dataclass(frozen=True)
class Estimate:
    """The flight computer's picture of the state: world position and velocity, attitude, body rates."""

    position: np.ndarray
    velocity: np.ndarray
    quaternion: np.ndarray
    angular_rate: np.ndarray

    @property
    def tilt(self) -> tuple[float, float]:
        return quaternion.tilt_angles(self.quaternion.astype(float))

    @property
    def tilt_rate(self) -> tuple[float, float]:
        return quaternion.tilt_rates(self.quaternion.astype(float), self.angular_rate.astype(float))

    @property
    def total_tilt(self) -> float:
        return quaternion.total_tilt(self.quaternion.astype(float))

    @property
    def roll(self) -> float:
        return quaternion.roll_angle(self.quaternion.astype(float))


def _multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=F32,
    )


def _rotation(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [ONE - TWO * (y * y + z * z), TWO * (x * y - w * z), TWO * (x * z + w * y)],
            [TWO * (x * y + w * z), ONE - TWO * (x * x + z * z), TWO * (y * z - w * x)],
            [TWO * (x * z - w * y), TWO * (y * z + w * x), ONE - TWO * (x * x + y * y)],
        ],
        dtype=F32,
    )


def attitude_from_gravity(specific_force: np.ndarray) -> np.ndarray:
    """Quaternion of a rocket at rest whose accelerometer reads this body vector, with zero roll."""
    measured = specific_force.astype(F32)
    direction = measured / np.sqrt(np.dot(measured, measured))
    axis = np.cross(direction, UP).astype(F32)
    sine = np.sqrt(np.dot(axis, axis))
    if sine < MIN_AXIS_NORM:
        return np.array([ONE, ZERO, ZERO, ZERO], dtype=F32)
    angle = np.arctan2(sine, np.dot(direction, UP))
    return np.concatenate([[np.cos(HALF * angle)], np.sin(HALF * angle) * axis / sine]).astype(F32)


class Estimator:
    def __init__(
        self, config: EstimatorConfig, gravity: float, imu_dt: float, pad_altitude: float, baro_lag: float = 0.0
    ) -> None:
        self.dt = F32(imu_dt)
        self.baro_lag = F32(baro_lag)
        self.gravity_down = np.array([0.0, 0.0, -gravity], dtype=F32)
        self.pad_altitude = F32(pad_altitude)
        self.altitude_gain = F32(config.baro_altitude_gain)
        self.velocity_gain = F32(config.baro_velocity_gain)
        self.pad_samples = max(1, int(round(config.pad_average_time / imu_dt)))
        self.reset()

    def reset(self) -> None:
        self.q = np.array([ONE, ZERO, ZERO, ZERO], dtype=F32)
        self.position = np.array([ZERO, ZERO, self.pad_altitude], dtype=F32)
        self.velocity = np.zeros(3, dtype=F32)
        self.gyro_bias = np.zeros(3, dtype=F32)
        self.angular_rate = np.zeros(3, dtype=F32)
        self.baro_offset = ZERO
        self.in_flight = False
        self._gyro_window: deque[np.ndarray] = deque(maxlen=self.pad_samples)
        self._accel_window: deque[np.ndarray] = deque(maxlen=self.pad_samples)
        self._baro_window: deque[np.float32] = deque(maxlen=self.pad_samples)

    @property
    def estimate(self) -> Estimate:
        return Estimate(self.position.copy(), self.velocity.copy(), self.q.copy(), self.angular_rate.copy())

    def update_imu(self, sample: ImuSample) -> None:
        """Integrate one IMU sample, or collect it for the pad calibration before liftoff."""
        force = sample.specific_force.astype(F32)
        rate = sample.angular_rate.astype(F32) - self.gyro_bias
        if not self.in_flight:
            self._gyro_window.append(sample.angular_rate.astype(F32))
            self._accel_window.append(force)
            self.angular_rate = rate
            return
        dq = _multiply(self.q, np.array([ZERO, rate[0], rate[1], rate[2]], dtype=F32))
        q = self.q + (HALF * self.dt) * dq
        self.q = q / np.sqrt(np.dot(q, q))
        accel_world = _rotation(self.q) @ force + self.gravity_down
        self.velocity = self.velocity + accel_world * self.dt
        self.position = self.position + self.velocity * self.dt
        self.angular_rate = rate

    def update_baro(self, sample: BaroSample) -> None:
        """Correct altitude and vertical speed with one barometer sample.

        The sample shows the altitude from `baro_lag` ago, so the altitude now is that plus
        the distance flown since, taken from the estimated vertical speed.
        """
        altitude = F32(sample.altitude)
        if not self.in_flight:
            self._baro_window.append(altitude)
            return
        altitude_now = altitude - self.baro_offset + self.velocity[2] * self.baro_lag
        error = altitude_now - self.position[2]
        self.position[2] = self.position[2] + self.altitude_gain * error
        self.velocity[2] = self.velocity[2] + self.velocity_gain * error

    def liftoff(self) -> None:
        """Freeze the pad calibration and start integrating."""
        if self._gyro_window:
            self.gyro_bias = np.mean(np.stack(self._gyro_window), axis=0).astype(F32)
        if self._accel_window:
            self.q = attitude_from_gravity(np.mean(np.stack(self._accel_window), axis=0))
        if self._baro_window:
            self.baro_offset = F32(np.mean(np.array(self._baro_window, dtype=F32))) - self.pad_altitude
        self.position = np.array([ZERO, ZERO, self.pad_altitude], dtype=F32)
        self.velocity = np.zeros(3, dtype=F32)
        self.in_flight = True
