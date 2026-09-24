"""The float32 estimator: pad calibration, attitude from gravity, integration and barometer fusion."""

import math

import numpy as np

from tests.conftest import ROTATE_Y
from rocketsim import quaternion
from rocketsim.estimator import Estimator, attitude_from_gravity
from rocketsim.guidance_config import EstimatorConfig
from rocketsim.sensors import BaroSample, ImuSample

G = 9.81
DT = 0.005
PAD_ALTITUDE = 0.6
CONFIG = EstimatorConfig(pad_average_time=0.5, baro_altitude_gain=0.1, baro_velocity_gain=0.5)


def make_estimator(baro_lag: float = 0.0) -> Estimator:
    return Estimator(CONFIG, G, DT, PAD_ALTITUDE, baro_lag=baro_lag)


def imu(t: float, force: tuple[float, float, float], rate: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> ImuSample:
    return ImuSample(t, np.array(force), np.array(rate))


def test_pad_calibration_removes_gyro_bias_and_barometer_offset() -> None:
    estimator = make_estimator()
    for i in range(200):
        estimator.update_imu(imu(i * DT, (0.0, 0.0, G), (0.01, -0.02, 0.0)))
        if i % 4 == 0:
            estimator.update_baro(BaroSample(i * DT, PAD_ALTITUDE + 2.0))
    estimator.liftoff()
    assert np.allclose(estimator.gyro_bias, [0.01, -0.02, 0.0], atol=1e-6)
    assert np.allclose(estimator.q, [1.0, 0.0, 0.0, 0.0])
    assert math.isclose(float(estimator.baro_offset), 2.0, abs_tol=1e-5)
    assert estimator.position.dtype == np.float32


def test_attitude_from_gravity_recovers_a_tilt() -> None:
    q_true = quaternion.from_axis_angle(ROTATE_Y, math.radians(10.0))  # nose leans toward +x
    force_body = quaternion.rotate_inverse(q_true, np.array([0.0, 0.0, G]))
    q = attitude_from_gravity(force_body)
    tilt_x, tilt_y = quaternion.tilt_angles(q.astype(float))
    assert math.isclose(tilt_x, math.radians(10.0), abs_tol=1e-5)
    assert math.isclose(tilt_y, 0.0, abs_tol=1e-5)


def test_flight_integration_of_acceleration_and_rate() -> None:
    estimator = make_estimator()
    estimator.liftoff()
    steps = 200
    for i in range(steps):
        estimator.update_imu(imu(i * DT, (0.0, 0.0, G + 5.0), (0.0, 0.1, 0.0)))
    estimate = estimator.estimate
    assert math.isclose(float(estimate.velocity[2]), 5.0, rel_tol=0.05)
    assert math.isclose(float(estimate.position[2]) - PAD_ALTITUDE, 2.5, rel_tol=0.05)
    assert math.isclose(estimate.tilt[0], 0.1, abs_tol=0.01)  # rate about +b_y leans the nose toward +x
    assert estimator.q.dtype == np.float32 and estimator.velocity.dtype == np.float32


def test_barometer_correction_uses_the_gains() -> None:
    estimator = make_estimator()
    estimator.liftoff()
    estimator.update_baro(BaroSample(0.0, PAD_ALTITUDE + 1.0))
    assert math.isclose(float(estimator.position[2]) - PAD_ALTITUDE, 0.1, abs_tol=1e-5)
    assert math.isclose(float(estimator.velocity[2]), 0.5, abs_tol=1e-5)


def test_barometer_lag_is_compensated_with_the_vertical_speed() -> None:
    estimator = make_estimator(baro_lag=0.02)
    estimator.liftoff()
    estimator.velocity[2] = -30.0  # falling: a 20 ms old reading is 0.6 m too high
    estimator.position[2] = PAD_ALTITUDE + 10.0
    estimator.update_baro(BaroSample(0.0, PAD_ALTITUDE + 10.6))
    assert math.isclose(float(estimator.position[2]), PAD_ALTITUDE + 10.0, abs_tol=1e-4)
    assert math.isclose(float(estimator.velocity[2]), -30.0, abs_tol=1e-4)
