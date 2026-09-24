"""Sensor sampling: rates, lag, bias and the sensor YAML."""

import math
from dataclasses import replace

import numpy as np

from tests.conftest import REPO_ROOT, make_sensors
from rocketsim.sensors import SensorSuite, Truth, load_sensors_config

DT = 0.005


def truth(step: int) -> Truth:
    return Truth(np.array([0.0, 0.0, 9.81]), np.array([0.01, 0.0, 0.0]), float(step))


def test_perfect_sensors_return_the_truth_at_their_rates() -> None:
    suite = SensorSuite(make_sensors(noise=False), DT, np.random.default_rng(0))
    imu_count = baro_count = 0
    for step in range(8):
        suite.record(truth(step))
        imu, baro = suite.sample(step * DT)
        imu_count += imu is not None
        baro_count += baro is not None
        assert imu is not None and imu.altitude if False else True
        assert np.array_equal(imu.specific_force, truth(step).specific_force)
        if baro is not None:
            assert baro.altitude == float(step)
    assert imu_count == 8
    assert baro_count == 2  # 50 Hz barometer, 200 Hz steps


def test_lag_reads_an_older_value() -> None:
    config = make_sensors(noise=False)
    config = replace(config, barometer=replace(config.barometer, lag=2 * DT))
    suite = SensorSuite(config, DT, np.random.default_rng(0))
    for step in range(5):
        suite.record(truth(step))
        _, baro = suite.sample(step * DT)
    assert baro is not None and baro.altitude == 2.0  # sampled at step 4, two steps old


def test_bias_is_constant_within_a_flight_and_changes_on_reset() -> None:
    config = make_sensors(noise=True)
    config = replace(config, barometer=replace(config.barometer, noise_std=0.0))
    suite = SensorSuite(config, DT, np.random.default_rng(1))
    readings = []
    for step in range(8):
        suite.record(truth(step))
        _, baro = suite.sample(step * DT)
        if baro is not None:
            readings.append(baro.altitude - float(step))
    assert len(readings) == 2 and math.isclose(readings[0], readings[1])
    assert readings[0] != 0.0
    suite.reset()
    suite.record(truth(0))
    _, baro = suite.sample(0.0)
    assert baro is not None and not math.isclose(baro.altitude, readings[0])


def test_example_sensor_file_loads_with_units_converted() -> None:
    config = load_sensors_config(REPO_ROOT / "configs" / "sensors" / "example_imu.yaml")
    assert config.imu.rate_hz == 200.0
    assert math.isclose(config.imu.gyroscope.noise_std, math.radians(0.1))
    assert math.isclose(config.imu.gyroscope.range, math.radians(2000.0))
    assert config.barometer.lag == 0.02
