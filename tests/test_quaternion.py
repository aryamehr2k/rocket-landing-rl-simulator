import math

import numpy as np
import pytest

from rocketsim import quaternion as quat
from tests.conftest import ROTATE_X, ROTATE_Y


def test_identity_is_upright() -> None:
    assert quat.body_axis(quat.IDENTITY) == pytest.approx([0.0, 0.0, 1.0])
    assert quat.tilt_angles(quat.IDENTITY) == (0.0, 0.0)
    assert quat.total_tilt(quat.IDENTITY) == 0.0
    assert quat.roll_angle(quat.IDENTITY) == 0.0


def test_tilt_signs_follow_conventions() -> None:
    # A positive rotation about y leans the nose toward +x.
    q = quat.from_axis_angle(ROTATE_Y, 0.1)
    assert quat.tilt_angles(q) == pytest.approx((0.1, 0.0))
    # A positive rotation about x leans the nose toward -y.
    q = quat.from_axis_angle(ROTATE_X, 0.1)
    assert quat.tilt_angles(q) == pytest.approx((0.0, -0.1))
    assert quat.total_tilt(q) == pytest.approx(0.1)


def test_rotation_round_trip_and_matrix() -> None:
    q = quat.normalize(np.array([0.9, 0.1, -0.3, 0.2]))
    v = np.array([1.0, -2.0, 3.0])
    assert quat.rotate_inverse(q, quat.rotate(q, v)) == pytest.approx(v)
    assert quat.to_matrix(q) @ v == pytest.approx(quat.rotate(q, v))
    assert np.linalg.det(quat.to_matrix(q)) == pytest.approx(1.0)


def test_roll_is_the_twist_about_the_body_axis() -> None:
    tilt = quat.from_axis_angle(ROTATE_Y, 0.4)
    roll = quat.from_axis_angle(np.array([0.0, 0.0, 1.0]), 0.7)
    assert quat.roll_angle(quat.multiply(tilt, roll)) == pytest.approx(0.7)
    assert quat.tilt_angles(quat.multiply(tilt, roll))[0] == pytest.approx(0.4)


def test_tilt_rates_match_finite_differences() -> None:
    q = quat.from_axis_angle(quat.normalize(np.array([0.3, -0.2, 0.9])), 0.5)
    omega = np.array([0.4, -0.7, 1.3])
    dt = 1e-7
    q_next = quat.normalize(q + dt * quat.derivative(q, omega))
    expected = (np.array(quat.tilt_angles(q_next)) - np.array(quat.tilt_angles(q))) / dt
    assert quat.tilt_rates(q, omega) == pytest.approx(tuple(expected), abs=1e-5)


def test_body_rate_integrates_about_body_axis() -> None:
    # Constant body rate about b_x for one second equals a rotation about the rotated x axis.
    q = quat.from_axis_angle(ROTATE_Y, math.pi / 2)
    q_end = quat.multiply(q, quat.from_axis_angle(ROTATE_X, 0.3))
    spun_axis = quat.rotate(quat.from_axis_angle(ROTATE_X, 0.3), quat.BODY_AXIS)
    assert quat.body_axis(q_end) == pytest.approx(quat.rotate(q, spun_axis))


def test_tilt_rate_is_finite_when_the_axis_lies_in_the_other_plane() -> None:
    q = quat.from_axis_angle(ROTATE_X, -math.pi / 2)
    rates = quat.tilt_rates(q, np.array([0.0, 1.0, 0.0]))
    assert all(math.isfinite(r) for r in rates)
    assert rates[0] == 0.0
