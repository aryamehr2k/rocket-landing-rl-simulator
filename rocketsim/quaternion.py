"""Unit quaternion helpers. q = (w, x, y, z) rotates body vectors into world vectors.

This is the only module that implements the attitude rotation, see docs/conventions.md.
"""

import math

import numpy as np

IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])
BODY_AXIS = np.array([0.0, 0.0, 1.0])
HALF = 0.5
# Below this the tilt angle of a plane is undefined (body axis lying in the other plane).
TILT_RATE_MIN_DENOMINATOR = 1e-12


def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cross product of two 3-vectors; numpy's general np.cross is many times slower for this case."""
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


def normalize(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q)


def multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product a * b."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def to_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrix R with v_world = R @ v_body."""
    w, x, y, z = q
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )


def rotate(q: np.ndarray, v_body: np.ndarray) -> np.ndarray:
    """Body vector to world vector."""
    return to_matrix(q) @ v_body


def rotate_inverse(q: np.ndarray, v_world: np.ndarray) -> np.ndarray:
    """World vector to body vector."""
    return to_matrix(q).T @ v_world


def from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    unit = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    return np.concatenate([[math.cos(HALF * angle)], math.sin(HALF * angle) * unit])


def derivative(q: np.ndarray, omega_body: np.ndarray) -> np.ndarray:
    """dq/dt = 0.5 * q * (0, omega) for a body frame angular velocity."""
    return HALF * multiply(q, np.concatenate([[0.0], omega_body]))


def body_axis(q: np.ndarray) -> np.ndarray:
    """World direction of the body axis (tail to nose)."""
    return to_matrix(q)[:, 2]


def tilt_angles(q: np.ndarray) -> tuple[float, float]:
    """(tilt_x, tilt_y): how far the nose leans toward +x and toward +y."""
    a = body_axis(q)
    return math.atan2(a[0], a[2]), math.atan2(a[1], a[2])


def total_tilt(q: np.ndarray) -> float:
    """Angle between the body axis and straight up."""
    return math.acos(min(1.0, max(-1.0, body_axis(q)[2])))


def roll_angle(q: np.ndarray) -> float:
    """Twist about the body axis after removing the tilt, positive right-handed about b_z."""
    return math.atan2(math.sin(2.0 * math.atan2(q[3], q[0])), math.cos(2.0 * math.atan2(q[3], q[0])))


def tilt_rates(q: np.ndarray, omega_body: np.ndarray) -> tuple[float, float]:
    """Time derivatives of tilt_x and tilt_y."""
    rotation = to_matrix(q)
    a = rotation[:, 2]
    a_dot = rotation @ cross(omega_body, BODY_AXIS)
    return (
        _plane_rate(a[0], a[2], a_dot[0], a_dot[2]),
        _plane_rate(a[1], a[2], a_dot[1], a_dot[2]),
    )


def _plane_rate(lateral: float, up: float, lateral_dot: float, up_dot: float) -> float:
    denominator = lateral ** 2 + up ** 2
    if denominator < TILT_RATE_MIN_DENOMINATOR:
        return 0.0
    return (up * lateral_dot - lateral * up_dot) / denominator
