"""Gimbal angle to servo pulse width and back, shared by the simulator and the firmware."""

import math
from dataclasses import dataclass

from rocketsim.units import rad_to_deg
from rocketsim.yaml_section import ConfigError, Section

SERVO_SIGNS = (-1, 1)
HALF = 0.5


@dataclass(frozen=True)
class ServoCalibration:
    """Gimbal angle to servo pulse width. Angles in radians, pulses in microseconds."""

    center_us: float
    pulse_us_per_rad: float
    sign: int
    linkage_ratio: float
    pulse_resolution_us: float
    min_us: float
    max_us: float

    def pulse_for_gimbal(self, gimbal_angle: float) -> float:
        """Unquantised pulse width for a gimbal angle."""
        servo_angle = gimbal_angle / self.linkage_ratio
        return self.center_us + self.sign * servo_angle * self.pulse_us_per_rad

    def quantize(self, pulse_us: float) -> float:
        """Round half up to the pulse resolution, then clip. The C firmware does the same."""
        steps = math.floor(pulse_us / self.pulse_resolution_us + HALF)
        return min(max(steps * self.pulse_resolution_us, self.min_us), self.max_us)

    def gimbal_for_pulse(self, pulse_us: float) -> float:
        """Gimbal angle the servo actually holds for a pulse width."""
        servo_angle = self.sign * (pulse_us - self.center_us) / self.pulse_us_per_rad
        return servo_angle * self.linkage_ratio


def load_servo_calibration(cal: Section, max_angle: float) -> ServoCalibration:
    """Read one servo block of the gimbal section and check its pulse range covers the gimbal limit."""
    cal.only_keys(
        "center_us", "pulse_us_per_deg", "sign", "linkage_ratio", "pulse_resolution_us", "min_us", "max_us"
    )
    sign = cal.integer("sign")
    if sign not in SERVO_SIGNS:
        raise ConfigError(f"{cal.where('sign')} must be 1 or -1, got {sign}")
    calibration = ServoCalibration(
        center_us=cal.number("center_us", above=0.0),
        pulse_us_per_rad=cal.number("pulse_us_per_deg", above=0.0),
        sign=sign,
        linkage_ratio=cal.number("linkage_ratio", above=0.0),
        pulse_resolution_us=cal.number("pulse_resolution_us", above=0.0),
        min_us=cal.number("min_us", above=0.0),
        max_us=cal.number("max_us", above=0.0),
    )
    if not calibration.min_us < calibration.center_us < calibration.max_us:
        raise ConfigError(f"{cal.where('center_us')} must lie between min_us and max_us")
    for key in ("center_us", "min_us", "max_us"):
        if getattr(calibration, key) % calibration.pulse_resolution_us != 0.0:
            raise ConfigError(f"{cal.where(key)} must be a multiple of pulse_resolution_us")
    for angle in (-max_angle, max_angle):
        pulse = calibration.pulse_for_gimbal(angle)
        if not calibration.min_us <= pulse <= calibration.max_us:
            raise ConfigError(
                f"{cal.where('min_us')}: gimbal angle {rad_to_deg(angle):.1f} deg needs a pulse of "
                f"{pulse:.0f} us, outside min_us..max_us"
            )
    return calibration
