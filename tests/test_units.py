import math

from rocketsim.units import split_unit_suffix, to_si


def test_suffix_conversions() -> None:
    assert to_si("length_mm", 1500.0) == 1.5
    assert to_si("dry_mass_g", 250.0) == 0.25
    assert math.isclose(to_si("max_angle_deg", 180.0), math.pi)
    assert math.isclose(to_si("rate_deg_per_s", 90.0), math.pi / 2)
    assert to_si("delay_s", 0.02) == 0.02
    assert to_si("speed_mps", 3.0) == 3.0
    assert to_si("center_us", 1500.0) == 1500.0
    assert to_si("inertia_kgm2", 0.1) == 0.1
    # Servo calibration: microseconds per servo degree becomes microseconds per radian.
    assert math.isclose(to_si("pulse_us_per_deg", 10.0), 10.0 * 180.0 / math.pi)
    assert to_si("linkage_ratio", 0.5) == 0.5


def test_longest_suffix_wins() -> None:
    base, _ = split_unit_suffix("servo_rate_limit_deg_per_s")
    assert base == "servo_rate_limit"
    base, _ = split_unit_suffix("scale_height_m")
    assert base == "scale_height"


def test_unknown_suffix_is_left_alone() -> None:
    assert to_si("drag_coefficient", 0.6) == 0.6
    assert split_unit_suffix("sign")[0] == "sign"
