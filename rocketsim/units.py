"""Unit conversion for values read from YAML: a key's suffix names its unit, and the code uses SI everywhere."""

import math
from typing import Callable

MM_PER_M = 1000.0
G_PER_KG = 1000.0
US_PER_S = 1.0e6
MS_PER_S = 1000.0
CM2_PER_M2 = 1.0e4
DEG_PER_RAD = 180.0 / math.pi
STANDARD_GRAVITY = 9.80665  # m/s^2, the g in thrust-to-weight and sensor ranges given in g


def mm_to_m(value: float) -> float:
    return value / MM_PER_M


def g_to_kg(value: float) -> float:
    return value / G_PER_KG


def kg_to_g(value: float) -> float:
    return value * G_PER_KG


def deg_to_rad(value: float) -> float:
    return value / DEG_PER_RAD


def rad_to_deg(value: float) -> float:
    return value * DEG_PER_RAD


def us_to_s(value: float) -> float:
    return value / US_PER_S


def cm2_to_m2(value: float) -> float:
    return value / CM2_PER_M2


def us_per_deg_to_us_per_rad(value: float) -> float:
    return value * DEG_PER_RAD


def identity(value: float) -> float:
    return value


Converter = Callable[[float], float]

# Longest suffixes first, so "_deg_per_s" wins over "_s".
SUFFIX_CONVERSIONS: list[tuple[str, Converter]] = [
    ("_us_per_deg", us_per_deg_to_us_per_rad),
    ("_deg_per_mps", deg_to_rad),
    ("_deg_per_m", deg_to_rad),
    ("_deg_per_s", deg_to_rad),
    ("_kgm2", identity),
    ("_cm2", cm2_to_m2),
    ("_mps2", identity),
    ("_mps", identity),
    ("_deg", deg_to_rad),
    ("_mm", mm_to_m),
    ("_kg", identity),
    ("_us", identity),
    ("_g", g_to_kg),
    ("_m", identity),
    ("_s", identity),
    ("_n", identity),
]


def split_unit_suffix(key: str) -> tuple[str, Converter]:
    """Return the key without its unit suffix and the function that converts its value to SI."""
    for suffix, convert in SUFFIX_CONVERSIONS:
        if key.endswith(suffix) and len(key) > len(suffix):
            return key[: -len(suffix)], convert
    return key, identity


def to_si(key: str, value: float) -> float:
    """Convert a YAML value to SI using the unit suffix of its key."""
    _, convert = split_unit_suffix(key)
    return convert(value)
