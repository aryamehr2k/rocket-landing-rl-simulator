"""Text of a built vehicle's training file: the base training file pointed at its vehicle, mission and site."""

import math
import re

import yaml

from rocketsim.hop.config_text import built_from, num
from rocketsim.hop.datasheet import BISECTION_STEPS, HALF, Datasheet, Derived
from rocketsim.hop.datasheet_checks import TIME_RESERVE, landing_height_needed
from rocketsim.hop.mission import MissionConfig, with_targets
from rocketsim.yaml_section import ConfigError

VARIATION_HEADROOM = 1.2  # training missions reach this far beyond the sheet's target height and hover time

Range = tuple[float, float]


def training_yaml(name: str, sheet: Datasheet, d: Derived, base_text: str, base_label: str) -> str:
    """The base training file pointed at this vehicle and mission, with the site, wind and mission ranges fitted."""
    base, site = yaml.safe_load(base_text), sheet["launch_site"]
    simulation, variation = base["simulation"], base["mission_variation"]
    budget = mission_time_budget(sheet["fan"]["battery_time_s"], d.mission, simulation["pad_hold_time_s"])
    altitudes, hovers = training_ranges(d.mission, variation["target_altitude_m"], variation["hover_time_s"], budget)
    lines = [line for line in base_text.splitlines() if not line.startswith("#")]
    while lines and not lines[0].strip():
        lines.pop(0)
    flight_time = max(simulation["max_flight_time_s"], d.mission.max_flight_time)
    for section, key, value in (
        ("", "vehicle", f"../vehicles/{name}.yaml"),
        ("", "mission", f"../missions/{name}.yaml"),
        ("simulation", "max_flight_time_s", num(flight_time)),
        ("environment", "site_elevation_m", num(site["elevation_m"])),
        ("mission_variation", "target_altitude_m", _pair(altitudes)),
        ("mission_variation", "hover_time_s", _pair(hovers)),
        ("evaluation", "wind_mps", num(site["wind_mps"])),
        ("evaluation", "gust_mps", num(site["gust_mps"])),
    ):
        _set_value(lines, section, key, value)
    header = (
        f"# Training the network that flies {name} through the whole mission: launch, climb, hover, descent, landing.\n"
        f"{built_from(f'{sheet.source} and {base_label}')}\n"
        f"# Run with: python scripts/train.py --training configs/training/{name}.yaml\n\n"
    )
    return header + "\n".join(lines) + "\n"


def mission_time_budget(battery_time: float, mission: MissionConfig, pad_hold_time: float) -> float:
    """Longest plan a training mission may have: the battery and the time limit, each with the reserve kept."""
    return min(battery_time, mission.max_flight_time - pad_hold_time) / (1.0 + TIME_RESERVE)


def training_ranges(mission: MissionConfig, altitudes: Range, hovers: Range, budget: float) -> tuple[Range, Range]:
    """Target height and hover time ranges of the training missions around the sheet's mission.

    The base ranges are widened to take in the sheet's mission. Their tops are then lowered toward it until
    the longest training mission fits the time budget. The lowest height leaves room for the landing slowdown.
    """
    target, hover = mission.target_altitude, mission.hover_time
    low_altitude = min(max(altitudes[0], 2.0 * landing_height_needed(mission)), target)
    top_altitude = max(altitudes[1], math.ceil(target * VARIATION_HEADROOM))
    top_hover = max(hovers[1], math.ceil(hover * VARIATION_HEADROOM))

    def longest(share: float) -> float:
        drawn = with_targets(mission, target + share * (top_altitude - target), hover + share * (top_hover - hover))
        return drawn.planned_duration

    if longest(1.0) > budget:
        low, high = 0.0, 1.0
        for _ in range(BISECTION_STEPS):
            middle = HALF * (low + high)
            low, high = (middle, high) if longest(middle) <= budget else (low, middle)
        top_altitude = max(target, math.floor(target + low * (top_altitude - target)))
        top_hover = max(hover, math.floor(hover + low * (top_hover - hover)))
    return (low_altitude, top_altitude), (min(hovers[0], hover), top_hover)


def _pair(values: Range) -> str:
    return f"[{values[0]:g}, {values[1]:g}]"


def _set_value(lines: list[str], section: str, key: str, value: str) -> None:
    """Replace the value of `section.key` (a top-level key when section is empty), keeping its comment."""
    current = ""
    indent = "  " if section else ""
    for index, line in enumerate(lines):
        top = re.match(r"^(\w+):", line)
        if top:
            current = top.group(1)
        match = re.match(rf"^{indent}{key}:\s*[^#]*?(\s*#.*)?$", line)
        if match and current == (section or key):
            lines[index] = f"{indent}{key}: {value}{match.group(1) or ''}"
            return
    raise ConfigError(f"the base training file has no {section + '.' if section else ''}{key} to set")
