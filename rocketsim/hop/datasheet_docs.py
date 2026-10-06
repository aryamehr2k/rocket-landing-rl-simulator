"""The fill-in template and the data request page, both written from the entry list in datasheet_fields."""

import re
import textwrap
from typing import Callable

from rocketsim.hop import datasheet as ds
from rocketsim.hop import datasheet_checks as checks
from rocketsim.hop.datasheet_fields import GROUPS, Entry
from rocketsim.hop.training_text import VARIATION_HEADROOM

COMMENT_COLUMN = 36
LINE_WIDTH = 108
TEMPLATE_PATH = "configs/datasheets/template.yaml"
BUILD_COMMAND = "python scripts/build_vehicle.py configs/datasheets/<your_name>.yaml"

TEMPLATE_HEADER = f"""\
# Data sheet of the electric test vehicle. Copy this file to configs/datasheets/<your_name>.yaml,
# fill in every entry marked required and replace the optional values you have measured, then run
#   {BUILD_COMMAND}
# Positions are measured from the top (nose) of the vehicle downward. An empty optional entry uses
# the default written next to it, or a value worked out from the rest of the sheet where it says so.
"""


INTRO = f"""\
# Electric test vehicle: data sheet

Everything the simulator needs to know about the electric test vehicle goes into one file.

1. Copy `{TEMPLATE_PATH}` to `configs/datasheets/<your_name>.yaml`.
2. Fill in every entry marked required. Optional entries carry a default; replace it once you have
   measured the value.
3. Run `{BUILD_COMMAND}`.

For the competition flight (climb to 50 m, hover 10 s, land within 10 m of the pad) the mission
entries are 50, 10 and 10.

The builder first checks the entries against each other and lists every one to change: a sensor or
control rate the simulator's 200 Hz step cannot run, a servo pulse range too short for the thrust angle, a
landing the legs or the time limit cannot take, and so on. It then writes the vehicle, motor, sensor,
mission and training files (`configs/vehicles/<name>.yaml`, `configs/motors/<name>_motor.yaml`,
`configs/sensors/<name>_sensors.yaml`, `configs/missions/<name>.yaml`, `configs/training/<name>.yaml`),
loads each one to make sure it is valid, prints a check of the vehicle (thrust to weight, hover throttle,
gimbal authority, controller gains, roll at hover, tip-over angle, battery time against the mission, the
training mission ranges) and the commands to fly the mission with the PID and to train. It never replaces
a file it did not write itself.
`configs/datasheets/example.yaml` is filled in with the stand-in numbers of the example vehicle.

Positions are measured from the top (nose) of the vehicle downward. A printable checklist of the
same entries is in [electric_vehicle_data_request.pdf](electric_vehicle_data_request.pdf).
"""


def template_text() -> str:
    """configs/datasheets/template.yaml: every entry, required ones empty, optional ones at their default."""
    return _sheet(TEMPLATE_HEADER, lambda group, entry: "" if entry.required or entry.default is None
                  else _yaml_number(entry.default))


def refreshed_sheet(text: str) -> str:
    """A filled-in sheet laid out like the template again, keeping its own header and values."""
    values: dict[tuple[str, str], str] = {}
    group = ""
    for line in text.splitlines():
        top, entry = re.match(r"^(\w+):", line), re.match(r"^  (\w+):[ \t]*(.*?)[ \t]*(#.*)?$", line)
        if top:
            group = top.group(1)
        elif entry:
            values[group, entry.group(1)] = entry.group(2)
    header = text[:text.index("\nvehicle:")]
    return _sheet(header, lambda group_key, entry: values.get((group_key, entry.key), ""))


def _sheet(header: str, value: Callable[[str, Entry], str]) -> str:
    lines = [header.rstrip("\n")]
    for group in GROUPS:
        lines += ["", f"{group.key}:"]
        for entry in group.entries:
            how = f" {entry.how[0].upper()}{entry.how[1:]}." if entry.how else ""
            lines += textwrap.wrap(f"{entry.what}.{how}", width=LINE_WIDTH, initial_indent="  # ",
                                   subsequent_indent="  # ")
            line = f"  {entry.key}: {value(group.key, entry)}".rstrip()
            lines.append(line.ljust(COMMENT_COLUMN) + f"# {_status(entry)}")
    return "\n".join(lines) + "\n"


def request_markdown() -> str:
    """docs/vehicle_data_request.md: what to measure, group by group, and what the builder works out."""
    parts = [INTRO]
    for group in GROUPS:
        parts.append(f"## {group.title}\n")
        parts.append("| What | Entry | Unit | Required or default | How |")
        parts.append("|---|---|---|---|---|")
        for entry in group.entries:
            default = f"**{entry.default_text}**" if entry.required else entry.default_text
            parts.append(f"| {entry.what} | `{group.key}.{entry.key}` | {entry.unit or '-'} | {default} | "
                         f"{entry.how or '-'} |")
        parts.append("")
    parts.append(worked_out_section())
    return "\n".join(parts)


def worked_out_section() -> str:
    """The derivations of rocketsim/hop/datasheet.py in words, with the constants it uses."""
    f, z = f"{ds.ATTITUDE_FREQUENCY:g}", f"{ds.ATTITUDE_DAMPING:g}"
    bullets = (
        "Mass and balance point without the fan unit, from the totals and the fan unit's mass and position.",
        "Pitch and roll inertia when they are empty: the body as a solid cylinder (pitch m (3 r² + L²) / 12, "
        "roll m r² / 2) plus the fan unit as a point mass. A swing test or CAD value is better.",
        "Gimbal authority A = hover thrust x (pivot - balance point) / pitch inertia: the pitch acceleration in "
        f"rad/s² for each radian of thrust deflection. The attitude gains put the tilt loop at {f} rad/s with "
        f"damping {z}: kp = {f}² / A, kd = 2 x {z} x {f} / A.",
        "Roll gains: the roll inertia I sits behind a lag T = roll response time + (command delay steps + "
        f"{ds.HELD_COMMAND_DELAY:g}) / control rate, so the roll loop has three poles. The gains put a pair with "
        f"damping {ds.ROLL_DAMPING:g} at w = min({ds.ROLL_FREQUENCY:g} rad/s, 1 / (3 x {ds.ROLL_DAMPING:g} x T)), "
        f"the stiffest loop the lag allows, and the third pole at p = 1 / T - 2 x {ds.ROLL_DAMPING:g} x w: "
        f"kp = I T p w², kd = I T (w² + 2 x {ds.ROLL_DAMPING:g} x w p). w is lowered further when the roll torque "
        f"would run out before {ds.ROLL_LINEAR_ANGLE_DEG:g} degrees of roll error.",
        f"Thrust vectoring speed: {ds.HORN_TRAVEL_DEG:g} degrees / servo time x linkage ratio. Touchdown speed the "
        "legs survive: sqrt(2 g h) from the drop height.",
        f"An empty flight time limit is the mission plan plus {ds.FLIGHT_TIME_MARGIN_S:g} s, rounded up to "
        f"{ds.FLIGHT_TIME_STEP_S:g} s.",
        "Training mission ranges: those of `configs/training/hop.yaml`, widened to reach "
        f"{VARIATION_HEADROOM:g} times the sheet's height and hover time, then cut until the longest training "
        f"mission plus a {checks.TIME_RESERVE:.0%} reserve fits the battery and the flight time limit.",
        "Position, altitude, estimator and safety settings are those of the example vehicle "
        "(`configs/vehicles/electric_hopper.yaml`).",
    )
    lines = ["## What the builder works out", ""]
    for bullet in bullets:
        lines += textwrap.wrap(bullet, width=LINE_WIDTH, initial_indent="- ", subsequent_indent="  ")
    lines += ["", f"This page, `{TEMPLATE_PATH}` and `electric_vehicle_data_request.pdf` are written from",
              "`rocketsim/hop/datasheet_fields.py` by `python scripts/write_data_request.py`.", ""]
    return "\n".join(lines)


def _status(entry: Entry) -> str:
    if entry.required:
        return "required"
    if entry.default is None and entry.fallback:
        return f"optional, {entry.fallback} when empty"
    return "optional"


def _yaml_number(value: float | int) -> str:
    return f"{value:g}" if isinstance(value, float) and not value.is_integer() else str(value)
