"""Checks across the entries of a data sheet: layouts, rates and flight plans the simulator cannot fly.

Every message names the sheet entries to change, in the sheet's own units.
"""

import math
from typing import Any

from rocketsim.hop.mission import MissionConfig
from rocketsim.simconfig import RATE_RATIO_TOLERANCE, SimConfig
from rocketsim.units import G_PER_KG, MM_PER_M

LANDING_HEIGHT_MARGIN_M = 1.0  # the plan should be down to the final speed at least this high above the pad
TOUCHDOWN_SPEED_MARGIN_MPS = 0.5  # flown touchdowns come in up to this much faster than the plan's final speed
TIME_RESERVE = 0.3  # share of the planned flight time kept in hand for wind and controller lag
SLOWEST_OFFERED_RATE_HZ = 20.0  # control rates below this are not suggested
SENSOR_RATES = ("imu_rate_hz", "barometer_rate_hz", "gps_rate_hz")

SheetValues = dict[str, dict[str, Any]]


def landing_height_needed(mission: MissionConfig) -> float:
    """Lowest final height from which the plan slows from the descent speed to the final speed, plus a margin."""
    slowing = (mission.descent_speed ** 2 - mission.final_speed ** 2) / (2.0 * mission.climb_acceleration)
    return max(slowing, 0.0) + LANDING_HEIGHT_MARGIN_M


def leg_touchdown_speed(drop_height_mm: float, gravity: float) -> float:
    """Touchdown speed in m/s the legs survive, from the drop height of the drop test."""
    return math.sqrt(2.0 * gravity * drop_height_mm / MM_PER_M)


def sheet_problems(sheet: SheetValues, plan: MissionConfig, sim: SimConfig) -> list[str]:
    """Everything wrong between the entries of a sheet whose entries are each valid on their own."""
    return (_layout(sheet, sim.environment.gravity) + _servos(sheet["thrust_vectoring"])
            + _rates(sheet["sensors"], sim.simulation.physics_rate_hz) + _mission(sheet, plan, sim))


def _layout(sheet: SheetValues, gravity: float) -> list[str]:
    vehicle, fan, vectoring = sheet["vehicle"], sheet["fan"], sheet["thrust_vectoring"]
    length, cg, mass = vehicle["length_mm"], vehicle["balance_point_mm"], vehicle["total_mass_g"]
    problems = []
    if fan["unit_mass_g"] >= mass:
        problems.append("fan.unit_mass_g must be less than vehicle.total_mass_g")
    for group, key in (("vehicle", "balance_point_mm"), ("fan", "position_mm"), ("thrust_vectoring", "pivot_mm"),
                       ("vehicle", "centre_of_pressure_mm")):
        value = sheet[group][key]
        if value is not None and value > length:
            problems.append(f"{group}.{key} ({value:g} mm) is beyond the body length ({length:g} mm)")
    if vectoring["pivot_mm"] <= cg:
        problems.append("thrust_vectoring.pivot_mm must be below the balance point (larger than balance_point_mm)")
    weight = mass / G_PER_KG * gravity
    if fan["max_thrust_n"] <= weight:
        problems.append(f"fan.max_thrust_n ({fan['max_thrust_n']:g} N) cannot lift the vehicle ({weight:.1f} N)")
    if fan["unit_mass_g"] < mass:
        dry_cg = (mass * cg - fan["unit_mass_g"] * fan["position_mm"]) / (mass - fan["unit_mass_g"])
        if not 0.0 < dry_cg < length:
            problems.append(f"the balance point without the fan unit comes out at {dry_cg:.0f} mm, outside the body; "
                            "check balance_point_mm, fan.unit_mass_g and fan.position_mm")
    return problems


def _servos(tv: dict[str, Any]) -> list[str]:
    center, low, high, step = tv["servo_center_us"], tv["servo_min_us"], tv["servo_max_us"], tv["pulse_resolution_us"]
    problems = []
    if not low < center < high:
        problems.append("thrust_vectoring.servo_center_us must lie between servo_min_us and servo_max_us")
    for key in ("servo_center_us", "servo_min_us", "servo_max_us"):
        if tv[key] % step != 0.0:
            problems.append(f"thrust_vectoring.{key} ({tv[key]:g} us) must be a multiple of pulse_resolution_us "
                            f"({step:g} us)")
    if tv["servo_deadband_deg"] >= tv["max_angle_deg"]:
        problems.append("thrust_vectoring.servo_deadband_deg must be smaller than max_angle_deg")
    travel = tv["max_angle_deg"] / tv["linkage_ratio"] * tv["servo_us_per_deg"]
    if low < center < high and travel > min(center - low, high - center):
        problems.append(
            f"thrust_vectoring.max_angle_deg ({tv['max_angle_deg']:g} deg) needs pulses {travel:.0f} us either side "
            f"of servo_center_us with linkage_ratio {tv['linkage_ratio']:g} and servo_us_per_deg "
            f"{tv['servo_us_per_deg']:g}, beyond servo_min_us..servo_max_us ({low:g}..{high:g} us)"
        )
    return problems


def _rates(sensors: dict[str, Any], physics_rate: float) -> list[str]:
    problems = []
    control = sensors["control_rate_hz"]
    steps = physics_rate / control
    if steps < 1.0 or abs(steps - round(steps)) > RATE_RATIO_TOLERANCE:
        offered = [physics_rate / n for n in range(1, int(physics_rate / SLOWEST_OFFERED_RATE_HZ) + 1)]
        allowed = ", ".join(f"{rate:g}" for rate in offered if rate.is_integer())
        problems.append(f"sensors.control_rate_hz ({control:g} Hz) must divide the simulator's {physics_rate:g} Hz "
                        f"step rate: use one of {allowed}")
    for key in SENSOR_RATES:
        if sensors[key] > physics_rate:
            problems.append(f"sensors.{key} ({sensors[key]:g} Hz) is faster than the simulator's {physics_rate:g} Hz "
                            f"step rate; enter {physics_rate:g} for a faster sensor")
    return problems


def _mission(sheet: SheetValues, plan: MissionConfig, sim: SimConfig) -> list[str]:
    mission = sheet["mission"]
    problems = []
    if plan.final_height >= plan.target_altitude:
        problems.append("mission.final_height_m must be below target_height_m")
    if plan.final_speed > plan.descent_speed:
        problems.append("mission.final_speed_mps must not be more than descent_speed_mps")
    needed = landing_height_needed(plan)
    if plan.final_height < needed:
        problems.append(
            f"mission.final_height_m ({plan.final_height:g} m) is too low to slow from descent_speed_mps to "
            f"final_speed_mps at climb_acceleration_mps2: make it at least {needed:.1f} m"
        )
    legs = leg_touchdown_speed(sheet["legs"]["max_drop_height_mm"], sim.environment.gravity)
    if plan.final_speed > legs - TOUCHDOWN_SPEED_MARGIN_MPS:
        problems.append(
            f"mission.final_speed_mps ({plan.final_speed:g} m/s) leaves too little margin to the {legs:.2f} m/s the "
            f"legs survive (legs.max_drop_height_mm): flown touchdowns come in up to "
            f"{TOUCHDOWN_SPEED_MARGIN_MPS:g} m/s faster, so make it at most {legs - TOUCHDOWN_SPEED_MARGIN_MPS:.2f} m/s"
        )
    pad, flight = sim.simulation.pad_hold_time, plan.planned_duration
    if mission["max_flight_time_s"] is not None and mission["max_flight_time_s"] < pad + flight:
        problems.append(f"mission.max_flight_time_s ({mission['max_flight_time_s']:g} s) is shorter than the "
                        f"{pad:g} s pad calibration plus the {flight:.0f} s plan")
    return problems
