"""A short check of a built vehicle: thrust margin, control authority, tip-over angle, battery, warnings."""

from dataclasses import dataclass

from rocketsim.hop.datasheet import ATTITUDE_DAMPING, ATTITUDE_FREQUENCY, ROLL_DAMPING, roll_frequency_limit
from rocketsim.hop.datasheet_checks import TIME_RESERVE
from rocketsim.hop.vehicle_files import BuiltVehicle
from rocketsim.touchdown import tip_over_angle
from rocketsim.units import MM_PER_M, MS_PER_S, deg_to_rad, rad_to_deg

MIN_THRUST_TO_WEIGHT = 1.4  # below this there is little thrust left to climb, fight wind or stop a descent
MAX_HOVER_THROTTLE = 0.7
HALF_THROTTLE = 0.5
THRUST_CURVE_TOLERANCE = 0.1  # share of full thrust the 50 % throttle point may be off a straight line
ROLL_FREQUENCY_TOLERANCE = 0.99
MAX_HOVER_ROLL_DEG = 90.0  # half way to the 180 deg at which the roll loop loses its sense of direction


@dataclass(frozen=True)
class VehicleCheck:
    mass: float  # kg
    thrust_to_weight: float
    hover_throttle: float
    authority: float  # rad/s2 per rad of gimbal at hover thrust
    tip_over: float  # rad
    hover_roll: float  # rad the fan's twist turns the vehicle at hover against the roll loop
    battery_time: float  # s
    mission_time: float  # s, planned launch to touchdown
    warnings: tuple[str, ...]


def check_vehicle(built: BuiltVehicle) -> VehicleCheck:
    """Numbers to look at before the first flight, and warnings for the ones out of the usual range."""
    vehicle, mission, sheet, derived = built.vehicle, built.mission, built.sheet, built.derived
    weight = vehicle.mass * built.sim.environment.gravity
    thrust_to_weight = vehicle.max_thrust / weight
    hover_throttle = 1.0 / thrust_to_weight
    tip_over = tip_over_angle(vehicle.body, vehicle.body.loaded_cg)
    battery, plan = vehicle.motor.spec.burn_time, mission.planned_duration
    hover_roll = vehicle.motor.spec.reaction_torque_per_n * weight / vehicle.roll.kp
    warnings = []
    if thrust_to_weight < MIN_THRUST_TO_WEIGHT:
        warnings.append(f"thrust to weight {thrust_to_weight:.2f} is below {MIN_THRUST_TO_WEIGHT}: little thrust "
                        "is left to climb, hold against wind or stop a descent")
    if hover_throttle > MAX_HOVER_THROTTLE:
        warnings.append(f"hover throttle {hover_throttle:.2f} is above {MAX_HOVER_THROTTLE}")
    if not sheet.has_gps:
        warnings.append(f"no GPS: the position drifts with the IMU alone, so landing inside the "
                        f"{mission.landing_radius:g} m circle is unlikely")
    if battery < plan * (1.0 + TIME_RESERVE):
        warnings.append(f"the battery lasts {battery:.0f} s, less than the {plan:.0f} s plan plus a "
                        f"{TIME_RESERVE:.0%} reserve ({plan * (1.0 + TIME_RESERVE):.0f} s)")
    pad = built.sim.simulation.pad_hold_time
    if mission.max_flight_time < pad + plan * (1.0 + TIME_RESERVE):
        warnings.append(f"the {mission.max_flight_time:g} s flight time limit leaves less than a {TIME_RESERVE:.0%} "
                        f"reserve over the {pad:g} s pad calibration and the {plan:.0f} s plan")
    reaction = vehicle.motor.spec.reaction_torque_per_n * vehicle.max_thrust
    if reaction >= vehicle.roll.max_torque:
        warnings.append(f"at full thrust the fan twists the vehicle with {reaction:.3f} N m, more than the roll "
                        f"control's {vehicle.roll.max_torque:.3f} N m")
    reachable = roll_frequency_limit(derived.roll_lag)
    if derived.roll_frequency < reachable * ROLL_FREQUENCY_TOLERANCE:
        warnings.append(f"the roll torque limits the roll loop to {derived.roll_frequency:.1f} rad/s "
                        f"instead of {reachable:.1f}")
    if hover_roll > deg_to_rad(MAX_HOVER_ROLL_DEG):
        warnings.append(f"at hover the fan's twist turns the vehicle {rad_to_deg(hover_roll):.0f} deg in roll before "
                        f"the roll loop holds it: a faster roll actuator (roll_control.response_ms) or a smaller fan "
                        f"twist (fan.reaction_torque_nm_per_n, worth measuring) brings it down")
    half = sheet["fan"]["half_throttle_thrust_n"]
    if half is not None and abs(half / vehicle.max_thrust - HALF_THROTTLE) > THRUST_CURVE_TOLERANCE:
        warnings.append(f"half throttle gives {half / vehicle.max_thrust:.0%} of full thrust; the simulator assumes "
                        "thrust follows the throttle in a straight line, so the flight code must map one to the other")
    if tip_over < vehicle.body.legs.max_tilt:
        warnings.append(f"the vehicle tips over at {rad_to_deg(tip_over):.1f} deg, before the legs' "
                        f"{rad_to_deg(vehicle.body.legs.max_tilt):g} deg limit: widen the foot circle")
    strongest = max((stage.level.wind_max or 0.0 for stage in built.training.curriculum), default=0.0)
    if built.training.curriculum and sheet["launch_site"]["wind_mps"] > strongest:
        warnings.append(f"the expected wind of {sheet['launch_site']['wind_mps']:g} m/s is stronger than the "
                        f"strongest training wind ({strongest:g} m/s)")
    return VehicleCheck(vehicle.mass, thrust_to_weight, hover_throttle, derived.authority, tip_over, hover_roll,
                        battery, plan, tuple(warnings))


def check_lines(built: BuiltVehicle, check: VehicleCheck) -> list[str]:
    """The check as aligned lines of text."""
    d = built.derived
    rows = [
        ("mass", f"{check.mass:.3f} kg, balance point {d.cg * MM_PER_M:.0f} mm from the top"),
        ("thrust to weight", f"{check.thrust_to_weight:.2f}"),
        ("hover throttle", f"{check.hover_throttle:.2f}"),
        ("gimbal authority", f"{check.authority:.1f} rad/s2 per rad of thrust angle "
                             f"({d.lever_arm * MM_PER_M:.0f} mm lever arm, pitch inertia {d.pitch_inertia:.4f} kg m2)"),
        ("attitude gains", f"kp {d.attitude_kp:.3f}, kd {d.attitude_kd:.3f} s "
                           f"({ATTITUDE_FREQUENCY:g} rad/s, damping {ATTITUDE_DAMPING:g})"),
        ("roll gains", f"kp {d.roll_kp:.3f} N m/rad, kd {d.roll_kd:.4f} N m s/rad ({d.roll_frequency:.1f} rad/s, "
                       f"damping {ROLL_DAMPING:g}, {d.roll_lag * MS_PER_S:.0f} ms roll lag)"),
        ("roll at hover", f"{rad_to_deg(check.hover_roll):.0f} deg from the fan's twist"),
        ("tip-over angle", f"{rad_to_deg(check.tip_over):.1f} deg"),
        ("battery", f"{check.battery_time:.0f} s for a {check.mission_time:.0f} s mission plan"),
        ("training missions", _training_missions(built)),
    ]
    width = max(len(label) for label, _ in rows) + 2
    lines = [f"  {label.ljust(width)}{value}" for label, value in rows]
    lines.append("warnings:" if check.warnings else "warnings: none")
    lines += [f"  - {warning}" for warning in check.warnings]
    return lines


def _training_missions(built: BuiltVehicle) -> str:
    variation = built.training.mission_variation
    parts = []
    if variation.target_altitude is not None:
        parts.append(f"{variation.target_altitude[0]:g} to {variation.target_altitude[1]:g} m")
    if variation.hover_time is not None:
        parts.append(f"hover {variation.hover_time[0]:g} to {variation.hover_time[1]:g} s")
    return ", ".join(parts) or "the sheet's mission only"
