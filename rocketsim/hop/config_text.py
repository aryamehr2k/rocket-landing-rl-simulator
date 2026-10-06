"""Text of the vehicle, motor, sensor, mission and training YAML files written from a data sheet."""

from typing import Any

import numpy as np

from rocketsim.hop.datasheet import (
    ATTITUDE_DAMPING, ATTITUDE_FREQUENCY, ROLL_DAMPING, Datasheet, Derived,
)
from rocketsim.units import G_PER_KG, MM_PER_M, MS_PER_S, STANDARD_GRAVITY, rad_to_deg

DIGITS = 4  # significant digits of worked-out numbers
MASS_DIGITS = 5  # worked-out masses and positions, good to a gram and a millimetre on a large vehicle
GAIN_DIGITS = 3
SHEET_DIGITS = 6  # numbers copied from the sheet keep what was typed
COMMENT_COLUMN = 33
BUILDER = "scripts/build_vehicle.py"  # named in every built file, which marks the files the builder may replace
SAFETY_GIMBAL_RATE_DEG_PER_S = 150.0  # kept below the servo speed so the commands stay smooth
MAX_THROTTLE_RATE_PER_S = 2.0
# Controller and estimator settings taken unchanged from configs/vehicles/electric_hopper.yaml.
ESTIMATOR = {"pad_average_time_s": 1.0, "baro_altitude_gain": 0.10, "baro_velocity_gain_per_s": 0.50}
GPS_POSITION_GAIN = 0.05
GPS_VELOCITY_GAIN = 0.10
POSITION_PID = {"kp_deg_per_m": 2.0, "kd_deg_per_mps": 4.0, "max_tilt_command_deg": 15.0}
ATTITUDE_KI_PER_S = 0.0
ATTITUDE_MAX_INTEGRAL_DEG = 3.0
ALTITUDE_PID = {"height_gain_per_s": 1.0, "max_speed_correction_mps": 2.0, "speed_gain_per_s": 2.0,
                "speed_integral_gain_per_s2": 0.5, "max_integral_accel_mps2": 2.0}
SENSOR_MODELS = (("IMU", "imu_model"), ("barometer", "barometer_model"), ("GPS", "gps_model"),
                 ("board", "flight_computer"))


def vehicle_yaml(name: str, sheet: Datasheet, d: Derived) -> str:
    """configs/vehicles/<name>.yaml: airframe, gimbal, legs, controller gains and safety limits."""
    v, tv, legs, sensors = sheet["vehicle"], sheet["thrust_vectoring"], sheet["legs"], sheet["sensors"]
    roll, mission = sheet["roll_control"], sheet["mission"]
    servos = "\n".join(_servo(axis, tv) for axis in ("pitch", "yaw"))
    if sheet.has_gps:
        sensor_note = "the GPS keeps the horizontal position"
        gps = (f"\n  gps_position_gain: {num(GPS_POSITION_GAIN)} # share of each GPS position error applied"
               f"\n  gps_velocity_gain: {num(GPS_VELOCITY_GAIN)}")
    else:
        sensor_note, gps = "no GPS: the horizontal position drifts and the landing with it", ""
    gimbal_rate = rad_to_deg(d.gimbal_rate)
    return aligned(f"""\
# {name}: electric test vehicle, one ducted fan with thrust vectoring and electric roll control.
{built_from(sheet.source)}
# Positions are stations from the nose tip toward the tail, as for the rocket (docs/conventions.md).

name: {name}

airframe:
  dry_mass_g: {num(d.dry_mass * G_PER_KG, MASS_DIGITS)} # everything except the fan unit, battery included
  dry_cg_from_nose_mm: {num(d.dry_cg * MM_PER_M, MASS_DIGITS)} # balance point without the fan unit
  dry_pitch_inertia_kgm2: {num(d.dry_pitch_inertia)} # without the fan unit, about its own balance point
  dry_roll_inertia_kgm2: {num(d.dry_roll_inertia)}
  length_mm: {num(v["length_mm"], SHEET_DIGITS)}
  reference_diameter_mm: {num(v["diameter_mm"], SHEET_DIGITS)}

aero:
  drag_coefficient: {num(v["drag_coefficient"])}
  normal_force_slope_per_rad: {num(v["side_force_slope_per_rad"])} # side force for wind across the whole body
  cp_from_nose_mm: {num(d.centre_of_pressure * MM_PER_M, MASS_DIGITS)}

motor:
  file: ../motors/{name}_motor.yaml
  position_from_nose_mm: {num(sheet["fan"]["position_mm"], SHEET_DIGITS)}
  gimbaled: true # thrust vectoring by gimbal or vanes

gimbal:
  pivot_from_nose_mm: {num(tv["pivot_mm"], SHEET_DIGITS)}
  max_angle_deg: {num(tv["max_angle_deg"])}
  servo_rate_limit_deg_per_s: {num(gimbal_rate)} # servo speed x linkage ratio
  servo_delay_s: {num(tv["servo_delay_ms"] / MS_PER_S)}
  servo_deadband_deg: {num(tv["servo_deadband_deg"])}
  servos:
{servos}

roll_control:{_note(roll["kind"])}
  max_torque_nm: {num(roll["max_torque_nm"])}
  time_constant_s: {num(roll["response_ms"] / MS_PER_S)}
  kp_nm_per_rad: {num(d.roll_kp, GAIN_DIGITS)} # roll loop at {d.roll_frequency:.1f} rad/s, damping {ROLL_DAMPING:g}, \
{d.roll_lag * MS_PER_S:.0f} ms lag
  kd_nm_s_per_rad: {num(d.roll_kd, GAIN_DIGITS)}

legs:
  span_mm: {num(legs["foot_circle_mm"], SHEET_DIGITS)}
  height_mm: {num(legs["below_body_mm"], SHEET_DIGITS)}
  count: {legs["count"]}
  max_touchdown_vertical_speed_mps: {num(d.touchdown_speed)} # {legs["max_drop_height_mm"]:g} mm drop test
  max_touchdown_lateral_speed_mps: {num(legs["max_sideways_speed_mps"])}
  max_touchdown_tilt_deg: {num(legs["max_tilt_deg"])}

control:
  control_rate_hz: {_whole(sensors["control_rate_hz"])}
  action_delay_steps: {sensors["command_delay_steps"]}

sensors:
  file: ../sensors/{name}_sensors.yaml # {sensor_note}

estimator:{_mapping(ESTIMATOR)}{gps}

controllers: # who flies: pid or policy (a trained model, scripts/fly_mission.py --model)
  steering: policy
  throttle: policy

attitude_pid: # baseline steering, also the fallback when the policy output is invalid
  attitude: # gimbal authority {d.authority:.1f} rad/s2 per rad; gains for {ATTITUDE_FREQUENCY:g} rad/s, damping \
{ATTITUDE_DAMPING:g}
    kp: {num(d.attitude_kp, GAIN_DIGITS)}
    ki_per_s: {num(ATTITUDE_KI_PER_S)}
    kd_s: {num(d.attitude_kd, GAIN_DIGITS)}
    max_integral_deg: {num(ATTITUDE_MAX_INTEGRAL_DEG)}
  position:{_mapping(POSITION_PID, indent=4)}

altitude_pid: # baseline throttle: follows the mission's reference height and speed{_mapping(ALTITUDE_PID)}

safety:
  max_gimbal_rate_deg_per_s: {num(min(SAFETY_GIMBAL_RATE_DEG_PER_S, gimbal_rate))}
  max_throttle_rate_per_s: {num(MAX_THROTTLE_RATE_PER_S)}
  abort_tilt_deg: {num(mission["abort_tilt_deg"])} # beyond this the fan is cut
  geofence_radius_m: {num(mission["geofence_radius_m"])} # and beyond this distance from the pad
""")


def motor_yaml(name: str, sheet: Datasheet) -> str:
    """configs/motors/<name>_motor.yaml: the fan unit as an electric motor."""
    fan = sheet["fan"]
    return aligned(f"""\
# Fan unit of {name}{_note(fan["model"], ": ")}.
{built_from(sheet.source)}
name: {name.upper()}_FAN
type: electric
mass_g: {num(fan["unit_mass_g"], SHEET_DIGITS)} # fan, motor and ESC; the battery belongs to the airframe dry mass
max_thrust_n: {num(fan["max_thrust_n"], SHEET_DIGITS)} # static thrust at full throttle with a charged battery
spin_up_time_constant_s: {num(fan["spin_up_ms"] / MS_PER_S)}
max_run_time_s: {num(fan["battery_time_s"], SHEET_DIGITS)} # battery endurance at hover
reaction_torque_nm_per_n: {num(fan["reaction_torque_nm_per_n"])} # roll torque of the spinning fan per newton
diameter_mm: {num(fan["diameter_mm"], SHEET_DIGITS)}
length_mm: {num(fan["length_mm"] or fan["diameter_mm"], SHEET_DIGITS)}
""")


def sensors_yaml(name: str, sheet: Datasheet) -> str:
    """configs/sensors/<name>_sensors.yaml: IMU, barometer and, when there is one, the GPS."""
    s = sheet["sensors"]
    models = ", ".join(f"{label} {s[key]}" for label, key in SENSOR_MODELS if s[key])
    full_scale = s["accel_range_mps2"] / STANDARD_GRAVITY
    if sheet.has_gps:
        gps = f"""
gps:
  rate_hz: {_whole(s["gps_rate_hz"])}
  lag_s: {num(s["gps_lag_ms"] / MS_PER_S)}
  position_noise_std_m: {num(s["gps_noise_m"])} # sample to sample scatter
  position_drift_std_m: {num(s["gps_drift_m"])} # slow wander of the position over a minute
  velocity_noise_std_mps: {num(s["gps_speed_noise_mps"])}
"""
    else:
        gps = "\n# No GPS: the horizontal position comes from the IMU alone and drifts during the flight.\n"
    return aligned(f"""\
# Sensors of {name}{f" ({models})" if models else ""}.
{built_from(sheet.source)}
# A bias is drawn once per flight from its standard deviation; the estimator calibrates it away on the pad.

name: {name}_sensors

imu:
  rate_hz: {_whole(s["imu_rate_hz"])}
  lag_s: {num(s["imu_lag_ms"] / MS_PER_S)} # from the true motion to the value in the register
  accelerometer:
    noise_std_mps2: {num(s["accel_noise_mps2"])}
    bias_std_mps2: {num(s["accel_bias_mps2"])}
    range_mps2: {num(s["accel_range_mps2"])} # {full_scale:.0f} g full scale
  gyroscope:
    noise_std_deg_per_s: {num(s["gyro_noise_deg_per_s"])}
    bias_std_deg_per_s: {num(s["gyro_bias_deg_per_s"])}
    range_deg_per_s: {num(s["gyro_range_deg_per_s"])}

barometer:
  rate_hz: {_whole(s["barometer_rate_hz"])}
  lag_s: {num(s["barometer_lag_ms"] / MS_PER_S)}
  noise_std_m: {num(s["barometer_noise_m"])}
  bias_std_m: {num(s["barometer_bias_m"])}
{gps}""")


def mission_yaml(name: str, sheet: Datasheet, d: Derived) -> str:
    """configs/missions/<name>.yaml: the flight plan and its pass criteria."""
    m = sheet["mission"]
    return aligned(f"""\
# Flight plan for {name}: climb to {m["target_height_m"]:g} m, hold {m["hover_time_s"]:g} s, land within \
{m["landing_radius_m"]:g} m of the pad.
{built_from(sheet.source)}
# Heights are of the landing feet above the pad.

name: {name}

ascent:
  target_altitude_m: {num(m["target_height_m"])}
  climb_speed_mps: {num(m["climb_speed_mps"])} # steady climb rate
  climb_acceleration_mps2: {num(m["climb_acceleration_mps2"])} # how fast the reference speeds up and slows down

hover:
  time_s: {num(m["hover_time_s"])} # required hold at the target height
  extra_time_s: {num(m["extra_hover_s"])} # the reference hovers this much longer, for margin

descent:
  descent_speed_mps: {num(m["descent_speed_mps"])}
  final_height_m: {num(m["final_height_m"])} # below this height the vehicle slows to the final speed
  final_speed_mps: {num(m["final_speed_mps"])} # touchdown speed the reference aims for

criteria:
  altitude_tolerance_m: {num(m["height_tolerance_m"])} # counts as at the target height when within this band
  landing_radius_m: {num(m["landing_radius_m"])} # touchdown must be this close to the pad

max_flight_time_s: {num(d.mission.max_flight_time)}
""")


def num(value: float, digits: int = DIGITS) -> str:
    """A number for YAML with `digits` significant digits, never in exponent form."""
    return np.format_float_positional(float(value), precision=digits, unique=False, fractional=False, trim="0")


def aligned(text: str) -> str:
    """Line up the end-of-line comments of a YAML text at COMMENT_COLUMN."""
    lines = []
    for line in text.splitlines():
        code, mark, comment = line.partition(" # ")
        if mark and code.strip() and not code.lstrip().startswith("#"):
            line = f"{code.ljust(COMMENT_COLUMN - 1)} # {comment}"
        lines.append(line)
    return "\n".join(lines) + "\n"


def built_from(source: str) -> str:
    """The header lines that name the sheet a file was built from."""
    return (f"# Built from {source} by {BUILDER}.\n"
            "# Change the data sheet and build again rather than editing this file.")


def _note(value: str | None, prefix: str = " # ") -> str:
    return f"{prefix}{value}" if value else ""


def _servo(axis: str, tv: dict[str, Any]) -> str:
    return (f"    {axis}:\n      center_us: {_whole(tv['servo_center_us'])}\n"
            f"      pulse_us_per_deg: {num(tv['servo_us_per_deg'])}\n      sign: {tv[f'{axis}_servo_sign']}\n"
            f"      linkage_ratio: {num(tv['linkage_ratio'])}\n"
            f"      pulse_resolution_us: {_whole(tv['pulse_resolution_us'])}\n"
            f"      min_us: {_whole(tv['servo_min_us'])}\n      max_us: {_whole(tv['servo_max_us'])}")


def _whole(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else num(value, SHEET_DIGITS)


def _mapping(values: dict[str, float], indent: int = 2) -> str:
    return "".join(f"\n{' ' * indent}{key}: {num(value)}" for key, value in values.items())
