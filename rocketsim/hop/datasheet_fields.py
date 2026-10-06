"""Every entry of the electric vehicle data sheet: unit, required or default, and how to measure it.

The template, docs/vehicle_data_request.md and the printable checklist are written from this list.
"""

from dataclasses import dataclass
from typing import Any

TEXT, NUMBER, INTEGER = "text", "number", "integer"
SIGNS = (1, -1)
MAX_ANGLE_DEG = 90.0  # the vehicle file loaders refuse larger gimbal, tilt and abort angles


@dataclass(frozen=True)
class Entry:
    key: str
    unit: str
    what: str
    how: str = ""
    required: bool = False
    default: Any = None
    fallback: str = ""  # what an empty optional entry is worked out from, when it has no fixed default
    kind: str = NUMBER
    minimum: float | None = 0.0
    inclusive: bool = False  # the minimum itself is allowed
    maximum: float | None = None
    choices: tuple[int, ...] = ()

    @property
    def default_text(self) -> str:
        """`required`, the default value, or what an empty entry falls back to."""
        if self.required:
            return "required"
        if self.default is not None:
            return f"{self.default:g}" if isinstance(self.default, float) else str(self.default)
        return self.fallback or "optional"


@dataclass(frozen=True)
class Group:
    key: str
    title: str
    entries: tuple[Entry, ...]


def required(key: str, unit: str, what: str, how: str = "", **limits: Any) -> Entry:
    return Entry(key, unit, what, how, required=True, **limits)


def optional(key: str, unit: str, what: str, how: str = "", default: Any = None, **limits: Any) -> Entry:
    return Entry(key, unit, what, how, default=default, **limits)


def text(key: str, what: str, fallback: str = "") -> Entry:
    return Entry(key, "", what, fallback=fallback, kind=TEXT, minimum=None)


GROUPS: tuple[Group, ...] = (
    Group("vehicle", "Vehicle", (
        text("name", "Short name for the config files: letters, digits, _", fallback="the sheet's file name"),
        required("total_mass_g", "g", "Mass ready to fly, battery and fan unit in", "scale"),
        required("balance_point_mm", "mm", "Balance point (centre of gravity) from the top, ready to fly",
                 "balance it across a ruler edge"),
        required("length_mm", "mm", "Body length from the top to the bottom end, legs not counted", "tape measure"),
        required("diameter_mm", "mm", "Body outside diameter", "calliper"),
        optional("pitch_inertia_kgm2", "kg m2", "Pitch inertia of the whole vehicle about its balance point",
                 "CAD (g mm2 / 1e9) or a swing test", fallback="solid cylinder"),
        optional("roll_inertia_kgm2", "kg m2", "Roll inertia of the whole vehicle about its long axis", "CAD",
                 fallback="solid cylinder"),
        optional("drag_coefficient", "", "Drag coefficient on the body diameter", "OpenRocket", 0.8),
        optional("centre_of_pressure_mm", "mm", "Centre of pressure from the top, wind from the side", "OpenRocket",
                 fallback="middle of the body"),
        optional("side_force_slope_per_rad", "1/rad", "Side force slope for wind across the body",
                 "keep unless you have wind tunnel data", 8.0, inclusive=True),
    )),
    Group("fan", "Fan and battery", (
        text("model", "Fan, motor and ESC model"),
        required("unit_mass_g", "g", "Fan, motor and ESC as mounted, battery not included", "scale before mounting"),
        required("position_mm", "mm", "Centre of the fan unit from the top", "tape measure"),
        required("diameter_mm", "mm", "Fan (duct) diameter", "calliper"),
        optional("length_mm", "mm", "Fan unit length", "calliper", fallback="same as the diameter"),
        required("max_thrust_n", "N", "Static thrust at full throttle, freshly charged battery", "thrust stand"),
        optional("half_throttle_thrust_n", "N", "Static thrust at 50 % throttle",
                 "same test; shows whether thrust follows the throttle in a straight line", fallback="not checked"),
        required("spin_up_ms", "ms", "Time for the thrust to reach 63 % of a throttle step",
                 "thrust stand log at 100 Hz or faster"),
        required("battery_time_s", "s", "Hover time on one charge down to the low voltage cut",
                 "thrust stand at hover throttle"),
        optional("reaction_torque_nm_per_n", "N m/N", "Roll torque of the spinning fan per newton of thrust",
                 "torque arm on the thrust stand", 0.003, inclusive=True),
    )),
    Group("thrust_vectoring", "Thrust vectoring and servos", (
        required("pivot_mm", "mm", "Gimbal pivot or vane hinge line from the top", "tape measure"),
        required("max_angle_deg", "deg", "Largest thrust deflection each way, each axis",
                 "phone level at full servo travel", maximum=MAX_ANGLE_DEG),
        required("linkage_ratio", "", "Thrust angle per degree of servo horn travel",
                 "move the horn 20 deg, divide the thrust angle by 20"),
        text("servo_model", "Servo model"),
        required("servo_time_per_60deg_s", "s", "Servo speed: time for 60 deg of horn travel at your voltage",
                 "servo datasheet"),
        optional("servo_delay_ms", "ms", "Time from a command to the horn starting to move", "slow motion video",
                 20.0, inclusive=True),
        optional("servo_deadband_deg", "deg", "Smallest thrust angle change the servo follows",
                 "step the command on the bench", 0.1, inclusive=True),
        optional("servo_center_us", "us", "Pulse that holds the thrust straight", "find it on the bench", 1500),
        optional("servo_us_per_deg", "us/deg", "Pulse change per degree of horn travel", "servo datasheet", 10.0),
        optional("servo_min_us", "us", "Shortest pulse the flight code may send", "servo datasheet", 1000),
        optional("servo_max_us", "us", "Longest pulse the flight code may send", "servo datasheet", 2000),
        optional("pulse_resolution_us", "us", "Smallest pulse step the board puts out", "board datasheet", 1),
        optional("pitch_servo_sign", "", "+1 if a longer pitch servo pulse swings the air jet toward +x, else -1",
                 "mark one side of the body the pitch servo swings the jet toward as +x, send servo_center_us + 100 "
                 "and watch the air leaving the duct. The simulator cannot catch a wrong sign, only this test can", 1,
                 kind=INTEGER, minimum=None, choices=SIGNS),
        optional("yaw_servo_sign", "", "+1 if a longer yaw servo pulse swings the air jet toward +y, else -1",
                 "+y is 90 deg anticlockwise from +x, seen from above. Same test with the yaw servo", 1,
                 kind=INTEGER, minimum=None, choices=SIGNS),
    )),
    Group("roll_control", "Roll control", (
        text("kind", "Roll fans, reaction wheel or vanes"),
        required("max_torque_nm", "N m", "Largest roll torque",
                 "roll fan thrust x its distance from the centre line x fans pushing the same way"),
        optional("response_ms", "ms", "Time to reach 63 % of a torque step", "thrust stand log", 50.0,
                 inclusive=True),
    )),
    Group("legs", "Landing legs", (
        required("count", "", "Number of legs", kind=INTEGER, minimum=3, inclusive=True),
        required("foot_circle_mm", "mm", "Diameter of the circle through the feet", "tape measure"),
        required("below_body_mm", "mm", "How far the feet reach below the bottom end of the body", "tape measure",
                 inclusive=True),
        required("max_drop_height_mm", "mm", "Highest drop onto the legs they survive, at flight mass",
                 "drop test; 115 mm lands at 1.5 m/s"),
        optional("max_sideways_speed_mps", "m/s", "Sideways speed at touchdown the legs survive",
                 "drop test with a push", 1.0),
        optional("max_tilt_deg", "deg", "Tilt at touchdown the legs survive", "drop test on a slope", 10.0,
                 maximum=MAX_ANGLE_DEG),
    )),
    Group("sensors", "Sensors and flight computer", (
        text("flight_computer", "Flight computer board"),
        required("control_rate_hz", "Hz", "How often the flight code runs the control loop",
                 "flight code; must divide the simulator's 200 Hz step: 200, 100, 50, 40 or 25"),
        optional("command_delay_steps", "steps", "Control steps from reading the sensors to the servo command",
                 "flight code", 1, kind=INTEGER, inclusive=True),
        text("imu_model", "IMU model"),
        required("imu_rate_hz", "Hz", "IMU sample rate the flight code reads",
                 "flight code; at most 200, the simulator's step rate, so enter 200 for a faster IMU"),
        optional("imu_lag_ms", "ms", "IMU filter delay", "datasheet, from the filter setting", 2.0, inclusive=True),
        optional("accel_noise_mps2", "m/s2", "Accelerometer noise",
                 "30 s log with the board still: standard deviation", 0.05, inclusive=True),
        optional("accel_bias_mps2", "m/s2", "Accelerometer bias change between power-ups",
                 "mean of several still logs", 0.02, inclusive=True),
        optional("accel_range_mps2", "m/s2", "Accelerometer full scale (16 g is 157)", "datasheet", 156.0),
        optional("gyro_noise_deg_per_s", "deg/s", "Gyroscope noise", "same still log", 0.1, inclusive=True),
        optional("gyro_bias_deg_per_s", "deg/s", "Gyroscope bias change between power-ups", "several still logs",
                 0.5, inclusive=True),
        optional("gyro_range_deg_per_s", "deg/s", "Gyroscope full scale", "datasheet", 2000.0),
        text("barometer_model", "Barometer model"),
        required("barometer_rate_hz", "Hz", "Barometer sample rate the flight code reads", "flight code; at most 200"),
        optional("barometer_lag_ms", "ms", "Barometer filter delay", "datasheet, from the oversampling setting",
                 20.0, inclusive=True),
        optional("barometer_noise_m", "m", "Barometer height noise", "same still log", 0.3, inclusive=True),
        optional("barometer_bias_m", "m", "Barometer height drift over a flight", "10 min still log", 1.0,
                 inclusive=True),
        text("gps_model", "GPS model"),
        required("gps_rate_hz", "Hz", "GPS update rate, 0 if there is no GPS", "flight code; at most 200",
                 inclusive=True),
        optional("gps_lag_ms", "ms", "GPS delay", "datasheet", 100.0, inclusive=True),
        optional("gps_noise_m", "m", "GPS position scatter from sample to sample",
                 "2 min log standing still at the field", 0.8, inclusive=True),
        optional("gps_drift_m", "m", "Slow GPS position wander over a minute", "same log", 1.0, inclusive=True),
        optional("gps_speed_noise_mps", "m/s", "GPS speed noise", "same log", 0.1, inclusive=True),
    )),
    Group("launch_site", "Launch site", (
        optional("elevation_m", "m", "Field height above sea level", "map", 300.0, minimum=None),
        optional("wind_mps", "m/s", "Steady wind expected on test day", "weather forecast", 4.0, inclusive=True),
        optional("gust_mps", "m/s", "Gust strength (standard deviation)", "weather forecast", 1.5, inclusive=True),
    )),
    Group("mission", "Mission", (
        required("target_height_m", "m", "Hover height of the feet above the pad", "competition rules"),
        required("hover_time_s", "s", "Hover time at the target height", "competition rules", inclusive=True),
        required("landing_radius_m", "m", "Touchdown must be within this distance of the pad", "competition rules"),
        optional("height_tolerance_m", "m", "Counts as at the target height within this band", "competition rules",
                 2.0),
        optional("climb_speed_mps", "m/s", "Planned climb speed", "", 4.0),
        optional("climb_acceleration_mps2", "m/s2", "How quickly the plan speeds up and slows down", "", 1.5),
        optional("descent_speed_mps", "m/s", "Planned descent speed", "", 3.0),
        optional("final_height_m", "m", "Height where the plan slows to the touchdown speed",
                 "at least (descent_speed^2 - final_speed^2) / (2 x climb_acceleration) + 1 m", 4.0),
        optional("final_speed_mps", "m/s", "Touchdown speed the plan aims for",
                 "at least 0.5 m/s below what the legs survive", 0.7),
        optional("extra_hover_s", "s", "Extra hover time planned so the required time is met with margin", "",
                 2.0, inclusive=True),
        optional("max_flight_time_s", "s", "Flight time limit", "competition rules; longer than the plan",
                 fallback="plan + 40 s"),
        optional("abort_tilt_deg", "deg", "Tilt at which the flight computer cuts the fan", "", 40.0,
                 maximum=MAX_ANGLE_DEG),
        optional("geofence_radius_m", "m", "Distance from the pad at which the flight computer cuts the fan", "",
                 40.0),
    )),
)


def find_entry(group: str, key: str) -> Entry:
    for candidate in GROUPS:
        if candidate.key == group:
            for entry in candidate.entries:
                if entry.key == key:
                    return entry
    raise KeyError(f"{group}.{key}")
