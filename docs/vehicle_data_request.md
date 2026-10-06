# Electric test vehicle: data sheet

Everything the simulator needs to know about the electric test vehicle goes into one file.

1. Copy `configs/datasheets/template.yaml` to `configs/datasheets/<your_name>.yaml`.
2. Fill in every entry marked required. Optional entries carry a default; replace it once you have
   measured the value.
3. Run `python scripts/build_vehicle.py configs/datasheets/<your_name>.yaml`.

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

## Vehicle

| What | Entry | Unit | Required or default | How |
|---|---|---|---|---|
| Short name for the config files: letters, digits, _ | `vehicle.name` | - | the sheet's file name | - |
| Mass ready to fly, battery and fan unit in | `vehicle.total_mass_g` | g | **required** | scale |
| Balance point (centre of gravity) from the top, ready to fly | `vehicle.balance_point_mm` | mm | **required** | balance it across a ruler edge |
| Body length from the top to the bottom end, legs not counted | `vehicle.length_mm` | mm | **required** | tape measure |
| Body outside diameter | `vehicle.diameter_mm` | mm | **required** | calliper |
| Pitch inertia of the whole vehicle about its balance point | `vehicle.pitch_inertia_kgm2` | kg m2 | solid cylinder | CAD (g mm2 / 1e9) or a swing test |
| Roll inertia of the whole vehicle about its long axis | `vehicle.roll_inertia_kgm2` | kg m2 | solid cylinder | CAD |
| Drag coefficient on the body diameter | `vehicle.drag_coefficient` | - | 0.8 | OpenRocket |
| Centre of pressure from the top, wind from the side | `vehicle.centre_of_pressure_mm` | mm | middle of the body | OpenRocket |
| Side force slope for wind across the body | `vehicle.side_force_slope_per_rad` | 1/rad | 8 | keep unless you have wind tunnel data |

## Fan and battery

| What | Entry | Unit | Required or default | How |
|---|---|---|---|---|
| Fan, motor and ESC model | `fan.model` | - | optional | - |
| Fan, motor and ESC as mounted, battery not included | `fan.unit_mass_g` | g | **required** | scale before mounting |
| Centre of the fan unit from the top | `fan.position_mm` | mm | **required** | tape measure |
| Fan (duct) diameter | `fan.diameter_mm` | mm | **required** | calliper |
| Fan unit length | `fan.length_mm` | mm | same as the diameter | calliper |
| Static thrust at full throttle, freshly charged battery | `fan.max_thrust_n` | N | **required** | thrust stand |
| Static thrust at 50 % throttle | `fan.half_throttle_thrust_n` | N | not checked | same test; shows whether thrust follows the throttle in a straight line |
| Time for the thrust to reach 63 % of a throttle step | `fan.spin_up_ms` | ms | **required** | thrust stand log at 100 Hz or faster |
| Hover time on one charge down to the low voltage cut | `fan.battery_time_s` | s | **required** | thrust stand at hover throttle |
| Roll torque of the spinning fan per newton of thrust | `fan.reaction_torque_nm_per_n` | N m/N | 0.003 | torque arm on the thrust stand |

## Thrust vectoring and servos

| What | Entry | Unit | Required or default | How |
|---|---|---|---|---|
| Gimbal pivot or vane hinge line from the top | `thrust_vectoring.pivot_mm` | mm | **required** | tape measure |
| Largest thrust deflection each way, each axis | `thrust_vectoring.max_angle_deg` | deg | **required** | phone level at full servo travel |
| Thrust angle per degree of servo horn travel | `thrust_vectoring.linkage_ratio` | - | **required** | move the horn 20 deg, divide the thrust angle by 20 |
| Servo model | `thrust_vectoring.servo_model` | - | optional | - |
| Servo speed: time for 60 deg of horn travel at your voltage | `thrust_vectoring.servo_time_per_60deg_s` | s | **required** | servo datasheet |
| Time from a command to the horn starting to move | `thrust_vectoring.servo_delay_ms` | ms | 20 | slow motion video |
| Smallest thrust angle change the servo follows | `thrust_vectoring.servo_deadband_deg` | deg | 0.1 | step the command on the bench |
| Pulse that holds the thrust straight | `thrust_vectoring.servo_center_us` | us | 1500 | find it on the bench |
| Pulse change per degree of horn travel | `thrust_vectoring.servo_us_per_deg` | us/deg | 10 | servo datasheet |
| Shortest pulse the flight code may send | `thrust_vectoring.servo_min_us` | us | 1000 | servo datasheet |
| Longest pulse the flight code may send | `thrust_vectoring.servo_max_us` | us | 2000 | servo datasheet |
| Smallest pulse step the board puts out | `thrust_vectoring.pulse_resolution_us` | us | 1 | board datasheet |
| +1 if a longer pitch servo pulse swings the air jet toward +x, else -1 | `thrust_vectoring.pitch_servo_sign` | - | 1 | mark one side of the body the pitch servo swings the jet toward as +x, send servo_center_us + 100 and watch the air leaving the duct. The simulator cannot catch a wrong sign, only this test can |
| +1 if a longer yaw servo pulse swings the air jet toward +y, else -1 | `thrust_vectoring.yaw_servo_sign` | - | 1 | +y is 90 deg anticlockwise from +x, seen from above. Same test with the yaw servo |

## Roll control

| What | Entry | Unit | Required or default | How |
|---|---|---|---|---|
| Roll fans, reaction wheel or vanes | `roll_control.kind` | - | optional | - |
| Largest roll torque | `roll_control.max_torque_nm` | N m | **required** | roll fan thrust x its distance from the centre line x fans pushing the same way |
| Time to reach 63 % of a torque step | `roll_control.response_ms` | ms | 50 | thrust stand log |

## Landing legs

| What | Entry | Unit | Required or default | How |
|---|---|---|---|---|
| Number of legs | `legs.count` | - | **required** | - |
| Diameter of the circle through the feet | `legs.foot_circle_mm` | mm | **required** | tape measure |
| How far the feet reach below the bottom end of the body | `legs.below_body_mm` | mm | **required** | tape measure |
| Highest drop onto the legs they survive, at flight mass | `legs.max_drop_height_mm` | mm | **required** | drop test; 115 mm lands at 1.5 m/s |
| Sideways speed at touchdown the legs survive | `legs.max_sideways_speed_mps` | m/s | 1 | drop test with a push |
| Tilt at touchdown the legs survive | `legs.max_tilt_deg` | deg | 10 | drop test on a slope |

## Sensors and flight computer

| What | Entry | Unit | Required or default | How |
|---|---|---|---|---|
| Flight computer board | `sensors.flight_computer` | - | optional | - |
| How often the flight code runs the control loop | `sensors.control_rate_hz` | Hz | **required** | flight code; must divide the simulator's 200 Hz step: 200, 100, 50, 40 or 25 |
| Control steps from reading the sensors to the servo command | `sensors.command_delay_steps` | steps | 1 | flight code |
| IMU model | `sensors.imu_model` | - | optional | - |
| IMU sample rate the flight code reads | `sensors.imu_rate_hz` | Hz | **required** | flight code; at most 200, the simulator's step rate, so enter 200 for a faster IMU |
| IMU filter delay | `sensors.imu_lag_ms` | ms | 2 | datasheet, from the filter setting |
| Accelerometer noise | `sensors.accel_noise_mps2` | m/s2 | 0.05 | 30 s log with the board still: standard deviation |
| Accelerometer bias change between power-ups | `sensors.accel_bias_mps2` | m/s2 | 0.02 | mean of several still logs |
| Accelerometer full scale (16 g is 157) | `sensors.accel_range_mps2` | m/s2 | 156 | datasheet |
| Gyroscope noise | `sensors.gyro_noise_deg_per_s` | deg/s | 0.1 | same still log |
| Gyroscope bias change between power-ups | `sensors.gyro_bias_deg_per_s` | deg/s | 0.5 | several still logs |
| Gyroscope full scale | `sensors.gyro_range_deg_per_s` | deg/s | 2000 | datasheet |
| Barometer model | `sensors.barometer_model` | - | optional | - |
| Barometer sample rate the flight code reads | `sensors.barometer_rate_hz` | Hz | **required** | flight code; at most 200 |
| Barometer filter delay | `sensors.barometer_lag_ms` | ms | 20 | datasheet, from the oversampling setting |
| Barometer height noise | `sensors.barometer_noise_m` | m | 0.3 | same still log |
| Barometer height drift over a flight | `sensors.barometer_bias_m` | m | 1 | 10 min still log |
| GPS model | `sensors.gps_model` | - | optional | - |
| GPS update rate, 0 if there is no GPS | `sensors.gps_rate_hz` | Hz | **required** | flight code; at most 200 |
| GPS delay | `sensors.gps_lag_ms` | ms | 100 | datasheet |
| GPS position scatter from sample to sample | `sensors.gps_noise_m` | m | 0.8 | 2 min log standing still at the field |
| Slow GPS position wander over a minute | `sensors.gps_drift_m` | m | 1 | same log |
| GPS speed noise | `sensors.gps_speed_noise_mps` | m/s | 0.1 | same log |

## Launch site

| What | Entry | Unit | Required or default | How |
|---|---|---|---|---|
| Field height above sea level | `launch_site.elevation_m` | m | 300 | map |
| Steady wind expected on test day | `launch_site.wind_mps` | m/s | 4 | weather forecast |
| Gust strength (standard deviation) | `launch_site.gust_mps` | m/s | 1.5 | weather forecast |

## Mission

| What | Entry | Unit | Required or default | How |
|---|---|---|---|---|
| Hover height of the feet above the pad | `mission.target_height_m` | m | **required** | competition rules |
| Hover time at the target height | `mission.hover_time_s` | s | **required** | competition rules |
| Touchdown must be within this distance of the pad | `mission.landing_radius_m` | m | **required** | competition rules |
| Counts as at the target height within this band | `mission.height_tolerance_m` | m | 2 | competition rules |
| Planned climb speed | `mission.climb_speed_mps` | m/s | 4 | - |
| How quickly the plan speeds up and slows down | `mission.climb_acceleration_mps2` | m/s2 | 1.5 | - |
| Planned descent speed | `mission.descent_speed_mps` | m/s | 3 | - |
| Height where the plan slows to the touchdown speed | `mission.final_height_m` | m | 4 | at least (descent_speed^2 - final_speed^2) / (2 x climb_acceleration) + 1 m |
| Touchdown speed the plan aims for | `mission.final_speed_mps` | m/s | 0.7 | at least 0.5 m/s below what the legs survive |
| Extra hover time planned so the required time is met with margin | `mission.extra_hover_s` | s | 2 | - |
| Flight time limit | `mission.max_flight_time_s` | s | plan + 40 s | competition rules; longer than the plan |
| Tilt at which the flight computer cuts the fan | `mission.abort_tilt_deg` | deg | 40 | - |
| Distance from the pad at which the flight computer cuts the fan | `mission.geofence_radius_m` | m | 40 | - |

## What the builder works out

- Mass and balance point without the fan unit, from the totals and the fan unit's mass and position.
- Pitch and roll inertia when they are empty: the body as a solid cylinder (pitch m (3 r² + L²) / 12, roll m
  r² / 2) plus the fan unit as a point mass. A swing test or CAD value is better.
- Gimbal authority A = hover thrust x (pivot - balance point) / pitch inertia: the pitch acceleration in
  rad/s² for each radian of thrust deflection. The attitude gains put the tilt loop at 6 rad/s with damping
  0.6: kp = 6² / A, kd = 2 x 0.6 x 6 / A.
- Roll gains: the roll inertia I sits behind a lag T = roll response time + (command delay steps + 0.5) /
  control rate, so the roll loop has three poles. The gains put a pair with damping 0.5 at w = min(10 rad/s,
  1 / (3 x 0.5 x T)), the stiffest loop the lag allows, and the third pole at p = 1 / T - 2 x 0.5 x w: kp =
  I T p w², kd = I T (w² + 2 x 0.5 x w p). w is lowered further when the roll torque would run out before 15
  degrees of roll error.
- Thrust vectoring speed: 60 degrees / servo time x linkage ratio. Touchdown speed the legs survive: sqrt(2
  g h) from the drop height.
- An empty flight time limit is the mission plan plus 40 s, rounded up to 10 s.
- Training mission ranges: those of `configs/training/hop.yaml`, widened to reach 1.2 times the sheet's
  height and hover time, then cut until the longest training mission plus a 30% reserve fits the battery and
  the flight time limit.
- Position, altitude, estimator and safety settings are those of the example vehicle
  (`configs/vehicles/electric_hopper.yaml`).

This page, `configs/datasheets/template.yaml` and `electric_vehicle_data_request.pdf` are written from
`rocketsim/hop/datasheet_fields.py` by `python scripts/write_data_request.py`.
