# Conventions

Every module in `rocketsim/`, every script, every test and the firmware follow this file.
If code and this file disagree, the code is wrong.

## Units

SI everywhere inside the code: metres, kilograms, seconds, radians, newtons, pascals.
YAML files may use millimetres, grams and degrees. A key's suffix says which unit the YAML
value is in and `rocketsim/units.py` converts it when the file is loaded:

| suffix         | YAML unit                     | stored as               |
|----------------|-------------------------------|-------------------------|
| `_mm`          | millimetres                   | metres                  |
| `_m`           | metres                        | metres                  |
| `_g`           | grams                         | kilograms               |
| `_kg`          | kilograms                     | kilograms               |
| `_deg`         | degrees                       | radians                 |
| `_deg_per_s`   | degrees per second            | radians per second      |
| `_us_per_deg`  | microseconds per degree       | microseconds per radian |
| `_deg_per_m`   | degrees per metre             | radians per metre       |
| `_deg_per_mps` | degrees per metre per second  | radians per metre per second |
| `_s`           | seconds                       | seconds                 |
| `_mps`, `_mps2`| metres per second (squared)   | unchanged               |
| `_us`          | microseconds                  | microseconds            |
| `_kgm2`, `_hz`, `_n`, `_kgpm3`, `_per_rad` | already SI | unchanged     |

Rules:

- The longest matching suffix wins, so `_deg_per_s` is a rate and `_us_per_deg` is a servo
  calibration, never a plain `_s` or `_deg`. Plain gains carry no suffix (`kp`) or a suffix
  that says what they multiply (`kd_s`, `ki_per_s`, `baro_velocity_gain_per_s`).
- A key whose suffix is not in the table (for example `drag_coefficient`, `sign`,
  `linkage_ratio`) is stored unchanged.
- Pulse widths keep their microsecond unit because they are servo command counts, not
  physics. Their field names keep the `_us` suffix.
- After conversion the field name drops the unit suffix (`dry_mass_g` becomes `dry_mass`,
  `pulse_us_per_deg` becomes `pulse_us_per_rad`). This file names YAML keys with their
  suffix when it talks about a file, and the plain name when it talks about a value in code.

## World frame

```
        z (up)
        ^
        |      nose
        |      /
        |     /  tilt_x (positive: nose leans toward +x)
        |    /
        |   o  <- centre of gravity
        |  /
        | /  tail
        |/
        +--------------------> x (downrange)
       pad                y (crossrange) points into the page for a right-handed frame
```

- `x`: horizontal, downrange. `y`: horizontal, crossrange. `z`: altitude above the pad
  surface, positive up. The frame is right-handed (`x` cross `y` = `z`). The pad is at the origin
  and its surface is `z = 0`. Launch site elevation above sea level is a config value used only
  for air density.
- Position and velocity (`x, y, z`, `vx, vy, vz`) are those of the centre of gravity of the
  mass still on board. Body points are placed around it with the current CG station, so as
  propellant burns the airframe slides a little toward the tail relative to the integrated
  point (about 2 cm over the ascent burn). That small variable-mass effect is accepted; the
  pad hold and the touchdown arming below make sure it never counts as ground contact.

## Body frame

- Body origin for configuration is the nose tip. Every position in a rocket YAML is a station
  measured from the nose tip along the body axis toward the tail (`_from_nose_mm`). A larger
  station is closer to the tail.
- For dynamics, forces and moments are taken about the current centre of gravity.
- Body axes: `b_z` points along the body axis from tail to nose. `b_x` and `b_y` are lateral
  and `b_x` cross `b_y` = `b_z`. Standing upright on the pad with no roll, `b_x = x`, `b_y = y`,
  `b_z = z`: that is the identity attitude.
- A point at station `s` sits at body position `(0, 0, x_cg - s)` relative to the centre of
  gravity, where `x_cg` is the centre of gravity station. Points behind the CG have negative
  `b_z` coordinates.
- Attitude is the unit quaternion `q = (w, x, y, z)` that rotates body vectors into world
  vectors: `v_world = q * v_body * conj(q)`. `rocketsim/quaternion.py` is the only place that
  implements this rotation; everything else calls it.
- Angular velocity `omega = (wx, wy, wz)` is expressed in the body frame. Attitude kinematics
  are `dq/dt = 0.5 * q * (0, omega)`.
- A moment is a body frame vector `(Mx, My, Mz)` from `r cross F`, with `r` the body position of
  the force relative to the CG. Rotational dynamics are `I * d(omega)/dt = M - omega cross (I * omega)`
  with the diagonal inertia `I = diag(I_pitch, I_pitch, I_roll)`: the rocket is treated as
  axisymmetric.

## Tilt, roll and the two control planes

- The body axis in world coordinates is `a = q * (0, 0, 1) * conj(q)`.
- `tilt_x = atan2(a_x, a_z)`: positive when the nose leans toward `+x`. This is the pitch plane.
- `tilt_y = atan2(a_y, a_z)`: positive when the nose leans toward `+y`. This is the yaw plane.
- `tilt = acos(a_z)`: total angle from vertical, used for touchdown grading.
- Roll is the twist of `q` about the body axis: `roll = 2 * atan2(q_z, q_w)` after the swing
  twist decomposition, positive for a right-hand rotation about `b_z`. It is zero on the pad.
- `tilt_rate_x` and `tilt_rate_y` are the time derivatives of the tilt angles, computed from
  `q` and `omega` in `quaternion.py`.
- Sign warning for anyone deriving by hand: a positive moment about `+b_y` leans the nose
  toward `+b_x` (positive `tilt_x` acceleration), but a positive moment about `+b_x` leans the
  nose toward `-b_y`. That is how right-handed frames work; the gimbal sign rules below hide it
  from the controller.

## Gimbal

- The gimbal pivot is at station `pivot_from_nose_mm`. The moment arm is
  `L = x_pivot - x_cg`, recomputed every step from the current centre of gravity, and the thrust
  is applied at body position `(0, 0, -L)`.
- Two deflections: `delta_p` from the pitch servo and `delta_y` from the yaw servo, both in
  radians, both limited to `+-max_angle`. The thrust direction in the body frame is
  `normalize(-tan(delta_p), -tan(delta_y), 1)`. With `delta_y = 0` this is
  `(-sin(delta_p), 0, cos(delta_p))`.
- Sign rule the controller relies on: positive `delta_p` leans the nose toward `+b_x` and
  positive `delta_y` leans the nose toward `+b_y`. Worked out: a thrust `(F_x, F_y, F_z)` at
  `(0, 0, -L)` gives the moment `(L * F_y, -L * F_x, 0)`. With `F_x = -T sin(delta_p)` that is
  `My = +L * T * sin(delta_p)`, and with `F_y = -T sin(delta_y)` it is `Mx = -L * T * sin(delta_y)`,
  which (see the sign warning above) leans the nose toward `+b_y`.
- The commanded angle, the actual angle after the servo model, and the servo pulse width are
  three different numbers and are logged separately for each servo.
- `max_angle` clips the commanded gimbal angle before the servo model sees it. The actual
  angle is further bounded by the pulse clip below.
- Servo calibration, per servo, all angles in radians, pulses in microseconds:
  `servo_angle = gimbal_angle / linkage_ratio`,
  `pulse = center_us + sign * servo_angle * pulse_us_per_rad`,
  `pulse = floor(pulse / pulse_resolution_us + 0.5) * pulse_resolution_us` (round half up),
  `pulse = clip(pulse, min_us, max_us)`.
  `linkage_ratio` is gimbal angle per servo angle and `sign` is `+1` or `-1` for the way the
  servo is mounted. `center_us`, `min_us` and `max_us` must be multiples of the resolution.
  Python and C round the same way so both emit the same pulse for the same command.
- The servo model (stage 2) works in gimbal angle terms and applies, in this order: clip to
  `max_angle`, deadband on the change of command, pulse quantisation, pure delay
  `servo_delay`, rate limit `servo_rate_limit`. Its output is the actual gimbal angle.
- Motors marked `gimbaled: false` thrust along the body axis through the pivot and produce no
  moment. All motor thrust acts at the pivot. No thrust produces a roll moment.

## Aerodynamics

- Air density `rho = rho_0 * exp(-(z + site_elevation) / scale_height)`.
- Wind is a horizontal world vector `(wind_x, wind_y, 0)`. Velocity relative to the air is
  `v_rel = v - wind`, rotated into the body frame for the loads below.
- Dynamic pressure `q = 0.5 * rho * |v_rel|^2`, reference area `S = pi * d^2 / 4`.
- Drag `D = -0.5 * rho * Cd * S * |v_rel| * v_rel` acts at the CG with no moment. Written this
  way it is exactly zero at rest instead of dividing zero by zero.
- Angle of attack `alpha` is the angle between the body axis and `v_rel`. The normal force
  opposes the lateral part of the relative velocity:
  `N = -0.5 * rho * S * CNa * |v_rel| * (v_rel_bx, v_rel_by, 0)`, which equals
  `q * S * CNa * sin(alpha)` in magnitude and never divides by speed. It acts at the centre of
  pressure station `cp_from_nose_mm`, body position `(0, 0, x_cg - x_cp)`.
- With the centre of pressure behind the centre of gravity the moment `r cross N` turns the
  nose toward the relative velocity, which is the restoring torque the fins give at speed. It
  vanishes as `|v_rel|^2` at low speed.
- The same moment makes tail-first flight (`alpha` near `pi`) an unstable equilibrium whenever
  `x_cp > x_cg`, however small the margin: a tilt `eps` from tail-first grows as
  `exp(t * sqrt((x_cp - x_cg) * q * S * CNa / I_pitch))`. A rocket that must fall tail-first
  therefore needs `x_cp <= x_cg` at ascent burnout, or a burn that starts before the tilt
  has had time to grow. Static margins must be judged at the burnout centre of gravity, which
  is ahead of the loaded one because the motors sit at the tail.
- No aerodynamic pitch damping and no roll damping or roll torque are modelled yet, so roll
  stays at whatever rate it starts with.

## Mass properties

- `dry_mass_g` is everything that is not a motor. `dry_cg_from_nose_mm` is its centre of
  gravity, `dry_pitch_inertia_kgm2` its inertia about a lateral axis through that point and
  `dry_roll_inertia_kgm2` its inertia about the body axis.
- Each motor is a point mass at `position_from_nose_mm` for pitch and yaw, and a solid
  cylinder of its diameter (`0.5 * m * r^2`) for roll. Its mass is the motor's total mass minus the
  propellant burned so far. Propellant burned follows the thrust curve:
  `d(burned)/dt = thrust_curve(t) / c`, with `c = total_impulse / propellant_mass`.
- Centre of gravity and inertia are recomputed every derivative evaluation from the parts:
  `x_cg = (m_dry * x_dry + sum(m_i * x_i)) / m`,
  `I_pitch = I_pitch_dry + m_dry * (x_cg - x_dry)^2 + sum(m_i * (x_cg - x_i)^2)`,
  `I_roll = I_roll_dry + sum(0.5 * m_i * r_i^2)`.

## Motors and igniters

- A motor is idle until an ignition command is accepted. The igniter model adds the ignition
  delay (mean plus a uniform spread of `+-spread`, never below zero). Thrust then follows the
  curve from time zero at the moment thrust starts.
- Solid motors cannot throttle, stop or restart. A second ignition command is ignored.
- A throttleable engine multiplies the curve by a throttle in `[0, 1]` that follows the command
  with a first order lag. Mass flow is proportional to actual thrust, and the engine stops when
  its propellant is gone or when the curve ends, whichever comes first.
- A random thrust scale from the training YAML multiplies thrust but not mass flow, so it
  models uncertainty in the total impulse per gram of propellant.

## Ground, pad and touchdown

- The rocket starts standing on the pad at the identity attitude. The leg feet plane is at
  station `length_mm + legs.height_mm`. The centre of gravity is therefore at
  `z = length + leg_height - x_cg` on the pad.
- The `legs.count` feet (at least three) sit in the feet plane on a circle of diameter
  `legs.span_mm`, foot `k` at the angle `2 * pi * k / count` measured from `b_x` toward `b_y`.
- Before liftoff the rocket is held at rest while net vertical acceleration is zero or
  negative: only the burned propellant advances (forward Euler, since nothing else moves)
  and the CG height is re-seated so the feet stay exactly on the pad. Liftoff happens on the
  first step with positive net vertical acceleration.
- Touchdown detection arms once the lowest of the nose tip and the feet has been above
  `z = 0`. After that, the flight ends the first physics step in which that lowest point is at
  or below `z = 0`.
- Touchdown is graded against the leg limits: vertical speed `-vz`, lateral speed
  `sqrt(vx^2 + vy^2)`, tilt (total angle from vertical). The tilt limit actually used is the
  smaller of `max_touchdown_tilt_deg` and the tip-over angle
  `atan((span / 2) * cos(pi / count) / h_cg)`, where `h_cg` is the CG height above the feet
  plane at touchdown and `(span / 2) * cos(pi / count)` is the inscribed radius of the foot
  polygon. Miss distance is `sqrt(x^2 + y^2)` at touchdown.

## Time and the control loop

- Physics integrates with RK4 at `physics_rate_hz` (200 Hz by default), step `dt`.
- The control loop runs at `control_rate_hz` (50 Hz by default). `physics_rate_hz` must be
  an integer multiple of it. Control step `k` happens at `t_k = k / control_rate_hz`.
- At control step `k` the controller receives the observation built from sensor samples taken
  up to and including `t_k` and computes action `a_k`.
- Action `a_k` is applied starting at control step `k + action_delay_steps`
  (`1` by default), that is during `[t_{k+1}, t_{k+2})`. Until then the previous action
  stays applied. The delay models compute time and servo command latency on the board.
- The flight computer sits on the pad for `pad_hold_time_s` (training YAML) calibrating its
  sensors; the launch command goes out at the first control step at or after that time.
- The action delay applies to the whole command, igniter commands included.
- Sensors are three axis (specific force and angular rate in the body frame) plus a
  barometer. They sample at their own rates from the sensor YAML. A sample taken at time `t` reflects
  the true state at `t - lag`. Between samples the last value is held.
- The log records one row per physics step. The firmware log has the same columns. Sensor
  columns hold the latest sample, estimate columns the estimate after that sample.

## Pitch and yaw on the flight computer

The policy is trained on plane quantities and runs twice per control step on the board, once
for each plane, sharing altitude and vertical velocity:

| plane quantity in the observation | pitch plane run | yaw plane run |
|-----------------------------------|-----------------|---------------|
| lateral position                  | `x`             | `y`           |
| lateral velocity                  | `vx`            | `vy`          |
| tilt                              | `tilt_x`        | `tilt_y`      |
| tilt rate                         | `tilt_rate_x`   | `tilt_rate_y` |
| gimbal command                    | `delta_x`       | `delta_y`     |

Both planes use the same sign rule: a positive gimbal command leans the nose toward the
positive lateral axis of that plane. The two commands are world-plane commands; the control
loop rotates them by the roll angle into the body servo commands,
`(delta_p, delta_y_servo) = rotate_by(-roll) * (delta_x, delta_y)`, which is the identity when
roll is zero. Each run also produces the continuous ignition output; the control loop averages
the two and applies the threshold once.

## Sensors

The sensor YAML (`configs/sensors/*.yaml`, referenced from the rocket file's `sensors.file`)
describes one IMU and one barometer.

- The IMU reads specific force `f = R^T (a - g)` with `g = (0, 0, -gravity)` and the angular
  rate `omega`, both in the body frame. At rest and upright it reads `(0, 0, +gravity)`.
  While the rocket is held on the pad its acceleration is zero whatever the thrust.
- Each sensor samples at `rate_hz`. A sample at time `t` shows the truth at `t - lag_s`, plus a
  bias drawn once per flight from a normal distribution with `bias_std`, plus white noise
  with `noise_std`, clipped to `+-range`. The barometer reads the altitude of the centre of
  gravity.
- The IMU rate must not exceed the physics rate; the estimator integrates with `1 / rate_hz`.

## Estimator

Runs in float32 with the same operations in the same order as the C version.

- On the pad it keeps the last `pad_average_time_s` of samples. At liftoff the gyro mean
  becomes the gyro bias, the accelerometer mean gives the initial attitude (the rotation with
  zero roll that takes the measured gravity direction to `+z`), and the barometer mean minus
  the known pad altitude of the centre of gravity becomes the barometer offset. Position
  starts at `(0, 0, pad_cg_height)` with zero velocity.
- Each IMU sample in flight: `rate = gyro - bias`; `q += 0.5 * dt * q * (0, rate)`, normalise;
  `a_world = R(q) f + g`; `v += a_world * dt`; `p += v * dt` (velocity first, then position).
- Each barometer sample in flight: `alt_now = baro - offset + vz * lag`;
  `err = alt_now - z`; `z += baro_altitude_gain * err`; `vz += baro_velocity_gain_per_s * err`.
- Height of the feet above the ground is `z - pad_cg_height`. Tilt angles, tilt rates and
  roll come from `quaternion.py` applied to the estimated quaternion and rate.

## Phases

`PAD -> BOOST -> COAST -> DESCENT -> LANDING_BURN -> LANDED`, with `ABORT` reachable from
`BOOST`. Thresholds live in the rocket YAML `phases` section.

- `PAD -> BOOST`: the IMU body axis specific force exceeds `liftoff_accel_mps2`. This is
  checked on every IMU sample so the estimator starts integrating at once.
- `BOOST -> COAST`: at least `min_boost_time_s` after liftoff and the body axis specific force
  below `burnout_accel_mps2`.
- `BOOST -> ABORT`: estimated tilt above `abort_tilt_deg`. In `ABORT` the gimbal is centred
  and no igniter command goes out.
- `COAST -> DESCENT`: estimated vertical speed below `apogee_vz_mps`.
- `COAST` or `DESCENT -> LANDING_BURN`: a landing igniter command passes the safety checks.
- `-> LANDED`: touchdown.

The gimbal is controlled in `BOOST` and `LANDING_BURN`; in every other phase it is centred and
the PID integrators are reset.

## Landing trigger

For a solid landing motor the only decision is when to send the igniter command.

- At start-up the flight computer integrates the landing burn in one dimension for every
  downward speed from 0 to 80 m/s in 0.5 m/s steps: thrust from the curve times
  `thrust_margin`, mass `dry + empty ascent case + full landing motor` minus propellant burned
  along the curve, gravity, and drag with the air density at ground level. The distance
  fallen until the speed drops to `target_speed_mps` (or the motor burns out) plus
  `target_height_m` is the required height for that speed.
- The table is consulted in `DESCENT` only. The command goes out when the height and speed the
  rocket will have when thrust starts,
  `delay = igniter mean delay + (action_delay_steps + 0.5) * control period` from now, satisfy
  `height_then <= required(speed_then)`.
- A solid cannot be throttled, so this only works when the hard part of the burn can take
  the descent speed at the crossing down to the target speed with a small margin. Too little
  impulse leaves speed the tail cannot remove; too much stops the rocket in the air and it
  climbs on the leftover thrust. Matching that impulse is part of the rocket design.

## PID baseline

Per plane, on the estimate, at the control rate, with `hold_position` in `BOOST` and
`LANDING_BURN`:

```
tilt_cmd = clip(-(kp_deg_per_m * lateral + kd_deg_per_mps * lateral_velocity), +-max_tilt_command)
error    = tilt_cmd - tilt
integral = clip(integral + error * dt, +-max_integral / ki)
delta    = kp * error + ki * integral - kd * tilt_rate
```

`delta_x` and `delta_y` are then rotated by `-roll` into the pitch and yaw servo commands.

## Safety

Applied to every command, PID or policy, before it reaches the actuators:

- Gimbal commands are clipped to `+-max_angle`.
- The ascent igniter may only be commanded in `PAD`.
- The landing igniter may only be commanded in `COAST` or `DESCENT`, at least
  `landing_ignition_lockout_s` after liftoff, with the estimated tilt at most
  `max_landing_ignition_tilt_deg` and the estimated feet height at most
  `max_landing_ignition_height_m`.
- In `ABORT` everything is zero and nothing ignites.

## Actuators

- Servo commands are accepted once per control step, after the action delay, and go through
  the servo model in the order given in the Gimbal section; the pure delay and the rate limit
  advance once per physics step, so the actual gimbal angle changes every physics step.
- An igniter command starts the motor after a delay drawn once per flight, uniform in
  `mean +- spread` and never negative. A second command to the same motor does nothing.

## State vector layout in code

`y = [x, y, z, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz, burned_0, burned_1, ...]` as a numpy
array, one `burned_i` entry per motor in the order the rocket YAML lists them (ascent first,
then landing). Index names live in `physics.py` and nothing else hard-codes the positions.
The quaternion is renormalised after every integration step.
