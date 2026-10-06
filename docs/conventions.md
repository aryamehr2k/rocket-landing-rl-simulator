# Conventions

Units, frames and sign rules used by the Python code and the firmware.

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
| `_cm2`         | square centimetres            | square metres           |
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
- The servo model works in gimbal angle terms and applies, in this order: clip to
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
- No aerodynamic pitch damping from the body and no roll damping or roll torque are modelled,
  so roll stays at whatever rate it starts with. The drag device below is the only damping.

## Drag device

An optional `drag_device` section describes a deployable brake (petals or flaps) as an ideal
drag area at one station. It is opened by a fraction `f` in `[0, 1]`; `0` is shut and adds
nothing, so a rocket without the section has no brake.

- `drag_area_cm2` is `Cd * A` of the fully open device on top of the body's drag; a flat plate
  has `Cd` about 1.2. `station_from_nose_mm` is where its force acts. `deploy_time_s` and
  `retract_time_s` are the times to open fully and to shut.
- The device sits at body position `r = (0, 0, x_cg - x_station)` relative to the centre of
  gravity. Its relative wind is the local one, `v_loc = v_rel_body + omega cross r`, so a
  rocket that turns feels the device resist the turn (pendulum damping).
- Force `F = -0.5 * rho * CdA * f * |v_loc| * v_loc` in body axes, moment `r cross F`.
- Sign of the moment for a tail-first fall: a device ahead of the centre of gravity (smaller
  station, toward the nose, the trailing end when falling) gives a restoring moment like the
  feathers of a shuttlecock; a device behind it (larger station, the legs) gives a diverging one
  and flips the rocket during the unpowered coast. The safety layer refuses to open a device
  behind the descent centre of gravity outside `LANDING_BURN`.
- The descent centre of gravity `descent_cg` is that of dry mass, empty ascent case and full
  landing motor; `descent_mass` is their sum. Both are properties of the rocket config.
- Terminal speed: with the device open the fall stops accelerating where drag equals weight,
  `v_t = sqrt(m g / k)` with `k = 0.5 rho (Cd_body S + CdA_device f)`. The device area for a
  wanted `v_t` is `CdA_total = 2 m g / (rho v_t^2)`; `rocketsim/dragdevice.py` has both.
- The actuator (`BrakeServo` in `actuators.py`) takes the fraction once per control step after
  the action delay, applies the gimbal servo's pure delay, then moves at `1 / deploy_time_s`
  per second opening and `1 / retract_time_s` closing, once per physics step. The log records
  the actual `brake_fraction` and the device force `device_drag_n`.
- `ControlCommand.brake_fraction` is the flight computer's command; a policy may set
  `PlaneAction.brake` to override the rules below, `None` leaves them in charge.

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

## Hidden errors per flight

The training YAML's optional `randomize` section lists uniform ranges `[low, high]` that
`Simulation.reset` draws once per flight from the seeded generator, before anything else:
`landing_thrust_scale` and `ascent_thrust_scale` (thrust scales as above), `dry_mass_g`
(added to the airframe dry mass) and `device_drag_area_scale` (multiplies the device
`Cd*A`). The physics flies the rocket with these errors: `Simulation.reset` builds a new
`RocketDynamics` from the drawn rocket every flight, so `sim.dynamics` must be read after the
reset, never cached across flights. The flight computer, its trigger table and its sensors keep
the nominal rocket file. Without the section nothing is drawn and the generator is untouched, so
older seeds reproduce. `rocketsim/randomize.py` holds the draw.

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
  along the curve, gravity, and drag with the air density at ground level. The drag counts the
  body and the drag device at the opening `brake.burn_fraction` it will have during the burn.
  The distance fallen until the speed drops to `target_speed_mps` (or the motor burns out) plus
  `target_height_m` is the required height for that speed. `integrate_burn` in
  `landing_trigger.py` is that integration, vectorised, and the design tool reuses it.
- The table is consulted in `DESCENT` only. The command goes out when the height and speed the
  rocket will have when thrust starts,
  `delay = igniter mean delay + (action_delay_steps + 0.5) * control period` from now, satisfy
  `height_then <= required(speed_then)`. The speed then is predicted with gravity and the drag
  at the current device opening, `speed_then = v + (g - k v^2 / m) * delay`, and the height with
  the mean of the two speeds. At terminal speed the prediction is `v` itself; without the drag
  term it would add `g * delay`, about 1.8 m/s, and the trigger would fire metres too early.
- `thrust_margin` may be up to 1.10. Above 1 the table assumes a motor stronger than the curve
  and lights later. A motor weaker than the table assumes leaves a little speed the tail can
  absorb; a stronger one stops the rocket too high and it climbs, which the tail cannot undo.
  Setting the margin at the strongest motor of a measured batch therefore fails safe.
- `can_reach_target_from` is the highest speed the burn can bring down to the target before
  burnout. Above it the table keeps the required height of that edge instead of the whole
  burn's fall distance (which would fire tens of metres early on speed noise), so the hard
  part still ends as low as it can with the least speed left. The trigger warns at start-up
  when the expected arrival speed (the terminal speed with the device open) is within 1 m/s
  of the edge, because a burn beyond it leaves speed the tail cannot remove.
- `calibrate_drag_in_flight: true` makes the trigger fit the drag from the accelerometer once
  the device has been fully open for its deploy time plus the servo delay. Falling tail first
  without thrust, the body axis specific force is the drag over the mass, `f = (k / m) v^2
  cos(tilt)` with `v` the estimated descent speed, so a least squares fit of `f` against
  `v^2 cos(tilt)` over 1 s of control steps gives `k / m` without waiting for terminal speed.
  Samples below 12 m/s or beyond 20 degrees of tilt are skipped. The device drag becomes
  `m k/m` minus the body drag, and the table is rebuilt for speeds within 5 m/s of the new
  terminal speed. `FlightComputer.reset` restores the nominal drag and table. In the simulator
  the fit lands within about 2 % of the true drag area in calm air and reads about 4 % high in
  a 4 m/s wind, because the airspeed is more than the descent speed. It corrects where the
  trigger fires, not the motor's impulse: with the brake area 15 % below or above the file the
  example landed 19 and 14 of 20 with the fit against 17 and 9 without. The fit cannot help a
  hard part that is too weak for the arrival speed. The example leaves it off.
- A solid cannot be throttled, so this only works when the hard part of the burn can take
  the descent speed at the crossing down to the target speed with a small margin. Too little
  impulse leaves speed the tail cannot remove; too much stops the rocket in the air and it
  climbs on the leftover thrust. Matching that impulse is part of the rocket design; with a
  drag device the arrival speed is the terminal speed whatever the apogee, and
  `scripts/design_landing_burn.py` sizes the hard part for it.

## Brake rules

The `brake` section of the rocket YAML (needs a `drag_device`) tells the flight computer when
to move the device. It keeps two latches, `deployed` and `retracted`, reset with the flight.

- In `DESCENT`: once the estimated descent speed exceeds `deploy_descent_speed_mps` the device
  opens fully and stays open. Waiting for that speed instead of opening at apogee keeps the
  horizontal relative wind from swinging the rocket while it is still slow.
- In `LANDING_BURN`, if it was deployed: the opening is `burn_fraction` until the estimated
  descent speed drops below `retract_descent_speed_mps` (the hand-over from the hard part to
  the tail), then the device shuts and stays shut so the wind cannot push on it during the
  slow sink. `burn_fraction` is what the trigger table assumes during the burn.
- In every other phase the command is 0. A `PlaneAction.brake` that is not `None` replaces
  the rule's output; the safety layer still applies.

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

- Gimbal commands are clipped to `+-max_angle`, and with `safety.max_gimbal_rate_deg_per_s` set
  they may move at most that rate between two control steps, measured from the last command
  that went out (reset to zero with the flight). This applies to the PID and to a policy alike,
  and the C firmware applies the same limit, so a policy cannot learn a square wave that a servo
  would not survive.
- The ascent igniter may only be commanded in `PAD`.
- The landing igniter may only be commanded in `COAST` or `DESCENT`, at least
  `landing_ignition_lockout_s` after liftoff, with the estimated tilt at most
  `max_landing_ignition_tilt_deg` and the estimated feet height at most
  `max_landing_ignition_height_m`.
- The brake fraction is clipped to `[0, 1]`. A device may only open in `DESCENT` or
  `LANDING_BURN`, and a device behind the descent centre of gravity only in `LANDING_BURN`.
- In `ABORT` everything is zero and nothing ignites.

## Actuators

- Servo commands are accepted once per control step, after the action delay, and go through
  the servo model in the order given in the Gimbal section; the pure delay and the rate limit
  advance once per physics step, so the actual gimbal angle changes every physics step.
- An igniter command starts the motor after a delay drawn once per flight, uniform in
  `mean +- spread` and never negative. A second command to the same motor does nothing.
- The brake servo is described in the Drag device section.

## Burn summary

`Simulation.burn_summary` collects, from the true state, the numbers that explain a touchdown
speed: the time of the igniter command and the estimator's height and speed errors at that
instant (estimate minus truth, feet height from the lowest point of the rocket), the height and
speed when thrust started, the stop height (feet height when the descent first dropped below
`target_speed_mps` during the burn), the climb after that stop, and the burn time left at
touchdown. `scripts/fly_scripted.py` prints it.

## One dimensional design model

`rocketsim/landing_design.py` and `rocketsim/landing_montecarlo.py` size a brake and a burn
and estimate the landing rate without the six degree of freedom physics. They use the same
gravity, ground level air density, body and device drag, thrust curve, mass flow and trigger
as the flight computer, in the vertical axis only: no wind, tilt or lateral speed. The
Monte Carlo runs many flights at once from apogee with the brake rules and the trigger on
noisy estimates, each flight with its own igniter delay, thrust scale, mass and device area
error. Its landing rate is an upper bound on the simulator's, which adds the attitude motion.

`rocketsim/landing_sensitivity.py` holds the closed forms the design tool prints next to that
model. The stop point is where the hard part has brought the descent to the target speed. Its
shift per error is `v dt` for a timing error, `d s (a + g) / a` for a relative thrust or mass
error `s` (with `d` the stopping distance and `a = v^2 / (2 d)` the mean deceleration) and
`sigma_v v / a` for a speed estimate error. The touchdown speed for a shift starts from the
nominal model flight: the stop height `h0`, the climb `c` on the rest of the hard ramp and the
effective tail sink acceleration `a_t = v_td^2 / (2 (h0 + c))`. A stop lower by `delta` sinks
from `h0 - delta + c`; below the ground it is `sqrt(v_t^2 + 2 a_end (delta - h0))` with `a_end`
the deceleration at the end of the hard part. A stop higher by `delta` sinks from
`h0 + delta + c` while the tail lasts and falls freely after burnout. The closed forms keep
the climb and the 20 ms control step nominal, so where they disagree with the model run with
that single error, the model is right; the tool prints both.

## State vector layout in code

`y = [x, y, z, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz, burned_0, burned_1, ...]` as a numpy
array, one `burned_i` entry per motor in the order the rocket YAML lists them (ascent first,
then landing). Index names live in `physics.py` and nothing else hard-codes the positions.
The quaternion is renormalised after every integration step.

## Electric vehicle

Everything above applies; these are the additions for `rocketsim/hop/`.

- **Motor**: a motor file with `type: electric` gives thrust `= throttle x max_thrust_n x thrust scale`,
  throttle in [0, 1]. The thrust follows the throttle command with a first order lag
  (`spin_up_time_constant_s`, `Throttle` in `actuators.py`). Mass does not change. The motor starts
  when the flight computer arms it at launch; there is no igniter delay.
- **Roll**: `Inputs.roll_torque` (N m, right handed about `b_z`) comes from the roll control
  (`RollActuator`: limited to `max_torque_nm`, first order lag). A spinning fan adds
  `reaction_torque_nm_per_n x thrust` about the same axis.
- **GPS** (optional section `gps` of the sensor file): horizontal position and velocity at `rate_hz`,
  delayed by `lag_s`, with white noise and a slow position drift (first order, 60 s time constant).
  The estimator averages the pad samples for the origin, then on each sample pulls the horizontal
  position toward `gps + velocity x lag` by `gps_position_gain` and the horizontal velocity by
  `gps_velocity_gain`.
- **Heights**: mission heights are of the landing feet above the pad, the same `height` the
  estimator reports (`z - pad_cg_height`).
- **Mission plan** (`Guidance`, once per control step): ascent speed `min(climb_speed, speed + a dt,
  sqrt(2 a (target - height)))`; at the target, hover for `time_s + extra_time_s`; descent speed
  ramps to `-descent_speed`; below `final_height_m` it ramps to `-final_speed` and continues to the
  ground. During the final phase the controller follows only the speed.
- **Order in one control step**: at the launch time the pad calibration ends and the plan starts;
  a policy (evaluated outside the flight computer) sees this step's estimate and the previous
  step's reference; the reference advances; the PID runs; the policy replaces (direct) or adds to
  (residual) the PID commands; abort checks; clip and rate limit. The C code in `firmware/hop/`
  does the same.
- **PID baseline**: lateral position and speed to a tilt command, tilt to gimbal (as for the
  rocket, `attitude_pid`); reference height error to a speed correction, speed error to an
  acceleration with an integral, throttle `= hover throttle x (1 + a / g) / max(cos tilt, 0.5)`;
  roll torque `= -(kp roll + kd roll rate)`.
- **Abort**: tilt beyond `abort_tilt_deg` or a distance from the pad beyond `geofence_radius_m`
  cuts the motor for the rest of the flight.
