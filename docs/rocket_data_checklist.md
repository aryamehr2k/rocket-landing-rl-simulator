# Rocket data checklist

What to measure on the real rocket, where it goes in the YAML files, how accurately it is
needed and why. Copy `configs/rockets/example_tvc.yaml` to a new file, fill in every line
below, and the simulator, the design tool and later the training run on your rocket. Every
number in the example file is a placeholder.

Accuracy column: how much error the landing tolerates, from the sensitivity study in the
README and `docs/physics.md`. Measure the first group before anything else.

## Measure these first

| what                          | YAML key                                   | unit  | how                                                       | accuracy | why |
|-------------------------------|--------------------------------------------|-------|-----------------------------------------------------------|----------|-----|
| landing motor thrust curve    | `configs/motors/<name>.yaml` `thrust_curve` | s, N | load cell test of 2 or more motors of the batch, 200 Hz or faster | 1 to 2 % on impulse | the burn's impulse is fixed; 3 % strong crashes |
| landing motor propellant and total mass | `propellant_mass_g`, `total_mass_g` | g   | scale, before and after a test burn                        | 1 g      | mass flow and centre of gravity during the burn |
| igniter delay                 | `motors.landing.ignition_delay_s` `mean`, `spread` | s | fire about 20 igniters with your own circuit, time from command to first thrust | 10 ms | 30 ms spread lands, 80 ms does not |
| rocket mass before the flight | `airframe.dry_mass_g` (without motors)     | g     | scale, every flight day                                    | 10 g     | 36 g light behaves like a strong motor |
| brake drag area               | `drag_device.drag_area_cm2`                | cm2   | plate area times 1.2, then a drop test or `calibrate_drag_in_flight: true` on early flights | 5 % | sets the arrival speed the burn is sized for |

## Airframe

| what                          | YAML key                         | unit   | how                                                        | accuracy |
|-------------------------------|----------------------------------|--------|------------------------------------------------------------|----------|
| dry mass (no motors)          | `airframe.dry_mass_g`            | g      | scale                                                      | 10 g     |
| dry centre of gravity from the nose tip | `airframe.dry_cg_from_nose_mm` | mm | balance the rocket without motors on an edge, or OpenRocket | 5 mm |
| pitch inertia about the dry CG | `airframe.dry_pitch_inertia_kgm2` | kg m2 | OpenRocket, or a bifilar pendulum swing test              | 10 %     |
| roll inertia about the axis   | `airframe.dry_roll_inertia_kgm2` | kg m2  | OpenRocket, or `m r^2` of the tube as an estimate           | 20 %     |
| length nose tip to tail end   | `airframe.length_mm`             | mm     | tape measure                                               | 2 mm     |
| body diameter                 | `airframe.reference_diameter_mm` | mm     | calliper                                                   | 0.5 mm   |

## Aerodynamics

| what                          | YAML key                              | how                                                         | accuracy |
|-------------------------------|---------------------------------------|-------------------------------------------------------------|----------|
| drag coefficient of the body  | `aero.drag_coefficient`               | OpenRocket at 20 to 40 m/s, legs and petals shut             | 20 %     |
| normal force slope per radian | `aero.normal_force_slope_per_rad`     | OpenRocket (nose cone plus fins)                            | 20 %     |
| centre of pressure from the nose | `aero.cp_from_nose_mm`             | OpenRocket; compare with the burnout CG, see docs/physics.md section 4 | 10 mm |

## Motors and their positions

| what                          | YAML key                                  | how                                            |
|-------------------------------|-------------------------------------------|------------------------------------------------|
| ascent motor curve            | `motors.ascent.file` (`.eng` from ThrustCurve.org or your own YAML) | manufacturer file, or load cell |
| ascent motor mass, propellant | in the `.eng` header or motor YAML         | scale                                          |
| ascent igniter delay          | `motors.ascent.ignition_delay_s`           | same as the landing igniter; less critical     |
| centre of mass of each motor from the nose | `motors.*.position_from_nose_mm` | tape measure with the motor installed          |
| does the motor thrust through the gimbal | `motors.*.gimbaled`             | yes for a motor in the gimbal mount            |

## Gimbal and servos

| what                          | YAML key                                   | unit    | how                                                    |
|-------------------------------|--------------------------------------------|---------|--------------------------------------------------------|
| pivot station from the nose   | `gimbal.pivot_from_nose_mm`                | mm      | tape measure to the gimbal's pivot axis                |
| maximum gimbal angle          | `gimbal.max_angle_deg`                     | deg     | move the servo to its limits and measure with a protractor or phone level |
| servo speed                   | `gimbal.servo_rate_limit_deg_per_s`        | deg/s   | datasheet (60 deg in 0.1 s is 600 deg/s), or film it   |
| servo delay                   | `gimbal.servo_delay_s`                     | s       | film the command LED and the horn at high frame rate; 15 to 30 ms is typical |
| servo deadband                | `gimbal.servo_deadband_deg`                | deg     | smallest command step that moves the horn              |
| pulse at centre               | `gimbal.servos.pitch.center_us`, `yaw`     | us      | the pulse that puts the nozzle straight (find it on the bench) |
| pulse change per servo degree | `pulse_us_per_deg`                         | us/deg  | servo datasheet, about 10 us per degree for 180 degree servos |
| direction                     | `sign`                                     | +1 / -1 | +1 if a longer pulse leans the nose toward +x (pitch) or +y (yaw) |
| linkage ratio                 | `linkage_ratio`                            | none    | gimbal degrees per servo degree, from the arm lengths  |
| pulse limits and resolution   | `min_us`, `max_us`, `pulse_resolution_us`  | us      | your servo driver                                      |

## Drag brake

| what                          | YAML key                               | how                                                   |
|-------------------------------|----------------------------------------|-------------------------------------------------------|
| drag area of the open petals  | `drag_device.drag_area_cm2`            | plate area times 1.2, then measure (see first group)  |
| station of the petals         | `drag_device.station_from_nose_mm`     | tape measure; must be ahead of the burnout CG          |
| time to open and to shut      | `deploy_time_s`, `retract_time_s`      | film it                                               |
| when to open and shut         | `brake.deploy_descent_speed_mps`, `retract_descent_speed_mps` | keep 8 and 3 unless the design tool says otherwise |

## Landing legs

| what                          | YAML key                                    | how                                                 |
|-------------------------------|---------------------------------------------|-----------------------------------------------------|
| foot circle diameter          | `legs.span_mm`                              | tape measure between opposite feet                  |
| feet below the tail end       | `legs.height_mm`                            | tape measure                                        |
| number of legs                | `legs.count`                                | count                                               |
| allowed touchdown speed down  | `legs.max_touchdown_vertical_speed_mps`     | drop test of the legs with the rocket's mass; 2 m/s is a hobby leg |
| allowed sideways speed        | `legs.max_touchdown_lateral_speed_mps`      | same drop test with a sideways push                 |
| allowed tilt                  | `legs.max_touchdown_tilt_deg`               | must be below the tip-over angle in docs/physics.md |

## Sensors (`configs/sensors/<name>.yaml`)

| what                          | YAML key                                    | how                                                          |
|-------------------------------|---------------------------------------------|--------------------------------------------------------------|
| IMU sample rate and lag       | `imu.rate_hz`, `imu.lag_s`                  | your driver's rate; lag from the datasheet filter settings   |
| accelerometer noise, bias, range | `imu.accelerometer.*`                    | log 30 s on a still table: standard deviation is the noise, the mean minus gravity the bias; range from the datasheet |
| gyro noise, bias, range       | `imu.gyroscope.*`                           | same still log, in deg/s                                     |
| barometer rate, lag, noise, bias | `barometer.*`                            | still log; lag from the datasheet oversampling setting       |

## Flight computer settings (`example_tvc.yaml`, lower half)

These are choices rather than measurements. Start from the example and let the design tool
and the simulator tell you if they need changing:

| section            | keys                                                                  | note |
|--------------------|-----------------------------------------------------------------------|------|
| `control`          | `control_rate_hz`, `action_delay_steps`                               | the rate your board runs the loop at, and how many steps its command lags |
| `estimator`        | `pad_average_time_s`, `baro_altitude_gain`, `baro_velocity_gain_per_s` | keep the example values unless the estimate drifts in the logs |
| `phases`           | liftoff, burnout, apogee thresholds, abort tilt                      | liftoff threshold must be below your ascent motor's initial acceleration |
| `landing_trigger`  | `target_speed_mps`, `target_height_m`, `thrust_margin`, `calibrate_drag_in_flight` | `thrust_margin` = strongest motor of your batch, for example 1.02 |
| `pid`              | attitude and position gains                                          | retune if the gimbal arm or inertia differ a lot from the example |
| `safety`           | lockout time, maximum ignition tilt and height, `max_gimbal_rate_deg_per_s` | height limit must be above your expected ignition height; the gimbal rate limit below the servo's own speed |

## Launch site (`configs/training/<name>.yaml`)

| what                    | YAML key                           | how                                                   |
|-------------------------|------------------------------------|-------------------------------------------------------|
| site elevation          | `environment.site_elevation_m`     | map or phone GPS                                      |
| wind on the day         | `wind.steady_mps`, `steady_direction_deg`, `gust_std_mps` | hand anemometer; train with the range you expect to fly in |
| motor batch spread      | `randomize.landing_thrust_scale`   | from the load cell tests: +-2 % for a good batch      |
| weighing error          | `randomize.dry_mass_g`             | your scale's error                                    |
| brake area uncertainty  | `randomize.device_drag_area_scale` | how well you know the petals' drag                    |

## Checking the file

```
python -c "from rocketsim.config import load_rocket_config; load_rocket_config('configs/rockets/my_rocket.yaml')"
python scripts/design_landing_burn.py --rocket configs/rockets/my_rocket.yaml --terminal-speed 20
python scripts/fly_scripted.py --rocket configs/rockets/my_rocket.yaml --episodes 20
```

The first line reports any missing or out of range value with the file and key. The second
sizes the brake and the landing motor for your rocket and prints which three measurements
would win the most landings. The third flies twenty simulated landings with the hidden
errors from the training YAML.
