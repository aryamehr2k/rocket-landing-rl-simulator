# Rocket landing simulator

A small Python project for a school rocket that is supposed to land on its tail.
You describe the rocket, its motors and its sensors in YAML files. The simulator flies the
whole thing from the pad to touchdown in three dimensions, trains a landing policy with
reinforcement learning, checks that policy against a plain PID controller, and exports it as
C code that runs on the rocket's flight computer.

The project is being built in stages. Stages 1 and 2 are done: configuration loading, motor
files, six degree of freedom flight physics, servo and sensor models, the onboard state
estimator, flight phases, a landing burn trigger, a PID controller, a safety layer, a closed
loop simulation with wind, flight logs, plots and two ways to watch a flight in 3D. After
stage 2 the descent was redesigned: the example rocket now opens a drag brake at the nose
after apogee so it falls at a steady 20 m/s instead of speeding up to 36 m/s, and the landing
motor was resized for that arrival. With that design the PID lands 20 flights in 20 in calm
air and 17 in 20 in a 4 m/s gusty crosswind (the free-fall design managed 14 and 12 on the
same seeds), and it keeps landing with motors 3 % weaker or 2 % stronger than the flight
computer expects. The sections on the landing burn and the drag brake explain the numbers.
Later stages add the Gymnasium environment, training, the C firmware and the hardware in the
loop bridge.

All the numbers in `configs/rockets/example_tvc.yaml` and `configs/motors/` are stand-ins.
Replace them with values measured from your rocket: masses from a scale, centre of gravity
and inertias from OpenRocket or a balance test, and the real motor `.eng` files from
ThrustCurve.org. Nothing that comes out of the simulator means anything until you do.

## Install

You need Python 3.10, 3.11 or 3.12.

```
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
pytest
```

The training extras (`pip install -e ".[train]"`) pull in PyTorch and Stable-Baselines3 and
are only needed from stage 4 on. On Linux the PyPI torch wheel drags in about 2 GB of CUDA
libraries; if you only have a laptop, install the CPU build first and the extras after:

```
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[train]"
```

`.[export]` adds onnx for the policy export in stage 5 and `.[hil]` adds pyserial for the
hardware bridge. Python 3.13 is not supported yet because the pinned numpy has no wheels
for it and Stable-Baselines3 2.4 needs numpy 1.x.

## How the pieces fit

- `docs/conventions.md` defines the world and body frames, the quaternion, the signs of tilt
  and gimbal angles, the units and the control loop timing. Read it before touching anything
  else.
- `rocketsim/units.py` converts YAML values (mm, g, deg) to SI when a file is loaded.
- `rocketsim/config.py` loads and validates rocket YAML files; `rocketsim/simconfig.py` loads
  the simulation, atmosphere and wind settings from a training YAML.
- `rocketsim/motors.py` reads RASP `.eng` files and YAML thrust curves.
- `rocketsim/quaternion.py` is the attitude maths: rotations, tilt angles, roll.
- `rocketsim/aero.py` holds air density, drag, the fin normal force and the wind model.
  `rocketsim/dragdevice.py` is the deployable drag brake: its force and moment at its station,
  and the terminal speed sums (how much brake area for a wanted descent speed).
- `rocketsim/physics.py` is the rigid body model with an RK4 integrator;
  `rocketsim/touchdown.py` has the leg geometry, ground contact and the landing grade.
  Neither imports anything from RL, so they can be tested on their own.
- `rocketsim/flightlog.py` writes every flight as a CSV with a fixed set of columns.
- `rocketsim/sensors.py` reads the sensor YAML and samples an IMU and a barometer with rate,
  lag, bias and noise. `rocketsim/actuators.py` is the servo model (clip, deadband, pulse
  steps, delay, rate limit), the brake servo that opens and shuts the drag device, and the
  igniter delay. `rocketsim/servo_calibration.py` turns gimbal angles into pulse widths.
- `rocketsim/estimator.py`, `rocketsim/phases.py`, `rocketsim/landing_trigger.py`,
  `rocketsim/pid.py` and `rocketsim/safety.py` are the parts of the flight computer, and
  `rocketsim/flightcomputer.py` wires them together. `rocketsim/guidance_config.py` reads
  their settings from the rocket YAML. None of this sees the true state, only sensors.
- `rocketsim/simulation.py` is the closed loop: sensors, flight computer, actuators, physics
  and log, one control step at a time. The scripted flight and, later, the Gymnasium
  environment both drive it. `rocketsim/randomize.py` draws the hidden errors of each flight
  (motor strength, mass, brake area) from the training YAML, and `rocketsim/burnsummary.py`
  collects the numbers that explain a touchdown speed: where the burn stopped the rocket, how
  much it climbed afterwards, how much burn was left.
- `rocketsim/landing_design.py`, `rocketsim/landing_montecarlo.py` and
  `rocketsim/landing_sensitivity.py` are the one dimensional design model behind
  `scripts/design_landing_burn.py`: brake sizing, the speed profile, the hard part and tail of
  the motor, thousands of quick landings with random errors, and the closed forms for how far
  the stop point moves per error. They reuse the flight computer's own burn integration.
- `scripts/fly_scripted.py` flies the rocket with the flight computer's PID and trigger, or
  open loop, once or over many seeds, with a hidden motor or mass error pinned if you want.
  `scripts/design_landing_burn.py` designs the brake and the motor for a rocket and prints the
  expected landing rate.
- `scripts/plot_flight.py` plots one flight log. `scripts/animate_flight.py` renders it as a
  GIF or MP4, and `viewer/flight_viewer.html` shows it in the browser. `scripts/bundle_viewer.py`
  packs the viewer and one flight into a single HTML file and `scripts/serve_viewer.py` serves
  the viewer from a remote machine.

## Adding a rocket

Copy `configs/rockets/example_tvc.yaml`, rename it and change the values. Every position is
a station measured from the nose tip toward the tail in millimetres. The keys carry their
unit in the suffix (`dry_mass_g`, `pivot_from_nose_mm`, `max_angle_deg`) and are converted
on load, so type the number in the unit the key names. Unknown keys and out of range values
are rejected with a message that names the file and the key.

The file has these sections:

- `airframe`: dry mass and its centre of gravity (the rocket with no motors in it), the
  pitch inertia about a lateral axis through that point, the roll inertia about the body
  axis, the overall length and the reference diameter.
- `aero`: drag coefficient, normal force slope per radian, and the centre of pressure.
  The centre of gravity is computed from the parts and moves as propellant burns.
- `motors`: an `ascent` and a `landing` motor, each with a motor file, its station, whether
  it thrusts through the gimbal, and the igniter delay as a mean and a spread in seconds.
- `drag_device` (optional): the drag brake. `drag_area_cm2` is the drag coefficient times the
  area of the open petals, `station_from_nose_mm` where they sit, `deploy_time_s` and
  `retract_time_s` how fast they open and shut. Put it near the nose; see the drag brake
  section for why the legs must never be the brake.
- `gimbal`: pivot station, maximum deflection, servo rate limit, delay and deadband, and one
  calibration block per servo (`pitch` leans the nose toward +x, `yaw` toward +y) from gimbal
  angle to pulse width in microseconds.
- `legs`: foot span, height of the tail end above the ground, number of legs, and the allowed
  touchdown vertical speed, lateral speed and tilt. The tip-over angle is computed from the
  footprint and the centre of gravity height, not typed in.
- `control`: control loop rate and how many control steps an action is delayed.
- `sensors`: the sensor YAML to use, see below.
- `estimator`, `phases`, `landing_trigger`, `pid`, `safety`: what runs on the flight
  computer. The README section on the landing burn and `docs/conventions.md` explain each
  value; the comments in the example file say what they do in one line. `landing_trigger`
  has `thrust_margin` (set it to the strongest motor of your batch, for example 1.02) and
  `calibrate_drag_in_flight` (fit the brake's drag from the accelerometer during the descent
  and rebuild the trigger table; a way to measure the brake on early flights, off by default).
- `brake` (needs a `drag_device`): the flight computer's rules for it. Open once the descent
  is faster than `deploy_descent_speed_mps`, hold `burn_fraction` of the opening during the
  hard part of the burn, shut once the descent is slower than `retract_descent_speed_mps`.

`configs/rockets/example_tvc_freefall.yaml` is the same rocket without the brake and with the
old landing motor, kept so you can compare the two designs with the same commands.

A note on fins, because it decides whether the rocket can land at all. A rocket with the
centre of pressure behind the centre of gravity is stable nose-first, which is what fins are
for on a normal model rocket. The same geometry makes tail-first flight unstable: once the
rocket starts falling, any small tilt grows exponentially until it is nose-down. The opposite
choice, centre of pressure ahead of the centre of gravity, makes the unpowered coast after
burnout unstable instead: the simulator showed a 14 mm margin turning a 1 degree tilt at
burnout into 30 degrees at apogee, and the landing motor then lit with the rocket far from
vertical. There is no thrust during the coast or the descent, so the gimbal cannot help in
either case. The example rocket therefore has only vestigial fins and its centre of pressure
sits at the centre of gravity after the ascent burn, where it is neutral: whatever tilt and
turning rate it has at burnout it keeps, slowly, until the landing burn takes over. It is
unstable during both burns, which is what the thrust vector control is for. Judge the margin
at burnout, not on the pad, because the motors sit at the tail and the centre of gravity moves
forward as propellant burns. The simulator models all of this, so you can see what your own
fins do before you build them.

## Adding a motor

Drop the `.eng` file from ThrustCurve.org into `configs/motors/` and point the rocket file at
it. The `.eng` header gives the propellant and total mass, and the pairs after it are the
thrust curve. If you measured a curve yourself, write it as YAML like
`configs/motors/example_g127_landing_brake.yaml`: a list of `[time_s, thrust_n]` pairs plus the
masses. Propellant mass flow follows the thrust curve, so mass and centre of gravity change
during the burn. A motor YAML may carry a default `ignition_delay_s`; the rocket file can
override it.

Solid motors cannot throttle or restart. For solids the landing decision is when to send the
ignition command and how to steer the gimbal during the burn. A YAML motor with
`throttleable: true` also gets a throttle input with a first order lag.

## The landing burn

This is the part of the project that decides everything else, so it gets its own section.

The flight computer builds a table at start-up: for every descent speed, how far the rocket
falls from the moment the landing motor lights until the burn has brought it down to
`target_speed_mps`. It does this by integrating the motor's thrust curve, the rocket's mass,
gravity and drag in one dimension. In flight it watches the estimated height and speed,
adds the igniter delay and its own control loop latency, and sends the igniter command the
moment the height it will have when thrust starts drops to the table value plus
`target_height_m`. That is all a solid motor allows: one decision, made once. Above the
fastest speed the burn can stop at all, the table keeps the height of that edge rather than
firing tens of metres early, so the burn still ends as low as it can.

The catch is that the burn's impulse is fixed. If the hard part of the burn is too weak for
the speed the rocket has when it lights, the rocket reaches the ground with speed left. If it
is too strong, the rocket stops in the air and climbs on the leftover thrust, then falls from
wherever the burn ends. The stop point is where the hard part has brought the descent down
to `target_speed_mps`; the example plans it 1.5 m up and the one dimensional model puts it
at 1.6 m. A stop that comes out a little too low is harmless: the tail has less height to
sink, and the rocket only hits hard if the stop would have been below the ground (for the
example, 1.8 m too low). A stop that comes out too high costs time: the rest of the hard ramp
throws the rocket up a little, and the weak tail then needs seconds to bring it down. The
tail must last that long. The example's 6.5 s tail survives a stop 1.1 m too high; the old
3.7 s tail did not, and that is what made the free fall fragile.

### Why the free fall was fragile

The first design let the rocket free-fall from its 106 m apogee. Drag on the slim body is
only about 2 N against 14 N of weight, so it kept speeding up and lit the motor at 36 m/s,
38 m above the ground, with a 46 N hard burn. Three things then had to be right at once:

| error, one at a time                 | moves the stop point by | touchdown (1D model)                  |
|--------------------------------------|-------------------------|---------------------------------------|
| igniter 30 ms early                  | 1.1 m up                | 4.8 m/s: the 3.7 s tail runs out      |
| igniter 30 ms late                   | 1.1 m down              | 0.3 m/s                               |
| motor 1 % stronger                   | 0.45 m up               | 5.9 m/s: climbs, tail runs out        |
| motor 1 % weaker                     | 0.45 m down             | 1.3 m/s                               |
| speed estimate 0.3 m/s fast or slow  | 0.5 m                   | 0.8 to 1.3 m/s                        |
| apogee 3 m lower                     | about 1 m               | 3.3 m/s                               |

The rows come from `scripts/design_landing_burn.py --rocket configs/rockets/example_tvc_freefall.yaml
--drag-area-cm2 0 --motor configs/motors/example_g120_landing.yaml`, which flies the free-fall
rocket in one dimension with one error at a time. The stop point moves by the speed times the
timing error, and 36 m/s is a lot of speed: the trigger's own 20 ms control step is 0.7 m of
fall. The whole-curve motor window was -3 % to 0 %. The motor's impulse was also matched to one
apogee: a different ascent motor, or the same motor on a hotter day, needed a different landing
motor. In the simulator this design landed 14 of 20 flights in calm air and 12 of 20 in a
4 m/s gusty wind, and with motors 5 % off or the rocket 40 g lighter than the flight computer
believed it landed none.

### The fix: arrive at the same speed every time

A solid motor cannot throttle, and tilting the rocket to throw thrust away costs 4 to 20 times
as much sideways push as the thrust it removes (0 of 30 landings in wind when we tried it). A
gentler, earlier burn is not the answer either: it lands less often, because a weak brake's
stopping distance is more sensitive to thrust and mass errors. Descending slowly under power
is out of the question with a solid: hovering costs 14 N s per second, and coming down 100 m
at 5 m/s would burn 280 N s, more than two of these motors.

What works is to make the rocket arrive at the burn at the same speed every flight. The
example rocket now opens a drag brake at the nose once it is falling faster than 8 m/s after
apogee. Drag then equals weight at 20 m/s, so the speed stops growing there: this is the
terminal speed, and the rocket reaches it whatever the apogee was. The landing motor is sized
for that one arrival speed: 31 N (2.2 times the weight) for 1.41 s, then a 12.6 N tail (0.92
of the weight) to 8 s. This is the speed profile the simulator gives, feet height to speed
down:

| height | 100 m | 80 m | 60 m | 40 m | 20 m | 17 m (motor lit) | 10 m | 5 m | 2 m | 1 m | touchdown |
|--------|-------|------|------|------|------|------------------|------|-----|-----|-----|-----------|
| speed  | 10    | 17   | 19   | 19.7 | 19.9 | 19.9             | 16   | 10  | 4   | 1   | 1.2-1.9 m/s |

The command goes out at about 17 m; thrust starts 0.18 s later at about 14 m; the hard part
has the rocket at walking pace after 1.4 s, 1.6 m up in the one dimensional model and between
0.3 and 2.5 m in the simulator's flights; the brake shuts and the tail sets it down over the
next 2 to 3 s. From apogee to touchdown takes about 10.5 s.

At 20 m/s the same errors cost about half as much: a 30 ms igniter error moves the stop
0.6 m instead of 1.1 m (touchdown 1.3 or 1.8 m/s instead of 0.3 or 4.8), a 1 % thrust error
0.2 m instead of 0.45 m (1.5 or 1.6 m/s instead of 1.3 or 5.9), and the brake's drag during
the burn adds deceleration that costs no propellant. The tail has 6.5 s to sink a high stop.
The design tool's fourth block prints these numbers for any rocket, closed form next to the
one dimensional model. The table of results, 20 flights
each, seeds 100 to 119, from `scripts/fly_scripted.py` (the first two rows without hidden
errors, the third with the training YAML's random motor, mass and brake errors):

| case                                            | free fall (old) | drag brake (new) |
|-------------------------------------------------|-----------------|------------------|
| calm air                                        | 14 of 20        | 20 of 20         |
| 4 m/s crosswind with 1.5 m/s gusts              | 12 of 20        | 17 of 20         |
| calm, motors +-2 %, mass +-12 g, brake +-5 %    | 5 of 20         | 19 of 20         |
| windy, same hidden errors                       | 6 of 20         | 18 of 20         |
| landing motor 5 % weak                          | 0 of 20         | 16 of 20         |
| landing motor 3 % weak                          | 9 of 20         | 19 of 20         |
| landing motor 2 % strong                        | 0 of 20         | 20 of 20         |
| landing motor 3 % strong                        | 0 of 20         | 2 of 20          |
| rocket 36 g lighter than the flight computer thinks | 1 of 20      | 1 of 20          |
| rocket 36 g heavier                             | 11 of 20        | 19 of 20         |

The three windy failures of the new design are sideways speed and tilt at touchdown after the
rocket swung in the wind, not touchdown speed. A rocket that is lighter than the flight
computer believes behaves like a stronger motor and still crashes; weigh the rocket before
every flight and put the number in the file.

## The drag brake

Four flat petals of about 109 by 109 mm at the nose, roughly 470 cm2 of plate, give the
example rocket a drag area (`Cd * A`) of about 570 cm2 on top of the body's 27 cm2 (the file
says 566; the tool rounds to 567 for exactly 20.0 m/s). The sum comes
from `Cd * A = 2 m g / (rho v_t^2)` with the descent mass of 1.43 kg, the air density at the
site and the wanted terminal speed `v_t` of 20 m/s. Each petal carries about 3.4 N at 20 m/s
and needs about 0.18 N m at its hinge; a hobby servo opens all four in half a second.

Why the nose: when the rocket falls tail-first the nose is the trailing end, and drag at the
trailing end steadies the fall like the feathers of a shuttlecock. Drag at the leading end
does the opposite. We tried the legs as brakes: in 119 of 120 simulated flights the rocket
flipped nose-down during the unpowered coast, because there is no thrust then to fight the
turning moment. The safety layer therefore refuses to open any device behind the descent
centre of gravity until the landing burn is running. The petals also stay shut during the
boost, where a 10 % leak would eat 10 % of the gimbal's authority.

Why 20 m/s: at 15 m/s the brake needs 855 cm2 of plate and the rocket swings so much in wind
that it landed 6 of 30 windy flights; at 25 m/s the stop point is 1.5 times more sensitive to
thrust errors. 20 m/s was the best of the three.

Three rules came out of the failures. Open the petals only once the rocket falls faster than
8 m/s, not at apogee, or the horizontal wind swings the rocket 40 to 60 degrees while it is
still slow. Shut them at the hand-over from the hard burn to the tail (the descent below
3 m/s), or the wind pushes on them during the slow sink; leaving them open cost 6 landings in
30 windy flights. And expect a swing of 30 to 50 degrees in a 4 m/s wind during the descent,
which is close to the 35 degree tilt at which the safety layer refuses to light the motor.

What the simulator does not know: the petals are an ideal drag area. The real drag
coefficient of petals sitting in the body's wake, their side forces and roll, and the true
damping of the pendulum swing need a drop test or a wind tunnel. The brake area must be known
to about 5 %: with 15 % more drag than the file says the rocket arrives at 18.6 m/s instead of
20, the hard part stops it too high, and 9 of 20 flights land. With 15 % less drag it arrives
at 21.7 m/s, faster than the 21.5 m/s the hard part can stop, and 17 of 20 land with a little
speed left. Log the terminal speed on the first flights, or set `calibrate_drag_in_flight:
true`: the flight computer then fits the drag from the accelerometer while it falls with the
petals open (the specific force along the body is the drag over the mass, so
`f = (k / m) v^2`, and one second of samples fixes `k / m`) and rebuilds its trigger table for
the drag it measured. In the simulator the fit is within about 2 % of the true area in calm
air and reads about 4 % high in a 4 m/s wind, because the wind adds to the airspeed. It
corrects where the trigger fires, not the motor's impulse, so it helps most when the burn can
still stop the rocket:

| brake area error, 20 flights, seeds 100 to 119 | trigger table from the file | table refitted in flight |
|------------------------------------------------|-----------------------------|--------------------------|
| none, calm                                     | 20 of 20                    | 20 of 20                 |
| none, 4 m/s gusty crosswind                    | 17 of 20                    | 18 of 20                 |
| 15 % less drag (arrives at 21.7 m/s)           | 17 of 20                    | 19 of 20                 |
| 10 % less drag                                 | 18 of 20                    | 20 of 20                 |
| 10 % more drag                                 | 17 of 20                    | 20 of 20                 |
| 15 % more drag (arrives at 18.6 m/s)           | 9 of 20                     | 14 of 20                 |

The example leaves the setting off, so the results in this README use the file's table; the
flight log of `scripts/fly_scripted.py` prints the fitted area when it is on. Size the hard
part for the low end of your drag estimate: a burn that cannot stop the arrival speed is not
rescued by knowing it.

## Your motor batch and the thrust window

The motor strength window is asymmetric. A motor weaker than the curve in the file leaves a
little speed the tail can absorb: 4 % weak still lands 18 of 20, 5 % weak 16 of 20. A motor
stronger than the curve stops the rocket too high and it climbs on the leftover hard burn; the
tail, at 0.92 of the weight, cannot bring it down before burnout: 3 % strong lands 2 of 20, 5 %
strong none. So:

- Measure several landing motors of your batch on a load cell. Total impulse and the impulse
  of the hard part to 1 or 2 %.
- Load the flight computer with the curve of the strongest motor, or set
  `landing_trigger.thrust_margin` to 1 plus half the spread (1.02 for a +-2 % batch). Every
  real motor is then nominal or weaker than the table assumes, which is the safe side. The
  loader allows a margin up to 1.10.
- Measure the igniter delay on about 20 igniters with your own firing circuit. +-30 ms is fine
  now; +-60 ms still landed 29 of 30 in calm air but 24 of 30 in wind.
- Weigh the rocket before each flight to 10 g and enter the mass.

## Designing your own landing

```
python scripts/design_landing_burn.py --rocket configs/rockets/example_tvc.yaml --terminal-speed 20
python scripts/design_landing_burn.py --rocket my_rocket.yaml --terminal-speed 18 --hard-n 28 --tail-ratio 0.92 --tail-s 6.5 --sim-flights 20
python scripts/design_landing_burn.py --rocket configs/rockets/example_tvc_freefall.yaml --drag-area-cm2 0 --motor configs/motors/example_g120_landing.yaml
```

The tool prints seven blocks, each with how it was computed: the brake area for the wanted
terminal speed and the speed profile from apogee; the hard part end time and the tail thrust
of the motor, with a thrust curve you can paste into a motor YAML; the trigger numbers (when
the command goes out, where thrust starts, where the rocket stops); how far the stop point
moves per error and what that costs at touchdown, as a closed form and as the one dimensional
model with that single error, side by side (where they differ, trust the model); the motor
strength window; a landing rate
from thousands of one dimensional flights with your error budget (`--igniter-spread-s`,
`--thrust-error`, `--mass-error-g`, `--drag-area-error`) and, with `--sim-flights N`, real
simulator flights in calm and windy air; and the three measurements that would win the most
landings. With `--motor` it checks an existing motor file instead of sizing one, and
`--drag-area-cm2 0` checks a rocket without a brake (the third command above is the free-fall
table in the section on the landing burn). The one dimensional model has no wind, tilt or
sideways speed, so its landing rate is an upper bound: 97 % on the example's error budget
where the simulator gives 19 of 20.

Two things the tool will tell you to watch. The hard part is sized for the terminal speed plus
a small margin; if your brake area might be smaller than you think, size for the low end of
its band, because a motor at the edge of its reach cannot be rescued. And the tail must last
longer than the slow sink from the highest stop it can survive, which is why the example tail
runs 6.5 s.

## Adding sensors

`configs/sensors/example_imu.yaml` describes the IMU and the barometer: sample rate, lag, the
standard deviation of the bias drawn once per flight, the noise per sample and the range.
The estimator calibrates the biases away on the pad during `pad_hold_time_s`, so what hurts
the landing is drift during the flight and the barometer lag, which it corrects with the
estimated vertical speed. Replace the numbers with what you measure with the board sitting
still on a table; the datasheet noise density times the square root of the bandwidth is a
fair start.

## Flying

```
python scripts/fly_scripted.py --out runs/landing.csv --plot
python scripts/fly_scripted.py --sim configs/training/windy.yaml --out runs/windy.csv --plot
python scripts/fly_scripted.py --episodes 20 --seed 1 --wind-mps 3 --gust-mps 1
```

The first command flies the whole thing closed loop: the flight computer calibrates on the
pad for two seconds, launches, holds the rocket upright through the boost, waits through the
coast, opens the brake in the descent, lights the landing motor from its table, shuts the
brake at the hand-over and steers the landing. It prints the phase times, anything the safety
layer refused, the hidden errors of this flight, a summary of the burn (estimate errors at
the command, where thrust started, where the rocket stopped, how much burn was left) and the
touchdown numbers, and writes the CSV and a PNG. The second uses a training YAML with a 4 m/s
gusty crosswind. The third repeats the flight over twenty seeds, each with its own sensor
errors, igniter delays, gusts and hidden errors, and prints one line per flight plus how many
landed. `--wind-mps`, `--wind-direction-deg` and `--gust-mps` override the training YAML.

Every flight also draws hidden errors from the `randomize` section of the training YAML:
motor strength, dry mass and brake area within the listed ranges. The flight computer never
sees them; that is the point. To sweep one of them, pin it:

```
python scripts/fly_scripted.py --episodes 20 --seed 100 --landing-thrust-scale 1.03
python scripts/fly_scripted.py --episodes 20 --seed 100 --dry-mass-offset-g -36
python scripts/fly_scripted.py --rocket configs/rockets/example_tvc_freefall.yaml --episodes 20 --seed 100
```

The last line flies the old free-fall design for comparison. Delete the `randomize` section
from a copy of the training YAML to fly with no hidden errors at all.

```
python scripts/fly_scripted.py --open-loop --out runs/vertical.csv --plot
```

`--open-loop` switches the PID and the trigger off: the gimbal is held at
`--gimbal-pitch-deg` and `--gimbal-yaw-deg` (the rocket then tips over, which is a good way
to see the sign conventions) and `--landing-ignite-at` sends the landing igniter command at a
given time, subject to the same safety checks as the flight computer. With the example rocket
the vertical flight reaches about 105 m and, with no landing burn, hits the ground at about
44 m/s.

To plot a log you already have:

```
python scripts/plot_flight.py runs/vertical.csv --out runs/vertical.png
```

## Watching a flight in 3D

Two options, both reading the flight log CSV:

```
python scripts/animate_flight.py runs/vertical.csv --out runs/vertical.gif
python scripts/animate_flight.py runs/vertical.csv --out runs/vertical.mp4 --follow
```

The GIF needs nothing extra. The MP4 needs `ffmpeg` on your PATH. `--follow` keeps the camera
in a small box around the rocket, which is the view you want for a flight that goes straight
up, because the full-extent view has to fit the whole 115 m and the rocket becomes a dot.
`--speed` changes playback speed and `--fps` the frame rate.

For an interactive view, open `viewer/flight_viewer.html` in a browser, choose the CSV and,
optionally, the rocket YAML so the drawing has the right proportions. You can orbit, zoom,
scrub through time and switch between the true and the estimated state. The page loads
three.js from a CDN, so it needs an internet connection the first time.

If the simulator runs on a remote machine you reach over SSH, the HTML file is not on the
computer that has the browser. Two ways around that:

```
python scripts/bundle_viewer.py runs/vertical.csv --out runs/vertical.html
```

writes one self-contained HTML file with the flight and the rocket geometry inside it.
Download that file (in VS Code: right click it, Download) and open it on any computer. It
starts playing with the camera following the rocket. Or

```
python scripts/serve_viewer.py runs/vertical.csv
```

serves the project folder on port 8000 and prints a link. VS Code forwards the port by itself
when it sees the link in the terminal; from a plain SSH session, connect with
`ssh -L 8000:localhost:8000 user@host` first, then open the link on your own computer. The
viewer reads `?log=` and `?rocket=` from its URL, which is what that link carries, so you can
edit the log name in the address bar after another run.

## Training, evaluating, exporting, comparing with a real flight

These parts arrive in later stages: `scripts/train.py`, `scripts/evaluate.py`,
`scripts/export_policy.py`, `scripts/hil_bridge.py`, `scripts/compare_real_flight.py` and
the `firmware/` tree. This section will be filled in as they land.

## Tests

```
pytest
```

The tests cover unit conversion, config validation messages, motor file parsing, the
quaternion maths, one derivative evaluation and one RK4 step against numbers worked out by
hand, energy and angular momentum conservation in a vacuum, the mirror symmetry of the pitch
and yaw planes, a vertical burn that must stay exactly vertical, the pad hold, the touchdown
geometry with four legs and the flight log round trip. Stage 2 adds the servo model step by
step, sensor rates, lag and bias, the estimator's pad calibration, integration and barometer
fusion, the phase sequence, the PID signs, the trigger table against a constant deceleration,
the safety refusals, and the whole closed loop: the example rocket must land in calm air,
the same seed must give the same flight, and the open loop action path must work. The drag
brake adds tests for the device force (13.4 N at 20 m/s), the sign of its moment at the nose
and at the tail, its damping, the brake servo's ramps, the deploy and retract rules and their
override by a policy, the safety refusals for the brake, the trigger's drag-aware delay
prediction, its accelerometer drag fit (on synthetic samples and in a closed loop flight, where
it must finish before the burn and land within 3 % of the true area), the new motor's impulse
and tail ratio, the hidden errors being drawn per seed and kept from the flight computer, the
burn summary and the log columns, and the design tool reproducing the example's brake area,
hard part end time, tail thrust, motor strength window and landing rates, its closed-form stop
sensitivities agreeing with the one dimensional model to 0.3 m/s, and its usage message for a
rocket without a brake.
