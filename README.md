# Rocket landing simulator

A small Python project for a school rocket that is supposed to land on its tail.
You describe the rocket, its motors and its sensors in YAML files. The simulator flies the
whole thing from the pad to touchdown in three dimensions, trains a landing policy with
reinforcement learning, checks that policy against a plain PID controller, and exports it as
C code that runs on the rocket's flight computer.

The project is being built in stages. Stages 1 and 2 are done: configuration loading, motor
files, six degree of freedom flight physics, servo and sensor models, the onboard state
estimator, flight phases, a landing burn trigger, a PID controller, a safety layer, a closed
loop simulation with wind, flight logs, plots and two ways to watch a flight in 3D. With the
example rocket the PID lands about 7 flights in 10 in calm air and 6 in 10 in a 4 m/s gusty
crosswind; the section on the landing burn explains why not more. Later stages add the
Gymnasium environment, training, the C firmware and the hardware in the loop bridge.

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
- `rocketsim/physics.py` is the rigid body model with an RK4 integrator;
  `rocketsim/touchdown.py` has the leg geometry, ground contact and the landing grade.
  Neither imports anything from RL, so they can be tested on their own.
- `rocketsim/flightlog.py` writes every flight as a CSV with a fixed set of columns.
- `rocketsim/sensors.py` reads the sensor YAML and samples an IMU and a barometer with rate,
  lag, bias and noise. `rocketsim/actuators.py` is the servo model (clip, deadband, pulse
  steps, delay, rate limit) and the igniter delay.
- `rocketsim/estimator.py`, `rocketsim/phases.py`, `rocketsim/landing_trigger.py`,
  `rocketsim/pid.py` and `rocketsim/safety.py` are the parts of the flight computer, and
  `rocketsim/flightcomputer.py` wires them together. `rocketsim/guidance_config.py` reads
  their settings from the rocket YAML. None of this sees the true state, only sensors.
- `rocketsim/simulation.py` is the closed loop: sensors, flight computer, actuators, physics
  and log, one control step at a time. The scripted flight and, later, the Gymnasium
  environment both drive it.
- `scripts/fly_scripted.py` flies the rocket with the flight computer's PID and trigger, or
  open loop, once or over many seeds.
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
  value; the comments in the example file say what they do in one line.

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
`configs/motors/example_g120_landing.yaml`: a list of `[time_s, thrust_n]` pairs plus the
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
`target_height_m`. That is all a solid motor allows: one decision, made once.

The catch is that the burn's impulse is fixed. If the hard part of the burn is too weak for
the speed the rocket has at that crossing, the rocket reaches the ground with speed left. If
it is too strong, the rocket stops in the air and climbs on the leftover thrust, then falls
from wherever the burn ends. With the example rocket the window between the two is about one
metre per second of impulse, or a metre of ignition height. The igniter delay spread of
0.03 s alone is worth a metre at 36 m/s. That is why the PID baseline lands 7 in 10, not 10
in 10, and why the example landing motor has the shape it has: a hard part matched to the
descent speed at the crossing, a very short ramp, and a long tail with thrust a little below
the weight so the rocket sinks the last metre and a half slowly. Read the comments in
`configs/motors/example_g120_landing.yaml`.

For your rocket this means three things. Measure your igniter delay and its spread; it
matters more than any gain. Pick the landing motor and the apogee together, because the
motor's hard impulse must match the speed the rocket has when its stopping distance equals
its height. And expect the trained policy to beat the trigger by doing things a fixed rule
cannot, such as tilting during the hard burn to throw away excess thrust.

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
coast and the descent, lights the landing motor from its table and steers the landing. It
prints the phase times, anything the safety layer refused, and the touchdown numbers, and
writes the CSV and a PNG. The second uses a training YAML with a 4 m/s gusty crosswind. The
third repeats the flight over twenty seeds, each with its own sensor errors, igniter delays
and gusts, and prints one line per flight plus how many landed. `--wind-mps`,
`--wind-direction-deg` and `--gust-mps` override the training YAML.

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
the same seed must give the same flight, and the open loop action path must work.
