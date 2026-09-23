# Rocket landing simulator

A small Python project for a school rocket that is supposed to land on its tail.
You describe the rocket, its motors and its sensors in YAML files. The simulator flies the
whole thing from the pad to touchdown in three dimensions, trains a landing policy with
reinforcement learning, checks that policy against a plain PID controller, and exports it as
C code that runs on the rocket's flight computer.

The project is being built in stages. Right now stage 1 is done: configuration loading,
motor files, six degree of freedom flight physics, an open loop vertical test flight with a
plot, and two ways to watch a flight as a 3D animation. Later stages add the servo and sensor
models, the estimator, the PID baseline, the Gymnasium environment, training, the C firmware
and the hardware in the loop bridge.

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
- `scripts/fly_scripted.py` flies the rocket without any controller (stage 1) and with the
  PID baseline (from stage 2).
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

A note on fins, because it decides whether the rocket can land at all. A rocket with the
centre of pressure behind the centre of gravity is stable nose-first, which is what fins are
for on a normal model rocket. The same geometry makes tail-first flight unstable: once the
rocket starts falling, any small tilt grows exponentially until it is nose-down, and the
margin only sets how fast. A solid landing motor cannot buy time to recover from that. The
example rocket therefore has only vestigial fins and its centre of pressure sits a little
ahead of the centre of gravity at ascent burnout. It is unstable on the way up, which is what
the thrust vector control is for, and settles tail-first on the way down. Judge the margin at
burnout, not on the pad, because the motors sit at the tail and the centre of gravity moves
forward as propellant burns. The simulator models both cases, so you can see what your own
fins do before you build them.

## Adding a motor

Drop the `.eng` file from ThrustCurve.org into `configs/motors/` and point the rocket file at
it. The `.eng` header gives the propellant and total mass, and the pairs after it are the
thrust curve. If you measured a curve yourself, write it as YAML like
`configs/motors/example_f30_landing.yaml`: a list of `[time_s, thrust_n]` pairs plus the
masses. Propellant mass flow follows the thrust curve, so mass and centre of gravity change
during the burn. A motor YAML may carry a default `ignition_delay_s`; the rocket file can
override it.

Solid motors cannot throttle or restart. For solids the landing decision is when to send the
ignition command and how to steer the gimbal during the burn. A YAML motor with
`throttleable: true` also gets a throttle input with a first order lag.

## Flying the open loop test flight

```
python scripts/fly_scripted.py --out runs/vertical.csv --plot
```

This ignites the ascent motor on the pad with both gimbal servos at zero and lets the rocket
fly ballistically until it hits the ground. It prints the apogee and the touchdown numbers
and writes `runs/vertical.csv` and `runs/vertical.png`. `--gimbal-pitch-deg` and
`--gimbal-yaw-deg` hold the gimbal at a fixed angle (the rocket then tips over, which is a
good way to see the sign conventions), and `--landing-ignite-at` lights the landing motor at
a given time, which is a quick way to feel how sensitive a landing burn is to timing. With the
example rocket the vertical flight reaches about 115 m and, with no landing burn, hits the
ground at about 45 m/s.

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
geometry with four legs and the flight log round trip.
