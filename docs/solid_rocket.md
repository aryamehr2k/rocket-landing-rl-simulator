# Solid-motor rocket

The second vehicle in this project: a 1.5 kg rocket that boosts on one solid motor, falls back
under a nose drag brake and lands on a second solid motor. A PID flies the boost, a stopping
distance table lights the landing motor, and a trained network can steer the landing burn.
This page collects the design notes and commands for it; the electric test vehicle is covered
in the main README.

## Details

### Adding a rocket

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
  `safety.max_gimbal_rate_deg_per_s` limits how fast the gimbal command may move, whoever
  issues it, so neither the PID nor a trained policy can whip the servos.
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

### Adding a motor

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

### The landing burn

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

#### Why the free fall was fragile

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

#### The fix: arrive at the same speed every time

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
one dimensional model. The table of results, 20 flights each, seeds 100 to 119, from
`scripts/fly_scripted.py`. The rows marked "none" were flown without hidden errors, with
`--sim configs/training/calm_exact.yaml` or `windy_exact.yaml`, so a pinned motor or mass
error is the only error; the two rows marked "training YAML" used `default.yaml` and
`windy.yaml`, whose `randomize` section draws motor, mass and brake errors for every flight:

| case                                                | hidden errors | free fall (old) | drag brake (new) |
|-----------------------------------------------------|---------------|-----------------|------------------|
| calm air                                            | none          | 14 of 20        | 20 of 20         |
| 4 m/s crosswind with 1.5 m/s gusts                  | none          | 12 of 20        | 17 of 20         |
| calm, motors +-2 %, mass +-12 g, brake +-5 %        | training YAML | 5 of 20         | 19 of 20         |
| windy, same hidden errors                           | training YAML | 5 of 20         | 18 of 20         |
| landing motor 5 % weak                              | none          | 0 of 20         | 16 of 20         |
| landing motor 3 % weak                              | none          | 9 of 20         | 19 of 20         |
| landing motor 2 % strong                            | none          | 0 of 20         | 20 of 20         |
| landing motor 3 % strong                            | none          | 0 of 20         | 2 of 20          |
| rocket 36 g lighter than the flight computer thinks | none          | 3 of 20         | 1 of 20          |
| rocket 36 g heavier                                 | none          | 11 of 20        | 19 of 20         |

The commands, one per kind of row (add `--rocket configs/rockets/example_tvc_freefall.yaml`
for the free fall column):

```
python scripts/fly_scripted.py --episodes 20 --seed 100 --sim configs/training/calm_exact.yaml
python scripts/fly_scripted.py --episodes 20 --seed 100 --sim configs/training/windy_exact.yaml
python scripts/fly_scripted.py --episodes 20 --seed 100
python scripts/fly_scripted.py --episodes 20 --seed 100 --sim configs/training/windy.yaml
python scripts/fly_scripted.py --episodes 20 --seed 100 --sim configs/training/calm_exact.yaml --landing-thrust-scale 0.95
python scripts/fly_scripted.py --episodes 20 --seed 100 --sim configs/training/calm_exact.yaml --dry-mass-offset-g -36
```

The three windy failures of the new design, without hidden errors, came after the rocket
swung in the wind: seed 106 touched down with 1.05 m/s of sideways speed, seed 116 with a
tilt of 11.3 degrees, and seed 119 with 2.08 m/s of vertical speed and 1.10 m/s sideways.
A rocket that is lighter than the flight computer believes behaves like a stronger motor and
still crashes; weigh the rocket before every flight and put the number in the file.

### The drag brake

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

### Your motor batch and the thrust window

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

### Designing your own landing

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

### Adding sensors

`configs/sensors/example_imu.yaml` describes the IMU and the barometer: sample rate, lag, the
standard deviation of the bias drawn once per flight, the noise per sample and the range.
The estimator calibrates the biases away on the pad during `pad_hold_time_s`, so what hurts
the landing is drift during the flight and the barometer lag, which it corrects with the
estimated vertical speed. Replace the numbers with what you measure with the board sitting
still on a table; the datasheet noise density times the square root of the bandwidth is a
fair start.

### Flying

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
sees them; that is the point. `configs/training/calm_exact.yaml` and `windy_exact.yaml` are
`default.yaml` and `windy.yaml` without that section, so nothing hidden is drawn; the tables
in this README were flown with them unless a row says otherwise. To sweep one error, pin it
on top of an exact YAML, or the other errors are still drawn around it:

```
python scripts/fly_scripted.py --episodes 20 --seed 100 --sim configs/training/calm_exact.yaml --landing-thrust-scale 1.03
python scripts/fly_scripted.py --episodes 20 --seed 100 --sim configs/training/calm_exact.yaml --dry-mass-offset-g -36
python scripts/fly_scripted.py --episodes 20 --seed 100 --sim configs/training/calm_exact.yaml --rocket configs/rockets/example_tvc_freefall.yaml
```

The last line flies the old free-fall design for comparison.

```
python scripts/fly_scripted.py --open-loop --out runs/vertical.csv --plot
```

`--open-loop` switches the PID and the trigger off: the gimbal is held at
`--gimbal-pitch-deg` and `--gimbal-yaw-deg` (the rocket then tips over, which is a good way
to see the sign conventions) and `--landing-ignite-at` sends the landing igniter command at a
given time, subject to the same safety checks as the flight computer. With the example rocket
the vertical flight reaches about 102 m; the brake rules are not part of the PID or the
trigger, so the petals still open in the descent and, with no landing burn, the rocket hits
the ground at the brake's 20 m/s. With `--rocket configs/rockets/example_tvc_freefall.yaml`
it free-falls and hits at about 42 m/s.

To plot a log you already have:

```
python scripts/plot_flight.py runs/vertical.csv --out runs/vertical.png
```

### Training the landing policy

The policy is a small neural network (two layers of 32 neurons) trained with PPO from
Stable-Baselines3 on the same simulation the PID flies. Install the training extras first
(see Install), then:

```
python scripts/train.py --rocket configs/rockets/example_tvc.yaml --training configs/training/default.yaml
```

Every run gets a folder `runs/<date>_<time>_ppo/` with the model, the policy as plain numpy
weights (`policy.npz`), the observation normalisation, the training log (`progress.csv`, one
line per update with the landing rate of the last hundred flights) and exact copies of every
YAML it used. Training the example for four million steps takes about an hour on a desktop;
`--timesteps` overrides the YAML for a quick try.

What the policy does is decided in YAML, not code:

- `controllers` in the rocket file says who steers in each phase: `BOOST` and `LANDING_BURN`
  are `pid` or `policy`, `landing_ignition` is `trigger` (the stopping distance table) or
  `policy`. The example lets the policy fly the landing burn and leaves the boost and the
  ignition to the rules. A training YAML may override the section, which is how
  `policy_boost.yaml` and `policy_ignition.yaml` try more authority. The PID stays the
  fallback: if the policy ever outputs a non-number the flight computer uses the PID's
  command for that step.
- `observation` in the training file lists what the policy sees, all from the estimator, each
  divided by a scale: height, vertical speed, the plane's lateral position and speed, tilt and
  tilt rate, the last gimbal command, the phase flags, how far the burn has progressed and
  whether the trigger table would light the motor now.
- `action`: the policy outputs the gimbal angle for its plane and a continuous ignite signal;
  above `ignite_threshold` the landing igniter is commanded. The same network runs twice per
  control step, once for the pitch plane and once for the yaw plane, exactly as it will on
  the flight computer, and the two ignite outputs are averaged.
- `rewards`: one weight per term. Touchdown speed, sideways speed, tilt and distance from the
  pad are penalties paid once; a landing inside the leg limits is a bonus and a crash a large
  penalty; small penalties every step for tilt, sideways motion, gimbal movement and gimbal
  angle shape the behaviour during the burn. The gimbal terms matter: with a weak movement
  penalty the first trained policy landed by flipping the gimbal between its limits every
  step, which a servo would not survive for long.
- `curriculum`: training starts in calm air with little sensor noise and no hidden errors, and
  the stages add wind (a steady speed in m/s and gusts, named outright) and turn the noise and
  the errors up as training progresses. The `wind` section of the file is for scripted
  flights; the policy trains with the curriculum's wind.
- `randomize` and `episode`: the hidden errors of every flight, and a new wind speed (up to the
  stage's value) and direction each flight, so the policy cannot learn one side.
- `ppo`: how many flights run at once, how many worker processes, the network size and the
  usual PPO settings.

### What the trained policy does so far

The current run (`configs/training/default.yaml`, four million steps, policy steering the
landing burn, the trigger table lighting the motor) against the PID on the same fifty seeds,
sensor noise and hidden errors on. The last column is how many degrees the two gimbal commands
move per control step during the burn:

| conditions                        | controller | landed   | speed down | speed sideways | tilt    | distance from pad | gimbal move |
|-----------------------------------|------------|----------|------------|----------------|---------|-------------------|-------------|
| calm air                          | policy     | 47 of 50 | 1.53 m/s   | 0.30 m/s       | 0.2 deg | 1.4 m             | 0.15 deg    |
| calm air                          | PID        | 47 of 50 | 1.53 m/s   | 0.23 m/s       | 0.4 deg | 1.3 m             | 0.07 deg    |
| 4 m/s crosswind, 1.5 m/s gusts    | policy     | 49 of 50 | 1.39 m/s   | 0.48 m/s       | 1.8 deg | 19.5 m            | 0.17 deg    |
| 4 m/s crosswind, 1.5 m/s gusts    | PID        | 47 of 50 | 1.36 m/s   | 0.60 m/s       | 1.3 deg | 18.6 m            | 0.34 deg    |

Equal in calm air; two landings better in wind, with less sideways speed and half the gimbal
movement. Three lessons from getting there:

- A policy trained without wind (the first attempt, before the curriculum named the training
  wind) landed 3 of 50 windy flights: it had never seen a gust.
- The next one landed 48 of 50 windy flights but flipped the gimbal between its limits every
  step, a habit a servo would not survive. Two causes: PPO's exploration noise was still 0.6
  of the action range at the end of training, which pushes the learned mean to the extremes,
  and the movement penalty was too small. Making the penalty forty times larger over-corrected
  and that policy never landed at all. The fix that worked is `log_std_init: -1.5` (a tenth of
  the noise), a moderate movement penalty plus a small gimbal angle term, and the safety
  layer's `max_gimbal_rate_deg_per_s`, which bounds the command rate for everyone.
- Letting the policy also steer the boost (`policy_boost.yaml`) lands 46 of 50 calm and 42 of
  50 windy, 6 m closer to the pad in wind, but it drives the gimbal in a square wave during the
  boost and lets the rocket tumble in the coast, so it is not a design to fly.

### Evaluating against the PID and flying the trained policy

```
python scripts/evaluate.py runs/<run> --episodes 50
python scripts/evaluate.py runs/<run> --episodes 50 --sim configs/training/windy.yaml --write-logs runs/policy_flights
python scripts/fly_scripted.py --policy runs/<run> --sim configs/training/windy.yaml --out runs/policy.csv --plot
python scripts/animate_flight.py runs/policy.csv --out runs/policy.gif --follow
```

The first two fly the policy and the PID on the same seeds, with sensor noise, hidden errors
and the wind of the chosen YAML, and print the landing rate, the mean touchdown speed, sideways
speed, tilt, distance from the pad and how many degrees the gimbal commands move per control
step during the burn (the PID moves them about 0.4 degrees per step; a policy that moves them
several degrees is steering bang-bang) for both, then one line per policy flight that failed.
The third flies one flight with the policy in the loop and writes the usual flight log, so the
plot, the GIF and the browser viewer work on it unchanged.

