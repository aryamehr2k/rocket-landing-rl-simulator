# The physics that decides the landing

This page explains every formula the simulator uses to land the example rocket, with the
example's own numbers put in, for a reader who has done first-year physics and nothing else. The
rocket is `configs/rockets/example_tvc.yaml` with the motors in `configs/motors/` and the launch
site in `configs/training/default.yaml`. Every number comes from those files or from running the
project's code on them, and none of it means anything for your rocket until you replace the
numbers with measurements.

## How to read this page

Every formula gets the same block:
- a heading that says the question it answers;
- the formula in a code box, written the way you would type it into a calculator;
- **What the symbols mean**: one line per symbol with its unit and where the number comes from (a
  YAML key, a measurement or an earlier formula);
- **Worked example**: the example rocket's numbers step by step with the result and its unit, and
  a note wherever a number differs slightly from the README;
- **What it tells you**: what changes if you change a number and what that means for the rocket
  you build.

Words used on this page:

- A **station** is a distance from the nose tip toward the tail: the `_from_nose_mm` keys, in
  millimetres in the YAML and in metres here.
- The **centre of gravity** `x_cg` is where the rocket balances. The **centre of pressure** `x_cp`
  is where the sideways air force acts.
- **Drag** is the air force against the motion. An **impulse** is a force times the time it acts,
  in N s; a motor's total impulse is the area under its thrust curve.
- A **moment** is a force times a lever arm, in N m; it turns the rocket. The **pitch inertia** `I`
  (kg m2) says how hard the rocket is to turn, the way mass says how hard it is to push.
- **Burnout** is the moment the ascent motor stops. The **coast** is the unpowered climb from
  burnout to the **apogee**, the highest point, flown nose-first. The **descent** is the fall from
  apogee, tail-first, with the brake open.
- The **brake** (`drag_device` in the YAML) is four flat plates, the **petals**, hinged at the
  nose, that swing open during the descent to add drag. The **gimbal** is the pivot that lets the
  motor tilt by an angle `delta` to steer.
- The landing motor's burn has two parts. The **hard part** is the strong first 1.41 s at about
  31 N that slows the rocket. The **tail** is the weak rest, 12.6 N until 8.0 s, a little less
  than the weight, that lets the rocket sink the last metres. The **stop point** is the height
  where the hard part has brought the descent speed down to 0.5 m/s; the example plans it 1.5 m up.

| symbol | meaning | unit | comes from |
|--------|---------|------|------------|
| `m` | mass on board right now | kg | formula 2 |
| `g` | gravity | 9.807 m/s2 | `environment.gravity_mps2`, training YAML |
| `rho` | air density | kg/m3 | formula 1 |
| `v` | speed through the air | m/s | the rocket's motion at that moment: the true value in the simulator, the flight computer's estimate on board |
| `S` | body reference area, `pi * d^2 / 4` | m2 | `reference_diameter_mm` |
| `Cd` | body drag coefficient | none | `aero.drag_coefficient` |
| `CNa` | normal force slope | per radian | `aero.normal_force_slope_per_rad` |
| `(Cd A)` | drag area of the open brake | m2 | `drag_device.drag_area_cm2` |
| `T` | thrust | N | the motor file's curve |
| `J` | a motor's total impulse, the area under its thrust curve | N s | the motor file |
| `I` | pitch inertia about the centre of gravity | kg m2 | formula 2 |
| `alpha` | angle of attack: between the body axis and the air flow | rad | the rocket's motion at that moment: the true value in the simulator, the flight computer's estimate on board |

## Before the flight

### 1. How thick the air is where you fly
Every drag force on this page is proportional to the air density, so the first number is how
dense the air is at the launch site.
```
rho = rho_0 * exp(-(z + h_site) / H)
```
**What the symbols mean:**
- `rho_0` = 1.225 kg/m3, the sea level density; `z` the height above the pad in m; `h_site` the
  site's elevation above sea level in m. `rho_0`, `h_site` and `H` are the `environment` section
  of the training YAML (`configs/training/default.yaml`), not of the rocket YAML.
- `H` = 8500 m (`environment.scale_height_m`): climb this far and the density falls to 37 % of
  what it was.

**Worked example:** site at 300 m, on the pad: `rho = 1.225 * exp(-300 / 8500) = 1.225 * 0.965 =
1.183 kg/m3`. At the 105 m apogee it is 1.168 kg/m3, 1.2 % less.

**What it tells you:** 100 m of flight changes the density by about 1 %, so the flight computer
uses the ground value for the whole flight. A site at 1000 m has 11 % thinner air than sea level
and every drag number on this page, the terminal speed included, shifts with it: put your launch
site's elevation in `environment.site_elevation_m` of the training YAML you fly with.

### 2. How heavy the rocket is and where it balances
The motors sit at the tail and lose mass as they burn, so the mass, the balance point and how hard
the rocket is to turn all change during the flight. The flight computer's tables depend on them.
```
m     = m_dry + sum(m_i)                      m_i = motor case + propellant still in motor i
x_cg  = (m_dry * x_dry + sum(m_i * x_i)) / m
I     = I_dry + m_dry * (x_cg - x_dry)^2 + sum(m_i * (x_cg - x_i)^2)
dm/dt = T / c,   c = J / m_prop           J = total impulse, m_prop = propellant mass
```
**What the symbols mean:**
- `m_dry` 1.200 kg (`airframe.dry_mass_g`, a scale), `x_dry` 0.450 m (`dry_cg_from_nose_mm`, a
  balance test), `I_dry` 0.085 kg m2 (`dry_pitch_inertia_kgm2`, OpenRocket or a swing test).
- `m_i`, `x_i`: each motor's mass and station (`position_from_nose_mm`). The motor file gives the
  total and propellant masses: ascent 125 g with 62 g of propellant, landing 169 g with 93 g.
- `c` in m/s, the exhaust speed: the motor's total impulse `J` (N s) divided by its propellant
  mass `m_prop` (kg). `dm/dt` is the propellant burned per second, in kg/s.

**Worked example:** on the pad `m = 1.200 + 0.125 + 0.169 = 1.494 kg` and
`x_cg = (1.200 * 0.450 + 0.125 * 0.810 + 0.169 * 0.690) / 1.494 = (0.540 + 0.101 + 0.117) / 1.494
= 0.507 m`. After the ascent burn its 62 g of propellant are gone and the ascent case weighs
`0.125 - 0.062 = 0.063 kg`: `m = 1.200 + 0.063 + 0.169 = 1.432 kg` and
`x_cg = (1.200 * 0.450 + 0.063 * 0.810 + 0.169 * 0.690) / 1.432 = (0.540 + 0.051 + 0.117) / 1.432
= 0.494 m`. The distances from this centre of gravity are `0.494 - 0.450 = 0.044 m` (dry rocket),
`0.810 - 0.494 = 0.316 m` (ascent case) and `0.690 - 0.494 = 0.196 m` (landing motor), so
`I = 0.085 + 1.200 * 0.044^2 + 0.063 * 0.316^2 + 0.169 * 0.196^2 = 0.085 + 0.0023 + 0.0063 +
0.0065 = 0.100 kg m2`. The landing motor's `J` is the area under the curve in
`example_g127_landing_brake.yaml`, 126.8 N s, so `c = 126.8 / 0.093 = 1363 m/s` and at 31 N it
burns `31 / 1363 = 0.023 kg/s`. By the end of the hard part 30 g are gone: `m = 1.402 kg` and
`x_cg = 0.490 m`. By touchdown, about 4.3 s after thrust on, 58 g are gone: `m = 1.374 kg` and
`x_cg = (0.540 + 0.051 + 0.111 * 0.690) / 1.374 = 0.486 m` (formula 12). With both motors empty
(landing case `0.169 - 0.093 = 0.076 kg`): `m = 1.339 kg` and `x_cg = 0.481 m`. The README says
"about 1.4 kg on the pad"; the file's masses add up to 1.494 kg.

**What it tells you:** the centre of gravity moves 13 mm forward during the ascent burn, which is
why formula 6 is judged at burnout, not on the pad. Weigh the rocket without its motors before
every flight and put the number in `airframe.dry_mass_g`; the motor masses come from the motor
files. The flight computer plans the burn with `m`, and formula 11 shows what 36 g cost.

### 3. The forces that act on the rocket
Five forces decide the flight: weight, thrust, the body's drag, the sideways air force on the body
and fins, and the brake's drag.
```
W   = m * g                                     down
T   = thrust from the curve                     along the gimbal direction, at the pivot
D   = 0.5 * rho * Cd * S * v^2                  against the motion, at the centre of gravity
N   = 0.5 * rho * S * CNa * v^2 * sin(alpha)    sideways, at the centre of pressure
D_b = 0.5 * rho * (Cd A) * v^2                  against the motion, at the brake's station
```
**What the symbols mean:**
- `S = pi * d^2 / 4` with `d` = 0.075 m (`reference_diameter_mm`): 0.00442 m2 = 44.2 cm2.
- `Cd` 0.60 and `CNa` 2.0 per radian, from OpenRocket or a wind tunnel (`aero` section).
- `(Cd A)` 566 cm2 = 0.0566 m2 (`drag_device.drag_area_cm2`): the petals' area times their drag
  coefficient, about 1.2 for a flat plate. `0.5 * rho * v^2` is the dynamic pressure, the
  pressure the moving air can exert. For `D_b` the simulator uses the airspeed at the brake's own
  station, the rocket's airspeed plus what its turning adds, so a swinging rocket feels the brake
  resist the swing; for a rocket falling straight this is `v`.

**Worked example:** at 20 m/s in the descent with `rho` = 1.183 kg/m3: `W = 1.432 * 9.807 =
14.0 N`. `D = 0.5 * 1.183 * 0.60 * 0.00442 * 400 = 0.63 N` (2.0 N at 36 m/s). `N` at `alpha` =
10 degrees: `0.5 * 1.183 * 0.00442 * 2.0 * 400 * 0.174 = 0.36 N`.
`D_b = 0.5 * 1.183 * 0.0566 * 400 = 13.4 N`: `13.4 / 4 = 3.35 N` on each of the four petals. A
petal is a square of 109 mm (formula 7 shows where that comes from), so the force acts about half
a petal, 0.055 m, from the hinge: `3.35 * 0.055 = 0.18 N m` at each hinge, which the brake servo
must hold. The landing motor's `T` is 31 N.

**What it tells you:** the body's own drag is a few percent of the weight, so without the brake
the rocket keeps speeding up all the way down. The brake's drag plus the body's,
`13.4 + 0.6 = 14.0 N`, equals the weight at 20 m/s: that is the idea of formula 7. Every drag
force grows with the square of the speed: twice the speed, four times the force. The body's `Cd`
is the one number on this page you need not measure well; the brake's `(Cd A)` is the one to
measure (formula 7).

## Ascent

### 4. Will it leave the pad, and how high does it go
The ascent motor has to lift the rocket and take it high enough for the brake to do its work.
```
T / W       must be above 1 to lift off; 3 to 4 is comfortable
v_burnout  ~ J_ascent / m - g * t_burn          ignoring drag and the mass burned off
h_apogee   ~ h_burnout + v_burnout^2 / (2 * g)  minus what drag takes during the coast
```
**What the symbols mean:**
- `J_ascent` 84.4 N s, the ascent motor's total impulse (the area under the curve in
  `example_g40.eng`), and `t_burn` 2.15 s, the curve's length. Peak thrust 55 N, mean thrust
  `84.4 / 2.15 = 39.3 N`.
- `m` 1.494 kg on the pad, `W` = 14.65 N (formulas 2 and 3).
- `~` means roughly: both lines ignore drag and the mass burned off during the burn.
- `h_burnout` the height at the end of the ascent burn, 45 m in the project's one dimensional
  flight. One dimensional means straight up and straight down: no tilt, no wind, no sideways
  motion. `rocketsim/landing_design.py` flies the rocket that way.

**Worked example:** `T / W = 55 / 14.65 = 3.75` at the peak, 2.7 on average.
`v_burnout = 84.4 / 1.494 - 9.807 * 2.15 = 56.5 - 21.1 = 35.4 m/s`. The project's one dimensional
flight with drag (`rocketsim.landing_design.ascent_apogee`) gives 35.3 m/s at 45 m, then
`45 + 35.3^2 / (2 * 9.807) = 45 + 64 = 109 m` minus 4 m of drag in the coast: 105 m (the README's
simulator flight of the first design reached 106 m).

**What it tells you:** with the brake the apogee hardly matters: it only needs to give the rocket
about 45 to 50 m of fall above the 18 m command height to be within 5 % of the terminal speed, so
about 65 m or more (formula 7 and the profile), and stay below the 120 m ignition safety limit; a
lower apogee only means a slightly slower arrival that the table still handles. The training
YAML's 5 % ascent motor error puts the apogee anywhere from 93 m to 117 m, and the landing does
not care. In the old design without the brake, 3 m less apogee moved the stop point (formula 8)
about 1 m.

### 5. Can the gimbal hold the rocket upright
The only way to turn the rocket is to tilt the motor. This is how fast that turns it.
```
M_gimbal = T * L * sin(delta),   L = x_pivot - x_cg
angular acceleration = M_gimbal / I
```
**What the symbols mean:**
- `x_pivot` 0.840 m (`gimbal.pivot_from_nose_mm`); `L` the lever arm from the pivot to the centre
  of gravity, 0.346 m during the landing burn (formula 2).
- `delta` the gimbal angle, at most 7 degrees (`gimbal.max_angle_deg`); `I` 0.100 kg m2.

**Worked example:** `M_gimbal = 31 * 0.346 * sin(7 deg) = 31 * 0.346 * 0.122 = 1.31 N m` and
`1.31 / 0.100 = 13.1 rad/s2`, about 750 degrees per second squared. Held for 0.1 s from rest this
turns the rocket `0.5 * 13.1 * 0.1^2 = 0.065 rad = 3.7 degrees` and leaves it turning at
75 degrees/s; turning through 7 degrees from rest takes 0.14 s. The servo needs `7 / 300 =
0.023 s` to reach 7 degrees at 300 degrees/s plus its 20 ms delay, so the servo is not the slow
part. During the ascent the arm is `0.840 - 0.507 = 0.333 m` and the 55 N peak gives
`55 * 0.333 * 0.122 = 2.2 N m`.

**What it tells you:** a longer arm `L` or more thrust corrects a tilt faster, so mount the gimbal
pivot as far behind the centre of gravity as the airframe allows. During the coast and the descent
there is no thrust, so the gimbal cannot turn the rocket at all (engineers say it has no control
authority). Whatever tilt the rocket picks up then must be small enough for the burn to remove,
and the flight computer's safety rules (`safety.max_landing_ignition_tilt_deg`) refuse to light
the motor when the tilt is beyond 35 degrees.

## Coast and descent

### 6. Why the fins must be tiny: falling tail-first
Fins keep a normal rocket pointing into the air flow. The coast is nose-first, so there fins
steady the rocket; the descent is tail-first, and the same fins make it flip over. This is the
moment they produce and how fast a tilt grows.
```
M_aero = (x_cp - x_cg) * 0.5 * rho * S * CNa * v^2 * sin(alpha)
sigma  = sqrt( (x_cp - x_cg) * 0.5 * rho * S * CNa * v^2 / I )      tilt grows as exp(sigma * t)
```
**What the symbols mean:**
- `x_cp` 0.494 m (`aero.cp_from_nose_mm`, from OpenRocket); `x_cp - x_cg` is the static margin,
  positive when the centre of pressure is behind the centre of gravity (stable nose-first).
- `sigma` in 1/s: the growth rate of a tilt while falling tail-first; `t` the seconds since the
  tilt started.

**Worked example:** the example puts `x_cp` at 0.494 m, exactly at `x_cg` after the ascent burn,
so the margin is 0 and `sigma` = 0: neutral, the rocket keeps whatever turning rate it had at
burnout. If bigger fins put the centre of pressure 14 mm behind: at 20 m/s
`0.5 * rho * v^2 = 236.5 Pa` and `sigma = sqrt(0.014 * 236.5 * 0.00442 * 2.0 / 0.100) =
sqrt(0.293) = 0.54 per s`. A tilt doubles every `ln(2) / 0.54 = 1.3 s`; over the roughly 5 s from
the brake opening (about 0.8 s after apogee, once the fall passes 8 m/s) to the motor lighting at
18 m, it grows `exp(0.54 * 5) = 15` times, 1 degree into 15 degrees. The other way, 14 mm ahead,
makes the coast unstable instead: the README's flight turned 1 degree at burnout into 30 degrees
at apogee.

**What it tells you:** aim for the centre of pressure at the burnout centre of gravity, to a few
millimetres: smaller fins move it toward the nose, bigger fins toward the tail, and OpenRocket
gives the number for `cp_from_nose_mm`. Judge it at burnout (formula 2), not on the pad. The
brake at the nose then supplies the steadying moment during the descent, like the feathers of a
shuttlecock, which is why it must never sit behind the centre of gravity.

### 7. How fast the rocket falls with the brake open
With the brake open the rocket stops speeding up where the drag equals the weight. That speed is
the terminal speed, and it sets the whole landing burn.
```
v_t          = sqrt( 2 * m * g / (rho * (Cd A)_total) )
(Cd A)_total = 2 * m * g / (rho * v_t^2)             = Cd * S + (Cd A)_brake
```
**What the symbols mean:**
- `m` 1.432 kg during the descent (dry mass, empty ascent case, full landing motor), `rho`
  1.183 kg/m3, `Cd * S` = 26.5 cm2 (formula 3).
- `(Cd A)_brake` is the brake's `(Cd A)` of the symbol table, 566 cm2 = 0.0566 m2;
  `(Cd A)_total` adds the body's `Cd * S`.
- `v_t` the wanted terminal speed, the `--terminal-speed` argument of the design tool.

**Worked example:** for 20 m/s: `2 * 1.432 * 9.807 = 28.09 N` and `rho * v_t^2 = 1.183 * 400 =
473 kg/(m s2)`, so `(Cd A)_total = 28.09 / 473 = 0.0594 m2 = 594 cm2`; minus the body's 26.5 cm2
the brake needs `594 - 26.5 = 567 cm2` of drag area, or `567 / 1.2 = 473 cm2` of flat plate at
Cd 1.2; `473 / 4 = 118 cm2` per petal and `sqrt(118 cm2) = 10.9 cm`: four squares of 109 mm. The
file rounds the brake to 566 cm2, so the total is `566 + 26.5 = 592.5 cm2 = 0.0592 m2` and
`v_t = sqrt(28.09 / (1.183 * 0.0592)) = sqrt(401) = 20.0 m/s`. The one dimensional descent
(`rocketsim.landing_design.descent_profile`) falls from the 105 m apogee and passes 100 m at
9.8 m/s, 80 m at 17.0 m/s, 60 m at 18.9 m/s, 40 m at 19.6 m/s and 20 m at 19.9 m/s: 95 % of `v_t`
by 58 m. An area 5 % off moves `v_t` to 19.6 or 20.5 m/s, 15 % off to 18.7 or 21.6 m/s (the
README's 18.6 and 21.7 are simulator flights).

**What it tells you:** the speed goes with one over the square root of the area, so 15 m/s would
need 858 cm2 of plate (855 in the README) and 25 m/s only 295 cm2. The burn can stop at most
21.5 m/s (formula 8), so the area must be known to about 5 %, from a drop test or from logging the
terminal speed. Without any brake the rocket reaches the ground at 43 m/s (one dimensional) and
passes 38 m at 35 m/s, where the README's first design lit its motor at 36 m/s.

## The landing burn

### 8. How hard the burn slows the rocket, and how far it needs
A solid motor gives one fixed push. This is how much it slows the rocket and how many metres of
fall the slowing takes.
```
a = (T + D + D_b) / m - g        deceleration during the hard part of the burn
d = v^2 / (2 * a_mean)           stopping distance from speed v
```
**What the symbols mean:**
- `T` 31 N for the hard part, 29 N at its end at 1.41 s (`example_g127_landing_brake.yaml`).
- `D` 0.63 N and `D_b` 13.4 N at 20 m/s, both falling with the speed squared (formula 3); `m`
  1.432 kg at ignition, 1.401 kg at the end of the hard part (formula 2).
- `a_mean` the average deceleration over the stop; `d` the fall from thrust start to the stop
  point, where the descent is down to `target_speed_mps`, 0.5 m/s.

**Worked example:** at thrust start `a = (31 + 0.63 + 13.4) / 1.432 - 9.807 = 31.4 - 9.8 =
21.6 m/s2`, the motor file's "about 22". At the end of the hard part the drag is gone:
`29 / 1.401 - 9.807 = 10.9 m/s2` (the file's 12 m/s2 uses the 31 N peak). Integrating the thrust
curve with the flight computer's own code (`LandingTrigger.stopping_distance(20.0)`) gives
`d = 13.1 m` in 1.40 s, so `a_mean = 20^2 / (2 * 13.1) = 15.3 m/s2`, between the two; from
19.9 m/s the design tool prints 13.0 m. From 22 m/s the burn never gets the speed down to 0.5 m/s
before it runs out, so 21.5 m/s is the edge of the burn's reach.

**What it tells you:** the stopping distance is what the trigger table holds (formula 9). The
impulse is fixed by the curve, so the hard part must end exactly when the speed is gone: half a
second too long and the rocket stops in the air and climbs, too short and it lands with speed.
Size the hard part for the smallest brake area you might really have, because that is the fastest
arrival: 5 % less area is 20.5 m/s (formula 7). The design tool,
`python scripts/design_landing_burn.py --rocket my_rocket.yaml --terminal-speed 20`, sizes the
hard part for the terminal speed plus a small margin (20.3 m/s for the example) and prints the
motor curve to order. A burn that cannot stop the arrival speed is not rescued by anything else on
this page.

### 9. When the flight computer sends the ignition command
The flight computer cannot throttle a solid motor, so its one decision is when to send the igniter
command. At start-up it makes a table: for every descent speed, the feet height at which thrust
must already be on, which is formula 8's stopping distance plus a margin. The igniter and the
control loop take time, so in flight it predicts where the rocket will be when thrust actually
starts and compares that with the table.
```
h_required(v) = d(v) + h_target                            the table, one entry per speed
delay   = igniter mean delay + (action_delay_steps + 0.5) / control_rate_hz
v_then  = v + (g - k * v^2 / m) * delay                    k = 0.5 * rho * (Cd A)_total
h_then  = h - 0.5 * (v + v_then) * delay
light when   h_then <= h_required(v_then)
```
**What the symbols mean:**
- `h_target` 1.5 m (`landing_trigger.target_height_m`), the feet height where the stop point is
  planned; `v`, `h` the descent speed and feet height as the flight computer estimates them from
  its sensors (not the true values; formula 11 says what the difference costs); `k` 0.0350 kg/m,
  the drag constant at the current brake opening: the drag force is `k * v^2`, and with the brake
  open `k = 0.5 * rho * (Cd A)_total = 0.5 * 1.183 * 0.0592 = 0.0350 kg/m` (formula 7).
- Igniter mean delay 0.15 s (`ignition_delay_s.mean`; measure it with your own firing circuit).
  The flight computer decides `control_rate_hz` = 50 times a second, so one decision step is
  `1 / 50 = 0.02 s`; a command takes effect `action_delay_steps` = 1 step later, and on average the
  rocket crosses the trigger height halfway between two decisions, the extra 0.5 step. Both numbers
  are in the `control` section of the rocket YAML.

**Worked example:** `delay = 0.15 + 1.5 / 50 = 0.18 s`. At 20 m/s `k * v^2 / m = 0.0350 * 400 /
1.432 = 9.79 m/s2`, almost all of gravity, so `v_then = 20 + (9.81 - 9.79) * 0.18 = 20.0 m/s` and
`h_then = h - 0.5 * 40 * 0.18 = h - 3.6 m`. The table gives `h_required(20) = 13.1 + 1.5 =
14.6 m`, so the command goes out when `h - 3.6 = 14.6`, at `h = 18.2 m`, and thrust starts at
14.6 m. Other entries of the table, required height for the speed: 15 m/s 9.9 m, 18 m/s 12.7 m,
21.5 m/s 16.0 m (they come from integrating the curve, not from `a_mean`, which changes with the
speed). Faster speeds keep the edge's 16.0 m, so the burn still ends as low as it can instead of
starting tens of metres early. Without the drag term the predicted speed would be
`20 + 9.81 * 0.18 = 21.8 m/s`, beyond the edge, so the table would give 16.0 m and the command
would go out at `16.0 + 0.5 * (20 + 21.8) * 0.18 = 16.0 + 3.8 = 19.8 m`, 1.6 m early. The
README's 17 m and 14 m are simulator flights, where the estimate is noisy and the rocket arrives
at 19.9 m/s.

**What it tells you:** every 10 ms of igniter delay the computer does not know about moves the
stop point 0.2 m (formula 11), so fire about 20 igniters with your own circuit, time them, and put
the mean and half the spread in `motors.landing.ignition_delay_s`. `thrust_margin` builds the
table for a motor a little stronger than the file's curve, so every real motor of the batch is as
strong as the table assumes or weaker. Weaker is the safe side; formula 11 says why.

### 10. How the tail sets the rocket down
After the hard part the motor keeps a weak tail, a little less than the weight, so the rocket sinks
the last metres slowly.
```
a_sink  = g * (1 - T_tail / W)
t_sink  = sqrt(2 * h_stop / a_sink),    v_touch = sqrt(2 * a_sink * h_stop)
```
**What the symbols mean:**
- `T_tail` 12.6 N from 1.56 s to 8.0 s in the motor file; `W` 13.72 N at tail start (1.399 kg,
  formula 2), so `T_tail / W = 12.6 / 13.72 = 0.918`, which the design tool's `--tail-ratio`
  rounds to 0.92.
- `h_stop` the height the sink starts from: the stop point plus the small climb after it.
- `t_sink` the seconds the sink takes and `v_touch` the touchdown speed, both for a sink that
  starts from rest (the 0.5 m/s at the stop point is ignored).

**Worked example:** `a_sink = 9.807 * (1 - 0.918) = 0.80 m/s2` at tail start. The tail burns
`12.6 / 1363 = 0.009 kg/s`, so the rocket keeps getting lighter and the sink gentler: 0.78 m/s2 at
the top of the sink 0.7 s later, `9.807 - 12.56 / 1.374 = 0.67 m/s2` at touchdown. The one
dimensional flight (the design tool with `--motor`) reaches the stop point at 1.67 m and climbs
0.14 m on the rest of the hard ramp (the 0.15 s in the file over which the thrust falls from 29 N
to 12.6 N), so `h_stop = 1.67 + 0.14 = 1.81 m`: `t_sink = sqrt(2 * 1.81 / 0.78) = 2.2 s` and
`v_touch = sqrt(2 * 0.78 * 1.81) = 1.68 m/s`. The tool's flight lands at 1.60 m/s because the sink
keeps getting gentler; `1.60^2 / (2 * 1.81) = 0.70 m/s2` is the effective value it quotes, and its
"2.9 s after the stop" includes about 0.7 s of climb and turn-around before the 2.2 s sink. The
tail lasts `8.0 - 1.56 = 6.4 s`. A stop 1.03 m too high sinks from `1.81 + 1.03 = 2.84 m`: at the
effective 0.70 m/s2 it needs `sqrt(2 * 2.84 / 0.70) = 2.85 s` and lands at
`sqrt(2 * 0.70 * 2.84) = 2.0 m/s`, the leg limit. The README's 1.6 m, 1.1 m and 6.5 s come from
the tool's sizing run (`--terminal-speed 20`), whose motor differs from the file's by rounding
(tail to 8.06 s).

**What it tells you:** a tail ratio near 1 sinks so slowly that the motor runs out in the air; a
ratio well below 0.9 lands hard. The tail must last longer than the slowest sink it may have to do,
which is why the example's tail runs 6.4 s and the old 3.7 s tail failed.

### 11. What a small error costs
The stop point is planned 1.5 m up and comes out at 1.67 m in the one dimensional flight. Three
kinds of error move it, and this is by how much.
```
timing error dt (igniter, control step):    stop moves by  v * dt
mass error s (fraction):                    stop moves by  d * s * (a + g) / a
thrust error s (fraction):                  stop moves by  d * s * T / (m * a)
speed estimate error sigma_v:               stop moves by  sigma_v * v / a
```
**What the symbols mean:**
- `v` 20 m/s at ignition; `d` 13.1 m and `a` = `a_mean` 15.3 m/s2 (formula 8); `T` 31 N, `m`
  1.432 kg; `dt` the igniter spread, 0.03 s (`ignition_delay_s.spread`); `sigma_v` the
  estimator's speed noise, about 0.3 m/s with the sensor YAML.
- The mass and thrust rows differ because a mass error also changes how much the brake's drag
  helps, so `(a + g) / a` = 1.64 for mass against `T / (m * a)` = 1.41 for thrust.

**Worked example:** igniter 30 ms off: `20 * 0.03 = 0.60 m`. One 20 ms control step: 0.40 m. If
the computer did not know about the 0.18 s delay at all: 3.6 m. Mass 1 % off:
`13.1 * 0.01 * (15.3 + 9.8) / 15.3 = 0.21 m`, so a rocket 36 g (2.5 %) lighter than the file says
stops 0.54 m high. Thrust 1 % off: `13.1 * 0.01 * 31 / (1.432 * 15.3) = 0.18 m`. Speed estimate
0.3 m/s off: `0.3 * 20 / 15.3 = 0.39 m` (the design tool prints the same four numbers). In the
project's first design, which fell without a brake and lit the motor at 36 m/s, the same 30 ms
cost `36 * 0.03 = 1.1 m`. The tool's one dimensional flights allow a stop 1.8 m too low or 1.0 m
too high before the touchdown exceeds 2 m/s (the README's 1.1 m is the sizing run's). A stop
1.8 m too low means the hard part would only have reached 0.5 m/s 0.2 m under the ground: the
rocket meets the ground while the hard part is still slowing it, at 2 m/s. The 30 ms error lands
at 1.3 m/s when the stop is low and 1.8 m/s when it is high.

**What it tells you:** a stop that comes out too low is cheap: the tail has less height to sink.
One that comes out too high costs a climb and seconds of tail (formula 10). So load the flight
computer with the strongest motor of your batch (`thrust_margin` 1.02 for a batch within 2 %), and
remember that a light rocket behaves like a strong motor: weigh it.

## Touchdown

### 12. How much tilt the legs allow
The rocket falls over if its centre of gravity passes outside the polygon of its feet.
```
tip-over angle = atan( r_in / h_cg ),    r_in = (span / 2) * cos(pi / n_legs)
```
**What the symbols mean:**
- `span` 0.300 m (`legs.span_mm`, the circle through the feet), `n_legs` 4 (`legs.count`);
  `r_in` is the radius of the largest circle that fits inside the shape the feet mark out.
- `h_cg` the centre of gravity's height above the feet: the feet station `length_mm + height_mm`
  = 1.020 m minus `x_cg` (formula 2).

**Worked example:** `r_in = 0.150 * cos(180 deg / 4) = 0.150 * cos(45 deg) = 0.106 m`; at
touchdown, with about 3.7 s of tail still burning, `x_cg = 0.486 m` (formula 2), so
`h_cg = 1.020 - 0.486 = 0.534 m` and the angle is `atan(0.106 / 0.534) = 11.2 degrees` (11.1 with
the motor fully burned out, 11.4 with the descent centre of gravity). The limit used is the smaller
of this and `max_touchdown_tilt_deg`, so 10 degrees. The other leg limits are 2 m/s down and
1 m/s sideways.

**What it tells you:** with three legs the angle drops to 8.0 degrees; a 400 mm span raises it to
14.8 degrees. So use four legs on the widest span the airframe allows and keep them short, and
keep `max_touchdown_tilt_deg` below the tip-over angle. The sideways speed limit is the one the
wind attacks: two of the three windy failures in the README were sideways speed (1.05 m/s) and
tilt (11.3 degrees) alone; the third exceeded both the 2 m/s descent limit (2.08 m/s) and the
sideways limit (1.10 m/s), after the rocket swung in the wind.

## The five numbers that matter most

1. **The landing motor's impulse, hard part and tail, to 1 or 2 %** of the batch you fly: 1 %
   moves the stop point 0.18 m and 3 % too strong lands 2 flights in 20 (formulas 8 and 11).
2. **The igniter delay and its spread**, measured with your own firing circuit: 30 ms is 0.6 m of
   stop point, and a delay the computer does not know about costs 0.2 m per 10 ms (formula 9).
3. **The mass before every flight, to 10 g**: 36 g lighter than the file behaves like a 2.5 %
   stronger motor and lands 1 in 20 (formulas 2 and 11).
4. **The brake's drag area, to 5 %**: it sets the arrival speed, the arrival speed sets the burn,
   and the burn cannot stop more than 21.5 m/s (formulas 7 and 8).
5. **The centre of pressure against the burnout centre of gravity, to a few millimetres**: a
   14 mm margin turns 1 degree of tilt into 15 during the descent or 30 during the coast, and the
   burn then starts far from vertical (formula 6).
