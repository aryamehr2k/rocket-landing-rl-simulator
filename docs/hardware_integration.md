# Running a trained model on the vehicle

## Where a trained model ends up

Training writes `runs/<run>/` (large, not in git) and publishes the result to `models/<run>/`:

| File | Format | Use |
|---|---|---|
| `policy.npz` | numpy float32 arrays: weights, biases, input mean and std, input names | the simulator and the dashboard |
| `policy.onnx` | ONNX opset 13, input `observation` [1, N] float32, output `action` [1, 2] | any ONNX runtime (Raspberry Pi, a laptop, a ground station) |
| `policy_weights.h` | C99 float32 arrays | microcontroller build with `firmware/policy/policy.c` |
| `policy_config.h` | which inputs, in which order, with which scales; output scaling | `firmware/hop/hop_control.c` |
| `hop_params.h` | mission plan, PID gains, safety limits, launch time as C initialisers | `firmware/hop/hop_control.c` |
| `model.json` | every input and output described in words, network size, check results, training summary, evaluation against the PID | people |
| `evaluation/` | comparison table and plots against the PID | people |
| `configs/` | the exact training, vehicle, mission, motor and sensor files used | reproducing the run |

Publish a run by hand (also works while it is still training, from its last checkpoint):

```
python scripts/export_policy.py runs/<run> --name hop_v1 --evaluate
```

Both the C and the ONNX versions are checked against Python on 1000 random inputs, and the C
controller is checked against the Python flight computer on a whole simulated flight. The
results are in `model.json` under `checks` and `firmware.replay_check`.

## C code for the flight computer

`firmware/` is plain C99 with no dynamic memory and no board-specific code:

| File | Contents |
|---|---|
| `policy/policy.c`, `policy.h` | the network forward pass |
| `hop/hop_mission.c`, `.h` | the mission plan (reference height and vertical speed) |
| `hop/hop_control.c`, `.h` | one control step: network or PID, mission plan, safety limits, abort |
| `hop/hop_fields.h` | the names of the inputs the network can use |
| `tests/hop_replay.c`, `tests/policy_check.c` | desktop checks |

To build it for a board, copy the model's headers in, then compile the four `.c` files with your
board's project:

```
python scripts/export_policy.py runs/<run> --name hop_v1 --install   # copies the headers into firmware/
cd firmware && make                                                   # desktop build of the checks
```

### What the board calls

```c
#include "hop_control.h"
#include "hop_params.h"

static hop_controller_t controller;
static const hop_mission_t mission = HOP_MISSION_INIT;
static const hop_vehicle_t vehicle = HOP_VEHICLE_INIT;

void setup(void) { hop_controller_init(&controller, &mission, HOP_LAUNCH_TIME); }

/* every 1 / control_rate_hz seconds (50 Hz), with your estimator's output */
void control_tick(float t, const hop_estimate_t *estimate) {
    hop_command_t c = hop_control_step(&controller, &vehicle, estimate, t);
    /* send to the servos, ESC and roll control now (the simulator assumes they act one tick later):
       c.gimbal_pitch, c.gimbal_yaw (rad, servo commands),
       c.throttle (0..1), c.roll_torque (N m), motor only when c.armed */
}
```

`hop_estimate_t` is what your estimator must provide, all in SI units:

| Field | Meaning |
|---|---|
| `height` | landing feet above the pad, m |
| `vertical_speed` | m/s, positive up |
| `position[2]`, `velocity[2]` | x, y from the pad and their rates; x, y horizontal, z up, right handed |
| `tilt[2]`, `tilt_rate[2]` | lean of the nose toward +x and toward +y, and their rates, rad and rad/s |
| `total_tilt` | angle between the body axis and vertical, rad |
| `roll`, `roll_rate` | rotation about the body axis, right handed, rad and rad/s |

Sign rules are in [conventions.md](conventions.md): a positive pitch gimbal command leans the nose
toward +x, a positive yaw gimbal command toward +y. Check both on the bench before flying: command
a small positive pitch angle and confirm the nozzle moves the way the simulator expects.

### Order inside one control step

The C code follows the Python flight computer exactly:

1. At the launch time the mission plan starts. End your estimator's pad calibration at the same moment.
2. The network runs for both planes on this step's estimate and the previous step's reference.
3. The mission reference advances one step.
4. The PID computes its commands; the network's commands replace them (direct mode) or are added
   (residual mode). An input or output that is not a finite number hands that plane to the PID.
5. Abort if the tilt exceeds `abort_tilt` or the vehicle leaves the geofence: everything goes to zero.
6. Commands are clipped and rate limited against the previous command.

## Before the first real flight

1. Fill in the vehicle data ([vehicle_data_request.md](vehicle_data_request.md)) and retrain. A
   model trained on the placeholder numbers in this repository is not safe to fly.
2. Check that the PID alone flies the mission in the simulator with the measured numbers
   (`python scripts/fly_mission.py --flights 20 --wind-mps 4 --hidden-errors`).
3. Bench test with the motor off: feed recorded estimates through `firmware/tests/hop_replay.c`
   and compare with the simulator (`make check`).
4. First flights tethered and low (set `ascent.target_altitude_m` to 2 to 3 m in the mission file),
   PID first, then the model in residual mode, then in direct mode.
5. Keep a hardware kill switch that cuts the motor independently of the flight computer.
6. Log the board's estimates and commands and compare them with a simulated flight
   (`scripts/plot_flight.py` reads the same CSV columns).
