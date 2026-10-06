# Rocket landing simulator

Simulation and reinforcement learning for a vertical take-off and landing test vehicle.
A six-degree-of-freedom simulator with sensor, actuator and flight computer models is used to
train a neural network controller with PPO, compare it with a PID baseline, and export it as C
code and ONNX for the vehicle's flight computer.

The current test vehicle is electric: a ducted fan with thrust vectoring and electric roll
control. Its mission is to climb to 50 m, hover for 10 s and land within 10 m of the pad. The
same code also models a solid-motor rocket with a drag brake and a landing motor
([docs/solid_rocket.md](docs/solid_rocket.md)).

**Every vehicle number in `configs/` is a placeholder.** Replace them with measured values before
using any result; [docs/vehicle_data_request.md](docs/vehicle_data_request.md) lists what is needed.

## Quick start

Python 3.10 to 3.12.

```
python3 -m venv .venv
. .venv/bin/activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu   # CPU build, much smaller
pip install -e ".[train,export,dev]"
pytest
```

Then:

```
python scripts/dashboard.py                      # control pad in the browser, http://127.0.0.1:8050
python scripts/fly_mission.py --plot --animate   # one mission flown by the PID, with plots and a 3D GIF
python scripts/train.py --training configs/training/hop.yaml   # train the network
```

On a remote machine, forward port 8050 (VS Code does this from the Ports tab) and open the link
on your own computer.

## Where to change things

All settings live in YAML files; no code changes are needed to change the vehicle, the mission or
the training.

| To change | Edit |
|---|---|
| Vehicle: mass, balance point, size, legs, gimbal limits, servos, roll control, PID gains, safety limits | `configs/vehicles/electric_hopper.yaml` |
| Motor: maximum thrust, spin-up time, battery time | `configs/motors/example_edf_90mm.yaml` |
| Sensors: IMU, barometer, GPS rates and noise | `configs/sensors/example_imu_gps.yaml` |
| Launch and landing: target height, climb and descent speeds, hover time, landing radius, pass criteria | `configs/missions/hop_50m.yaml` |
| Who flies (PID or the network, direct or residual) | `controllers` in the vehicle file, overridden by the training file |
| Training: what the network sees, rewards, randomisation, curriculum, network size | `configs/training/hop.yaml` |
| How many flights are simulated at once | `ppo.n_sims` (flights at once) and `ppo.workers` (processes) in the training file |
| Simulation rate, wind for test flights, launch site | `simulation`, `wind`, `environment` in the training file |
| Plots and animation after training | `evaluation.plots` and `evaluation.animation` in the training file |

Units are in the key names (`_mm`, `_g`, `_deg`, `_s`, `_mps`) and are converted on load. Unknown
keys and out-of-range values stop with a message that names the file and the key. Frames and sign
conventions are defined in [docs/conventions.md](docs/conventions.md).

## Dashboard

`python scripts/dashboard.py` starts a local web page with four tabs:

- **Fly**: launch a live flight with the PID or any trained model and watch it in 3D with live
  charts of height, speed, throttle, gimbal and tilt against the mission reference. Wind, gusts,
  thrust and mass errors, sensor noise and the mission targets are set before launch; wind can be
  changed and the vehicle pushed sideways during the flight. Charts can be switched off.
- **Many flights**: fly the same configuration on many seeds in parallel, optionally against the
  PID, and see the success rate and every path from above.
- **Training**: progress of every training run (success rate, landing rate, reward, miss distance)
  updating while it trains, the console log, and a form to start a new training.
- **Models**: every trained model with its inputs, outputs and evaluation; fly it with one click.

## Flying missions from the command line

```
python scripts/fly_mission.py                                   # PID, calm air, plots on
python scripts/fly_mission.py --model models/<name> --wind-mps 4 --gust-mps 1.5 --animate
python scripts/fly_mission.py --model models/<name> --flights 50 --compare --hidden-errors
python scripts/fly_mission.py --no-plot                         # numbers only
```

Each flight writes a CSV log with sensor readings, estimates, true state, commands and the
mission reference (`runs/flights/` by default). `--plot` (on by default) draws inputs, outputs
and the reference; `--animate` writes a 3D GIF; `--flights N` flies N seeds and prints how many
met every mission criterion. Existing logs: `scripts/plot_flight.py`, `scripts/animate_flight.py`,
`scripts/bundle_viewer.py` (single HTML file with a 3D viewer).

## Training

```
python scripts/train.py --training configs/training/hop.yaml            # network flies everything
python scripts/train.py --training configs/training/hop_residual.yaml   # network corrects the PID
```

Each run writes `runs/<date>_<name>/` with the training log (`progress.csv`), the network
(`policy.npz`, saved every `ppo.checkpoint_every` samples so it is usable before training ends),
the full PPO model and copies of every config used. When training finishes, the network is
published to `models/<run>/` and flown against the PID on the same seeds; the comparison is
stored with it. How the training works: [docs/architecture.md](docs/architecture.md).

## Trained models and the flight computer

`models/<name>/` holds everything needed to fly a model: `policy.npz` (simulator), `policy.onnx`
(any ONNX runtime), `policy_weights.h` with `policy_config.h` and `hop_params.h` (C), `model.json`
(every input and output described, check results, evaluation) and the configs it was trained
with. `firmware/` has the C99 controller that runs the network, the mission plan, the PID fallback
and the safety limits on the board; it is checked against the Python flight computer on whole
simulated flights. Wiring it into the flight software, the input format and a checklist for the
first flights: [docs/hardware_integration.md](docs/hardware_integration.md).

```
python scripts/export_policy.py runs/<run> --name hop_v1 --evaluate --install
```

## Repository layout

```
configs/      vehicles, motors, sensors, missions, rockets, training files
rocketsim/    simulator, flight computers, training environments, export   (rocketsim/hop: electric vehicle)
scripts/      command line tools
dashboard/    browser front end of scripts/dashboard.py
firmware/     C99 code for the flight computer and its desktop checks
models/       published trained models
docs/         architecture, conventions, physics, hardware integration, data requests
tests/        pytest suite (python -m pytest)
```

## Status

- Electric vehicle: simulator, mission plan, PID baseline, training, evaluation, model export and
  the C controller work. With GPS the PID completes the mission in 9 or 10 of 10 simulated flights
  in a 4 m/s gusty wind. All vehicle numbers are placeholders until measured.
- Solid rocket: simulator, landing burn design tool, PID and trained landing policy
  ([docs/solid_rocket.md](docs/solid_rocket.md)).
- Not modelled: ground effect near the pad, battery voltage sag, fan inflow at side wind beyond a
  constant side-force coefficient, servo backlash.
