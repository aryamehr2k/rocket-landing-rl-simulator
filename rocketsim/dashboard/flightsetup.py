"""One flight's settings from the dashboard form, and the simulation built from them.

`prepare` loads the files once; `new_simulation` then builds as many flights from them as needed.
"""

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from rocketsim.curriculum import scale_sensors
from rocketsim.dashboard.paths import PathError, ProjectPaths
from rocketsim.guidance_config import PID
from rocketsim.hop.env import vehicle_for_training
from rocketsim.hop.evaluate import HopModel, load_hop_model, with_mission_wind
from rocketsim.hop.mission import MissionConfig, load_mission_config, with_targets
from rocketsim.hop.simulation import HopErrors, HopSimulation
from rocketsim.hop.training import HiddenErrorRanges, load_hop_training_config
from rocketsim.hop.vehicle import HopVehicleConfig, load_vehicle_config
from rocketsim.policy import POLICY_FILE
from rocketsim.simconfig import SimConfig, load_sim_config
from rocketsim.units import deg_to_rad, g_to_kg, rad_to_deg

PID_CONTROLLER = "pid"
PID_LABEL = "PID"
DEFAULT_VEHICLE = "configs/vehicles/electric_hopper.yaml"
DEFAULT_MISSION = "configs/missions/hop_50m.yaml"
DEFAULT_WORLD = "configs/training/hop.yaml"
DEFAULT_SEED = 100
SEED_LIMIT = 2 ** 31 - 1
TIME_LIMIT_FACTOR = 1.5  # a changed mission keeps at least this many times its planned length before it times out

# (default, lowest, highest) of every number in the form, in the form's units.
NUMBERS: dict[str, tuple[float, float, float]] = {
    "wind_speed": (0.0, 0.0, 20.0),
    "wind_direction_deg": (0.0, -360.0, 360.0),
    "gust_std": (0.0, 0.0, 10.0),
    "thrust_scale": (1.0, 0.5, 1.5),
    "dry_mass_offset_g": (0.0, -1000.0, 1000.0),
    "sensor_noise": (1.0, 0.0, 10.0),
}
# (lowest, highest) of the mission overrides; a missing or empty override keeps the mission file's value.
OVERRIDES: dict[str, tuple[float, float]] = {
    "target_altitude": (1.0, 500.0),
    "hover_time": (0.0, 120.0),
    "landing_radius": (0.5, 100.0),
    "climb_speed": (0.2, 20.0),
    "descent_speed": (0.2, 20.0),
}


class SetupError(ValueError):
    """A form value or configuration file the flight cannot be built from."""


@dataclass(frozen=True)
class MissionOverrides:
    target_altitude: float | None = None
    hover_time: float | None = None
    landing_radius: float | None = None
    climb_speed: float | None = None
    descent_speed: float | None = None


@dataclass(frozen=True)
class FlightSetup:
    vehicle: Path
    mission: Path
    world: Path
    model: Path | None
    seed: int
    wind_speed: float
    wind_direction_deg: float
    gust_std: float
    thrust_scale: float
    dry_mass_offset_g: float
    sensor_noise: float
    overrides: MissionOverrides


@dataclass
class PreparedFlight:
    """The loaded files of a setup, ready to build simulations from."""

    vehicle: HopVehicleConfig
    mission: MissionConfig
    sim: SimConfig
    model: HopModel | None
    errors: HopErrors
    error_ranges: HiddenErrorRanges

    @property
    def controller_name(self) -> str:
        return self.model.name if self.model is not None else PID_LABEL


def parse_setup(body: dict[str, Any], paths: ProjectPaths) -> FlightSetup:
    """Check the form values the browser sent and turn them into a FlightSetup."""
    try:
        configs = paths.configs
        vehicle = paths.resolve_inside(_text(body, "vehicle", DEFAULT_VEHICLE), configs)
        mission = paths.resolve_inside(_text(body, "mission", DEFAULT_MISSION), configs)
        world = paths.resolve_inside(_text(body, "world", DEFAULT_WORLD), configs)
        model = _model_folder(_text(body, "controller", PID_CONTROLLER), paths)
    except PathError as error:
        raise SetupError(str(error)) from error
    numbers = {key: _number(body, key, *limits) for key, limits in NUMBERS.items()}
    overrides = MissionOverrides(**{key: _optional_number(body, key, *limits) for key, limits in OVERRIDES.items()})
    return FlightSetup(
        vehicle=vehicle, mission=mission, world=world, model=model, seed=_seed(body), overrides=overrides, **numbers,
    )


def prepare(setup: FlightSetup) -> PreparedFlight:
    """Load the vehicle, mission, world and model files and apply the form's changes to them."""
    try:
        vehicle = load_vehicle_config(setup.vehicle)
        mission = apply_overrides(load_mission_config(setup.mission), setup.overrides)
        sim = load_sim_config(setup.world)
        ranges = load_hop_training_config(setup.world).hidden_errors
        model = load_hop_model(setup.model) if setup.model is not None else None
    except (ValueError, OSError, KeyError) as error:
        raise SetupError(str(error)) from error
    vehicle = replace(vehicle, sensors=scale_sensors(vehicle.sensors, setup.sensor_noise))
    if model is None:
        vehicle = replace(vehicle, controllers=replace(vehicle.controllers, steering=PID, throttle=PID))
    else:
        vehicle = vehicle_for_training(vehicle, model.training)
    sim = with_mission_wind(sim, setup.wind_speed, deg_to_rad(setup.wind_direction_deg), setup.gust_std)
    mission, sim = fit_time_limits(mission, sim)
    errors = HopErrors(thrust_scale=setup.thrust_scale, dry_mass_offset=g_to_kg(setup.dry_mass_offset_g))
    return PreparedFlight(vehicle, mission, sim, model, errors, ranges)


def apply_overrides(mission: MissionConfig, overrides: MissionOverrides) -> MissionConfig:
    """The mission with the form's target height, hover time, landing radius and speeds where given."""
    mission = with_targets(mission, overrides.target_altitude, overrides.hover_time)
    changes = {
        name: value for name, value in (
            ("landing_radius", overrides.landing_radius), ("climb_speed", overrides.climb_speed),
            ("descent_speed", overrides.descent_speed),
        ) if value is not None
    }
    return replace(mission, **changes)


def fit_time_limits(mission: MissionConfig, sim: SimConfig) -> tuple[MissionConfig, SimConfig]:
    """Longer time limits when the changed mission needs them; the files' limits otherwise."""
    needed = TIME_LIMIT_FACTOR * mission.planned_duration + sim.simulation.pad_hold_time
    mission = replace(mission, max_flight_time=max(mission.max_flight_time, needed))
    simulation = replace(sim.simulation, max_flight_time=max(sim.simulation.max_flight_time, mission.max_flight_time))
    return mission, replace(sim, simulation=simulation)


def new_simulation(prepared: PreparedFlight, seed: int, errors: HopErrors | None = None) -> HopSimulation:
    """A flight on the pad, ready for its first control step."""
    simulation = HopSimulation(prepared.vehicle, prepared.mission, prepared.sim, seed=seed)
    simulation.reset(seed, errors=errors if errors is not None else prepared.errors)
    if prepared.model is not None:
        prepared.model.controller.reset()
    return simulation


def control_step(prepared: PreparedFlight, simulation: HopSimulation) -> None:
    """One control step with the model's action, or the PID's when there is no model."""
    model = prepared.model
    action = model.controller.action(simulation.computer, simulation.t) if model is not None else None
    simulation.control_step(action)


def describe(prepared: PreparedFlight, setup: FlightSetup) -> dict[str, Any]:
    """What the browser needs to draw the vehicle and grade the mission."""
    body, mission = prepared.vehicle.body, prepared.mission
    wind = prepared.sim.wind
    return {
        "controller": prepared.controller_name,
        "seed": setup.seed,
        "vehicle": vehicle_geometry(prepared.vehicle),
        "legs": {
            "max_vertical_speed": body.legs.max_vertical_speed, "max_lateral_speed": body.legs.max_lateral_speed,
            "max_tilt_deg": rad_to_deg(body.legs.max_tilt),
        },
        "mission": {
            "name": mission.name, "target_altitude": mission.target_altitude, "altitude_tolerance": mission.altitude_tolerance,
            "hover_time": mission.hover_time, "landing_radius": mission.landing_radius, "climb_speed": mission.climb_speed,
            "descent_speed": mission.descent_speed, "max_flight_time": mission.max_flight_time,
        },
        "wind": {"speed": wind.steady, "direction_deg": rad_to_deg(wind.steady_direction), "gust_std": wind.gust_std},
        "errors": {
            "thrust_scale": setup.thrust_scale, "dry_mass_offset_g": setup.dry_mass_offset_g,
            "sensor_noise": setup.sensor_noise,
        },
    }


def vehicle_geometry(vehicle: HopVehicleConfig) -> dict[str, Any]:
    """Sizes in metres for drawing the vehicle, and its thrust and gimbal limits."""
    body = vehicle.body
    return {
        "name": vehicle.name, "length": body.airframe.length, "diameter": body.airframe.reference_diameter,
        "cg": body.loaded_cg, "pivot": body.gimbal.pivot, "leg_span": body.legs.span, "leg_height": body.legs.height,
        "leg_count": body.legs.count, "max_thrust": vehicle.max_thrust,
        "max_gimbal_deg": rad_to_deg(body.gimbal.max_angle), "pad_cg_height": body.pad_cg_height,
    }


def _model_folder(controller: str, paths: ProjectPaths) -> Path | None:
    if controller == PID_CONTROLLER:
        return None
    folder = paths.resolve(controller)
    if not any(folder.is_relative_to(root) for root in (paths.models, paths.runs)):
        raise PathError(f"{controller!r}: a model must be inside models/ or runs/")
    if not (folder / POLICY_FILE).exists():
        raise PathError(f"{controller!r} has no {POLICY_FILE} (still training?)")
    return folder


def _text(body: dict[str, Any], key: str, default: str) -> str:
    value = body.get(key) or default
    if not isinstance(value, str):
        raise SetupError(f"{key} must be text")
    return value


def _to_float(key: str, raw: Any) -> float:
    if isinstance(raw, bool):
        raise SetupError(f"{key} must be a number")
    try:
        value = float(raw)
    except (TypeError, ValueError) as error:
        raise SetupError(f"{key} must be a number, got {raw!r}") from error
    if not math.isfinite(value):
        raise SetupError(f"{key} must be a finite number")
    return value


def _number(body: dict[str, Any], key: str, default: float, low: float, high: float) -> float:
    raw = body.get(key)
    value = default if raw is None or raw == "" else _to_float(key, raw)
    if not low <= value <= high:
        raise SetupError(f"{key} must be between {low:g} and {high:g}, got {value:g}")
    return value


def _optional_number(body: dict[str, Any], key: str, low: float, high: float) -> float | None:
    raw = body.get(key)
    if raw is None or raw == "":
        return None
    return _number(body, key, low, low, high)


def _seed(body: dict[str, Any]) -> int:
    raw = body.get("seed")
    value = DEFAULT_SEED if raw is None or raw == "" else _to_float("seed", raw)
    if value != int(value) or not 0 <= value <= SEED_LIMIT:
        raise SetupError(f"seed must be a whole number from 0 to {SEED_LIMIT}")
    return int(value)
