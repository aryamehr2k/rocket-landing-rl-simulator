"""Write the vehicle, motor, sensor, mission and training files of a data sheet, then load them back."""

import re
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

from rocketsim.hop.config_text import BUILDER, mission_yaml, motor_yaml, sensors_yaml, vehicle_yaml
from rocketsim.hop.datasheet import Datasheet, DatasheetError, Derived, derive, load_datasheet
from rocketsim.hop.mission import MissionConfig, load_mission_config
from rocketsim.hop.training import HopTrainingConfig, load_hop_training_config
from rocketsim.hop.training_text import training_yaml
from rocketsim.hop.vehicle import HopVehicleConfig, load_vehicle_config
from rocketsim.motors import MotorFileError
from rocketsim.simconfig import SimConfig, load_sim_config, physics_steps_per_control_step
from rocketsim.yaml_section import ConfigError

NAME_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")
HEADER_LINES = 5  # the builder's mark sits in the first lines of every file it writes


@dataclass(frozen=True)
class ConfigPaths:
    vehicle: Path
    motor: Path
    sensors: Path
    mission: Path
    training: Path

    def all(self) -> tuple[Path, ...]:
        return self.vehicle, self.motor, self.sensors, self.mission, self.training


@dataclass(frozen=True)
class BuiltVehicle:
    name: str
    paths: ConfigPaths
    sheet: Datasheet
    derived: Derived
    vehicle: HopVehicleConfig
    mission: MissionConfig
    training: HopTrainingConfig
    sim: SimConfig


def config_paths(configs: Path, name: str) -> ConfigPaths:
    """Where the files of vehicle `name` go inside a configs folder."""
    return ConfigPaths(
        configs / "vehicles" / f"{name}.yaml", configs / "motors" / f"{name}_motor.yaml",
        configs / "sensors" / f"{name}_sensors.yaml", configs / "missions" / f"{name}.yaml",
        configs / "training" / f"{name}.yaml",
    )


def vehicle_name(sheet: Datasheet, given: str | None = None) -> str:
    """The name the files get: `given`, else the sheet's vehicle.name, else the sheet's file name."""
    name = given or sheet["vehicle"]["name"] or Path(sheet.source).stem
    if not NAME_PATTERN.match(name):
        raise DatasheetError(f"vehicle name {name!r} may only hold letters, digits and _")
    return name


def build_vehicle(
    sheet_path: str | Path, configs: Path, training_base: Path, name: str | None = None, force: bool = False,
) -> BuiltVehicle:
    """Read the sheet, write the five config files and load them with the simulator's own loaders."""
    sheet = load_datasheet(sheet_path)
    name = vehicle_name(sheet, name)
    # The files name their sources relative to the project, never by a path that only exists on this computer.
    project = configs.resolve().parent
    sheet = replace(sheet, source=project_label(Path(sheet_path), project))
    derived = derive(sheet, load_sim_config(training_base))
    paths = config_paths(configs, name)
    _check_overwrite(paths, training_base, project, force)
    texts = (
        vehicle_yaml(name, sheet, derived), motor_yaml(name, sheet), sensors_yaml(name, sheet),
        mission_yaml(name, sheet, derived),
        training_yaml(name, sheet, derived, training_base.read_text(), project_label(training_base, project)),
    )
    # A trial copy is loaded first so that a sheet the loaders reject leaves no files behind.
    with tempfile.TemporaryDirectory() as scratch:
        trial = config_paths(Path(scratch), name)
        _write(trial, texts)
        try:
            _load(trial)
        except (ConfigError, MotorFileError) as error:
            label = project_label(configs, project)
            message = str(error).replace(str(Path(scratch).resolve()), label).replace(scratch, label)
            raise DatasheetError(f"{sheet.source} builds a file the simulator refuses:\n  {message}") from error
    _write(paths, texts)
    return BuiltVehicle(name, paths, sheet, derived, *_load(paths))


def project_label(path: Path, project: Path) -> str:
    """`path` relative to the project folder, or just its file name when it lies outside."""
    path = path.resolve()
    return path.relative_to(project).as_posix() if path.is_relative_to(project) else path.name


def written_by_builder(path: Path) -> bool:
    """Whether the file at `path` carries the builder's header, so building again may replace it."""
    with path.open() as file:
        return any(line.startswith("# Built from ") and BUILDER in line for _, line in zip(range(HEADER_LINES), file))


def _check_overwrite(paths: ConfigPaths, training_base: Path, project: Path, force: bool) -> None:
    existing = [path for path in paths.all() if path.exists()]
    foreign = [path for path in existing if not written_by_builder(path)]
    if paths.training.resolve() == training_base.resolve() and paths.training not in foreign:
        foreign.append(paths.training)
    if foreign:
        raise ConfigError("these files were not written by the builder or are its training base, so it never "
                          "replaces them; give the vehicle another name (vehicle.name or --name):\n  "
                          + "\n  ".join(project_label(path, project) for path in foreign))
    if existing and not force:
        raise ConfigError("these files exist already; use --force to replace them:\n  "
                          + "\n  ".join(project_label(path, project) for path in existing))


def _write(paths: ConfigPaths, texts: tuple[str, ...]) -> None:
    for path, text in zip(paths.all(), texts):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def _load(paths: ConfigPaths) -> tuple[HopVehicleConfig, MissionConfig, HopTrainingConfig, SimConfig]:
    vehicle = load_vehicle_config(paths.vehicle)
    mission = load_mission_config(paths.mission)
    training = load_hop_training_config(paths.training)
    if training.vehicle_path.resolve() != paths.vehicle.resolve():
        raise ConfigError(f"{paths.training} does not point at {paths.vehicle}")
    sim = load_sim_config(paths.training)
    physics_steps_per_control_step(sim.simulation, vehicle.body.control)
    return vehicle, mission, training, sim
