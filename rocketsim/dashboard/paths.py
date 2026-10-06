"""The folders the dashboard reads and writes, and the check that keeps browser paths inside them.

The browser names files with paths relative to the project such as
`configs/vehicles/electric_hopper.yaml` or `runs/20261006_070357_hop`. The first part picks one
of the allowed folders (configs, models, runs); anything that resolves outside it is refused.
"""

from dataclasses import dataclass
from pathlib import Path

CONFIGS = "configs"
MODELS = "models"
RUNS = "runs"
TRAIN_LOGS = "train_logs"
TRAINING = "training"
STATIC = "dashboard"
SCRIPTS = "scripts"


class PathError(ValueError):
    """A path from the browser that is missing or lies outside the allowed folders."""


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    runs: Path
    models: Path

    @classmethod
    def for_project(cls, root: Path, runs: Path | None = None, models: Path | None = None) -> "ProjectPaths":
        """The project's own folders, with the runs and models folders optionally elsewhere (tests)."""
        root = root.resolve()
        return cls(
            root=root,
            runs=(runs if runs is not None else root / RUNS).resolve(),
            models=(models if models is not None else root / MODELS).resolve(),
        )

    @property
    def configs(self) -> Path:
        return self.root / CONFIGS

    @property
    def training_files(self) -> Path:
        return self.configs / TRAINING

    @property
    def static(self) -> Path:
        return self.root / STATIC

    @property
    def scripts(self) -> Path:
        return self.root / SCRIPTS

    @property
    def train_logs(self) -> Path:
        return self.runs / TRAIN_LOGS

    def folders(self) -> dict[str, Path]:
        return {CONFIGS: self.configs, MODELS: self.models, RUNS: self.runs}

    def resolve(self, client_path: str, must_exist: bool = True) -> Path:
        """The file or folder a browser path names, refused unless it lies inside configs, models or runs."""
        if not isinstance(client_path, str) or not client_path:
            raise PathError("a path is required")
        parts = Path(client_path).parts
        folder = self.folders().get(parts[0]) if parts else None
        if folder is None or Path(client_path).is_absolute():
            raise PathError(f"{client_path!r} must start with one of {sorted(self.folders())}/")
        resolved = folder.joinpath(*parts[1:]).resolve()
        if not resolved.is_relative_to(folder.resolve()):
            raise PathError(f"{client_path!r} points outside {parts[0]}/")
        if must_exist and not resolved.exists():
            raise PathError(f"{client_path!r} does not exist")
        return resolved

    def client_path(self, path: Path) -> str:
        """The browser name of a file inside configs, models or runs."""
        resolved = path.resolve()
        for name, folder in self.folders().items():
            if resolved.is_relative_to(folder.resolve()):
                return str(Path(name) / resolved.relative_to(folder.resolve()))
        raise PathError(f"{path} is not inside {sorted(self.folders())}")

    def resolve_inside(self, client_path: str, folder: Path) -> Path:
        """Like resolve, and also refused unless the result lies inside `folder`."""
        resolved = self.resolve(client_path)
        if not resolved.is_relative_to(folder.resolve()):
            raise PathError(f"{client_path!r} must be inside {self.client_path(folder)}/")
        return resolved
