"""Small helper for reading YAML mappings with bounds checks and unit conversion.

Every problem raises ConfigError with the file and the key path so the user can find it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from rocketsim.units import split_unit_suffix, to_si


class ConfigError(ValueError):
    """Raised when a configuration file is missing, malformed or fails validation."""


def read_yaml_mapping(path: str | Path) -> tuple[dict[str, Any], str]:
    """Read a YAML file whose top level is a mapping. Returns the data and the path as text."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"{path}: file not found")
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return data, str(path)


class Section:
    """A YAML mapping with typed getters that check bounds and convert units to SI."""

    def __init__(self, data: dict[str, Any], source: str, path: str = "") -> None:
        self.data = data
        self.source = source
        self.path = path

    def where(self, key: str) -> str:
        return f"{self.source}: {self.path}.{key}" if self.path else f"{self.source}: {key}"

    def sub(self, key: str) -> Section:
        value = self.data.get(key)
        if not isinstance(value, dict):
            raise ConfigError(f"{self.where(key)} must be a mapping with named values")
        return Section(value, self.source, f"{self.path}.{key}" if self.path else key)

    def has(self, key: str) -> bool:
        return key in self.data

    def number(
        self,
        key: str,
        *,
        minimum: float | None = None,
        above: float | None = None,
        maximum: float | None = None,
        default: float | None = None,
    ) -> float:
        """Return a numeric value converted to SI. Bounds are in the YAML unit of the key."""
        raw = self._raw(key, default)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ConfigError(f"{self.where(key)} must be a number, got {raw!r}")
        if minimum is not None and raw < minimum:
            raise ConfigError(f"{self.where(key)} must be >= {minimum}, got {raw}")
        if above is not None and raw <= above:
            raise ConfigError(f"{self.where(key)} must be > {above}, got {raw}")
        if maximum is not None and raw > maximum:
            raise ConfigError(f"{self.where(key)} must be <= {maximum}, got {raw}")
        return to_si(key, float(raw))

    def integer(self, key: str, *, minimum: int | None = None, default: int | None = None) -> int:
        raw = self._raw(key, default)
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise ConfigError(f"{self.where(key)} must be an integer, got {raw!r}")
        if minimum is not None and raw < minimum:
            raise ConfigError(f"{self.where(key)} must be >= {minimum}, got {raw}")
        return raw

    def boolean(self, key: str, *, default: bool | None = None) -> bool:
        raw = self._raw(key, default)
        if not isinstance(raw, bool):
            raise ConfigError(f"{self.where(key)} must be true or false, got {raw!r}")
        return raw

    def string(self, key: str, *, default: str | None = None) -> str:
        raw = self._raw(key, default)
        if not isinstance(raw, str) or not raw:
            raise ConfigError(f"{self.where(key)} must be a non-empty string, got {raw!r}")
        return raw

    def only_keys(self, *allowed: str) -> None:
        """Reject keys that are not expected, which catches typos and wrong unit suffixes."""
        unknown = sorted(set(self.data) - set(allowed))
        if unknown:
            raise ConfigError(
                f"{self.where(unknown[0])} is not a known setting; expected one of {sorted(allowed)}"
            )

    def _raw(self, key: str, default: Any) -> Any:
        if key in self.data:
            return self.data[key]
        if default is not None:
            return default
        base, _ = split_unit_suffix(key)
        raise ConfigError(f"{self.where(key)} is required ({base})")
