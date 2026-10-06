"""JSON for the browser: NaN and infinity become null, numpy scalars become plain numbers."""

import json
import math
from typing import Any

import numpy as np


def json_safe(value: Any) -> Any:
    """A copy of `value` that json.dumps can write as strict JSON."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    return value


def dumps(value: Any) -> str:
    """Strict JSON text of `value`, compact."""
    return json.dumps(json_safe(value), allow_nan=False, separators=(",", ":"))
