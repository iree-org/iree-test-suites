"""Read case manifests connecting source MLIR with shared tensor files."""

import json
from pathlib import Path
from typing import Any

import numpy as np


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_tensor(path: Path) -> np.ndarray:
    tensor = read_json(path)
    return np.array(tensor["values"], dtype=tensor["dtype"]).reshape(tensor["shape"])
