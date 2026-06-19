"""Config loader. Config-driven: nada de hardcodear hiperparámetros en el código."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Carga un YAML de configuración a dict."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)
