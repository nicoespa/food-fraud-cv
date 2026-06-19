"""Entrenamiento de detectores.

- Slice runnable (forensic + zero-shot): se entrena dentro de `scripts/run_experiment.py`
  (end-to-end con la evaluación). Corré ese.
- Camino a escala (fine-tune CNN/ViT): `src/detection/finetune.py`, requiere
  `uv sync --extra dl` + GPU/MPS.

    uv run --extra ml python scripts/run_experiment.py      # slice runnable
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    print("El slice runnable entrena dentro de scripts/run_experiment.py (forensic + zero-shot).")
    print("Para fine-tune de backbones (escala): uv sync --extra dl  +  src/detection/finetune.py")


if __name__ == "__main__":
    main()
