"""Adaptadores de datasets públicos REALES (camino a escala).

El slice runnable usa `corpus.build_corpus` (sintético). Para el experimento real,
implementar acá los loaders de los datasets públicos verificados (ver CLAUDE.md §4):
  - Food-101            → genuine-undamaged
  - Fresh&Rotten/etc.   → genuine-damaged
y usar `src/generation/diffusion.py` para producir fake-damaged.

VERIFICAR licencia + estructura en la primera carga (registrar en un dataset card).
"""
from __future__ import annotations

from pathlib import Path


def load_food101(raw_dir: str | Path):
    """Carga Food-101 desde data/raw. Stub: implementar al wirear datos reales."""
    raw = Path(raw_dir) / "food-101"
    if not raw.exists():
        raise FileNotFoundError(
            f"No encontré Food-101 en {raw}. Bajalo (HF `food101` o torchvision) a "
            f"data/raw/food-101/ y validá licencia/estructura. Ver docs/01-...md."
        )
    raise NotImplementedError("TODO(Fase 1 real): parsear Food-101 → lista de paths/labels.")
