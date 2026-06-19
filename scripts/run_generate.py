"""Fase 1 — Pipeline de generación del set `fake-damaged` (STUB).

Toma imágenes `genuine-undamaged`, las edita con cada generador configurado para
simular daño (moho / crudo / objeto extraño), post-procesa (strip EXIF, recompresión)
y cachea en data/generated/<generator>/. Diseño completo en
docs/01-data-generation-and-models.md.

    uv run --extra gen python scripts/run_generate.py --config configs/default.yaml --limit 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--limit", type=int, default=None, help="imágenes por generador (smoke)")
    args = ap.parse_args()
    cfg = load_config(args.config)

    print("[run_generate] STUB — pendiente de implementar (Fase 1).")
    print(f"  generadores: {[g['name'] for g in cfg['generation']['generators']]}")
    print(f"  edits: {cfg['generation']['edits']}")
    print(f"  held-out (solo test): {cfg['data']['holdout_generator']}")
    print("  → ver docs/01-data-generation-and-models.md para el diseño del pipeline.")
    # TODO(Fase 1):
    #   1. cargar genuine-undamaged desde data/raw (src/data)
    #   2. por cada generador: para inpaint, segmentar comida → máscara → editar;
    #      para img2img/instruct, aplicar edición global; tag = {generator, edit}
    #   3. post-proceso (src/generation): strip_exif, jpeg_recompress
    #   4. guardar en data/generated/<generator>/ + manifest.csv (label, generator, edit, src)


if __name__ == "__main__":
    main()
