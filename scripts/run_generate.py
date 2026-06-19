"""Fase 1 — Construye el corpus de 3 clases (camino runnable, sintético).

Genera genuine-undamaged / genuine-damaged / fake-damaged (multi-generador, con
held-out solo en test) y escribe imágenes + manifest.csv en data/generated/.
El camino real (Food-101 + difusión) está en src/data/sources.py y src/generation/diffusion.py.

    uv run --extra ml python scripts/run_generate.py --config configs/default.yaml --n-sources 160
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config  # noqa: E402
from src.data.corpus import build_corpus  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--n-sources", type=int, default=160)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out_dir = Path(cfg["paths"]["generated"])

    df = build_corpus(cfg, out_dir, n_sources=args.n_sources)
    print(f"corpus generado en {out_dir} → {len(df)} imágenes")
    print(df.groupby(["split", "label"]).size().to_string())
    print(f"\ngeneradores: {sorted(set(df['generator']) - {'none'})} "
          f"(held-out solo en test: {cfg['data']['holdout_generator']})")


if __name__ == "__main__":
    main()
