"""Fase 1/2 — Entrenamiento de detectores (STUB).

Fine-tune de un backbone (resnet50/efficientnet/vit vía timm) sobre el corpus de
3 clases, con calibración posterior. Diseño en docs/01-data-generation-and-models.md.

    uv run --extra ml python scripts/run_train.py --config configs/default.yaml --model resnet50
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
    ap.add_argument("--model", default=None, help="override de detection.model")
    args = ap.parse_args()
    cfg = load_config(args.config)
    model = args.model or cfg["detection"]["model"]

    print("[run_train] STUB — pendiente de implementar (Fase 1/2).")
    print(f"  modelo: {model}  | calibración: {cfg['detection']['calibration']}")
    print(f"  clases: {cfg['data']['classes']}  | image_size: {cfg['data']['image_size']}")
    # TODO(Fase 1/2):
    #   1. dataloaders con split GENERADOR-DISJUNTO (holdout_generator solo en test)
    #   2. fine-tune backbone (transfer learning, MPS); early stopping en PR-AUC val
    #   3. calibrar (temperature/isotonic/platt) sobre val
    #   4. guardar pesos (data/ o results/, gitignored) + scores de val/test en results/


if __name__ == "__main__":
    main()
