"""Fase 2/3 — Evaluación + experimento estrella (STUB).

Carga scores de modelos (results/) y computa PR-AUC, TPR@FPR, ECE, Brier y el costo
esperado bajo cada política, con breakdown por generador (incluyendo el held-out).
Es el motor del experimento estrella (modelo × régimen-de-info × generador).

    uv run --extra ml python scripts/run_evaluate.py --config configs/default.yaml
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
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)

    print("[run_evaluate] STUB — pendiente de implementar (Fase 2/3).")
    print(f"  métricas: {cfg['evaluation']['metrics']}")
    print(f"  TPR@FPR objetivo: {cfg['evaluation']['tpr_at_fpr']}")
    print(f"  breakdown por generador: {cfg['evaluation']['per_generator_breakdown']}")
    # TODO(Fase 2/3):
    #   1. leer scores por modelo desde results/
    #   2. métricas (src/evaluation/metrics.py) globales + por generador (held-out aparte)
    #   3. costo esperado bajo D0/D1/D2 (src/decision/policy.py)
    #   4. tests de significancia (bootstrap CIs, McNemar) → tabla del experimento estrella


if __name__ == "__main__":
    main()
