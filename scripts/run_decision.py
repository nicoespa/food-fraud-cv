"""Demo runnable de la Capa 3 (decisión cost-sensitive).

Sin datos reales todavía: simula probabilidades de fraude calibradas + un set de
etiquetas, y compara políticas D0 (accuracy) / D1 (costo esperado) / D2 (tres zonas).
Sirve para validar el núcleo de decisión y mostrar el insight de la Tesis 2.

    uv run python scripts/run_decision.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config  # noqa: E402
from src.decision.policy import Costs, decide, realized_cost  # noqa: E402


def simulate(n: int = 5000, seed: int = 42):
    """Genera probabilidades de fraude CALIBRADAS por construcción + etiquetas.

    p = P(fraude) ~ Beta(mean≈0.15); is_fraud ~ Bernoulli(p). Así p es la posterior
    calibrada real, y la regla de Bayes sensible al costo (D1/D2) sí minimiza el costo.
    """
    rng = np.random.default_rng(seed)
    probs = rng.beta(1.5, 8.5, n)              # prob de fraude calibrada, media ~0.15
    is_fraud = (rng.random(n) < probs).astype(int)
    return probs, is_fraud


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    costs = Costs.from_config(cfg)
    probs, is_fraud = simulate(seed=cfg.get("seed", 42))

    print(f"costos: c_fn={costs.c_fn} c_fp={costs.c_fp} c_review={costs.c_review}")
    print(f"n={len(probs)}  fraude={is_fraud.mean():.1%}\n")
    print(f"{'policy':<6} {'cost/case':>10} {'FN':>6} {'FP':>6} {'review%':>8}")
    for policy in ("D0", "D1", "D2"):
        actions = decide(probs, costs, policy=policy)
        r = realized_cost(actions, is_fraud, costs)
        print(f"{policy:<6} {r['cost_per_case']:>10.4f} {r['false_negatives']:>6} "
              f"{r['false_positives']:>6} {r['review_rate']:>7.1%}")
    print("\nEsperado: D1/D2 < D0 en costo/caso → 'mejor accuracy ≠ mejor decisión'.")


if __name__ == "__main__":
    main()
