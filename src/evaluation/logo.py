"""Leave-One-Generator-Out (LOGO) — la medida rigurosa de generalización cross-generator.

Para CADA generador g: se entrena con TODOS los datos menos los fakes de g, y se testea
sobre los fakes de g (un generador nunca visto). Reproduce el setup de FraudBench: mide
cuánto cae la detección frente a una familia de edición nueva. Devuelve una matriz
held-out → TPR. Usa el clasificador forense (rápido, sklearn) → corre sin GPU.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.decision.policy import Costs, cost_threshold
from src.detection.baseline import ForensicClassifier


def leave_one_generator_out(feats: np.ndarray, df: pd.DataFrame, costs: Costs,
                            seed: int = 42, fake_id: int | None = None) -> pd.DataFrame:
    """Sweep LOGO con el clasificador forense. Una fila por generador held-out.
    `fake_id` = id de la clase fake-damaged (2 en frutas 3-clases, 1 en cocido 2-clases)."""
    if fake_id is None:
        from src.data.corpus import LABEL2ID
        fake_id = LABEL2ID["fake-damaged"]
    thr = cost_threshold(costs)

    is_fake = (df["label"] == "fake-damaged").to_numpy()
    is_genuine = ~is_fake
    gen = df["generator"].to_numpy()
    label_id = df["label_id"].to_numpy()
    split = df["split"].to_numpy()

    # genuinas: split fijo (no se filtran entre folds)
    gen_train = is_genuine & (split != "test")
    gen_test = is_genuine & (split == "test")

    rows = []
    for g in sorted(set(gen[is_fake])):
        held = is_fake & (gen == g)
        other_fakes = is_fake & (gen != g)
        tr = gen_train | other_fakes              # entreno: genuinas(train) + fakes de OTROS gen
        te_fake = held                            # testeo fakes SOLO del generador held-out

        clf = ForensicClassifier(seed=seed, fake_id=fake_id).fit(feats[tr], label_id[tr])
        p_held = clf.prob_fake(feats[te_fake])
        p_gen_test = clf.prob_fake(feats[gen_test])

        tpr = float((p_held >= thr).mean()) if len(p_held) else float("nan")
        fpr = float((p_gen_test >= thr).mean()) if gen_test.any() else float("nan")
        rows.append({"held_out_generator": g, "n_test_fakes": int(held.sum()),
                     "TPR_held_out": tpr, "FPR_genuine": fpr})
    return pd.DataFrame(rows).set_index("held_out_generator")
