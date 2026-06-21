"""Fase 3 — Evaluación: discriminación, costo, calibración, breakdown por generador,
significancia. Combina sklearn (PR/ROC-AUC) con las métricas propias de metrics.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.decision.policy import Costs, cost_threshold, decide, realized_cost
from src.evaluation.metrics import brier_score, expected_calibration_error, tpr_at_fpr


def discrimination_metrics(is_fraud: np.ndarray, scores: np.ndarray, target_fpr: float) -> dict:
    """Métricas de ranking — invariantes a calibración monótona → usar el score CRUDO."""
    return {
        "pr_auc": float(average_precision_score(is_fraud, scores)),
        "roc_auc": float(roc_auc_score(is_fraud, scores)),
        "tpr_at_fpr": float(tpr_at_fpr(is_fraud, scores, target_fpr)),
    }


def calibration_metrics(is_fraud: np.ndarray, probs: np.ndarray) -> dict:
    """Métricas de calibración — sobre la probabilidad CALIBRADA."""
    return {
        "ece": expected_calibration_error(is_fraud, probs),
        "brier": brier_score(is_fraud, probs),
    }


def decision_costs(probs: np.ndarray, is_fraud: np.ndarray, costs: Costs) -> dict:
    return {p: realized_cost(decide(probs, costs, p), is_fraud, costs)["cost_per_case"]
            for p in ("D0", "D1", "D2")}


def per_generator_breakdown(df_test: pd.DataFrame, probs: np.ndarray, costs: Costs,
                            holdout: str) -> dict:
    """TPR (recall) por generador entre los fake, y FPR sobre las clases genuinas,
    al umbral de costo. El generador held-out mide generalización cross-generator."""
    thr = cost_threshold(costs)
    pred_fraud = probs >= thr
    gens = df_test["generator"].to_numpy()
    labels = df_test["label"].to_numpy()
    out = {}
    for g in sorted(set(gens) - {"none"}):
        m = (gens == g) & (labels == "fake-damaged")
        if m.any():
            tag = f"tpr[{g}]" + ("*held-out" if g == holdout else "")
            out[tag] = float(pred_fraud[m].mean())
    for cls in sorted(set(labels) - {"fake-damaged"}):  # cualquier clase genuina (frutas o cocido)
        m = labels == cls
        if m.any():
            out[f"fpr[{cls}]"] = float(pred_fraud[m].mean())
    return out


def bootstrap_cost_ci(probs: np.ndarray, is_fraud: np.ndarray, costs: Costs,
                      policy_a: str = "D0", policy_b: str = "D2",
                      iters: int = 1000, seed: int = 0) -> dict:
    """IC bootstrap de la mejora de costo policy_a → policy_b (cost_a - cost_b)."""
    rng = np.random.default_rng(seed)
    n = len(probs)
    diffs = []
    for _ in range(iters):
        idx = rng.integers(0, n, n)
        p, y = probs[idx], is_fraud[idx]
        ca = realized_cost(decide(p, costs, policy_a), y, costs)["cost_per_case"]
        cb = realized_cost(decide(p, costs, policy_b), y, costs)["cost_per_case"]
        diffs.append(ca - cb)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"mean_improvement": float(np.mean(diffs)), "ci95": [float(lo), float(hi)]}
