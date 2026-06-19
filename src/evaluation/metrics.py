"""Métricas de evaluación. No usar accuracy como métrica principal (clases
desbalanceadas + costo asimétrico). Solo numpy para mantener el core liviano."""
from __future__ import annotations

import numpy as np


def tpr_at_fpr(y_true: np.ndarray, scores: np.ndarray, target_fpr: float = 0.05) -> float:
    """TPR alcanzable manteniendo el FPR <= target_fpr. y_true: 1=fraude."""
    y_true = np.asarray(y_true).astype(bool)
    scores = np.asarray(scores, dtype=float)
    pos, neg = scores[y_true], scores[~y_true]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # umbral más bajo que respeta el FPR objetivo
    thr = np.quantile(neg, 1.0 - target_fpr)
    return float(np.mean(pos >= thr))


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    """ECE: brecha promedio |confianza - accuracy| por bin. Mide calibración."""
    y_true = np.asarray(y_true).astype(float)
    probs = np.asarray(probs, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(probs, bins) - 1, 0, n_bins - 1)
    ece, n = 0.0, len(probs)
    for b in range(n_bins):
        m = idx == b
        if not np.any(m):
            continue
        conf, acc = probs[m].mean(), y_true[m].mean()
        ece += (np.sum(m) / n) * abs(conf - acc)
    return float(ece)


def brier_score(y_true: np.ndarray, probs: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    probs = np.asarray(probs, dtype=float)
    return float(np.mean((probs - y_true) ** 2))
