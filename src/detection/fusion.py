"""Capa L4 — Fusión multi-señal → probabilidad de fraude calibrada.

Combina la señal de PÍXELES (CNN/forense) con PROCEDENCIA (reuso/EXIF) y COMPORTAMIENTO
en un solo clasificador. Es lo que permite atrapar fraude que ninguna capa sola detecta
(p.ej. el píxel no ve una foto reusada; la procedencia sí).
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from src.detection.behavioral import BEHAVIORAL_NAMES
from src.detection.provenance import PROVENANCE_NAMES

FUSION_FEATURE_NAMES = ["pixel_prob", *PROVENANCE_NAMES, *BEHAVIORAL_NAMES]


def build_fusion_features(pixel_prob: np.ndarray, provenance: np.ndarray,
                          behavioral: np.ndarray) -> np.ndarray:
    """X = [pixel_prob | reuse_score, exif_missing | refund_count, account_age, device_risk]."""
    return np.column_stack([np.asarray(pixel_prob).reshape(-1, 1), provenance, behavioral]).astype(np.float32)


class FusionClassifier:
    name = "fusion-layered"

    def __init__(self, seed: int = 42):
        self.scaler = StandardScaler()
        self.clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=seed)

    def fit(self, X: np.ndarray, is_fraud: np.ndarray) -> "FusionClassifier":
        self.clf.fit(self.scaler.fit_transform(X), is_fraud.astype(int))
        self._col = list(self.clf.classes_).index(1)
        return self

    def prob_fraud(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(self.scaler.transform(X))[:, self._col]
