"""Fase 2 — Detectores (camino runnable, sklearn sobre features forensics).

- ZeroShotFreqDetector: representa un detector AIGC genérico "off-the-shelf" que se
  apoya en artefactos de alta frecuencia. Sin entrenar con etiquetas. Esperamos que
  dispare también sobre daño REAL (falsos positivos) → evidencia de la Tesis 1.
- ForensicClassifier: clasificador 3-clases entrenado sobre TODAS las features
  forensics. Usa la huella + doble compresión → separa fake de daño real.
- Calibrador isotónico para llevar el score a probabilidad calibrada (la decisión
  cost-sensitive lo necesita).
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler

from src.data.corpus import LABEL2ID
from src.detection.forensic import FEATURE_NAMES

FAKE_ID = LABEL2ID["fake-damaged"]


class ZeroShotFreqDetector:
    """Heurística genérica: score = energía de alta frecuencia estandarizada (sin labels)."""

    name = "zero-shot-aigc"

    def fit_reference(self, feats: np.ndarray) -> "ZeroShotFreqDetector":
        self.mu_ = feats.mean(axis=0)
        self.sd_ = feats.std(axis=0) + 1e-6
        # heurística genérica: energía en las dos bandas de MÁS alta frecuencia
        self._hi_idx = [FEATURE_NAMES.index(n) for n in ("fft_band4", "fft_band5")]
        return self

    def prob_fake(self, feats: np.ndarray) -> np.ndarray:
        z = (feats - self.mu_) / self.sd_
        score = z[:, self._hi_idx].sum(axis=1)
        return 1.0 / (1.0 + np.exp(-score))


class ForensicClassifier:
    """GBM sobre features forensics → P(fake-damaged). Soporta 2 o 3 clases (vía fake_id)."""

    name = "forensic-gbm"

    def __init__(self, seed: int = 42, fake_id: int = FAKE_ID):
        self.scaler = StandardScaler()
        self.clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=seed)
        self.fake_id = fake_id

    def fit(self, feats: np.ndarray, label_ids: np.ndarray) -> "ForensicClassifier":
        self.clf.fit(self.scaler.fit_transform(feats), label_ids)
        self._fake_col = list(self.clf.classes_).index(self.fake_id)
        return self

    def prob_fake(self, feats: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(self.scaler.transform(feats))[:, self._fake_col]


def fit_calibrator(scores_val: np.ndarray, is_fraud_val: np.ndarray) -> IsotonicRegression:
    """Calibración isotónica score→P(fraude) sobre validación."""
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(scores_val, is_fraud_val.astype(float))
    return iso


def apply_calibrator(iso: IsotonicRegression, scores: np.ndarray) -> np.ndarray:
    return np.clip(iso.predict(scores), 1e-6, 1 - 1e-6)
