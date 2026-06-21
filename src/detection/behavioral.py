"""Capa L3 — Señales de comportamiento/cuenta.

⚠️ **SIMULADO.** No hay datos públicos de cuentas/reembolsos, así que estas features se
simulan con correlación MODERADA al fraude + RUIDO fuerte (no son un oráculo). En
producción (Rappi) saldrían de datos reales: frecuencia de reembolsos, antigüedad de cuenta,
device-sharing, geo. Se incluyen para demostrar la fusión multi-señal del sistema en capas;
quedan claramente etiquetadas como simuladas en el informe.
"""
from __future__ import annotations

import numpy as np

BEHAVIORAL_NAMES = ["refund_count", "account_age_days", "device_risk"]


def simulate(is_fraud: np.ndarray, seed: int = 42) -> np.ndarray:
    """Genera features de comportamiento por reclamo. Correlación moderada + ruido fuerte."""
    rng = np.random.default_rng(seed)
    n = len(is_fraud)
    f = is_fraud.astype(float)

    # los fraudsters tienden a más reembolsos, pero con MUCHO solapamiento (Poisson ruidosa)
    refund_count = rng.poisson(lam=1.0 + 2.0 * f) + rng.poisson(lam=1.0, size=n)
    # cuentas más nuevas → algo más de fraude, muy ruidoso
    account_age = np.clip(rng.normal(400 - 150 * f, 250), 1, None)
    # device risk: bernoulli con prob ligeramente mayor en fraude
    device_risk = (rng.random(n) < (0.12 + 0.18 * f)).astype(float)

    return np.stack([refund_count, account_age, device_risk], axis=1).astype(np.float32)
