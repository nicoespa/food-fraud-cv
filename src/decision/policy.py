"""Capa 3 — Política de decisión sensible al costo.

Toma una probabilidad de fraude CALIBRADA p = P(fake-damaged) por instancia y decide
la acción sobre el reembolso. Es el corazón de la Tesis 2: la decisión óptima sale del
costo esperado, no de maximizar accuracy.

Modelo de costo (acción × verdad), con p = P(fraude):
  - approve (pagar reembolso):  E[cost] = p * c_fn        (si es fraude, pagamos c_fn)
  - deny    (rechazar):         E[cost] = (1 - p) * c_fp  (si era legítimo, c_fp)
  - review  (revisión humana):  E[cost] = c_review        (asume que resuelve bien)

Políticas:
  - D0: umbral fijo 0.5 (proxy de "maximizar accuracy"). A propósito ingenuo.
  - D1: umbral por costo esperado  t* = c_fp / (c_fn + c_fp); deny si p >= t*.
  - D2: tres zonas (approve / review / deny) — argmin del costo esperado por instancia.
        Zona de review = [c_review/c_fn, 1 - c_review/c_fp] cuando es válida.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

APPROVE, REVIEW, DENY = "approve", "review", "deny"


@dataclass(frozen=True)
class Costs:
    c_fn: float  # pagar un reembolso fraudulento (falso negativo)
    c_fp: float  # rechazar a un cliente honesto (falso positivo)
    c_review: float = 0.0  # mandar a revisión humana

    @classmethod
    def from_config(cls, cfg: dict) -> "Costs":
        c = cfg["decision"]["costs"]
        return cls(c_fn=c["c_fn"], c_fp=c["c_fp"], c_review=c.get("c_review", 0.0))


def cost_threshold(costs: Costs) -> float:
    """Umbral óptimo de deny para la decisión binaria (D1)."""
    return costs.c_fp / (costs.c_fn + costs.c_fp)


def decide(probs: np.ndarray, costs: Costs, policy: str = "D2") -> np.ndarray:
    """Devuelve un array de acciones (approve/review/deny) por instancia."""
    probs = np.asarray(probs, dtype=float)
    if policy == "D0":
        return np.where(probs >= 0.5, DENY, APPROVE)
    if policy == "D1":
        return np.where(probs >= cost_threshold(costs), DENY, APPROVE)
    if policy == "D2":
        e_approve = probs * costs.c_fn
        e_deny = (1.0 - probs) * costs.c_fp
        e_review = np.full_like(probs, costs.c_review)
        stack = np.vstack([e_approve, e_review, e_deny])  # filas = acciones
        idx = stack.argmin(axis=0)
        return np.array([APPROVE, REVIEW, DENY])[idx]
    raise ValueError(f"política desconocida: {policy!r}")


def realized_cost(actions: np.ndarray, is_fraud: np.ndarray, costs: Costs) -> dict:
    """Costo REALIZADO de un conjunto de decisiones contra la verdad (is_fraud ∈ {0,1}).

    review se cobra c_review y se asume resuelta correctamente (sin fn/fp).
    """
    actions = np.asarray(actions)
    is_fraud = np.asarray(is_fraud).astype(bool)

    approved = actions == APPROVE
    denied = actions == DENY
    reviewed = actions == REVIEW

    fn = int(np.sum(approved & is_fraud))     # aprobé un fraude
    fp = int(np.sum(denied & ~is_fraud))      # rechacé a un legítimo
    n_review = int(np.sum(reviewed))

    total = fn * costs.c_fn + fp * costs.c_fp + n_review * costs.c_review
    n = len(actions)
    return {
        "total_cost": total,
        "cost_per_case": total / n if n else 0.0,
        "false_negatives": fn,
        "false_positives": fp,
        "n_review": n_review,
        "review_rate": n_review / n if n else 0.0,
        "n": n,
    }
