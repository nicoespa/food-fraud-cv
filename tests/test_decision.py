"""Tests del núcleo de decisión cost-sensitive."""
import numpy as np

from src.decision.policy import APPROVE, DENY, REVIEW, Costs, cost_threshold, decide, realized_cost


def test_cost_threshold_basic():
    # con c_fn alto el umbral de deny baja (somos más agresivos rechazando)
    costs = Costs(c_fn=10.0, c_fp=3.0)
    assert cost_threshold(costs) == 3.0 / 13.0


def test_d1_uses_cost_threshold():
    costs = Costs(c_fn=10.0, c_fp=3.0)
    t = cost_threshold(costs)
    probs = np.array([t - 0.01, t + 0.01])
    actions = decide(probs, costs, policy="D1")
    assert list(actions) == [APPROVE, DENY]


def test_d2_has_review_zone():
    costs = Costs(c_fn=10.0, c_fp=3.0, c_review=1.0)
    # p intermedio donde review es más barato que approve y que deny
    actions = decide(np.array([0.01, 0.5, 0.99]), costs, policy="D2")
    assert list(actions) == [APPROVE, REVIEW, DENY]


def test_d2_no_review_when_review_expensive():
    # si la revisión es carísima, D2 nunca revisa (colapsa a approve/deny)
    costs = Costs(c_fn=10.0, c_fp=3.0, c_review=100.0)
    actions = decide(np.linspace(0, 1, 11), costs, policy="D2")
    assert REVIEW not in set(actions)


def test_realized_cost_counts():
    costs = Costs(c_fn=10.0, c_fp=3.0, c_review=1.0)
    actions = np.array([APPROVE, DENY, REVIEW, APPROVE])
    is_fraud = np.array([1, 0, 1, 0])  # approve+fraude=FN, deny+legit=FP
    r = realized_cost(actions, is_fraud, costs)
    assert r["false_negatives"] == 1
    assert r["false_positives"] == 1
    assert r["n_review"] == 1
    assert r["total_cost"] == 10.0 + 3.0 + 1.0


def test_cost_sensitive_beats_naive_on_imbalanced_costs():
    # Tesis 2: con probabilidades CALIBRADAS y costo asimétrico, la regla de Bayes
    # sensible al costo (D1/D2) no es peor que el umbral ingenuo 0.5 (D0).
    rng = np.random.default_rng(0)
    n = 8000
    probs = rng.beta(1.5, 8.5, n)              # P(fraude) calibrada
    is_fraud = (rng.random(n) < probs).astype(int)
    costs = Costs(c_fn=10.0, c_fp=3.0, c_review=1.0)
    c0 = realized_cost(decide(probs, costs, "D0"), is_fraud, costs)["total_cost"]
    c1 = realized_cost(decide(probs, costs, "D1"), is_fraud, costs)["total_cost"]
    c2 = realized_cost(decide(probs, costs, "D2"), is_fraud, costs)["total_cost"]
    assert c1 <= c0
    assert c2 <= c1
