"""Smoke test end-to-end: forensic le gana al detector genérico, y la política
cost-sensitive no es peor que el umbral ingenuo."""
import numpy as np
from sklearn.metrics import average_precision_score

from src.data.corpus import FAKE_ID, build_corpus, load_images
from src.decision.policy import Costs, decide, realized_cost
from src.detection.baseline import ForensicClassifier, ZeroShotFreqDetector
from src.detection.forensic import extract_batch


def _cfg():
    return {
        "seed": 7,
        "data": {"image_size": 64, "holdout_generator": "instruct-edit",
                 "split": {"train": 0.6, "val": 0.2, "test": 0.2}},
        "generation": {"generators": [{"name": "sd-inpaint"}, {"name": "img2img"},
                                       {"name": "instruct-edit"}]},
    }


def test_forensic_beats_zeroshot_and_cost_policy_helps(tmp_path):
    df = build_corpus(_cfg(), tmp_path, n_sources=60)
    feats = extract_batch(load_images(df, tmp_path))
    is_fraud = (df["label_id"].to_numpy() == FAKE_ID).astype(int)
    tr = (df["split"] == "train").to_numpy()
    te = (df["split"] == "test").to_numpy()

    fc = ForensicClassifier(seed=7).fit(feats[tr], df["label_id"].to_numpy()[tr])
    zs = ZeroShotFreqDetector().fit_reference(feats[tr])

    ap_fc = average_precision_score(is_fraud[te], fc.prob_fake(feats[te]))
    ap_zs = average_precision_score(is_fraud[te], zs.prob_fake(feats[te]))
    assert ap_fc > ap_zs          # Tesis 1: forensic > detector genérico
    assert ap_fc > 0.7            # y es razonablemente bueno

    costs = Costs(c_fn=10.0, c_fp=3.0, c_review=1.0)
    p = fc.prob_fake(feats[te])
    c0 = realized_cost(decide(p, costs, "D0"), is_fraud[te], costs)["total_cost"]
    c2 = realized_cost(decide(p, costs, "D2"), is_fraud[te], costs)["total_cost"]
    assert c2 <= c0               # Tesis 2: cost-sensitive con review no es peor
