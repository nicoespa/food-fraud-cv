"""Sistema en capas end-to-end: PÍXEL + PROCEDENCIA + COMPORTAMIENTO → FUSIÓN.

Demuestra lo central de producción: la fusión atrapa fraude que el CNN/píxel NO puede.
Inyecta un tipo de fraude **reuso** (foto REAL duplicada de un reclamo histórico): el píxel
la ve genuina (lo es), pero la procedencia (perceptual-hash) la detecta. La fusión combina
ambas señales y gana.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import average_precision_score

from src.data.corpus import load_images
from src.decision.policy import Costs, cost_threshold
from src.detection.behavioral import simulate
from src.detection.forensic import extract_batch
from src.detection.fusion import FusionClassifier, build_fusion_features
from src.detection.provenance import ReuseDetector, exif_missing


def _prov(images, reuse: ReuseDetector) -> np.ndarray:
    return np.stack([[reuse.reuse_score(im), exif_missing(im)] for im in images]).astype(np.float32)


def run_layered_demo(df: pd.DataFrame, feats: np.ndarray, forensic, db_images: list,
                     out_dir, cfg: dict, costs: Costs, seed: int = 42) -> dict:
    """db_images = 'reclamos históricos' (base del detector de reuso), distintos del corpus."""
    size = cfg["data"]["image_size"]
    thr = cost_threshold(costs)
    reuse = ReuseDetector().fit(db_images)

    # --- corpus actual (genuine + fakes-AI): píxel + procedencia + comportamiento ---
    corpus_pil = [Image.fromarray(a) for a in load_images(df, out_dir)]
    prov = _prov(corpus_pil, reuse)
    pixel = forensic.prob_fake(feats)                                   # capa píxel
    is_fraud = (df["label"] == "fake-damaged").to_numpy().astype(int)
    behav = simulate(is_fraud, seed)
    split = df["split"].to_numpy()
    subtype = np.where(is_fraud == 1, "ai", "genuine")

    # --- fraude por REUSO: fotos reales duplicadas de la base histórica ---
    k = min(len(db_images), max(10, len(df) // 8))
    db_sel = db_images[:k]
    db_feats = extract_batch(np.stack([np.array(im.convert("RGB").resize((size, size))) for im in db_sel]))
    rng = np.random.default_rng(seed + 1)
    reused_pixel = forensic.prob_fake(db_feats)                          # el píxel las ve genuinas
    reused_prov = _prov(db_sel, reuse)                                   # reuse_score ≈ 1 (están en la base)
    reused_behav = simulate(np.ones(k, dtype=int), seed + 2)
    reused_split = np.where(rng.random(k) < 0.6, "train", "test")        # algunas en train para que la fusión aprenda

    # --- ensamblar matriz de fusión ---
    X = np.vstack([build_fusion_features(pixel, prov, behav),
                   build_fusion_features(reused_pixel, reused_prov, reused_behav)])
    y = np.concatenate([is_fraud, np.ones(k, dtype=int)])
    sp = np.concatenate([split, reused_split])
    sub = np.concatenate([subtype, np.full(k, "reused")])
    pixel_all = np.concatenate([pixel, reused_pixel])                    # baseline solo-píxel

    tr, te = sp == "train", sp == "test"
    fusion = FusionClassifier(seed=seed).fit(X[tr], y[tr])
    p_fusion = fusion.prob_fraud(X[te])
    p_pixel = pixel_all[te]                                              # baseline: solo CNN/forense

    def _tpr(scores, mask):
        m = mask & (y[te] == 1)
        return float((scores[m] >= thr).mean()) if m.any() else float("nan")

    sub_te = sub[te]
    rows = []
    for name, scores in [("pixel-only (CNN/forense)", p_pixel), ("fusion-layered", p_fusion)]:
        rows.append({
            "model": name,
            "PR_AUC": float(average_precision_score(y[te], scores)),
            "TPR_ai": _tpr(scores, sub_te == "ai"),
            "TPR_reused": _tpr(scores, sub_te == "reused"),
            "FPR_genuine": float((scores[(sub_te == "genuine")] >= thr).mean()) if (sub_te == "genuine").any() else float("nan"),
        })
    table = pd.DataFrame(rows).set_index("model")
    return {"table": table, "n_test": int(te.sum()), "n_reused_test": int(((sub == "reused") & te).sum())}
