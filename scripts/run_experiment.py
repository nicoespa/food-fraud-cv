"""Fase 1→3 end-to-end — EXPERIMENTO ESTRELLA (slice runnable).

Construye el corpus, extrae features forensics, entrena/calibra los detectores y compara
zero-shot (AIGC genérico) vs forensic en la matriz: discriminación × costo × generador.
Demuestra las dos tesis sin GPU.

    uv run --extra ml python scripts/run_experiment.py --config configs/default.yaml
    uv run --extra ml python scripts/run_experiment.py --n-sources 60 --rebuild   # smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config  # noqa: E402
from src.data.corpus import FAKE_ID, build_corpus, load_images  # noqa: E402
from src.detection.baseline import (  # noqa: E402
    ForensicClassifier, ZeroShotFreqDetector, apply_calibrator, fit_calibrator)
from src.detection.forensic import extract_batch  # noqa: E402
from src.decision.policy import Costs  # noqa: E402
from src.evaluation.report import (  # noqa: E402
    bootstrap_cost_ci, calibration_metrics, decision_costs, discrimination_metrics,
    per_generator_breakdown)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--n-sources", type=int, default=160)
    ap.add_argument("--rebuild", action="store_true", help="regenerar el corpus")
    args = ap.parse_args()

    cfg = load_config(args.config)
    gen_dir = Path(cfg["paths"]["generated"])
    results_dir = Path(cfg["paths"]["results"]); results_dir.mkdir(exist_ok=True)
    target_fpr = cfg["evaluation"]["tpr_at_fpr"]
    holdout = cfg["data"]["holdout_generator"]
    costs = Costs.from_config(cfg)

    # 1. corpus
    manifest = gen_dir / "manifest.csv"
    if args.rebuild or not manifest.exists():
        print(f"[1/4] generando corpus sintético (n_sources={args.n_sources})...")
        df = build_corpus(cfg, gen_dir, n_sources=args.n_sources)
    else:
        df = pd.read_csv(manifest)
    print(f"      corpus: {len(df)} imgs | " + " ".join(f"{k}={v}" for k, v in df['split'].value_counts().items()))

    # 2. features forensics
    print("[2/4] extrayendo features forensics...")
    feats = extract_batch(load_images(df, gen_dir))
    is_fraud = (df["label_id"].to_numpy() == FAKE_ID).astype(int)
    tr = (df["split"] == "train").to_numpy()
    va = (df["split"] == "val").to_numpy()
    te = (df["split"] == "test").to_numpy()
    df_te = df[te].reset_index(drop=True)

    # 3. modelos
    print("[3/4] entrenando y calibrando detectores...")
    zs = ZeroShotFreqDetector().fit_reference(feats[tr])
    fc = ForensicClassifier(seed=cfg.get("seed", 42)).fit(feats[tr], df["label_id"].to_numpy()[tr])

    rows = []
    for model in (zs, fc):
        raw_te = model.prob_fake(feats[te])  # score crudo → métricas de ranking
        iso = fit_calibrator(model.prob_fake(feats[va]), is_fraud[va])
        p_te = apply_calibrator(iso, raw_te)  # prob calibrada → calibración + costo
        row = {"model": model.name}
        row.update(discrimination_metrics(is_fraud[te], raw_te, target_fpr))
        row.update(calibration_metrics(is_fraud[te], p_te))
        row.update({f"cost_{k}": v for k, v in decision_costs(p_te, is_fraud[te], costs).items()})
        row.update(per_generator_breakdown(df_te, p_te, costs, holdout))
        rows.append(row)

    # 4. reporte
    res = pd.DataFrame(rows).set_index("model")
    pd.set_option("display.width", 160, "display.max_columns", 30, "display.float_format", lambda x: f"{x:.3f}")
    print("\n[4/4] === EXPERIMENTO ESTRELLA ===")
    print(res[["pr_auc", "roc_auc", "tpr_at_fpr", "ece", "cost_D0", "cost_D1", "cost_D2"]].to_string())
    gen_cols = [c for c in res.columns if c.startswith(("tpr[", "fpr["))]
    print("\nBreakdown por generador / clase (al umbral de costo):")
    print(res[gen_cols].to_string())

    # significancia: la política de 3 zonas vs umbral ingenuo, para forensic
    p_fc = apply_calibrator(fit_calibrator(fc.prob_fake(feats[va]), is_fraud[va]),
                            fc.prob_fake(feats[te]))
    ci = bootstrap_cost_ci(p_fc, is_fraud[te], costs, "D0", "D2", iters=cfg["evaluation"]["bootstrap_iters"])
    print(f"\nMejora de costo D0→D2 (forensic): {ci['mean_improvement']:.3f} "
          f"[IC95 {ci['ci95'][0]:.3f}, {ci['ci95'][1]:.3f}] por caso")

    out = {"metrics": rows, "cost_improvement_D0_to_D2_forensic": ci,
           "n_test": int(te.sum()), "holdout_generator": holdout}
    (results_dir / "experiment.json").write_text(json.dumps(out, indent=2))
    res.to_csv(results_dir / "experiment.csv")
    print(f"\nGuardado en {results_dir}/experiment.json y .csv")


if __name__ == "__main__":
    main()
