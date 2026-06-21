"""PIPELINE REAL end-to-end (full local) — datos reales + difusión real + fine-tune real.

  fresh/rotten reales (HF Densu341) ──► genuine-undamaged / genuine-damaged
  fresh ──► difusión inpainting (SD 1.5) ──► fake-damaged (varias variantes + held-out)
  ──► fine-tune ResNet/ViT  +  baselines (forensic, zero-shot)  ──► evaluación cost-sensitive

Requiere `uv sync --extra real` + MPS/GPU + disco (~6-8 GB con pesos del modelo).
Lento (la difusión es el cuello de botella) → conviene correrlo en background.

    uv run --extra real python scripts/run_real.py --config configs/real.yaml
    uv run --extra real python scripts/run_real.py --n-per-group 40        # prueba chica
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config  # noqa: E402
from src.data.corpus import FAKE_ID, LABEL2ID  # noqa: E402


def _emit(rows, out_dir, label, generator, edit, source_id, split, img: Image.Image, tag):
    rel = Path(label) / f"src{source_id:05d}_{tag}.jpg"
    (out_dir / label).mkdir(parents=True, exist_ok=True)
    img.save(out_dir / rel, format="JPEG", quality=90)
    rows.append(dict(path=str(rel), label=label, label_id=LABEL2ID[label],
                     generator=generator, edit=edit, source_id=source_id, split=split))


def _split_of(rng, ratios):
    r = rng.random()
    if r < ratios["train"]:
        return "train"
    return "val" if r < ratios["train"] + ratios["val"] else "test"


def build_real_manifest(cfg, out_dir: Path, n_per_group: int) -> pd.DataFrame:
    from src.data.sources import load_fresh_rotten
    from src.generation.diffusion import REAL_GENERATORS, make_inpaint_pipeline, generate_fake

    seed = cfg.get("seed", 42)
    size = cfg["data"]["image_size"]
    ratios = cfg["data"]["split"]
    rng = np.random.default_rng(seed)
    train_gens = [g for g, c in REAL_GENERATORS.items() if not c["held_out"]]
    held = [g for g, c in REAL_GENERATORS.items() if c["held_out"]]

    print(f"[1/4] descargando imágenes reales fresh/rotten (n_per_group={n_per_group})...")
    data = load_fresh_rotten(n_per_group, image_size=size, seed=seed)
    rows = []

    # genuine-damaged = rotten reales
    for i, img in enumerate(data["rotten"]):
        _emit(rows, out_dir, "genuine-damaged", "none", "real-rot", 10_000 + i, _split_of(rng, ratios), img, "rot")

    # genuine-undamaged = fresh reales; guardamos también para derivar fakes
    fresh_meta = []
    for i, img in enumerate(data["fresh"]):
        sp = _split_of(rng, ratios)
        _emit(rows, out_dir, "genuine-undamaged", "none", "real-fresh", i, sp, img, "fresh")
        fresh_meta.append((i, sp, img))

    print(f"[2/4] generando fake-damaged con difusión real (SD 1.5 inpainting)...")
    pipe = make_inpaint_pipeline()
    steps = cfg["real"].get("steps")
    total = sum(len(train_gens) + (1 if sp == "test" else 0) for _, sp, _ in fresh_meta)
    done = 0
    for sid, sp, img in fresh_meta:
        gens = train_gens + (held if sp == "test" else [])  # held-out solo en test
        for g in gens:
            fake = generate_fake(pipe, img, g, np.random.default_rng(seed + sid * 17 + hash(g) % 1000), size, steps=steps)
            _emit(rows, out_dir, "fake-damaged", g, "diffusion", sid, sp, fake, g)
            done += 1
            if done % 20 == 0 or done == total:
                print(f"      difusión {done}/{total}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "manifest.csv", index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/real.yaml")
    ap.add_argument("--n-per-group", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None, help="override pasos de difusión")
    ap.add_argument("--reuse-corpus", action="store_true", help="no regenerar; usar manifest existente")
    ap.add_argument("--generate-only", action="store_true", help="solo generar el corpus y salir")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.steps is not None:
        cfg["real"]["steps"] = args.steps
    out_dir = Path(cfg["paths"]["generated"])
    results_dir = Path(cfg["paths"]["results"]); results_dir.mkdir(exist_ok=True)
    holdout = cfg["data"]["holdout_generator"]
    n_per_group = args.n_per_group or cfg["real"]["n_per_group"]
    target_fpr = cfg["evaluation"]["tpr_at_fpr"]

    manifest = out_dir / "manifest.csv"
    if args.reuse_corpus and manifest.exists():
        df = pd.read_csv(manifest)
    else:
        df = build_real_manifest(cfg, out_dir, n_per_group)
    print(f"      corpus real: {len(df)} imgs | " + df.groupby(['split', 'label']).size().to_string().replace(chr(10), ' | '))
    if args.generate_only:
        print("--generate-only: corpus listo, saliendo. Re-corré con --reuse-corpus para entrenar/evaluar.")
        return

    # ---- modelos: ViT/ResNet fine-tune (real) + baselines forensic / zero-shot ----
    from src.decision.policy import Costs
    from src.detection.baseline import (ForensicClassifier, ZeroShotFreqDetector,
                                         apply_calibrator, fit_calibrator)
    from src.detection.finetune import train as train_backbone
    from src.detection.forensic import extract_batch
    from src.data.corpus import load_images
    from src.evaluation.report import (bootstrap_cost_ci, calibration_metrics, decision_costs,
                                        discrimination_metrics, per_generator_breakdown)

    costs = Costs.from_config(cfg)
    tr = (df["split"] == "train").to_numpy()
    va = (df["split"] == "val").to_numpy()
    te = (df["split"] == "test").to_numpy()
    is_fraud = (df["label_id"].to_numpy() == FAKE_ID).astype(int)
    df_te = df[te].reset_index(drop=True)

    print("[3/4] features forensics + baselines (forensic, zero-shot)...")
    feats = extract_batch(load_images(df, out_dir))
    zs = ZeroShotFreqDetector().fit_reference(feats[tr])
    fc = ForensicClassifier(seed=cfg.get("seed", 42)).fit(feats[tr], df["label_id"].to_numpy()[tr])

    rows = []

    def _row(name, raw_va, raw_te):
        iso = fit_calibrator(raw_va, is_fraud[va])
        p_te = apply_calibrator(iso, raw_te)
        r = {"model": name}
        r.update(discrimination_metrics(is_fraud[te], raw_te, target_fpr))
        r.update(calibration_metrics(is_fraud[te], p_te))
        r.update({f"cost_{k}": v for k, v in decision_costs(p_te, is_fraud[te], costs).items()})
        r.update(per_generator_breakdown(df_te, p_te, costs, holdout))
        return r, p_te

    rows.append(_row("zero-shot-aigc", zs.prob_fake(feats[va]), zs.prob_fake(feats[te]))[0])
    rows.append(_row("forensic-gbm", fc.prob_fake(feats[va]), fc.prob_fake(feats[te]))[0])

    print("[4/4] fine-tune del backbone (real)...")
    ft = train_backbone(df, out_dir, cfg)
    row_vit, p_vit = _row(f"finetune-{ft['backbone']}", ft["val_prob_fake"], ft["test_prob_fake"])
    rows.append(row_vit)

    res = pd.DataFrame(rows).set_index("model")
    pd.set_option("display.width", 180, "display.max_columns", 40, "display.float_format", lambda x: f"{x:.3f}")
    print("\n=== EXPERIMENTO ESTRELLA (DATOS REALES) ===")
    print(res[["pr_auc", "roc_auc", "tpr_at_fpr", "ece", "cost_D0", "cost_D1", "cost_D2"]].to_string())
    gen_cols = [c for c in res.columns if c.startswith(("tpr[", "fpr["))]
    print("\nBreakdown por generador / clase:")
    print(res[gen_cols].to_string())

    ci = bootstrap_cost_ci(p_vit, is_fraud[te], costs, "D0", "D2", iters=cfg["evaluation"]["bootstrap_iters"])
    print(f"\nMejora de costo D0→D2 ({ft['backbone']}): {ci['mean_improvement']:.3f} "
          f"[IC95 {ci['ci95'][0]:.3f}, {ci['ci95'][1]:.3f}]")

    out = {"metrics": rows, "cost_improvement_D0_to_D2": ci, "n_test": int(te.sum()),
           "holdout_generator": holdout, "backbone": ft["backbone"], "data": "real (Densu341 + SD1.5 inpaint)"}
    (results_dir / "experiment_real.json").write_text(json.dumps(out, indent=2, default=float))
    res.to_csv(results_dir / "experiment_real.csv")
    print(f"\nGuardado en {results_dir}/experiment_real.json y .csv")


if __name__ == "__main__":
    main()
