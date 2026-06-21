"""PIPELINE REAL — COMIDA COCIDA (autenticidad binaria) — full local / Colab GPU.

  Food-101 cocido real (pizza/hamburguesa/...) ──► clase `genuine`
  esas fotos ──► difusión multi-generador ──► clase `fake-damaged`
  ──► fine-tune ResNet/ViT + baselines (forensic, zero-shot) + LOGO + adversarial
  ──► evaluación cost-sensitive

Pregunta: ¿la foto del pedido es AUTÉNTICA o manipulada por AI? (no hay dataset público
de comida cocida realmente dañada a escala → ver docs/05). Requiere `uv sync --extra real`.

    uv run --extra real python scripts/run_cooked.py --config configs/cooked.yaml
    uv run --extra real python scripts/run_cooked.py --n 30 --steps 8        # prueba chica
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config  # noqa: E402


def _split_of(sid: int, seed: int, ratios: dict) -> str:
    r = random.Random(f"{seed}-{sid}").random()
    if r < ratios["train"]:
        return "train"
    return "val" if r < ratios["train"] + ratios["val"] else "test"


def build_cooked_corpus(cfg: dict, out_dir: Path, n: int) -> pd.DataFrame:
    from src.data.sources import load_food101_cooked
    from src.generation.diffusion import REAL_GENERATORS, generate_fake

    seed = cfg.get("seed", 42)
    size = cfg["data"]["image_size"]
    ratios = cfg["data"]["split"]
    classes = cfg["data"]["classes"]            # ["genuine", "fake-damaged"]
    steps = cfg["real"].get("steps")
    train_gens = [g for g, c in REAL_GENERATORS.items() if not c["held_out"]]
    held = [g for g, c in REAL_GENERATORS.items() if c["held_out"]]

    print(f"[1/4] descargando comida cocida real de Food-101 (n={n})...")
    imgs = load_food101_cooked(n, image_size=size, seed=seed)
    rows, pipes = [], {}

    def _emit(label, generator, img, sid, tag):
        rel = Path(label) / f"src{sid:05d}_{tag}.jpg"
        (out_dir / label).mkdir(parents=True, exist_ok=True)
        img.save(out_dir / rel, format="JPEG", quality=90)
        rows.append(dict(path=str(rel), label=label, label_id=classes.index(label),
                         generator=generator, source_id=sid, split=_split_of(sid, seed, ratios)))

    print("[2/4] generando fake-damaged con difusión multi-generador...")
    total = 0
    for sid, img in enumerate(imgs):
        sp = _split_of(sid, seed, ratios)
        _emit("genuine", "none", img, sid, "real")
        gens = train_gens + (held if sp == "test" else [])
        for g in gens:
            fake = generate_fake(pipes, img, g, np.random.default_rng(seed + sid * 17 + hash(g) % 1000), size, steps=steps)
            _emit("fake-damaged", g, fake, sid, g)
            total += 1
            if total % 20 == 0:
                print(f"      difusión {total}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "manifest.csv", index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cooked.yaml")
    ap.add_argument("--n", type=int, default=None, help="imágenes cocidas (override n_per_group)")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--reuse-corpus", action="store_true")
    ap.add_argument("--adversarial", action="store_true", help="evaluar robustez FGSM del resnet")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.steps is not None:
        cfg["real"]["steps"] = args.steps
    out_dir = Path(cfg["paths"]["generated"])
    results_dir = Path(cfg["paths"]["results"]); results_dir.mkdir(exist_ok=True)
    classes = cfg["data"]["classes"]
    fake_id = classes.index("fake-damaged")
    holdout = cfg["data"]["holdout_generator"]
    n = args.n or cfg["real"]["n_per_group"]
    target_fpr = cfg["evaluation"]["tpr_at_fpr"]

    manifest = out_dir / "manifest.csv"
    if args.reuse_corpus and manifest.exists():
        df = pd.read_csv(manifest)
    else:
        df = build_cooked_corpus(cfg, out_dir, n)
    print(f"      corpus cocido: {len(df)} imgs | " + df.groupby(['split', 'label']).size().to_string().replace(chr(10), ' | '))

    from src.data.corpus import load_images
    from src.decision.policy import Costs
    from src.detection.baseline import (ForensicClassifier, ZeroShotFreqDetector,
                                        apply_calibrator, fit_calibrator)
    from src.detection.finetune import train as train_backbone
    from src.detection.forensic import extract_batch
    from src.evaluation.logo import leave_one_generator_out
    from src.evaluation.report import (bootstrap_cost_ci, calibration_metrics, decision_costs,
                                       discrimination_metrics, per_generator_breakdown)

    costs = Costs.from_config(cfg)
    tr = (df["split"] == "train").to_numpy()
    va = (df["split"] == "val").to_numpy()
    te = (df["split"] == "test").to_numpy()
    is_fraud = (df["label_id"].to_numpy() == fake_id).astype(int)
    df_te = df[te].reset_index(drop=True)

    print("[3/4] features forensics + baselines...")
    feats = extract_batch(load_images(df, out_dir))
    zs = ZeroShotFreqDetector().fit_reference(feats[tr])
    fc = ForensicClassifier(seed=cfg.get("seed", 42), fake_id=fake_id).fit(feats[tr], df["label_id"].to_numpy()[tr])

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
    row_vit, _ = _row(f"finetune-{ft['backbone']}", ft["val_prob_fake"], ft["test_prob_fake"])
    rows.append(row_vit)

    res = pd.DataFrame(rows).set_index("model")
    pd.set_option("display.width", 200, "display.max_columns", 40, "display.float_format", lambda x: f"{x:.3f}")
    print("\n=== AUTENTICIDAD — COMIDA COCIDA (Food-101 + difusión) ===")
    print(res[["pr_auc", "roc_auc", "tpr_at_fpr", "ece", "cost_D0", "cost_D1", "cost_D2"]].to_string())

    print("\n=== LEAVE-ONE-GENERATOR-OUT (forense; generalización cross-generator) ===")
    logo = leave_one_generator_out(feats, df, costs, seed=cfg.get("seed", 42), fake_id=fake_id)
    print(logo.to_string())

    print("\n=== SISTEMA EN CAPAS: píxel + procedencia + comportamiento → FUSIÓN ===")
    from src.data.sources import load_food101_cooked
    from src.detection.layered import run_layered_demo
    db_images = load_food101_cooked(max(20, n // 5), image_size=cfg["data"]["image_size"], seed=cfg.get("seed", 42) + 1000)
    layered = run_layered_demo(df, feats, fc, db_images, out_dir, cfg, costs, seed=cfg.get("seed", 42))
    print(layered["table"].to_string())
    print(f"(test={layered['n_test']}; de ellos fraude-por-reuso={layered['n_reused_test']}). "
          "El píxel NO atrapa el reuso; la fusión sí.")

    out = {"metrics": rows, "logo": logo.reset_index().to_dict(orient="records"),
           "layered": layered["table"].reset_index().to_dict(orient="records"),
           "n_test": int(te.sum()), "backbone": ft["backbone"], "domain": "cooked (Food-101)"}

    if args.adversarial:
        print("\n=== ROBUSTEZ ADVERSARIAL (FGSM sobre el CNN) ===")
        import torch
        from src.detection.adversarial import evaluate_fgsm
        from src.detection.finetune import _build_transforms
        tfm = _build_transforms(cfg["data"]["image_size"])
        fakes_te = df_te[df_te["label"] == "fake-damaged"]   # atacamos los fakes de test
        imgs_t = torch.stack([tfm(Image.open(out_dir / p).convert("RGB")) for p in fakes_te["path"]])
        adv = evaluate_fgsm(ft["model"], imgs_t, fake_id, device=ft["device"])  # reusa el modelo ya entrenado
        print("TPR sobre fakes a cada epsilon (0=limpio):")
        for eps, tpr in adv.items():
            print(f"  eps={eps:<6} TPR={tpr:.3f}")
        out["adversarial"] = adv

    (results_dir / "experiment_cooked.json").write_text(json.dumps(out, indent=2, default=float))
    res.to_csv(results_dir / "experiment_cooked.csv")
    print(f"\nGuardado en {results_dir}/experiment_cooked.json y .csv")


if __name__ == "__main__":
    main()
