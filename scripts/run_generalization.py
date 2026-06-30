"""Generalización a datos NUEVOS — responde "¿es confiable o sobreajusta?".

Dos pruebas fuera de distribución (no es testear contra lo que entrenó):
  1. CROSS-DOMAIN: entrenar en UN dominio (frutas) y testear en OTRO (comida cocida),
     y viceversa. Comida distinta → ¿generaliza?
  2. CNN LEAVE-ONE-GENERATOR-OUT: entrenar el CNN SIN la familia `classic-splice` y
     testear sobre ella (un generador nunca visto). ¿El CNN también colapsa como el forense (0.25)?

Requiere ambos corpus ya generados (corré run_real.py y run_cooked.py antes, o con --reuse-corpus).
    uv run --extra real python scripts/run_generalization.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config  # noqa: E402
from src.data.corpus import load_images  # noqa: E402
from src.detection.baseline import ForensicClassifier  # noqa: E402
from src.detection.forensic import extract_batch  # noqa: E402


def _isf(df):
    return (df["label"] == "fake-damaged").astype(int).to_numpy()


def _forensic(feats_tr, y_tr, feats_te, y_te):
    clf = ForensicClassifier(fake_id=1).fit(feats_tr, y_tr)
    p = clf.prob_fake(feats_te)
    return float(average_precision_score(y_te, p)), float(roc_auc_score(y_te, p))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fruit-config", default="configs/real.yaml")
    ap.add_argument("--cooked-config", default="configs/cooked.yaml")
    ap.add_argument("--skip-cnn", action="store_true", help="solo forense (sin GPU)")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    cf_f, cf_c = load_config(args.fruit_config), load_config(args.cooked_config)
    dir_f, dir_c = Path(cf_f["paths"]["generated"]), Path(cf_c["paths"]["generated"])
    for d in (dir_f, dir_c):
        if not (d / "manifest.csv").exists():
            sys.exit(f"Falta el corpus en {d}. Corré run_real.py y run_cooked.py primero.")

    df_f = pd.read_csv(dir_f / "manifest.csv")
    df_c = pd.read_csv(dir_c / "manifest.csv")
    print(f"frutas: {len(df_f)} imgs · cocido: {len(df_c)} imgs")
    print("extrayendo features forensics de ambos dominios...")
    Xf, Xc = extract_batch(load_images(df_f, dir_f)), extract_batch(load_images(df_c, dir_c))
    yf, yc = _isf(df_f), _isf(df_c)
    tef = (df_f["split"] == "test").to_numpy(); trf = ~tef
    tec = (df_c["split"] == "test").to_numpy(); trc = ~tec

    out = {}

    # ---- FORENSE: in-domain vs cross-domain ----
    print("\n=== FORENSE · generalización (PR-AUC / ROC-AUC) ===")
    rows = []
    ap_id_f, roc_id_f = _forensic(Xf[trf], yf[trf], Xf[tef], yf[tef])
    ap_id_c, roc_id_c = _forensic(Xc[trc], yc[trc], Xc[tec], yc[tec])
    ap_xf2c, roc_xf2c = _forensic(Xf, yf, Xc, yc)   # train frutas → test cocido
    ap_xc2f, roc_xc2f = _forensic(Xc, yc, Xf, yf)   # train cocido → test frutas
    rows = [
        ["in-domain · frutas", ap_id_f, roc_id_f],
        ["in-domain · cocido", ap_id_c, roc_id_c],
        ["cross · frutas→cocido", ap_xf2c, roc_xf2c],
        ["cross · cocido→frutas", ap_xc2f, roc_xc2f],
    ]
    print(f"{'setup':<26}{'PR-AUC':>9}{'ROC-AUC':>10}")
    for n, a, r in rows:
        print(f"{n:<26}{a:>9.3f}{r:>10.3f}")
    out["forensic"] = [{"setup": n, "pr_auc": a, "roc_auc": r} for n, a, r in rows]
    print("Lectura: si cross << in-domain, el forense NO generaliza a comida nueva.")

    if args.skip_cnn:
        (Path(cf_c["paths"]["results"]).mkdir(exist_ok=True), None)
        Path("results/generalization.json").write_text(json.dumps(out, indent=2, default=float))
        print("\n(--skip-cnn) guardado results/generalization.json"); return

    # ---- CNN: cross-domain ----
    from src.detection.finetune import cross_domain_cnn
    print("\n=== CNN (ResNet) · cross-domain ===")
    r1 = cross_domain_cnn(df_f, dir_f, df_c, dir_c, cf_c, epochs=args.epochs)  # train frutas → test cocido
    r2 = cross_domain_cnn(df_c, dir_c, df_f, dir_f, cf_c, epochs=args.epochs)  # train cocido → test frutas
    ap1 = float(average_precision_score(r1["is_fraud"], r1["prob"]))
    ap2 = float(average_precision_score(r2["is_fraud"], r2["prob"]))
    print(f"  train frutas → test cocido : PR-AUC {ap1:.3f}")
    print(f"  train cocido → test frutas : PR-AUC {ap2:.3f}")
    out["cnn_cross"] = {"fruit_to_cooked": ap1, "cooked_to_fruit": ap2, "backbone": r1["backbone"]}

    # ---- CNN: domain mixing (entrenar AMBOS dominios juntos) ----
    print("\n=== CNN · domain mixing (train frutas+cocido) ===")
    fa = df_f.copy(); fa["path"] = [str((dir_f / p).resolve()) for p in df_f["path"]]; fa["dom"] = "frutas"
    ca = df_c.copy(); ca["path"] = [str((dir_c / p).resolve()) for p in df_c["path"]]; ca["dom"] = "cocido"
    train_comb = pd.concat([fa[trf], ca[trc]], ignore_index=True)
    test_comb = pd.concat([fa[tef], ca[tec]], ignore_index=True)
    rc = cross_domain_cnn(train_comb, "/", test_comb, "/", cf_c, epochs=args.epochs)
    dom = test_comb["dom"].to_numpy()
    mix = {}
    for d in ("frutas", "cocido"):
        m = dom == d
        mix[d] = float(average_precision_score(rc["is_fraud"][m], rc["prob"][m]))
        print(f"  train AMBOS → test {d}: PR-AUC {mix[d]:.3f}   (comparar vs cross-domain de arriba)")
    out["cnn_domain_mixing"] = mix

    # ---- CNN: leave-one-generator-out sobre classic-splice (cocido) ----
    print("\n=== CNN · leave-one-generator-out (held-out = classic-splice) ===")
    held = "classic-splice"
    gen = df_c["generator"].to_numpy(); lab = df_c["label"].to_numpy(); spl = df_c["split"].to_numpy()
    tr_mask = ((lab == "genuine") & (spl != "test")) | ((lab == "fake-damaged") & (gen != held))
    te_mask = ((lab == "genuine") & (spl == "test")) | (gen == held)
    if te_mask.sum() > 0 and (gen == held).sum() > 0:
        rl = cross_domain_cnn(df_c[tr_mask], dir_c, df_c[te_mask], dir_c, cf_c, epochs=args.epochs)
        sub = lab[te_mask] == "fake-damaged"  # los fakes del test son todos de classic (held-out)
        thr = 0.5
        tpr_classic = float((rl["prob"][sub] >= thr).mean())
        ap_logo = float(average_precision_score(rl["is_fraud"], rl["prob"]))
        print(f"  CNN sin classic-splice → TPR sobre classic-splice (no visto): {tpr_classic:.3f}  (PR-AUC {ap_logo:.3f})")
        print(f"  (referencia: el forense colapsó a 0.25 en este mismo held-out)")
        out["cnn_logo_classic"] = {"tpr_classic_unseen": tpr_classic, "pr_auc": ap_logo}

    Path(cf_c["paths"]["results"]).mkdir(exist_ok=True)
    Path("results/generalization.json").write_text(json.dumps(out, indent=2, default=float))
    print("\nGuardado en results/generalization.json")


if __name__ == "__main__":
    main()
