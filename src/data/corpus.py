"""Fase 1 — Construcción del corpus de 3 clases + splits sin leakage.

Camino RUNNABLE: genera un corpus sintético chico (PIL/numpy) para validar toda la
tubería. El camino REAL (Food-101 + fresh/rotten + difusión) se documenta en
docs/01-data-generation-and-models.md y `src/data/sources.py`.

Regla anti-leakage: cada `source_id` cae en UN solo split (train/val/test). El generador
held-out se produce SOLO para sources del split test → garantiza que ese generador nunca
se vio en train (prueba de generalización cross-generator).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.generation import classical as C

CLASSES = ["genuine-undamaged", "genuine-damaged", "fake-damaged"]
LABEL2ID = {c: i for i, c in enumerate(CLASSES)}
FAKE_ID = LABEL2ID["fake-damaged"]  # clase positiva = fraude


def _assign_source_splits(n_sources: int, ratios: dict, rng: np.random.Generator) -> np.ndarray:
    idx = rng.permutation(n_sources)
    n_train = int(ratios["train"] * n_sources)
    n_val = int(ratios["val"] * n_sources)
    split = np.empty(n_sources, dtype=object)
    split[idx[:n_train]] = "train"
    split[idx[n_train:n_train + n_val]] = "val"
    split[idx[n_train + n_val:]] = "test"
    return split


def build_corpus(cfg: dict, out_dir: str | Path, n_sources: int = 120) -> pd.DataFrame:
    """Construye el corpus, escribe imágenes a out_dir y devuelve el manifest."""
    out_dir = Path(out_dir)
    seed = cfg.get("seed", 42)
    size = cfg["data"]["image_size"]
    holdout = cfg["data"]["holdout_generator"]
    gens = [g["name"] for g in cfg["generation"]["generators"]]
    train_gens = [g for g in gens if g != holdout]
    ratios = cfg["data"]["split"]

    rng = np.random.default_rng(seed)
    source_split = _assign_source_splits(n_sources, ratios, rng)
    rows = []

    for sid in range(n_sources):
        split = source_split[sid]
        base = C.make_base_image(np.random.default_rng(seed + sid), size=size)

        def _emit(label, generator, edit, arr, tag):
            rel = Path(label) / f"src{sid:04d}_{tag}.jpg"
            (out_dir / label).mkdir(parents=True, exist_ok=True)
            C.save_image(arr, out_dir / rel)
            rows.append(dict(path=str(rel), label=label, label_id=LABEL2ID[label],
                             generator=generator, edit=edit, source_id=sid, split=split))

        _emit("genuine-undamaged", "none", "none", base, "clean")
        _emit("genuine-damaged", "none", "real", C.damage_real(base, np.random.default_rng(seed + 7 + sid)), "dmg")
        # generadores vistos en train: para todos los sources (siguen el split del source)
        for g in train_gens:
            fake = C.generate_fake(base, g, np.random.default_rng(seed + 13 + sid))
            _emit("fake-damaged", g, "mold", fake, g)
        # generador held-out: SOLO para sources de test → nunca visto en train
        if split == "test":
            fake = C.generate_fake(base, holdout, np.random.default_rng(seed + 99 + sid))
            _emit("fake-damaged", holdout, "mold", fake, holdout)

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "manifest.csv", index=False)
    return df


def load_images(df: pd.DataFrame, root: str | Path) -> np.ndarray:
    """Carga las imágenes del manifest como array (N,H,W,3) uint8."""
    from PIL import Image
    root = Path(root)
    return np.stack([np.array(Image.open(root / p).convert("RGB")) for p in df["path"]])
