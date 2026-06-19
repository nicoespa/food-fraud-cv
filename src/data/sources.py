"""Carga de datos REALES (camino full local).

Fuente: `Densu341/Fresh-rotten-fruit` (HF, 30.4K imgs reales, labels `fresh*`/`rotten*`).
Usamos `fresh*` → genuine-undamaged y `rotten*` → genuine-damaged. Los `fake-damaged`
se fabrican editando las `fresh` con difusión real (ver src/generation/diffusion.py).

Streaming: solo bajamos las imágenes que muestreamos (no los 3 GB completos).
"""
from __future__ import annotations

import numpy as np
from PIL import Image

DATASET_ID = "Densu341/Fresh-rotten-fruit"


def _to_rgb(img: Image.Image, size: int) -> Image.Image:
    return img.convert("RGB").resize((size, size))


def load_fresh_rotten(n_per_group: int, image_size: int, seed: int = 42):
    """Muestrea n_per_group imágenes `fresh` y `rotten` (streaming).

    Devuelve dict: {"fresh": [PIL,...], "rotten": [PIL,...]}.
    """
    from datasets import load_dataset

    ds = load_dataset(DATASET_ID, split="train", streaming=True)
    try:
        names = ds.features["label"].names
    except Exception:  # fallback al esquema conocido
        names = ["freshapples", "freshbanana", "freshbittergroud", "freshcapsicum",
                 "freshcucumber", "freshokra", "freshoranges", "freshpatato", "freshpotato",
                 "freshtamto", "freshtomato", "rottenapples", "rottenbanana",
                 "rottenbittergroud", "rottencapsicum", "rottencucumber", "rottenokra",
                 "rottenoranges", "rottenpatato", "rottenpotato", "rottentamto", "rottentomato"]

    ds = ds.shuffle(seed=seed, buffer_size=2000)
    buckets: dict[str, list] = {"fresh": [], "rotten": []}
    for ex in ds:
        group = "fresh" if names[ex["label"]].startswith("fresh") else "rotten"
        if len(buckets[group]) < n_per_group:
            buckets[group].append(_to_rgb(ex["image"], image_size))
        if all(len(v) >= n_per_group for v in buckets.values()):
            break
    return buckets
