"""Carga de datos REALES (camino full local).

Fuente: `Densu341/Fresh-rotten-fruit` (HF, 30.4K imgs reales, labels `fresh*`/`rotten*`).
`fresh*` → genuine-undamaged, `rotten*` → genuine-damaged. Los `fake-damaged` se fabrican
editando las `fresh` con difusión real (src/generation/diffusion.py).

Implementación: usamos el **datasets-server de HF por HTTP** (endpoint /rows) en vez de la
librería `datasets` — más liviano (bajamos solo las imágenes que muestreamos) y evita un bug
de decodificación de labels de la librería v5 con este repo. Muestreo desde offsets random
para cubrir tanto clases fresh (al inicio) como rotten (más adelante).
"""
from __future__ import annotations

import io
import random

import requests  # usa certifi → evita el SSL CERTIFICATE_VERIFY_FAILED de urllib en macOS
from PIL import Image

DATASET_ID = "Densu341/Fresh-rotten-fruit"
FOOD101_ID = "ethz/food101"
# Categorías de Food-101 representativas de delivery (comida cocida).
DELIVERY_CATEGORIES = [
    "pizza", "hamburger", "hot_dog", "tacos", "french_fries", "nachos", "lasagna",
    "fried_rice", "dumplings", "spring_rolls", "club_sandwich", "grilled_cheese_sandwich",
    "pulled_pork_sandwich", "chicken_wings", "fish_and_chips", "ramen", "pad_thai",
    "onion_rings", "macaroni_and_cheese", "waffles", "pancakes", "donuts",
]
_ROWS = "https://datasets-server.huggingface.co/rows"
_HEADERS = {"User-Agent": "food-fraud-cv"}


def _get_json(url: str) -> dict:
    r = requests.get(url, headers=_HEADERS, timeout=90)
    r.raise_for_status()
    return r.json()


def _fetch_image(src: str, size: int) -> Image.Image:
    r = requests.get(src, headers=_HEADERS, timeout=90)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB").resize((size, size))


def _rows_url(offset: int, length: int = 100, dataset: str = DATASET_ID, split: str = "train") -> str:
    params = {"dataset": dataset, "config": "default", "split": split,
              "offset": offset, "length": length}
    return _ROWS + "?" + "&".join(f"{k}={requests.utils.quote(str(v))}" for k, v in params.items())


def load_fresh_rotten(n_per_group: int, image_size: int, seed: int = 42):
    """Muestrea n_per_group imágenes `fresh` y `rotten`. Devuelve {"fresh":[...], "rotten":[...]}.

    El dataset está ordenado por label (clases fresh primero, rotten después), así que
    muestreamos cada grupo desde su REGIÓN (mitad baja / mitad alta) para llenarlos de forma
    confiable incluso con n grande. Filtra por el label real, no por la región.
    """
    rng = random.Random(seed)
    first = _get_json(_rows_url(0, 1))
    total = int(first.get("num_rows_total", 30000))
    names = next(f["type"]["names"] for f in first["features"] if f["name"] == "label")

    buckets: dict[str, list] = {"fresh": [], "rotten": []}
    regions = {"fresh": (0, total // 2), "rotten": (total // 2, total)}
    for group, (lo, hi) in regions.items():
        seen: set[int] = set()
        attempts = 0
        while len(buckets[group]) < n_per_group and attempts < 400:
            attempts += 1
            offset = rng.randint(lo, max(lo, hi - 100))
            if offset in seen:
                continue
            seen.add(offset)
            try:
                data = _get_json(_rows_url(offset, 100))
            except Exception:
                continue
            for item in data.get("rows", []):
                row = item["row"]
                if names[row["label"]].startswith(group) and len(buckets[group]) < n_per_group:
                    try:
                        buckets[group].append(_fetch_image(row["image"]["src"], image_size))
                    except Exception:
                        pass
    return buckets


def load_food101_cooked(n: int, image_size: int, seed: int = 42,
                        categories: list[str] | None = None) -> list:
    """Muestrea n imágenes de comida cocida (delivery) reales desde Food-101 (`ethz/food101`).

    Devuelve una lista de PIL.Image (todas `genuine`). Los fakes se fabrican editándolas
    con difusión (src/generation/diffusion.py). Filtra a categorías tipo delivery
    (pizza, hamburguesa, etc.).
    """
    categories = categories or DELIVERY_CATEGORIES
    rng = random.Random(seed)
    first = _get_json(_rows_url(0, 1, dataset=FOOD101_ID))
    total = int(first.get("num_rows_total", 75000))
    names = next(f["type"]["names"] for f in first["features"] if f["name"] == "label")
    target = {i for i, nm in enumerate(names) if nm in set(categories)}

    imgs: list = []
    seen: set[int] = set()
    attempts = 0
    while len(imgs) < n and attempts < 800:
        attempts += 1
        offset = rng.randint(0, max(0, total - 100))
        if offset in seen:
            continue
        seen.add(offset)
        try:
            data = _get_json(_rows_url(offset, 100, dataset=FOOD101_ID))
        except Exception:
            continue
        for item in data.get("rows", []):
            row = item["row"]
            if row["label"] in target and len(imgs) < n:
                try:
                    imgs.append(_fetch_image(row["image"]["src"], image_size))
                except Exception:
                    pass
    return imgs
