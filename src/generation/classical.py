"""Operaciones de imagen para el corpus (camino RUNNABLE, sin GPU).

Esto es un STAND-IN clásico (PIL/numpy) del pipeline real de difusión, pensado para
validar la metodología end-to-end sin descargar pesos ni GPU. Modela fielmente el
fenómeno que importa:

  - `genuine-damaged`  = daño REAL (manchas, desaturación), SIN huella de manipulación.
  - `fake-damaged`     = daño + **huella de generador** (patrón espectral) + **doble
                         compresión JPEG** → deja artefactos forenses, como una edición
                         por difusión real.

Cada "generador" usa una frecuencia de huella distinta; el held-out usa otra → permite
medir generalización cross-generator de verdad. El camino real de difusión está en
`src/generation/diffusion.py` (requiere `--extra gen` + GPU).
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

# Huella por generador: (frecuencia, familia). instruct-edit (held-out) usa otra
# FAMILIA (grid) → frecuencia y firma espectral no vistas en train.
GENERATOR_FINGERPRINT = {
    "sd-inpaint": (12.0, "diagonal"),
    "img2img": (20.0, "diagonal"),
    "instruct-edit": (28.0, "grid"),
}


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def make_base_image(rng: np.random.Generator, size: int = 96) -> np.ndarray:
    """Imagen procedural tipo 'plato de comida' (RGB uint8)."""
    # fondo de baja frecuencia (gradientes suaves)
    low = rng.random((6, 6, 3))
    bg = np.array(Image.fromarray((low * 120 + 60).astype(np.uint8)).resize((size, size)))
    img = bg.astype(np.float32)
    # 'comida': blobs cálidos
    yy, xx = np.mgrid[0:size, 0:size]
    for _ in range(rng.integers(3, 6)):
        cy, cx = rng.integers(0, size, 2)
        r = rng.integers(size // 8, size // 3)
        color = np.array([rng.integers(150, 240), rng.integers(90, 180), rng.integers(40, 120)])
        bump = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r ** 2))
        img += bump[..., None] * color[None, None, :]
    img += rng.normal(0, 4, img.shape)  # textura
    return np.clip(img, 0, 255).astype(np.uint8)


def _apply_damage(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Daño visual (manchas marrones + desaturación + blur). Compartido por
    genuine-damaged y fake-damaged → el daño NO distingue las clases."""
    size = arr.shape[0]
    out = arr.astype(np.float32)
    yy, xx = np.mgrid[0:size, 0:size]
    for _ in range(rng.integers(2, 5)):
        cy, cx = rng.integers(0, size, 2)
        r = rng.integers(size // 12, size // 6)
        brown = np.array([90, 60, 30])
        blob = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r ** 2))
        out = out * (1 - 0.7 * blob[..., None]) + brown[None, None, :] * 0.7 * blob[..., None]
    gray = out.mean(axis=2, keepdims=True)
    out = out * 0.6 + gray * 0.4  # desaturar
    out = gaussian_filter(out, sigma=(0.5, 0.5, 0))
    return np.clip(out, 0, 255).astype(np.uint8)


def _finalize(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Cadena de compresión (variable, a veces doble). IDÉNTICA para damaged y fake →
    la compresión tampoco distingue las clases (la ELA no es un delator)."""
    out = jpeg_recompress(arr, quality=int(rng.integers(80, 93)))
    if rng.random() < 0.8:
        out = jpeg_recompress(out, quality=int(rng.integers(74, 86)))
    return out


def damage_real(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """genuine-damaged: daño + compresión, SIN huella de generador."""
    return _finalize(_apply_damage(arr, rng), rng)


def _embed_fingerprint(arr: np.ndarray, freq: float, kind: str = "diagonal", strength: float = 6.0) -> np.ndarray:
    """Huella espectral del 'generador'. `kind` distingue FAMILIAS de generador:
    diagonal (sinusoide en x+y) vs grid (sinusoides en ejes). El held-out usa otra
    familia → la generalización cross-generator es genuina, no trucada."""
    size = arr.shape[0]
    yy, xx = np.mgrid[0:size, 0:size]
    if kind == "grid":
        pattern = strength * (np.sin(2 * np.pi * freq * xx / size) + np.sin(2 * np.pi * freq * yy / size))
    else:  # diagonal
        pattern = strength * np.sin(2 * np.pi * freq * (xx + yy) / size)
    return np.clip(arr.astype(np.float32) + pattern[..., None], 0, 255).astype(np.uint8)


def generate_fake(arr: np.ndarray, generator: str, rng: np.random.Generator) -> np.ndarray:
    """fake-damaged: MISMO daño y compresión que genuine-damaged, difiere SOLO en la
    huella espectral del generador. Así la única señal discriminativa es la firma del
    generador → el modelo debe aprenderla por generador y falla en familias no vistas."""
    dmg = _apply_damage(arr, rng)
    freq, kind = GENERATOR_FINGERPRINT[generator]
    dmg = _embed_fingerprint(dmg, freq, kind=kind, strength=float(rng.uniform(1.5, 3.0)))
    return _finalize(dmg, rng)


def jpeg_recompress(arr: np.ndarray, quality: int = 85) -> np.ndarray:
    """Round-trip JPEG en memoria (imita la captura/edición del atacante)."""
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


def save_image(arr: np.ndarray, path) -> None:
    """Guarda como JPEG SIN EXIF (strip de metadata, como una app de edición)."""
    Image.fromarray(arr).save(path, format="JPEG", quality=90)
