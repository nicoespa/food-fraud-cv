"""Capa L0 — Procedencia / no-píxel. Cubre el mayor punto ciego del CNN: fraude que NO
edita píxeles (fotos reusadas/robadas, screenshots).

- **Perceptual hash (REAL):** dhash 64-bit + matching contra una base de reclamos previos →
  detecta REUSO (una foto idéntica/casi a una ya vista). Esto el CNN no lo ve.
- **EXIF (REAL):** flag de metadata ausente/strippeada (las apps de edición la borran). En
  el corpus sintético los JPEG no llevan EXIF → poco informativo acá, pero el código es real
  y sirve sobre fotos reales de producción.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ExifTags  # noqa: F401


def dhash(image: Image.Image, hash_size: int = 8) -> int:
    """Difference hash perceptual (64 bits)."""
    im = image.convert("L").resize((hash_size + 1, hash_size))
    a = np.asarray(im, dtype=np.int16)
    diff = a[:, 1:] > a[:, :-1]
    bits = 0
    for v in diff.flatten():
        bits = (bits << 1) | int(v)
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


class ReuseDetector:
    """Base de hashes de reclamos ya vistos → score de reuso de una foto nueva."""

    def fit(self, images) -> "ReuseDetector":
        self.db = [dhash(im) for im in images]
        return self

    def reuse_score(self, image: Image.Image) -> float:
        """1.0 = idéntica a una foto ya vista (reuso); ~0 = nunca vista."""
        if not self.db:
            return 0.0
        h = dhash(image)
        d = min(hamming(h, x) for x in self.db)
        return 1.0 - d / 64.0


def exif_missing(image: Image.Image) -> float:
    """1.0 si la imagen NO tiene EXIF (señal de edición/screenshot); 0.0 si tiene."""
    exif = getattr(image, "getexif", lambda: {})()
    return 0.0 if exif and len(exif) > 0 else 1.0


def provenance_features(image: Image.Image, reuse_detector: ReuseDetector) -> np.ndarray:
    """Vector de features de procedencia: [reuse_score, exif_missing]."""
    return np.array([reuse_detector.reuse_score(image), exif_missing(image)], dtype=np.float32)


PROVENANCE_NAMES = ["reuse_score", "exif_missing"]
