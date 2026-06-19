"""Fase 2 — Señales forensics (la huella de manipulación, no el contenido).

Extrae un vector de features por imagen: ELA, energía espectral por bandas (FFT),
residuo de ruido y estadísticas de color. Estas señales capturan la huella del
generador (pico espectral + doble compresión) que distingue fake-damaged de daño real.
Puro numpy/scipy/PIL → corre sin GPU.
"""
from __future__ import annotations

import io

import numpy as np
from matplotlib.colors import rgb_to_hsv
from PIL import Image
from scipy.ndimage import gaussian_filter

# Bandas espectrales FINAS y SIN un "pico" agnóstico a frecuencia: el detector debe
# aprender en qué banda vive la firma de cada generador (no un atajo universal).
N_FFT_BANDS = 6
FEATURE_NAMES = [
    "ela_mean", "ela_std",
    *[f"fft_band{i}" for i in range(N_FFT_BANDS)],
    "noise_std",
    "sat_mean", "val_mean", "val_std",
]


def _ela(arr: np.ndarray, quality: int = 90) -> tuple[float, float]:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    recompressed = np.array(Image.open(buf).convert("RGB")).astype(np.float32)
    diff = np.abs(arr.astype(np.float32) - recompressed).mean(axis=2)
    return float(diff.mean()), float(diff.std())


def _fft_bands(gray: np.ndarray, n_bands: int = N_FFT_BANDS) -> np.ndarray:
    """Energía espectral media por banda radial (la firma del generador cae en una banda)."""
    f = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.log1p(np.abs(f))
    h, w = gray.shape
    cy, cx = h / 2, w / 2
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r = r / r.max()
    edges = np.linspace(0, 1, n_bands + 1)
    return np.array([mag[(r >= edges[i]) & (r < edges[i + 1])].mean() for i in range(n_bands)])


def extract_features(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr).astype(np.uint8)
    gray = arr.mean(axis=2)
    ela_mean, ela_std = _ela(arr)
    bands = _fft_bands(gray)
    residual = gray - gaussian_filter(gray, sigma=1.0)
    noise_std = float(residual.std())
    hsv = rgb_to_hsv(arr.astype(np.float32) / 255.0)
    feats = [ela_mean, ela_std, *bands.tolist(), noise_std,
             float(hsv[..., 1].mean()), float(hsv[..., 2].mean()), float(hsv[..., 2].std())]
    return np.array(feats, dtype=np.float32)


def extract_batch(images: np.ndarray) -> np.ndarray:
    return np.stack([extract_features(im) for im in images])
