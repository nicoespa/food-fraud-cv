"""Generación REAL de fake-damaged por difusión (inpainting). Camino full local.

Toma una imagen `fresh` real, enmascara una región y la regenera con un prompt de
podrido/moho usando un modelo de difusión REAL (SD 1.5 inpainting). Varias "variantes"
de generador (prompt/máscara/guidance distintos) + un held-out solo en test → permite
medir generalización cross-generator de verdad.

Requiere `uv sync --extra real` + MPS/GPU. Imports perezosos (no rompen el slice liviano).
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

DEFAULT_MODEL = "stable-diffusion-v1-5/stable-diffusion-inpainting"  # verificado en HF
GEN_SIZE = 512  # SD 1.5 inpainting opera en 512x512

# Variantes de generador. `held_out=True` → solo se usa en el split test.
REAL_GENERATORS = {
    "sd-mold": dict(prompt="rotten food covered in green and white mold, decayed, spoiled, fuzzy mold spots",
                    mask="blobs", guidance=7.5, steps=25, held_out=False),
    "sd-rot": dict(prompt="spoiled rotten brown mushy decayed food, dark rot, slime",
                   mask="center", guidance=9.0, steps=25, held_out=False),
    "sd-fungus": dict(prompt="fuzzy white fungus and mold spores spreading over food, biohazard, decomposed",
                      mask="scatter", guidance=6.0, steps=30, held_out=True),  # HELD-OUT
}
NEG_PROMPT = "fresh, clean, appetizing, high quality, ripe"


def pick_device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def make_inpaint_pipeline(model_id: str = DEFAULT_MODEL, device: str | None = None):
    try:
        import torch
        from diffusers import AutoPipelineForInpainting
    except ImportError as e:  # pragma: no cover
        raise ImportError("Instalá el stack real: `uv sync --extra real`") from e
    device = device or pick_device()
    dtype = torch.float16 if device == "cuda" else torch.float32  # fp32 estable en MPS/CPU
    pipe = AutoPipelineForInpainting.from_pretrained(model_id, torch_dtype=dtype, safety_checker=None)
    pipe.set_progress_bar_config(disable=True)
    return pipe.to(device)


def make_mask(kind: str, rng: np.random.Generator, size: int = GEN_SIZE) -> Image.Image:
    """Máscara (blanco = región a regenerar) que define DÓNDE 'pudrir' la comida."""
    mask = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(mask)
    if kind == "center":
        r = int(size * 0.3)
        c = size // 2
        d.ellipse([c - r, c - r, c + r, c + r], fill=255)
    elif kind == "scatter":
        for _ in range(rng.integers(12, 22)):
            x, y = rng.integers(0, size, 2)
            r = int(rng.integers(size // 20, size // 10))
            d.ellipse([x - r, y - r, x + r, y + r], fill=255)
    else:  # blobs
        for _ in range(rng.integers(3, 6)):
            x, y = rng.integers(size // 5, size * 4 // 5, 2)
            r = int(rng.integers(size // 8, size // 4))
            d.ellipse([x - r, y - r, x + r, y + r], fill=255)
    return mask


def generate_fake(pipe, image: Image.Image, generator: str, rng: np.random.Generator,
                  out_size: int) -> Image.Image:
    """Edita `image` (fresh) → fake-damaged según la variante `generator`."""
    import torch
    cfg = REAL_GENERATORS[generator]
    base = image.convert("RGB").resize((GEN_SIZE, GEN_SIZE))
    mask = make_mask(cfg["mask"], rng)
    seed = int(rng.integers(0, 2**31 - 1))
    gen = torch.Generator(device=pipe.device.type).manual_seed(seed)
    result = pipe(prompt=cfg["prompt"], negative_prompt=NEG_PROMPT, image=base, mask_image=mask,
                  num_inference_steps=cfg["steps"], guidance_scale=cfg["guidance"], generator=gen).images[0]
    return result.resize((out_size, out_size))
