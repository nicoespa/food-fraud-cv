"""Generación REAL de fake-damaged — MULTI-GENERADOR (varias familias). Camino full local.

Familias de generador (para medir generalización cross-generator de verdad):
  - inpaint   : SD 1.5 inpainting (edita una región enmascarada)        [difusión A]
  - instruct  : InstructPix2Pix (edita por instrucción en lenguaje nat.) [difusión B]
  - classic   : copy-move + overlay con PIL (edición tipo Photoshop/app) [NO-AI]

Cada familia deja huellas distintas → un detector entrenado en unas puede fallar en otra.
Funciona sobre cualquier dominio (frutas o comida cocida). Requiere `uv sync --extra real`
+ GPU/MPS. Imports perezosos (no rompen el slice liviano).
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

INPAINT_MODEL = "stable-diffusion-v1-5/stable-diffusion-inpainting"  # verificado en HF
INSTRUCT_MODEL = "timbrooks/instruct-pix2pix"                        # verificado en HF
GEN_SIZE = 512
NEG_PROMPT = "fresh, clean, appetizing, high quality"

# `held_out=True` se reserva para test en run_real/run_cooked (el sweep LOGO lo ignora).
REAL_GENERATORS = {
    "sd-mold": dict(kind="inpaint", mask="blobs", guidance=7.5, steps=25, held_out=False,
                    prompt="rotten food covered in green and white mold, decayed, spoiled"),
    "sd-rot": dict(kind="inpaint", mask="center", guidance=9.0, steps=25, held_out=False,
                   prompt="spoiled rotten brown mushy decayed food, dark rot, slime"),
    "ip2p-rot": dict(kind="instruct", guidance=7.5, img_guidance=1.5, steps=25, held_out=False,
                     prompt="make the food look rotten, moldy and spoiled"),
    "classic-splice": dict(kind="classic", held_out=False),
    "sd-fungus": dict(kind="inpaint", mask="scatter", guidance=6.0, steps=25, held_out=True,
                      prompt="fuzzy white fungus and mold spores spreading over food, decomposed"),
}


def pick_device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_pipe(kind: str, device: str):
    import torch
    dtype = torch.float16 if device == "cuda" else torch.float32
    if kind == "inpaint":
        from diffusers import AutoPipelineForInpainting
        pipe = AutoPipelineForInpainting.from_pretrained(INPAINT_MODEL, torch_dtype=dtype, safety_checker=None)
    elif kind == "instruct":
        from diffusers import StableDiffusionInstructPix2PixPipeline
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(INSTRUCT_MODEL, torch_dtype=dtype, safety_checker=None)
    else:
        raise ValueError(f"kind sin pipeline: {kind}")
    pipe.set_progress_bar_config(disable=True)
    return pipe.to(device)


def get_pipe(pipes: dict, kind: str, device: str | None = None):
    """Lazy-load + cache de pipelines por familia. `classic` no usa pipeline."""
    if kind == "classic":
        return None
    if kind not in pipes:
        try:
            pipes[kind] = _load_pipe(kind, device or pick_device())
        except ImportError as e:  # pragma: no cover
            raise ImportError("Instalá el stack real: `uv sync --extra real`") from e
    return pipes[kind]


def make_mask(kind: str, rng: np.random.Generator, size: int = GEN_SIZE) -> Image.Image:
    mask = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(mask)
    if kind == "center":
        r = int(size * 0.3); c = size // 2
        d.ellipse([c - r, c - r, c + r, c + r], fill=255)
    elif kind == "scatter":
        for _ in range(rng.integers(12, 22)):
            x, y = rng.integers(0, size, 2); r = int(rng.integers(size // 20, size // 10))
            d.ellipse([x - r, y - r, x + r, y + r], fill=255)
    else:  # blobs
        for _ in range(rng.integers(3, 6)):
            x, y = rng.integers(size // 5, size * 4 // 5, 2); r = int(rng.integers(size // 8, size // 4))
            d.ellipse([x - r, y - r, x + r, y + r], fill=255)
    return mask


def _classic_fake(image: Image.Image, rng: np.random.Generator, out_size: int) -> Image.Image:
    """Edición NO-AI: copy-move de un parche oscurecido + overlay de moho verde (estilo app)."""
    arr = np.array(image.convert("RGB").resize((out_size, out_size))).astype(np.float32)
    h, w, _ = arr.shape
    ps = out_size // 4
    sx, sy = int(rng.integers(0, w - ps)), int(rng.integers(0, h - ps))
    dx, dy = int(rng.integers(0, w - ps)), int(rng.integers(0, h - ps))
    patch = arr[sy:sy + ps, sx:sx + ps] * 0.5 + np.array([60, 45, 30]) * 0.5   # parche podrido
    arr[dy:dy + ps, dx:dx + ps] = patch
    yy, xx = np.mgrid[0:h, 0:w]
    for _ in range(rng.integers(3, 7)):
        cy, cx = rng.integers(0, h), rng.integers(0, w); r = rng.integers(h // 12, h // 6)
        blob = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r ** 2))
        arr = arr * (1 - 0.6 * blob[..., None]) + np.array([90, 120, 70]) * 0.6 * blob[..., None]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def generate_fake(pipes: dict, image: Image.Image, generator: str, rng: np.random.Generator,
                  out_size: int, steps: int | None = None) -> Image.Image:
    """Genera un fake-damaged con la familia del `generator`. `pipes` cachea pipelines."""
    cfg = REAL_GENERATORS[generator]
    kind = cfg["kind"]
    if kind == "classic":
        return _classic_fake(image, rng, out_size)

    import torch
    pipe = get_pipe(pipes, kind)
    n_steps = steps or cfg["steps"]
    base = image.convert("RGB").resize((GEN_SIZE, GEN_SIZE))
    g = torch.Generator(device=pipe.device.type).manual_seed(int(rng.integers(0, 2**31 - 1)))
    if kind == "inpaint":
        out = pipe(prompt=cfg["prompt"], negative_prompt=NEG_PROMPT, image=base,
                   mask_image=make_mask(cfg["mask"], rng), num_inference_steps=n_steps,
                   guidance_scale=cfg["guidance"], generator=g).images[0]
    else:  # instruct
        out = pipe(cfg["prompt"], image=base, num_inference_steps=n_steps,
                   guidance_scale=cfg["guidance"], image_guidance_scale=cfg["img_guidance"],
                   generator=g).images[0]
    return out.resize((out_size, out_size))
