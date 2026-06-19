"""Generación REAL de fake-damaged por difusión (camino a escala).

Requiere `uv sync --extra gen` (diffusers/transformers/accelerate) + GPU/MPS. No se
ejecuta en el slice runnable; reemplaza a `classical.generate_fake` cuando se escala.
Diseño completo en docs/01-data-generation-and-models.md.
"""
from __future__ import annotations


def make_inpaint_pipeline(model_id: str | None = None, device: str = "mps"):
    """Crea un pipeline de inpainting de difusión. Verificar `model_id` en HF antes de usar."""
    try:
        import torch
        from diffusers import AutoPipelineForInpainting
    except ImportError as e:  # pragma: no cover - camino a escala
        raise ImportError("Instalá las deps de generación: `uv sync --extra gen`") from e

    if model_id is None:
        raise ValueError("Pasá un model_id de inpainting verificado en HF (no asumir IDs).")
    pipe = AutoPipelineForInpainting.from_pretrained(model_id, torch_dtype=torch.float16)
    return pipe.to(device)


def generate_fake_diffusion(pipe, image, mask, prompt: str = "moldy spoiled food, green mold"):
    """Edita la región enmascarada para fabricar daño falso (fake-damaged)."""
    return pipe(prompt=prompt, image=image, mask_image=mask).images[0]

# TODO(escala):
#   - segmentación de la comida → máscara (SAM / segmentador off-the-shelf)
#   - varios generadores (familias distintas) + uno held-out
#   - post-proceso (strip EXIF, recompresión) reutilizando classical.jpeg_recompress
