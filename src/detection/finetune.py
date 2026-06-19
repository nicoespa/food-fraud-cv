"""Fase 1/2 — Fine-tune de backbones CNN/ViT (camino a escala).

Requiere `uv sync --extra dl` (torch/torchvision/timm) + MPS/GPU. No corre en el slice
runnable. Reemplaza/compite con ForensicClassifier cuando se escala a datos reales.
"""
from __future__ import annotations


def build_model(name: str = "resnet50", num_classes: int = 3):
    """Crea un backbone pre-entrenado con cabeza de `num_classes`."""
    try:
        import timm
    except ImportError as e:  # pragma: no cover - camino a escala
        raise ImportError("Instalá las deps de DL: `uv sync --extra dl`") from e
    return timm.create_model(name, pretrained=True, num_classes=num_classes)


# TODO(escala):
#   - DataLoader sobre el corpus REAL con split generador-disjunto (ver data/corpus.py)
#   - loop de entrenamiento (MPS), early stopping en PR-AUC de validación
#   - calibración (temperature scaling) sobre val
#   - exportar P(fake) de val/test a results/ para el mismo run_evaluate que el slice
def train(*args, **kwargs):  # pragma: no cover - camino a escala
    raise NotImplementedError("Implementar el loop de fine-tune al escalar (uv sync --extra dl).")
