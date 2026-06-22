"""Robustez adversarial — ataque FGSM contra el CNN fine-tuneado.

Un fraudster sofisticado puede perturbar la imagen para evadir el detector. Acá medimos
cuánto cae la TPR sobre los fakes bajo FGSM a distintos epsilon (sobre píxeles, en espacio
normalizado). Requiere torch + el modelo entrenado → corre en Colab/GPU.
"""
from __future__ import annotations

import numpy as np

from src.detection.finetune import IMAGENET_MEAN, IMAGENET_STD


def evaluate_fgsm(model, images, fake_id: int, eps_list=(0.0, 0.005, 0.01, 0.02, 0.04),
                  device: str = "cpu", target_id: int | None = None) -> dict:
    """TPR sobre fakes (predecir 'fake') bajo FGSM **dirigido** a la clase genuina.

    Ataque realista de evasión: el fraudster quiere que su fake sea clasificado como genuino.
    Hacemos descenso de gradiente hacia la clase `target_id` (genuina) → la detección debería
    caer monótonamente. `images`: tensor (N,3,H,W) normalizado (ImageNet) de fakes. eps en píxel.
    """
    import torch
    model.eval()
    if target_id is None:
        target_id = 0 if fake_id != 0 else 1   # una clase genuina
    x = images.to(device)
    y_target = torch.full((x.shape[0],), target_id, dtype=torch.long, device=device)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    loss_fn = torch.nn.CrossEntropyLoss()

    x_req = x.clone().detach().requires_grad_(True)
    loss = loss_fn(model(x_req), y_target)
    grad = torch.autograd.grad(loss, x_req)[0].sign()

    out = {}
    for eps in eps_list:
        x_adv = (x - (eps / std) * grad).detach()   # descenso → empuja hacia 'genuino'
        with torch.no_grad():
            pred_fake = model(x_adv).softmax(1)[:, fake_id] >= 0.5
        out[float(eps)] = float(pred_fake.float().mean().cpu())
    return out
