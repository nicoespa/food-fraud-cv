"""Robustez adversarial — ataque FGSM contra el CNN fine-tuneado.

Un fraudster sofisticado puede perturbar la imagen para evadir el detector. Acá medimos
cuánto cae la TPR sobre los fakes bajo FGSM a distintos epsilon (sobre píxeles, en espacio
normalizado). Requiere torch + el modelo entrenado → corre en Colab/GPU.
"""
from __future__ import annotations

import numpy as np

from src.detection.finetune import IMAGENET_MEAN, IMAGENET_STD


def evaluate_fgsm(model, images, fake_id: int, eps_list=(0.0, 0.005, 0.01, 0.02, 0.03),
                  device: str = "cpu") -> dict:
    """TPR sobre fakes (predecir 'fake') bajo FGSM a cada epsilon.

    `images`: tensor (N,3,H,W) ya normalizado (ImageNet) de imágenes fake-damaged.
    eps está en unidades de píxel [0,1]; se escala por canal según el std de ImageNet.
    """
    import torch
    model.eval()
    x = images.to(device)
    y = torch.full((x.shape[0],), fake_id, dtype=torch.long, device=device)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    loss_fn = torch.nn.CrossEntropyLoss()

    # gradiente del loss respecto a la entrada (una sola backward; FGSM es one-step)
    x_req = x.clone().detach().requires_grad_(True)
    loss = loss_fn(model(x_req), y)
    grad = torch.autograd.grad(loss, x_req)[0].sign()

    out = {}
    for eps in eps_list:
        # eps en píxeles → en espacio normalizado se divide por std
        x_adv = (x + (eps / std) * grad).detach()
        with torch.no_grad():
            pred_fake = model(x_adv).softmax(1)[:, fake_id] >= 0.5
        out[float(eps)] = float(pred_fake.float().mean().cpu())
    return out
