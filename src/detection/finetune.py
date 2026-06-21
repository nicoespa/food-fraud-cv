"""Fine-tune REAL de un backbone CNN/ViT sobre el corpus de 3 clases (camino full local).

Entrena por transfer learning (timm), trackea PR-AUC de validación, y exporta la
probabilidad P(fake-damaged) de val y test para que run_real calibre y evalúe con el
mismo `src/evaluation`/`src/decision` que el slice. Requiere `uv sync --extra real` + MPS/GPU.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.corpus import FAKE_ID

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_transforms(image_size: int):
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _dataset(df: pd.DataFrame, root: Path, tfm):
    import torch
    from PIL import Image

    class _DS(torch.utils.data.Dataset):
        def __len__(self): return len(df)

        def __getitem__(self, i):
            row = df.iloc[i]
            img = Image.open(root / row["path"]).convert("RGB")
            return tfm(img), int(row["label_id"])

    return _DS()


def build_model(name: str, num_classes: int = 3):
    import timm
    return timm.create_model(name, pretrained=True, num_classes=num_classes)


def train(df: pd.DataFrame, root: str | Path, cfg: dict, device: str | None = None) -> dict:
    import torch
    from sklearn.metrics import average_precision_score
    from torch.utils.data import DataLoader

    root = Path(root)
    dcfg = cfg["detection"]
    name = dcfg.get("backbone", "resnet50")
    epochs = dcfg.get("epochs", 8)
    bs = dcfg.get("batch_size", 32)
    lr = dcfg.get("lr", 3e-4)
    image_size = cfg["data"]["image_size"]
    classes = cfg["data"]["classes"]
    num_classes = len(classes)
    fake_id = classes.index("fake-damaged")
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    tfm = _build_transforms(image_size)
    splits = {s: df[df["split"] == s].reset_index(drop=True) for s in ("train", "val", "test")}
    loaders = {s: DataLoader(_dataset(d, root, tfm), batch_size=bs, shuffle=(s == "train"))
               for s, d in splits.items()}

    model = build_model(name, num_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    def prob_fake(loader) -> np.ndarray:
        model.eval()
        out = []
        with torch.no_grad():
            for x, _ in loader:
                p = model(x.to(device)).softmax(1)[:, fake_id]
                out.append(p.cpu().numpy())
        return np.concatenate(out)

    best_ap, best_state = -1.0, None
    y_val = (splits["val"]["label_id"].to_numpy() == fake_id).astype(int)
    for ep in range(epochs):
        model.train()
        for x, y in loaders["train"]:
            opt.zero_grad()
            loss = loss_fn(model(x.to(device)), y.to(device))
            loss.backward()
            opt.step()
        ap = average_precision_score(y_val, prob_fake(loaders["val"]))
        print(f"  epoch {ep + 1}/{epochs}  val PR-AUC(fake)={ap:.3f}")
        if ap > best_ap:
            best_ap = ap
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "val_prob_fake": prob_fake(loaders["val"]),
        "val_is_fraud": y_val,
        "test_prob_fake": prob_fake(loaders["test"]),
        "test_is_fraud": (splits["test"]["label_id"].to_numpy() == fake_id).astype(int),
        "df_test": splits["test"],
        "backbone": name,
        "best_val_pr_auc": float(best_ap),
        "model": model,          # expuesto para evaluación adversarial
        "device": device,
        "fake_id": fake_id,
    }
