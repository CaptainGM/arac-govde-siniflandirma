"""
Train klasorundeki yanlis etiketli gorselleri bulur.
Model, klasor etiketinden farkli ve yuksek guvenle tahmin ediyorsa raporlar.

Kullanim:
  python audit_mislabeled_images.py

Cikti: reports/suspicious_labels.csv
Bu dosyalari kontrol edip dogru klasore tasiyin, sonra yeniden egitin.
"""

import csv
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm

from class_config import CANONICAL_CLASSES, find_class_directory, CANONICAL_TO_PDF_ID
from car_dataset import CarDataset, IMAGE_EXTENSIONS
from transforms_config import build_eval_transform


def get_device():
    try:
        import torch_directml
        return torch_directml.device()
    except ImportError:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(device):
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(2048, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, len(CANONICAL_CLASSES)),
    )
    state = torch.load("models/best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def main():
    device = get_device()
    model = load_model(device)
    transform = build_eval_transform()

    rows = []
    split = "train"

    for true_idx, true_cls in enumerate(CANONICAL_CLASSES):
        cls_dir = find_class_directory(split, true_cls)
        if cls_dir is None:
            continue
        paths = [p for p in cls_dir.glob("*.*") if p.suffix.lower() in IMAGE_EXTENSIONS]
        for path in tqdm(paths, desc=true_cls, leave=False):
            try:
                from car_dataset import CarDataset as CD
                img = CD._open_rgb(path)
                tensor = transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    probs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
                pred_idx = int(probs.argmax())
                conf = float(probs[pred_idx])
                pred_cls = CANONICAL_CLASSES[pred_idx]
                if pred_idx != true_idx and conf >= 0.55:
                    rows.append({
                        "path": str(path),
                        "true_class": true_cls,
                        "predicted_class": pred_cls,
                        "confidence": f"{conf:.3f}",
                        "true_pdf_id": CANONICAL_TO_PDF_ID[true_cls],
                        "pred_pdf_id": CANONICAL_TO_PDF_ID[pred_cls],
                    })
            except Exception as exc:
                rows.append({
                    "path": str(path),
                    "true_class": true_cls,
                    "predicted_class": "ERROR",
                    "confidence": "0",
                    "true_pdf_id": CANONICAL_TO_PDF_ID[true_cls],
                    "pred_pdf_id": "",
                    "error": str(exc),
                })

    out = Path("reports/suspicious_labels.csv")
    out.parent.mkdir(exist_ok=True)
    if rows:
        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f"\nSupheli etiket sayisi: {len(rows)}")
    print(f"Rapor: {out.resolve()}")
    if rows:
        print("\nEn sik karisikliklar (true -> pred):")
        from collections import Counter
        pairs = Counter((r["true_class"], r["predicted_class"]) for r in rows)
        for (t, p), n in pairs.most_common(15):
            print(f"  {t} -> {p}: {n}")


if __name__ == "__main__":
    main()
