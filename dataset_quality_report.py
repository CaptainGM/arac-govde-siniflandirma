"""
Train/valid klasor etiketleri ile model tahminini karsilastirir.
Yuksek uyumsuzluk = yanlis klasor veya cok zor ornek.

  python dataset_quality_report.py --split train
  python dataset_quality_report.py --split train --export reports/label_noise.csv
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from tqdm import tqdm

from class_config import CANONICAL_CLASSES, find_class_directory
from transforms_config import build_eval_transform, resolve_preprocess

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_model(device, num_classes=8):
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
        nn.Linear(512, num_classes),
    )
    state = torch.load("models/best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "valid", "test"])
    parser.add_argument("--min-confidence", type=float, default=0.65)
    parser.add_argument("--max-per-class", type=int, default=None, help="Hiz icin sinif basina limit")
    parser.add_argument("--export", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device("cpu")
    info = {}
    if Path("models/model_info.json").exists():
        with open("models/model_info.json", encoding="utf-8") as f:
            info = json.load(f)
    preprocess = resolve_preprocess(info)
    transform = build_eval_transform(preprocess)
    print(f"Model preprocess: {preprocess}")

    model = load_model(device)
    confusion = defaultdict(Counter)
    suspicious = []
    per_class_total = Counter()
    per_class_agree = Counter()

    for true_idx, true_cls in enumerate(CANONICAL_CLASSES):
        cls_dir = find_class_directory(args.split, true_cls)
        if cls_dir is None:
            continue
        files = [p for p in cls_dir.iterdir() if p.suffix.lower() in IMAGE_EXT]
        if args.max_per_class:
            files = files[: args.max_per_class]
        for path in tqdm(files, desc=true_cls, leave=False):
            try:
                img = Image.open(path).convert("RGB")
            except OSError:
                continue
            x = transform(img).unsqueeze(0)
            with torch.no_grad():
                logits = model(x)
                prob = torch.softmax(logits, dim=1)[0]
                pred_idx = int(prob.argmax())
                conf = float(prob[pred_idx])
            pred_cls = CANONICAL_CLASSES[pred_idx]
            per_class_total[true_cls] += 1
            confusion[true_cls][pred_cls] += 1
            if pred_cls == true_cls:
                per_class_agree[true_cls] += 1
            elif conf >= args.min_confidence:
                suspicious.append(
                    (str(path), true_cls, pred_cls, f"{conf:.3f}")
                )

    print("\n=== Klasor etiketi vs model (aynı preprocess) ===")
    for cls in CANONICAL_CLASSES:
        t = per_class_total[cls]
        if t == 0:
            continue
        a = per_class_agree[cls]
        print(f"  {cls:16s}  uyum: {a}/{t} ({100*a/t:.1f}%)  top yanlis: {confusion[cls].most_common(3)}")

    print("\n=== Supheli (model emin, etiket farkli) ===")
    print(f"  {len(suspicious)} dosya (conf >= {args.min_confidence})")
    pair_c = Counter((t, p) for _, t, p, _ in suspicious)
    for (t, p), n in pair_c.most_common(10):
        print(f"    {t} -> {p}: {n}")

    if args.export:
        args.export.parent.mkdir(parents=True, exist_ok=True)
        with open(args.export, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["path", "folder_label", "model_pred", "confidence"])
            w.writerows(suspicious)
        print(f"\n  CSV: {args.export}")


if __name__ == "__main__":
    main()
