from pathlib import Path
import numpy as np


CANONICAL_CLASSES = [
    "SUV",
    "VAN",
    "STATION_WAGON",
    "MICRO",
    "ACIK_TEKERLEK",
    "SEDAN",
    "HATCHBACK",
    "PICKUP",
]


CLASS_DIR_ALIASES = {
    "SUV": ["SUV", "suv"],
    "VAN": ["VAN", "van"],
    "STATION_WAGON": ["STATION_WAGON", "STATION WAGON", "station_wagon"],
    "MICRO": ["MICRO", "micro"],
    "ACIK_TEKERLEK": ["ACIK_TEKERLEK", "AÇIK_TEKERLEK", "acik_tekerlek"],
    "SEDAN": ["SEDAN", "Sedan", "sedan"],
    "HATCHBACK": ["HATCHBACK", "Hatchback", "hatchback"],
    "PICKUP": ["PICKUP", "Pick-Up", "PICK-UP", "pickup", "pick-up"],
}


def get_canonical_classes():
    return list(CANONICAL_CLASSES)



CANONICAL_TO_PDF_ID = {cls: i + 1 for i, cls in enumerate(CANONICAL_CLASSES)}
PDF_ID_TO_CANONICAL = {i + 1: cls for i, cls in enumerate(CANONICAL_CLASSES)}


def find_class_directory(base_dir, canonical_class):
    
    base_path = Path(base_dir)
    for alias in CLASS_DIR_ALIASES.get(canonical_class, [canonical_class]):
        candidate = base_path / alias
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None



RECALL_FOCUS_BOOST = {
    "MICRO": 1.0,
    "STATION_WAGON": 1.0,
    "SEDAN": 1.0,
    "SUV": 1.0,
    "PICKUP": 1.0,
    "HATCHBACK": 1.0,
    "VAN": 1.0,
    "ACIK_TEKERLEK": 1.0,
}
MAX_CLASS_WEIGHT = 2.5
MIN_CLASS_WEIGHT = 0.4


def count_images_per_class(split_dir):
    
    counts = {}
    for cls in CANONICAL_CLASSES:
        cls_dir = find_class_directory(split_dir, cls)
        if cls_dir is None:
            counts[cls] = 0
        else:
            counts[cls] = sum(
                1 for p in cls_dir.glob("*.*")
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            )
    return counts


def print_split_distribution(split_name, split_dir):
    counts = count_images_per_class(split_dir)
    total = sum(counts.values()) or 1
    print(f"\n  [{split_name}] sınıf dağılımı:")
    for cls, n in counts.items():
        print(f"    {cls:16s}: {n:4d} ({100 * n / total:5.1f}%)")
    return counts


def compute_class_weights(targets, num_classes, device=None):
    
    import torch

    counts = np.bincount(np.array(targets, dtype=np.int64), minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    total = counts.sum()
    weights = np.log(total / counts)
    weights = weights / weights.sum() * num_classes

    for idx, cls in enumerate(CANONICAL_CLASSES):
        weights[idx] *= RECALL_FOCUS_BOOST.get(cls, 1.0)

    weights = np.clip(weights, MIN_CLASS_WEIGHT, MAX_CLASS_WEIGHT)

    tensor = torch.tensor(weights, dtype=torch.float32)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def sampler_weights_for_targets(targets, class_weight_tensor):
    
    import torch

    cw = class_weight_tensor.detach().cpu().numpy()
    return [float(cw[t]) for t in targets]
