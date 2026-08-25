"""train/valid/test icindeki bozuk gorselleri listeler."""

from pathlib import Path

from class_config import CANONICAL_CLASSES, find_class_directory
from car_dataset import IMAGE_EXTENSIONS, is_readable_image


def scan_split(split_dir):
    print(f"\n=== {split_dir} ===")
    bad = []
    for cls in CANONICAL_CLASSES:
        cls_dir = find_class_directory(split_dir, cls)
        if cls_dir is None:
            continue
        for img in cls_dir.glob("*.*"):
            if img.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if not is_readable_image(img):
                bad.append(img)
                print(f"  BOZUK: {img}")
    print(f"Toplam bozuk: {len(bad)}")
    return bad


if __name__ == "__main__":
    all_bad = []
    for split in ("train", "valid", "test"):
        if Path(split).exists():
            all_bad.extend(scan_split(split))
    print(f"\nGenel toplam bozuk dosya: {len(all_bad)}")
