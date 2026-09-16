"""
Harici veri setini train/valid/test klasor yapisina kopyalar.

Ornek:
  python merge_external_dataset.py --source data_import/sprint48 --map-profile sprint48 --split train
  python merge_external_dataset.py --source data_import/extra/SUV --target-class SUV --split train --max-per-class 200
"""

import argparse
import shutil
from pathlib import Path

from class_config import CANONICAL_CLASSES, find_class_directory

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Data Sprint 48 / benzeri klasor adlari
SPRINT48_MAP = {
    "suv": "SUV",
    "sedan": "SEDAN",
    "hatchback": "HATCHBACK",
    "wagon": "STATION_WAGON",
    "estate": "STATION_WAGON",
    "station_wagon": "STATION_WAGON",
    "station wagon": "STATION_WAGON",
    "truck": "PICKUP",
    "pickup": "PICKUP",
    "pick-up": "PICKUP",
    "van": "VAN",
    "muv": "VAN",
    "minivan": "VAN",
    "convertible": "ACIK_TEKERLEK",
    "coupe": "ACIK_TEKERLEK",
    "micro": "MICRO",
    "mini": "MICRO",
    "bus": None,
}

COMPCARS_MAP = {
    "suv": "SUV",
    "crossover": "SUV",
    "sedan": "SEDAN",
    "fastback": "SEDAN",
    "hatchback": "HATCHBACK",
    "estate": "STATION_WAGON",
    "pickup": "PICKUP",
    "mpv": "VAN",
    "minibus": "VAN",
    "convertible": "ACIK_TEKERLEK",
    "sports": "ACIK_TEKERLEK",
    "hardtop convertible": "ACIK_TEKERLEK",
}


def normalize_key(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def map_folder_name(folder_name: str, profile: str) -> str | None:
    key = normalize_key(folder_name)
    table = SPRINT48_MAP if profile == "sprint48" else COMPCARS_MAP if profile == "compcars" else {}
    if key in table:
        return table[key]
    if key.upper() in CANONICAL_CLASSES:
        return key.upper()
    for alias, canonical in table.items():
        if alias in key or key in alias:
            return canonical
    return None


def unique_dest(dest_dir: Path, src: Path) -> Path:
    dest = dest_dir / src.name
    if not dest.exists():
        return dest
    stem, suf = src.stem, src.suffix
    n = 1
    while True:
        cand = dest_dir / f"{stem}_ext{n}{suf}"
        if not cand.exists():
            return cand
        n += 1


def copy_images(src_files, dest_dir: Path, dry_run: bool, max_count: int | None):
    dest_dir.mkdir(parents=True, exist_ok=True)
    existing = sum(1 for p in dest_dir.glob("*.*") if p.suffix.lower() in IMAGE_EXT)
    copied = 0
    for src in src_files:
        if max_count is not None and existing + copied >= max_count:
            break
        dest = unique_dest(dest_dir, src)
        if dry_run:
            print(f"  PLAN {src} -> {dest}")
        else:
            shutil.copy2(src, dest)
        copied += 1
    return copied


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="Kaynak kok veya sinif klasoru")
    parser.add_argument("--split", choices=["train", "valid", "test"], default="train")
    parser.add_argument("--map-profile", choices=["sprint48", "compcars", "identity"], default="identity")
    parser.add_argument("--target-class", type=str, help="Tek klasor kaynagi icin hedef canonical sinif")
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    src = args.source
    if not src.exists():
        print(f"Kaynak yok: {src}")
        return

    # Tek sinif dosyalari: source/*.jpg + --target-class SUV
    if args.target_class:
        canonical = args.target_class.upper()
        if canonical not in CANONICAL_CLASSES:
            print(f"Gecersiz sinif: {canonical}")
            return
        dest_dir = find_class_directory(args.split, canonical)
        if dest_dir is None:
            dest_dir = Path(args.split) / canonical
        files = [p for p in src.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXT]
        n = copy_images(files, dest_dir, args.dry_run, args.max_per_class)
        print(f"{canonical}: {n} dosya -> {dest_dir}")
        return

    # Alt klasorler = siniflar
    total = 0
    for sub in sorted(src.iterdir()):
        if not sub.is_dir():
            continue
        canonical = map_folder_name(sub.name, args.map_profile)
        if canonical is None:
            print(f"  ATLA (eslesme yok): {sub.name}")
            continue
        dest_dir = find_class_directory(args.split, canonical)
        if dest_dir is None:
            dest_dir = Path(args.split) / canonical
        files = [p for p in sub.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXT]
        n = copy_images(files, dest_dir, args.dry_run, args.max_per_class)
        print(f"  {sub.name} -> {canonical}: {n} kopya")
        total += n
    print(f"Toplam: {total} (dry_run={args.dry_run})")


if __name__ == "__main__":
    main()
