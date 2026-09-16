"""
Proje test/ klasorunden hocanin testdata/1..8 yapisini olusturur.
Kendi bilgisayarinda Predict + True.txt ile entegrasyonu test etmek icin.

Kullanim:
  python prepare_testdata_from_test.py
  python verify_submission.py
"""

from pathlib import Path
import shutil

from class_config import CANONICAL_CLASSES

# Canonical sinif -> hocanin testdata klasor numarasi (1-8)
CANONICAL_TO_TESTDATA_FOLDER = {cls: i + 1 for i, cls in enumerate(CANONICAL_CLASSES)}

SOURCE_SPLIT = Path("test")
TARGET_ROOT = Path("testdata")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_class_dir(split_dir, canonical_class):
    from class_config import find_class_directory
    return find_class_directory(split_dir, canonical_class)


def main():
    if not SOURCE_SPLIT.exists():
        print(f"Hata: {SOURCE_SPLIT} bulunamadi.")
        return

    if TARGET_ROOT.exists():
        print(f"Uyari: {TARGET_ROOT} zaten var; dosyalar uzerine yazilabilir.")

    for folder_id in range(1, 9):
        (TARGET_ROOT / str(folder_id)).mkdir(parents=True, exist_ok=True)

    total = 0
    for cls in CANONICAL_CLASSES:
        folder_id = CANONICAL_TO_TESTDATA_FOLDER[cls]
        src_dir = find_class_dir(SOURCE_SPLIT, cls)
        if src_dir is None:
            print(f"  Atlandi (kaynak yok): {cls}")
            continue
        dst_dir = TARGET_ROOT / str(folder_id)
        count = 0
        for img in src_dir.iterdir():
            if img.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            dst = dst_dir / img.name
            if not dst.exists():
                shutil.copy2(img, dst)
            count += 1
            total += 1
        print(f"  testdata/{folder_id} <- {cls}: {count} gorsel")

    print(f"\nToplam {total} gorsel -> {TARGET_ROOT.resolve()}")
    print("\nSonraki adim:")
    print("  python verify_submission.py")


if __name__ == "__main__":
    main()
