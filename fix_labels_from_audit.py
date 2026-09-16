"""
suspicious_labels.csv raporuna gore train klasorunde etiket duzeltme.

KURAL (onemli):
  Satir: true_class -> predicted_class
  Anlami: Dosya su an train/TRUE/ altinda ama model PREDICT diyor.
  Islem: train/TRUE/dosya.jpg  -->  train/PRED/ klasorune TASI

Ornek:
  PICKUP,SUV  => train/Pick-Up/xxx.jpg -> train/SUV/xxx.jpg

Guvenli kullanim:
  1) Once dry-run (varsayilan)
  2) Yuksek guven + az sayida cift ile basla
  3) Sonra yeniden egit

Ornekler:
  python fix_labels_from_audit.py
  python fix_labels_from_audit.py --min-confidence 0.80 --pairs PICKUP,SUV
  python fix_labels_from_audit.py --execute --min-confidence 0.85 --pairs PICKUP,SUV SEDAN,HATCHBACK
"""

import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path

from class_config import find_class_directory

REPORT_PATH = Path("reports/suspicious_labels.csv")
TRAIN_ROOT = Path("train")

# Ilk turda sadece cok tekrarlayan ve net karisikliklar (hepsini birden tasima)
RECOMMENDED_PAIRS = [
    ("PICKUP", "SUV"),
    ("SEDAN", "HATCHBACK"),
    ("SEDAN", "SUV"),
    ("PICKUP", "HATCHBACK"),
]


def parse_pairs(pair_args):
    pairs = set()
    for item in pair_args or []:
        a, b = item.split(",", 1)
        pairs.add((a.strip().upper(), b.strip().upper()))
    return pairs


def unique_dest(dest_dir: Path, filename: str) -> Path:
    dest = dest_dir / filename
    if not dest.exists():
        return dest
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    n = 1
    while True:
        candidate = dest_dir / f"{stem}_moved{n}{suffix}"
        if not candidate.exists():
            return candidate
        n += 1


def main():
    parser = argparse.ArgumentParser(description="Audit CSV'den etiket duzeltme")
    parser.add_argument("--csv", type=Path, default=REPORT_PATH)
    parser.add_argument("--min-confidence", type=float, default=0.75)
    parser.add_argument(
        "--pairs",
        nargs="*",
        help="Ornek: PICKUP,SUV SEDAN,HATCHBACK (bos birakilirsa recommended kullanilir)",
    )
    parser.add_argument(
        "--all-pairs",
        action="store_true",
        help="Tum supheli satirlari uygula (onerilmez, 600+ dosya)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Gercekten tasi (yoksa sadece dry-run)",
    )
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"Hata: {args.csv} bulunamadi. Once: python audit_mislabeled_images.py")
        return

    if args.all_pairs:
        allowed_pairs = None
        print("Mod: TUM supheli ciftler (dikkatli ol)")
    elif args.pairs:
        allowed_pairs = parse_pairs(args.pairs)
        print(f"Mod: Secilen ciftler ({len(allowed_pairs)})")
    else:
        allowed_pairs = set(RECOMMENDED_PAIRS)
        print("Mod: Onerilen ilk tur ciftler (PICKUP->SUV, SEDAN->HATCHBACK, ...)")

    rows = []
    with open(args.csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    planned = []
    skipped = Counter()

    for row in rows:
        try:
            conf = float(row["confidence"])
        except ValueError:
            skipped["invalid_conf"] += 1
            continue

        if conf < args.min_confidence:
            skipped["low_confidence"] += 1
            continue

        true_cls = row["true_class"].strip().upper()
        pred_cls = row["predicted_class"].strip().upper()

        if allowed_pairs is not None and (true_cls, pred_cls) not in allowed_pairs:
            skipped["pair_filtered"] += 1
            continue

        src = Path(row["path"])
        if not src.exists():
            skipped["missing_file"] += 1
            continue

        dst_dir = find_class_directory(TRAIN_ROOT, pred_cls)
        if dst_dir is None:
            skipped["no_dest_dir"] += 1
            continue

        dst = unique_dest(dst_dir, src.name)
        planned.append((src, dst, true_cls, pred_cls, conf))

    print("\n" + "=" * 70)
    print("ETIKET DUZELTME PLANI")
    print("=" * 70)
    print(f"Min guven: {args.min_confidence}")
    print(f"Tasinacak dosya: {len(planned)}")
    print(f"Atlanan: {dict(skipped)}")

    if planned:
        pair_counts = Counter((t, p) for _, _, t, p, _ in planned)
        print("\nBu turda tasinacak ciftler:")
        for (t, p), n in pair_counts.most_common():
            print(f"  {t} -> {p}: {n} dosya  (train/{t}/  ==>  train/{p}/)")

        print("\nOrnek (ilk 8):")
        for src, dst, t, p, c in planned[:8]:
            print(f"  [{c:.2f}] {src.name}")
            print(f"       {t} -> {p}")
            print(f"       {src}")
            print(f"    => {dst}\n")

    if not args.execute:
        print("=" * 70)
        print("DRY-RUN: Hicbir dosya tasinmadi.")
        print("Onayliyorsan:")
        print("  python fix_labels_from_audit.py --execute --min-confidence 0.80 --pairs PICKUP,SUV")
        print("=" * 70)
        return

    print("\nTasima basliyor...")
    moved = 0
    for src, dst, t, p, c in planned:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        moved += 1

    log_path = Path("reports/label_moves_log.csv")
    log_path.parent.mkdir(exist_ok=True)
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["src", "dst", "from_class", "to_class", "confidence"])
        for src, dst, t, p, c in planned:
            w.writerow([str(src), str(dst), t, p, f"{c:.3f}"])

    print(f"\nTamamlandi: {moved} dosya tasindi.")
    print(f"Log: {log_path.resolve()}")
    print("\nSonraki adim:")
    print("  python train_model_pytorch.py --device rocm")


if __name__ == "__main__":
    main()
