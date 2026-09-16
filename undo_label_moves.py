"""
reports/label_moves_log.csv icindeki tasimalari geri alir (dst -> src).

Kullanim:
  python undo_label_moves.py              # dry-run
  python undo_label_moves.py --execute
"""

import argparse
import csv
import shutil
from pathlib import Path

LOG_PATH = Path("reports/label_moves_log.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Dosyalari gercekten tasi")
    args = parser.parse_args()

    if not LOG_PATH.exists():
        print(f"Log bulunamadi: {LOG_PATH}")
        return

    rows = list(csv.DictReader(LOG_PATH.open(encoding="utf-8")))
    print(f"Geri alinacak kayit: {len(rows)}")
    if not rows:
        return

    ok, skip, err = 0, 0, 0
    for row in rows:
        src = Path(row["src"])
        dst = Path(row["dst"])
        # Tasima: dst konumundan eski src konumuna
        if not dst.exists():
            print(f"  ATLA (hedef yok): {dst}")
            skip += 1
            continue
        if src.exists():
            print(f"  ATLA (kaynak zaten var): {src}")
            skip += 1
            continue
        print(f"  {'TASI' if args.execute else 'PLAN'}: {dst} -> {src}")
        if args.execute:
            src.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(dst), str(src))
        ok += 1

    print(f"\n{'Geri alindi' if args.execute else 'Dry-run'}: {ok} | atlandi: {skip} | hata: {err}")
    if not args.execute:
        print("Onay icin: python undo_label_moves.py --execute")


if __name__ == "__main__":
    main()
