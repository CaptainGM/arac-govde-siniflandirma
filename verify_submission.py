"""
1) PredictionScript.txt icindeki Predict() fonksiyonunu calistirir -> Preds.txt
2) Varsa True.txt ile hocanin Test.txt mantiginda karsilastirir

True.txt formati Preds.txt ile ayni:
  resim.jpg | Pred: 3

Kendi test klasorun icin True.txt olusturmak:
  python prepare_testdata_from_test.py  sonrasi
  testdata/4 icindeki her resim icin dogru etiket 4 (MICRO) vb.
  veya asagidaki generate_true_from_folders.py
"""

from pathlib import Path
import runpy
import sys

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


def read_predictions(file_path):
    data = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" | ")
            filename = parts[0]
            value = int(parts[1].split(":")[1].strip())
            data[filename] = value
    return data


def evaluate_predictions(pred_txt, true_txt):
    class_labels = [1, 2, 3, 4, 5, 6, 7, 8]
    pred_dict = read_predictions(pred_txt)
    true_dict = read_predictions(true_txt)
    common_files = sorted(set(pred_dict.keys()) & set(true_dict.keys()))

    if not common_files:
        print("Hata: Preds.txt ve True.txt arasinda ortak dosya adi yok.")
        return None

    y_true = [true_dict[n] for n in common_files]
    y_pred = [pred_dict[n] for n in common_files]

    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    print("Confusion Matrix (satir=gercek, sutun=tahmin, etiket 1-8):")
    print(cm)

    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"\nAccuracy: {acc:.4f}")
    print(f"Macro F1:   {f1m:.4f}")
    print("\nClassification Report:")
    print(classification_report(
        y_true, y_pred, labels=class_labels,
        target_names=[str(c) for c in class_labels], zero_division=0,
    ))
    return {"accuracy": acc, "f1_macro": f1m}


def generate_true_from_testdata_folders(testdata_dir="testdata"):
    """testdata/1..8 klasor yapisindan True.txt uretir (kendi testin icin)."""
    lines = []
    root = Path(testdata_dir)
    for folder_id in range(1, 9):
        sub = root / str(folder_id)
        if not sub.exists():
            continue
        for img in sorted(sub.iterdir()):
            if img.is_file():
                lines.append(f"{img.name} | Pred: {folder_id} ")
    Path("True.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    print(f"True.txt yazildi ({len(lines)} satir) - klasor numarasi = dogru etiket varsayimi")


def run_predict():
    ns = runpy.run_path("PredictionScript.txt", run_name="prediction_script")
    if "Predict" not in ns:
        raise RuntimeError("PredictionScript.txt icinde Predict() bulunamadi")
    ns["Predict"]("testdata")


def main():
    if not Path("testdata").exists():
        print("testdata/ yok. Once: python prepare_testdata_from_test.py")
        sys.exit(1)

    if not Path("True.txt").exists():
        print("True.txt yok; testdata klasorlerinden uretiliyor...")
        generate_true_from_testdata_folders()

    print("Predict() calistiriliyor...\n")
    run_predict()

    if not Path("Preds.txt").exists():
        print("Hata: Preds.txt olusmadi.")
        sys.exit(1)

    print("\n--- Degerlendirme (hocanin Test.txt mantigi) ---\n")
    evaluate_predictions("Preds.txt", "True.txt")


if __name__ == "__main__":
    main()
