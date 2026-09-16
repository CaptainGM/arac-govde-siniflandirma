"""
EVALUATE MODEL: Test set üzerinde metrikleri hesapla
Fine-tuned model'in güncel performance'ını göster
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from car_dataset import CarDataset
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    f1_score, precision_score, recall_score, accuracy_score
)
import numpy as np
from pathlib import Path
from PIL import Image
import json
import matplotlib.pyplot as plt
import seaborn as sns
from class_config import get_canonical_classes, print_split_distribution
from transforms_config import build_eval_transform, resolve_preprocess

# ============================================================================
# DEVICE SETUP
# ============================================================================
def get_device():
    try:
        import torch_directml
        device = torch_directml.device()
        print("✓ Using DirectML (AMD GPU)")
    except:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    return device

# ============================================================================
# TRANSFORMS
# ============================================================================

if Path("models/model_info.json").exists():
    with open("models/model_info.json", encoding="utf-8") as _f:
        _preprocess = resolve_preprocess(json.load(_f))
test_transform = build_eval_transform(_preprocess)
print(f"  On-isleme modu: {_preprocess}")

# ============================================================================
# MAIN
# ============================================================================
def main():
    device = get_device()
    
    classes = get_canonical_classes()
    print_split_distribution("test", "test")
    
    print("\n" + "="*70)
    print("📊 MODEL EVALUATION (Fine-tuned)")
    print("="*70)
    
    # Load model
    print("\n📦 Loading model...")
    model = models.resnet50(pretrained=False)
    
    model.fc = nn.Sequential(
        nn.Linear(2048, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.4),
        
        nn.Linear(512, len(classes))
    )
    
    checkpoint = torch.load("models/best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    
    print("✓ Model loaded")
    
    # Load test data
    print("\n📊 Loading test data...")
    test_dataset = CarDataset("test", classes, transform=test_transform, split_name="test")
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    print(f"✓ Test: {len(test_dataset)} images")
    
    # Evaluate
    print("\n🔍 Evaluating on test set...")
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            if (batch_idx + 1) % 20 == 0:
                print(f"  Batch {batch_idx+1}/{len(test_loader)}")
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    print("\n" + "="*70)
    print("📈 OVERALL METRICS")
    print("="*70)
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    precision_macro = precision_score(all_labels, all_preds, average='macro')
    recall_macro = recall_score(all_labels, all_preds, average='macro')
    
    print(f"\n✅ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"✅ F1-Score (Macro): {f1_macro:.4f}")
    print(f"✅ F1-Score (Weighted): {f1_weighted:.4f}")
    print(f"✅ Precision (Macro): {precision_macro:.4f}")
    print(f"✅ Recall (Macro): {recall_macro:.4f}")
    
    # Per-class metrics
    print("\n" + "="*70)
    print("📊 PER-CLASS METRICS")
    print("="*70)
    
    class_f1 = f1_score(all_labels, all_preds, average=None)
    class_precision = precision_score(all_labels, all_preds, average=None)
    class_recall = recall_score(all_labels, all_preds, average=None)
    
    per_class_metrics = {}
    for idx, class_name in enumerate(classes):
        per_class_metrics[class_name] = {
            "F1-Score": float(class_f1[idx]),
            "Precision": float(class_precision[idx]),
            "Recall": float(class_recall[idx])
        }
        print(f"\n{class_name}:")
        print(f"  F1-Score: {class_f1[idx]:.4f}")
        print(f"  Precision: {class_precision[idx]:.4f}")
        print(f"  Recall: {class_recall[idx]:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Save metrics to JSON
    metrics_dict = {
        "preprocess": _preprocess,
        "test_accuracy": float(accuracy),
        "test_f1_macro": float(f1_macro),
        "test_f1_weighted": float(f1_weighted),
        "test_precision": float(precision_macro),
        "test_recall": float(recall_macro),
        "classes": classes,
        "num_classes": len(classes),
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": cm.tolist()
    }
    
    # Save to model_info.json
    with open("models/model_info.json", "w") as f:
        json.dump(metrics_dict, f, indent=2)
    
    print("\n" + "="*70)
    print("✅ METRICS SAVED TO: models/model_info.json")
    print("="*70)
    
    # Print confusion matrix
    print("\n📊 CONFUSION MATRIX:")
    print("\nPredicted →")
    print("Actual ↓")
    print(cm)
    
    # Show confusion pairs (high errors)
    print("\n⚠️  HIGH CONFUSION PAIRS:")
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and cm[i, j] > 5:  # More than 5 misclassifications
                print(f"  {classes[i]} → {classes[j]}: {cm[i, j]} times")
    
    print("\n" + "="*70)
    # evaluate_model.py'nin sonuna (print("🚀 Next...") satırından önce) ekle:
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(12,10)); sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Blues', xticklabels=classes, yticklabels=classes); plt.savefig('reports/confusion_matrix.png', dpi=300)
    print("  ✓ reports/confusion_matrix.png kaydedildi")
    print("🚀 Next: Run app.py to see updated metrics!")
    print("="*70)
   

if __name__ == "__main__":
    main()
