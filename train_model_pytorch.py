"""
ARABA GÖVDE SINIFI TAHMINI - FULL EĞİTİM
PyTorch + ResNet50 Transfer Learning
Metrikleri: F1-Score (Primary), Accuracy, Precision, Recall, Confusion Matrix
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from car_dataset import CarDataset
import torchvision.transforms as transforms
import torchvision.models as models
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from class_config import (
    get_canonical_classes,
    print_split_distribution,
    compute_class_weights,
    sampler_weights_for_targets,
)
from transforms_config import (
    build_train_transform,
    build_eval_transform,
    DEFAULT_TRAIN_PREPROCESS,
    PREPROCESS_STRETCH,
    PREPROCESS_LETTERBOX,
)

# Yeni egitim: letterbox. Mevcut checkpoint stretch ise app otomatik stretch kullanir.
PREPROCESS_MODE = DEFAULT_TRAIN_PREPROCESS
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    f1_score,
    accuracy_score,
    precision_score,
    recall_score
)

print("\n" + "="*80)
print("🚗 ARABA GÖVDE SINIFI - FULL EĞİTİM (PyTorch + ResNet50)")
print("="*80)

# ============================================================================
# KOMUT SATIRI PARAMETRELERI
# ============================================================================

parser = argparse.ArgumentParser(description='Araba Gövde Sınıflandırması')
parser.add_argument('--device', type=str, choices=['cuda', 'rocm', 'cpu'], default=None,
                    help='Kullanılacak device (cuda/rocm/cpu, default: otomatik)')
args = parser.parse_args()

# ============================================================================
# AYARLAR
# ============================================================================

# Device seçimi
if args.device == 'dml' or args.device == 'rocm':
    # AMD DirectML / ROCm GPU (Windows'ta RX 6700XT için DirectML devreye girer)
    try:
        import torch_directml
        device = torch_directml.device()
        print(f"\n💻 Device: AMD GPU (DirectML)")
        print(f"   ⚡ Eğitim hızlandırılıyor (~20-30x daha hızlı)")
    except ImportError:
        # Eğer DirectML yüklenmediyse güvenli liman olarak CPU'ya döner
        device = torch.device('cpu')
        print(f"\n⚠️  torch-directml bulunamadı! CPU moduna geçiliyor...")
elif args.device == 'cuda':
    # NVIDIA CUDA GPU
    device = torch.device('cuda')
    if not torch.cuda.is_available():
        print("⚠️  NVIDIA CUDA istendi ama CUDA mevcut değil. CPU'ya geçiliyor...")
        device = torch.device('cpu')
    else:
        print(f"\n💻 Device: NVIDIA CUDA GPU")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   ⚡ Eğitim hızlandırılıyor (~20-30x daha hızlı)")
elif args.device == 'cpu':
    device = torch.device('cpu')
    print(f"\n💻 Device: CPU")
    print("   ℹ️  CPU modu seçildi (yavaş ama güvenli)")
else:
    # Otomatik seçim
    try:
        import torch_directml
        device = torch_directml.device()
        print(f"\n💻 Device: AMD GPU (DirectML - Otomatik Seçim)")
    except ImportError:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\n💻 Device: {device}")
        if device.type == 'cuda':
            try:
                print(f"   GPU: {torch.cuda.get_device_name(0)}")
                print(f"   ⚡ Eğitim hızlandırılıyor (~20-30x daha hızlı)")
            except:
                pass

IMAGE_SIZE = 224
BATCH_SIZE = 16  # 32 → 16 (daha sık update)
EPOCHS = 100  # 50 → 100 (daha derin eğitim)
LEARNING_RATE = 0.001  # Head layer için
LEARNING_RATE_BASE = 0.0001  # Base layer için (10x daha küçük)
PATIENCE = 12  # macro F1 ile izle; val loss dalgalanmasına tolerans
LABEL_SMOOTHING = 0.05

# ============================================================================
# DATASET SINIFI
# ============================================================================

# ============================================================================
# VERI VE TRANSFORMS
# ============================================================================

print("\n📊 Veri hazırlanıyor...")

CLASSES = get_canonical_classes()
NUM_CLASSES = len(CLASSES)

print(f"  ✓ Sınıflar ({NUM_CLASSES}): {', '.join(CLASSES)}")
print_split_distribution("train", "train")
print_split_distribution("valid", "valid")
print_split_distribution("test", "test")

train_transform = build_train_transform(PREPROCESS_MODE)
val_transform = build_eval_transform(PREPROCESS_MODE)
print(f"  ✓ On-isleme: {PREPROCESS_MODE} (train/eval ayni)")

# Datasets
train_ds = CarDataset('train', CLASSES, train_transform, split_name='train')
val_ds = CarDataset('valid', CLASSES, val_transform, split_name='valid')
test_ds = CarDataset('test', CLASSES, val_transform, split_name='test')

class_weights = compute_class_weights(train_ds.targets, NUM_CLASSES, device=device)
sample_weights = sampler_weights_for_targets(train_ds.targets, class_weights)
train_sampler = WeightedRandomSampler(
    sample_weights,
    num_samples=len(sample_weights),
    replacement=True,
)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    sampler=train_sampler,
    shuffle=False,
    num_workers=0,
)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"\n  ✓ Train: {len(train_ds)} resim")
print(f"  ✓ Val: {len(val_ds)} resim")
print(f"  ✓ Test: {len(test_ds)} resim")

print(f"\n  ✓ Class Weights (log + recall-focus boost): {class_weights.cpu().numpy().round(3)}")
print("  ✓ Train sampler: WeightedRandomSampler (az örnekli sınıflar daha sık görülür)")

# ============================================================================
# MODEL
# ============================================================================

print("\n🧠 Model oluşturuluyor (ResNet50 + Custom Head)...")

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

# Asama 1: ResNet omurgasi donuk; sadece FC head ogrenir (finetune_fast asama 2)
for param in model.parameters():
    param.requires_grad = False

print("  ✓ ResNet50 omurgasi donduruldu (layer3/4 kapali)")

model.fc = nn.Sequential(
    nn.Linear(2048, 1024),
    nn.BatchNorm1d(1024),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(1024, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, NUM_CLASSES),
)

for param in model.fc.parameters():
    param.requires_grad = True

model = model.to(device)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"  ✓ Model ready (egitilebilir: {trainable:,} / {total:,} parametre)")

# ============================================================================
# OPTIMIZER VE CRITERION (sadece FC head)
# ============================================================================

optimizer = optim.Adam(model.fc.parameters(), lr=0.001, weight_decay=1e-3)
optimizer.param_groups[0]['initial_lr'] = 0.001

criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8)
warmup_epochs = 3

# ============================================================================
# TRAIN LOOP
# ============================================================================

print("\n" + "="*80)
print(f"🚀 EĞİTİM BAŞLANIYOR ({EPOCHS} Epoch)...")
print("="*80)

history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': []
}

best_val_f1 = 0.0
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(EPOCHS):
    # ========== TRAIN ==========
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.3f}'})
    
    train_loss_avg = train_loss / len(train_loader)
    train_acc = train_correct / train_total
    history['train_loss'].append(train_loss_avg)
    history['train_acc'].append(train_acc)
    
    # ========== VALIDATION ==========
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    val_preds = []
    val_true = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            val_preds.extend(predicted.cpu().numpy())
            val_true.extend(labels.cpu().numpy())
    
    val_loss_avg = val_loss / len(val_loader)
    val_acc = val_correct / val_total
    val_f1_macro = f1_score(val_true, val_preds, average='macro', zero_division=0)
    history['val_loss'].append(val_loss_avg)
    history['val_acc'].append(val_acc)
    
    # Warmup: ilk epoch'larda LR kademeli artış (optimizer.step SONRASI)
    if epoch < warmup_epochs:
        warmup_factor = (epoch + 1) / warmup_epochs
        for pg in optimizer.param_groups:
            base_lr = pg.get('initial_lr', pg['lr'])
            pg['lr'] = base_lr * (0.1 + 0.9 * warmup_factor)
    else:
        scheduler.step(val_loss_avg)
    
    print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss_avg:.4f} | Train Acc: {train_acc*100:5.2f}% | "
          f"Val Loss: {val_loss_avg:.4f} | Val Acc: {val_acc*100:5.2f}% | Val F1(macro): {val_f1_macro:.4f}")
    
    # En iyi model: birincil metrik macro F1 (proje gereksinimi)
    if val_f1_macro > best_val_f1:
        best_val_f1 = val_f1_macro
        best_val_loss = val_loss_avg
        patience_counter = 0
        Path('models').mkdir(exist_ok=True)
        torch.save(model.state_dict(), 'models/best_model.pt')
        print(f"  ✓ Best model! (Val F1 macro: {val_f1_macro:.4f}, Val Acc: {val_acc*100:.2f}%)")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\n⏹️  Early Stopping (macro F1 iyileşmedi, patience={PATIENCE})")
            break

# ============================================================================
# EN İYİ MODELI YÜKLE
# ============================================================================

print("\n📂 Best model yükleniyor...")
model.load_state_dict(torch.load('models/best_model.pt', weights_only=False))

# ============================================================================
# TEST VE METRIKLER
# ============================================================================

print("\n" + "="*80)
print("📊 TEST SONUÇLARI VE METRİKLER")
print("="*80)

model.eval()
all_predictions = []
all_targets = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        all_predictions.extend(predicted.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())

all_predictions = np.array(all_predictions)
all_targets = np.array(all_targets)

# F1-Score (Primary - Macro Average)
f1_macro = f1_score(all_targets, all_predictions, average='macro')
f1_weighted = f1_score(all_targets, all_predictions, average='weighted')

# Accuracy
accuracy = accuracy_score(all_targets, all_predictions)

# Precision ve Recall
precision_macro = precision_score(all_targets, all_predictions, average='macro', zero_division=0)
precision_weighted = precision_score(all_targets, all_predictions, average='weighted', zero_division=0)

recall_macro = recall_score(all_targets, all_predictions, average='macro', zero_division=0)
recall_weighted = recall_score(all_targets, all_predictions, average='weighted', zero_division=0)

print(f"\n✓ Accuracy (Genel): {accuracy*100:.2f}%")
print(f"✓ F1-Score (Macro): {f1_macro:.4f}")
print(f"✓ F1-Score (Weighted): {f1_weighted:.4f}")
print(f"✓ Precision (Macro): {precision_macro:.4f}")
print(f"✓ Recall (Macro): {recall_macro:.4f}")

# Classification Report
print("\n" + "="*80)
print("DETAYLI SINIFLAMA RAPORU (Per-Class)")
print("="*80)
print(classification_report(all_targets, all_predictions, target_names=CLASSES, digits=4))

# ============================================================================
# CONFUSION MATRIX
# ============================================================================

cm = confusion_matrix(all_targets, all_predictions)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

print(f"\nConfusion Matrix (Normalized):")
for i, cls in enumerate(CLASSES):
    print(f"  {cls}: {cm_normalized[i]}")

print("\n⚠️  En sık karışan çiftler (test):")
for i, true_cls in enumerate(CLASSES):
    for j, pred_cls in enumerate(CLASSES):
        if i != j and cm[i, j] >= 5:
            print(f"  {true_cls} → {pred_cls}: {cm[i, j]} kez")

# ============================================================================
# GRAFİKLER
# ============================================================================

print("\n📈 Grafikler oluşturuluyor...")

Path('reports').mkdir(exist_ok=True)

# 1. Loss ve Accuracy Grafiği (Büyük, Sunuma Uygun)
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Loss Grafiği
axes[0].plot(history['train_loss'], label='Training Loss', linewidth=3, color='#1f77b4', marker='o', markersize=4)
axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=3, color='#ff7f0e', marker='s', markersize=4)
axes[0].set_xlabel('Epoch', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Loss', fontsize=14, fontweight='bold')
axes[0].set_title('Training & Validation Loss', fontsize=16, fontweight='bold', pad=20)
axes[0].legend(fontsize=12, loc='best')
axes[0].grid(True, alpha=0.3, linewidth=1.5)
axes[0].tick_params(axis='both', which='major', labelsize=11)

# Accuracy Grafiği
axes[1].plot(np.array(history['train_acc']) * 100, label='Training Accuracy', linewidth=3, color='#2ca02c', marker='o', markersize=4)
axes[1].plot(np.array(history['val_acc']) * 100, label='Validation Accuracy', linewidth=3, color='#d62728', marker='s', markersize=4)
axes[1].set_xlabel('Epoch', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
axes[1].set_title('Training & Validation Accuracy', fontsize=16, fontweight='bold', pad=20)
axes[1].legend(fontsize=12, loc='best')
axes[1].grid(True, alpha=0.3, linewidth=1.5)
axes[1].tick_params(axis='both', which='major', labelsize=11)
axes[1].set_ylim([0, 105])

plt.tight_layout()
plt.savefig('reports/training_history.png', dpi=300, bbox_inches='tight')
print("  ✓ reports/training_history.png kaydedildi (300 DPI, 20x8 inch)")
plt.close()

# 2. Confusion Matrix Heatmap (Büyük, Net, Sunuma Uygun)
plt.figure(figsize=(14, 12))
sns.set_style("whitegrid")
ax = sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES,
                cbar_kws={'label': 'Normalized Prediction Rate', 'shrink': 0.8},
                linewidths=1.5, linecolor='gray',
                annot_kws={'size': 11, 'weight': 'bold'})
ax.set_xlabel('Predicted Class', fontsize=14, fontweight='bold', labelpad=10)
ax.set_ylabel('True Class', fontsize=14, fontweight='bold', labelpad=10)
ax.set_title('Confusion Matrix (Normalized) - Test Set', fontsize=16, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(rotation=0, fontsize=12)
plt.tight_layout()
plt.savefig('reports/confusion_matrix.png', dpi=300, bbox_inches='tight')
print("  ✓ reports/confusion_matrix.png kaydedildi (300 DPI, 14x12 inch)")
plt.close()

print("\n📊 Grafikler başarıyla oluşturuldu ve reports/ klasörüne kaydedildi!")

# ============================================================================
# MODEL BİLGİSİ KAYDET - PER-CLASS METRİKS İLE
# ============================================================================

# Per-class metrikleri sklearn'den al
from sklearn.metrics import classification_report as cr_dict
report_dict = cr_dict(all_targets, all_predictions, target_names=CLASSES, output_dict=True, zero_division=0)

# Per-class F1 scores
per_class_metrics = {
    cls: {
        "F1-Score": float(report_dict[cls]["f1-score"]),
        "Precision": float(report_dict[cls]["precision"]),
        "Recall": float(report_dict[cls]["recall"]),
    }
    for cls in CLASSES
}

model_info = {
    "classes": CLASSES,
    "num_classes": NUM_CLASSES,
    "preprocess": PREPROCESS_MODE,
    "input_size": IMAGE_SIZE,
    "test_accuracy": float(accuracy),
    "test_f1_macro": float(f1_macro),
    "test_f1_weighted": float(f1_weighted),
    "test_precision": float(precision_macro),
    "test_recall": float(recall_macro),
    "architecture": "ResNet50 Transfer Learning with Fine-tuning",
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "learning_rate_base": LEARNING_RATE_BASE,
    "epochs_trained": len(history['train_loss']),
    # Per-class metrics
    "per_class_metrics": per_class_metrics,
    "confusion_matrix": cm.tolist()
}

with open('models/model_info.json', 'w') as f:
    json.dump(model_info, f, indent=4)

print("\n  ✓ models/model_info.json kaydedildi (per-class metrics ile)")

# ============================================================================
# TAMAMLANDI
# ============================================================================

print("\n" + "="*80)
print("✅ EĞİTİM TAMAMLANDI!")
print("="*80)

print(f"\n📊 ÖZETLEMESİ:")
print(f"  - Accuracy: {accuracy*100:.2f}%")
print(f"  - F1-Score (Primary): {f1_macro:.4f}")
print(f"  - Epochs: {len(history['train_loss'])}")

print(f"\n📁 Çıktılar:")
print(f"  - models/best_model.pt (Model weights)")
print(f"  - models/model_info.json (Metadata)")
print(f"  - reports/training_history.png (Loss & Accuracy graphs)")
print(f"  - reports/confusion_matrix.png (Confusion Matrix)")

print(f"\n📝 Sonraki Adım: Web arayüzü başlatılıyor")
print(f"  streamlit run app.py")
