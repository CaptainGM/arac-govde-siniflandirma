"""
FAST FINE-TUNE: SGD Optimizer (Adam'dan 10x hızlı)
Sadece 5 epoch - layer3/4/FC'yi iyileştir
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from car_dataset import CarDataset
from class_config import get_canonical_classes, find_class_directory, compute_class_weights
import json
from pathlib import Path
from transforms_config import (
    build_train_transform,
    build_eval_transform,
    resolve_preprocess,
    DEFAULT_TRAIN_PREPROCESS,
)

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
def _load_preprocess_mode():
    info_path = Path("models/model_info.json")
    if info_path.exists():
        with open(info_path, encoding="utf-8") as f:
            return resolve_preprocess(json.load(f))
    return DEFAULT_TRAIN_PREPROCESS

# ============================================================================
# MAIN
# ============================================================================
def main():
    device = get_device()
    preprocess = _load_preprocess_mode()
    train_transform = build_train_transform(preprocess)
    val_transform = build_eval_transform(preprocess)
    print(f"  On-isleme: {preprocess}")

    classes = get_canonical_classes()
    
    print("\n" + "="*70)
    print("🚀 FAST FINE-TUNE (SGD - 10x hızlı)")
    print("="*70)
    print(
        "UYARI: Ana egitimde Val F1(macro) >= 0.88 olmadan calistirma.\n"
        "      Kilitli katmanlar optimizer disinda; sadece layer3/4/fc guncellenir."
    )
    
    # Load model - dünkü hafıza KORUNUYOR
    print("\n📦 Model yükleniyor (dünkü %96.82 hafıza)...")
    model = models.resnet50(pretrained=False)
    
    model.fc = nn.Sequential(
        nn.Linear(2048, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Dropout(0.5),  # train_model_pytorch.py ile AYNI
        
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.4),  # train_model_pytorch.py ile AYNI
        
        nn.Linear(512, len(classes))
    )
    
    checkpoint = torch.load("models/best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint)
    model.to(device)
    
    # BÜTÜN KATMANLARIN KİLİDİNİ AÇIYORUZ (Hafıza bükülmesin diye)
    for param in model.parameters():
        param.requires_grad = True
    
    # Erken katmanları donduran o satırları tamamen TRUE yapıyoruz:
    for param in model.conv1.parameters():
        param.requires_grad = False
    for param in model.bn1.parameters():
        param.requires_grad = False
    for param in model.layer1.parameters():
        param.requires_grad = False
    for param in model.layer2.parameters():
        param.requires_grad = False
    
    model.train()
    print("✓ Model yüklendi")
    print("✓ Frozen: conv1, bn1, layer1, layer2")
    print("🔓 Trainable: layer3, layer4, FC")
    
    train_labels = []
    for class_idx, class_name in enumerate(classes):
        class_path = find_class_directory("train", class_name)
        if class_path is not None:
            n = sum(1 for p in class_path.glob("*.*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"])
            train_labels.extend([class_idx] * n)
    class_weights = compute_class_weights(train_labels, len(classes), device=device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    
    # Sadece acik (requires_grad=True) parametreler - kilitli katmanlar optimizer'da OLMAMALI
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"  Trainable param tensors: {len(trainable_params)}")
    optimizer = optim.SGD(
        trainable_params,
        lr=0.00005,
        momentum=0.9,
        weight_decay=5e-3,
    )
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=False)
    
    # Load data
    print("\n📊 Data yükleniyor...")
    train_dataset = CarDataset("train", classes, transform=train_transform, split_name="train")
    val_dataset = CarDataset("valid", classes, transform=val_transform, split_name="valid")
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    print(f"✓ Train: {len(train_dataset)} images")
    print(f"✓ Val: {len(val_dataset)} images")
    
    # Training - hızlı kısa fine-tune
    print("\n🎯 HIZLI FINE-TUNE (10 EPOCH):")
    print("="*70)
    
    best_loss = float('inf')
    patience = 3
    patience_counter = 0
    
    for epoch in range(40):
        print(f"\n📍 Epoch {epoch+1}/10")
        
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        batch_count = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
            
            batch_count += 1
            # Progress every 50 batches
            if (batch_idx + 1) % 50 == 0:
                avg_loss = train_loss / batch_count
                acc = 100 * train_correct / train_total
                print(f"  Batch {batch_idx+1}/{len(train_loader)} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")
        
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Validation
        print(f"  🔍 Validating...")
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        print(f"  ✅ Epoch {epoch+1}: TrLoss={train_loss:.4f} TrAcc={train_acc:.2f}% | ValLoss={val_loss:.4f} ValAcc={val_acc:.2f}%")
        
        scheduler.step(val_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "models/best_model.pt")
            print(f"  🎉 BEST MODEL SAVED! (ValLoss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏹️  Early stopping at epoch {epoch+1}")
                break
    
    print("\n" + "="*70)
    print("✅ FINE-TUNING COMPLETE!")
    print("💾 Model: models/best_model.pt (updated)")
    print("🚀 App.py'de test etmeye hazır!")
    print("="*70)

if __name__ == "__main__":
    main()
