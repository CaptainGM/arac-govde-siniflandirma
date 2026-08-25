"""
RESUME TRAINING: Load best_model.pt ve 10 epoch devam et
Class weights iyileştir: MICRO:1.8, HATCHBACK:1.5
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
import json
from class_config import get_canonical_classes, find_class_directory, compute_class_weights
from transforms_config import build_train_transform, build_eval_transform

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
# TRANSFORMS (minimal aug for fine-tune)
# ============================================================================
train_transform = build_train_transform()
val_transform = build_eval_transform()

# ============================================================================
# MAIN
# ============================================================================
def main():
    device = get_device()
    
    
    classes = get_canonical_classes()
    
    # Load existing model
    print("\n📦 Loading pre-trained model from best_model.pt...")
    # Ham modeli boş olarak oluşturuyoruz (pretrained=False = ImageNet'i INDIRMIYORUZ!)
    model = models.resnet50(pretrained=False)
    
    # Dünkü 10 sınıflı yapına uygun hale getiriyoruz (Sequential FC Layer - dünkü ile birebir aynı)
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
    
    # ŞİMDİ dünkü efsanevi %96.82'lik beyni üzerine yüklüyoruz (model hafızası korunuyor!)
    checkpoint = torch.load("models/best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint)
    model.to(device)
    
    # Sadece ust katmanlar (finetune_fast ile ayni strateji - tum modeli acma!)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer3.parameters():
        param.requires_grad = True
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True

    model.train()
    print("✓ Model yuklendi (trainable: layer3, layer4, fc)")
    
    train_labels = []
    for class_idx, class_name in enumerate(classes):
        class_path = find_class_directory("train", class_name)
        if class_path is not None:
            n = sum(1 for p in class_path.glob("*.*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"])
            train_labels.extend([class_idx] * n)
    class_weights = compute_class_weights(train_labels, len(classes), device=device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(
        trainable_params,
        lr=0.00005,
        momentum=0.9,
        weight_decay=5e-3,
    )
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=False)
    
    # Load data
    print("📊 Loading data...")
    train_dataset = CarDataset("train", classes, transform=train_transform, split_name="train")
    val_dataset = CarDataset("valid", classes, transform=val_transform, split_name="valid")
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    print(f"✓ Train: {len(train_dataset)}")
    print(f"✓ Val: {len(val_dataset)}")
    
    # Training loop - resume for 10 epochs
    print("\n🚀 RESUME TRAINING (10 epochs with improved class weights)...")
    print("=" * 70)
    
    best_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(10):
        model.train()
        train_loss = 0
        train_acc = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_acc += (predicted == labels).sum().item()
            total += labels.size(0)
        
        train_loss /= len(train_loader)
        train_acc = 100 * train_acc / total
        
        # Validation
        model.eval()
        val_loss = 0
        val_acc = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                val_acc += (predicted == labels).sum().item()
                val_total += labels.size(0)
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_acc / val_total
        
        print(f"Epoch {epoch+1:2d} | TrLoss: {train_loss:.4f} TrAcc: {train_acc:.2f}% | "
              f"ValLoss: {val_loss:.4f} ValAcc: {val_acc:.2f}%")
        
        scheduler.step(val_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "models/best_model.pt")
            print(f"  ✓ ✓ ✓ NEW BEST MODEL SAVED!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n✅ Early stopping at epoch {epoch+1}")
                break
    
    print("\n" + "=" * 70)
    print("✅ RESUME TRAINING COMPLETE!")
    print("💾 Updated model: models/best_model.pt")
    print("🚀 Ready to test in app.py")

if __name__ == "__main__":
    main()
