"""
Calculate class weights based on actual dataset distribution
Inverse frequency weighting with smoothing
"""

from pathlib import Path
import numpy as np
from class_config import get_canonical_classes, find_class_directory

classes = get_canonical_classes()

print("\n" + "="*70)
print("📊 CLASS DISTRIBUTION IN TRAIN SET")
print("="*70)

class_counts = []
total_images = 0

for cls_name in classes:
    class_path = find_class_directory("train", cls_name)
    if class_path is None:
        class_counts.append(0)
        print(f"{cls_name:20} : {0:5} images")
        continue
    count = len(list(class_path.glob("*.*")))
    class_counts.append(count)
    total_images += count
    percentage = (count / total_images * 100) if total_images else 0
    print(f"{cls_name:20} : {count:5} images ({percentage:5.2f}%)")

print(f"\nTotal: {total_images} images")

class_counts = np.array(class_counts, dtype=float)
safe_counts = np.maximum(class_counts, 1.0)

# Method 1: Inverse Frequency (smooth)
print("\n" + "="*70)
print("⚖️  METHOD 1: INVERSE FREQUENCY (Smooth)")
print("="*70)
weights = 1.0 / np.sqrt(safe_counts)
weights = weights / weights.sum() * len(weights)

print("\nWeights:")
for cls_name, w in zip(classes, weights):
    print(f"  {cls_name:20} : {w:.4f}")

# Method 2: Inverse Frequency (no smooth) - more aggressive
print("\n" + "="*70)
print("⚖️  METHOD 2: INVERSE FREQUENCY (Aggressive)")
print("="*70)
weights2 = 1.0 / safe_counts
weights2 = weights2 / weights2.sum() * len(weights2)

print("\nWeights:")
for cls_name, w in zip(classes, weights2):
    print(f"  {cls_name:20} : {w:.4f}")

# Method 3: Log-based (balanced)
print("\n" + "="*70)
print("⚖️  METHOD 3: LOG-BASED (Balanced)")
print("="*70)
weights3 = np.log(max(total_images, 1) / (safe_counts + 1))
weights3 = weights3 / weights3.sum() * len(weights3)

print("\nWeights:")
for cls_name, w in zip(classes, weights3):
    print(f"  {cls_name:20} : {w:.4f}")

# RECOMMENDED: Use Method 3 (log-based)
print("\n" + "="*70)
print("✅ RECOMMENDED: Use Method 3 (Log-based)")
print("="*70)
print("\nPython tensor format:")
print(f"class_weights = torch.tensor([{', '.join([f'{w:.4f}' for w in weights3])}],")
print("                              dtype=torch.float32).to(device)")

print("\n" + "="*70)
print("📝 NOTES:")
print("="*70)
print("- Method 1: Gentle smoothing (sqrt) - good for mild imbalance")
print("- Method 2: Aggressive (1/x) - can overfit rare classes")
print("- Method 3: Log-based - best balance, prevents torpedoing")
print("="*70)
