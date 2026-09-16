
import random
from pathlib import Path
import shutil
from class_config import CANONICAL_CLASSES, find_class_directory

random.seed(42)

for cls in CANONICAL_CLASSES:
    # Kaynak klasör (train şu an tüm veriyi içeriyor olabilir)
    src_dir = find_class_directory("train", cls)
    if src_dir is None:
        continue
    
    # Tüm resimleri al
    images = list(src_dir.glob("*.*"))
    random.shuffle(images)
    
    total = len(images)
    train_end = int(total * 0.7)
    valid_end = int(total * 0.85)
    
    train_images = images[:train_end]
    valid_images = images[train_end:valid_end]
    test_images = images[valid_end:]
    
    # Hedef klasörler
    train_dst = Path(f"train_clean/{cls}")
    valid_dst = Path(f"valid/{cls}")
    test_dst = Path(f"test/{cls}")
    
    train_dst.mkdir(parents=True, exist_ok=True)
    valid_dst.mkdir(parents=True, exist_ok=True)
    test_dst.mkdir(parents=True, exist_ok=True)
    
    for img in train_images:
        shutil.copy(img, train_dst / img.name)
    for img in valid_images:
        shutil.copy(img, valid_dst / img.name)
    for img in test_images:
        shutil.copy(img, test_dst / img.name)
    
    print(f"{cls}: Train={len(train_images)}, Valid={len(valid_images)}, Test={len(test_images)}")