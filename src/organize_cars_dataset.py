"""
Kaggle 'cars-body-type-cropped' Dataseti Organize Scripti
https://www.kaggle.com/datasets/ademboukhris/cars-body-type-cropped
"""

import os
from pathlib import Path
import shutil

# Bu dataset'i organize etme parametreleri
# cars-body-type-cropped veri seti zaten kategorize edilmiş yapıda geliyor

def organize_cars_body_type_dataset(source_dir='cars-body-type-cropped', target_dir='data'):
    """
    cars-body-type-cropped dataseti'ni proje yapısına organize et
    
    Bu veri seti genelde şu yapıda gelir:
    cars-body-type-cropped/
    ├── images/
    │   ├── SEDAN/
    │   ├── SUV/
    │   ├── HATCHBACK/
    │   ├── VAN/
    │   ├── MICRO/           <- Bu sınıf bu datasette VAR!
    │   ├── STATION_WAGON/   <- Bu sınıf bu datasette VAR!
    │   └── ... diğer sınıflar
    """
    
    source_path = Path(source_dir)
    target_train = Path(target_dir) / 'train'
    target_val = Path(target_dir) / 'validation'
    target_test = Path(target_dir) / 'test'
    
    print("=" * 70)
    print("🚗 CARS-BODY-TYPE-CROPPED DATASETI ORGANIZE EDILIYOR")
    print("=" * 70)
    
    if not source_path.exists():
        print(f"\n❌ HATA: {source_dir} klasörü bulunamadı!")
        print(f"   Lütfen cars-body-type-cropped dataseti'ni indirin ve")
        print(f"   'proje klasörüne yerleştirin")
        return False
    
    # Kaynak klasörleri bul
    images_dir = source_path / 'images'
    if not images_dir.exists():
        # Belki doğrudan class klasörleri var
        images_dir = source_path
    
    print(f"\nKaynak: {images_dir}")
    print(f"Hedef: {target_train}")
    
    if not images_dir.exists():
        print(f"\n❌ images/ klasörü bulunamadı!")
        return False
    
    # Hangi sınıflar var
    classes = [d.name for d in images_dir.iterdir() if d.is_dir()]
    print(f"\nBulunan sınıflar: {classes}")
    
    # Her sınıfta kaç görüntü var
    print("\nSınıf İstatistikleri:")
    for class_name in sorted(classes):
        class_path = images_dir / class_name
        images = list(class_path.glob('*.jpg')) + list(class_path.glob('*.png'))
        print(f"  {class_name:20s}: {len(images)} görüntü")
    
    print("\n" + "=" * 70)
    print("✅ HAZIR! Bu veri setini data/ klasörüne taşımak için:")
    print("=" * 70)
    print("""
Aşağıdaki Python kodunu çalıştır:

from pathlib import Path
import shutil

source = Path('cars-body-type-cropped/images')  # Doğru yolu gir
target_train = Path('data/train')

for class_dir in source.iterdir():
    if class_dir.is_dir():
        class_name = class_dir.name.upper()
        
        # Her sınıf için klasör oluştur
        target_class = target_train / class_name
        target_class.mkdir(parents=True, exist_ok=True)
        
        # Görüntüleri kopyala
        for img in class_dir.glob('*.[jJ][pP][gG]'):
            shutil.copy2(img, target_class / img.name)
            
        print(f"✓ {class_name}: {len(list(class_dir.glob('*.[jJ][pP][gG]')))} görüntü")

print("✅ Taşıma tamamlandı!")
    """)
    
    return True

if __name__ == "__main__":
    # Dataseti kontrol et
    organize_cars_body_type_dataset()
