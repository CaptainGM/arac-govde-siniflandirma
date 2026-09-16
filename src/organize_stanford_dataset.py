"""
Stanford Cars Dataset'i Proje Veri Setine Dönüştürme Scripti
Convert Stanford dataset to project structure (8 classes)
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict

# Stanford veri setinde 7 sınıf var, projede 8 gerekli
# Mapping tablosu:
# Stanford Sınıf → Proje Sınıfı
CLASS_MAPPING = {
    'SUV': 'SUV',
    'VAN': 'VAN',
    'Sedan': 'SEDAN',
    'Hatchback': 'HATCHBACK',
    'Pick-Up': 'PICKUP',
    'Convertible': 'ACIK_TEKERLEK',  # Açık tekerlekli (F1 benzeri)
    'Coupe': 'SEDAN'  # Coupe'u Sedan'a ekle
}

STANFORD_BASE = Path('../train')  # Stanford train klasörü
STANFORD_VALID = Path('../valid')  # Stanford valid klasörü

PROJECT_TRAIN = Path('../../data/train')
PROJECT_VALID = Path('../../data/validation')
PROJECT_TEST = Path('../../data/test')

def count_images_in_source():
    """
    Kaynak klasörlerdeki görüntü sayısını say
    """
    print("📊 STANFORD VERİ SETİ ANALİZİ")
    print("=" * 70)
    
    stats = {'train': {}, 'valid': {}}
    
    for base_path, set_name in [(STANFORD_BASE, 'train'), (STANFORD_VALID, 'valid')]:
        print(f"\n{set_name.upper()} klasöründe bulunan sınıflar:")
        print("-" * 70)
        
        if base_path.exists():
            for class_dir in sorted(base_path.iterdir()):
                if class_dir.is_dir():
                    images = list(class_dir.glob('*.jpg'))
                    count = len(images)
                    stats[set_name][class_dir.name] = count
                    print(f"  {class_dir.name:20s}: {count:4d} görüntü")
            
            total = sum(stats[set_name].values())
            print(f"  {'TOPLAM':20s}: {total:4d} görüntü")

    # Genel toplam
    train_total = sum(stats['train'].values())
    valid_total = sum(stats['valid'].values())
    print(f"\n{'='*70}")
    print(f"GENEL TOPLAM: {train_total + valid_total} görüntü")
    print(f"  Train: {train_total}")
    print(f"  Valid: {valid_total}")
    
    return stats

def organize_data_to_project():
    """
    Stanford veri setini proje yapısına organize et
    """
    print("\n\n🔄 VERİ SETİ DÖNÜŞTÜRÜLÜYOR...")
    print("=" * 70)
    
    # Hedef klasörleri oluştur
    for base_dir in [PROJECT_TRAIN, PROJECT_VALID, PROJECT_TEST]:
        base_dir.mkdir(parents=True, exist_ok=True)
    
    # TRAIN ve VALID verilerini işle
    for base_path, project_path, split_name in [
        (STANFORD_BASE, PROJECT_TRAIN, 'TRAIN'),
        (STANFORD_VALID, PROJECT_VALID, 'VALID')
    ]:
        if not base_path.exists():
            print(f"⚠️  {base_path} bulunamadı!")
            continue
        
        print(f"\n{split_name} seti işleniyor...")
        
        for stanford_class, project_class in CLASS_MAPPING.items():
            source_dir = base_path / stanford_class
            target_dir = project_path / project_class
            
            if not source_dir.exists():
                continue
            
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Görüntüleri kopyala
            image_files = list(source_dir.glob('*.jpg'))
            
            for img_file in image_files:
                try:
                    target_file = target_dir / img_file.name
                    if not target_file.exists():
                        shutil.copy2(img_file, target_file)
                except Exception as e:
                    print(f"  ⚠️  Hata ({img_file.name}): {e}")
            
            print(f"  ✓ {stanford_class:20s} → {project_class:20s}: {len(image_files)} görüntü")

def analyze_organized_data():
    """
    Organize edilen veriyi analiz et
    """
    print("\n\n📈 ORGANIZE EDİLMİŞ VERİ ANALİZİ")
    print("=" * 70)
    
    for set_type in ['train', 'validation']:
        if set_type == 'train':
            base_path = PROJECT_TRAIN
        else:
            base_path = PROJECT_VALID
        
        print(f"\n{set_type.upper()} SETİ:")
        print("-" * 70)
        
        total = 0
        for class_dir in sorted(base_path.iterdir()):
            if class_dir.is_dir():
                images = list(class_dir.glob('*.jpg'))
                count = len(images)
                total += count
                print(f"  {class_dir.name:20s}: {count:4d} görüntü")
        
        print(f"  {'TOPLAM':20s}: {total:4d} görüntü")
        
        # Uyarı: Eksik sınıflar
        expected_classes = {
            'SUV', 'VAN', 'STATION_WAGON', 'MICRO', 
            'ACIK_TEKERLEK', 'SEDAN', 'HATCHBACK', 'PICKUP'
        }
        found_classes = {d.name for d in base_path.iterdir() if d.is_dir()}
        missing = expected_classes - found_classes
        
        if missing:
            print(f"\n⚠️  EKSİK SINIFLAR: {', '.join(sorted(missing))}")

def save_mapping_info():
    """
    Sınıf eşleştirme bilgisini kaydet
    """
    info = """# Stanford Cars Dataset - Proje Veri Seti Mapping

## Sinif Donusturme Tablosu

| Stanford Sinif | Proje Sinifi | Aciklama |
|---|---|---|
| SUV | SUV | Dogrudan esleme |
| VAN | VAN | Dogrudan esleme |
| Sedan | SEDAN | Dogrudan esleme |
| Hatchback | HATCHBACK | Dogrudan esleme |
| Pick-Up | PICKUP | Dogrudan esleme |
| Convertible | ACIK_TEKERLEK | Acik araclar (F1 benzeri) |
| Coupe | SEDAN | 2 kapili spor araclar |

## Eksik Siniflar
- STATION_WAGON: Stanford veri setinde yok, ek veri gerekli
- MICRO: Stanford veri setinde yok, ek veri gerekli

## Sonraki Adimlar
1. Kaggle'dan STATION_WAGON ve MICRO goruntuleri toplayin
2. data/train/STATION_WAGON/ ve data/train/MICRO/ klasorlerine koyun
3. validation/test klasorune de benzer sekilde ekleyin
"""
    
    with open('../../data/DATASET_MAPPING.md', 'w', encoding='utf-8') as f:
        f.write(info)
    
    print("\nMapping bilgisi kaydedildi: data/DATASET_MAPPING.md")

def main():
    print("\n" + "="*70)
    print("🚗 STANFORD CARS DATASET → PROJE VERİ SETİ DÖNÜŞTÜRME")
    print("="*70)
    
    # İstatistikleri kontrol et
    stats = count_images_in_source()
    
    # Veriyi organize et
    organize_data_to_project()
    
    # Organize edilen veriyi analiz et
    analyze_organized_data()
    
    # Mapping bilgisini kaydet
    save_mapping_info()
    
    print("\n" + "="*70)
    print("✅ DÖNÜŞTÜRME TAMAMLANDI!")
    print("="*70)
    print("\n📝 SONRAKI ADIMLAR:")
    print("1. Kaggle'dan STATION_WAGON görüntüleri toplayın (~200+)")
    print("2. Kaggle'dan MICRO görüntüleri toplayın (~200+)")
    print("3. data/train/ klasörlerine yerleştirin")
    print("4. validation/ klasörüne benzer şekilde ekleyin")
    print("5. Data exploration notebook'unu çalıştırın")

if __name__ == "__main__":
    main()
