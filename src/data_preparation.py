"""
Araba Gövde Tipi Sınıflandırması - Veri Hazırlama Scripti
Data preparation and organization
"""

import os
import shutil
from pathlib import Path
import json

# 8 sınıf tanımı
CLASSES = [
    'SUV',
    'VAN', 
    'STATION_WAGON',
    'MICRO',
    'ACIK_TEKERLEK',  # Açık Tekerlek / F1 Araçları
    'SEDAN',
    'HATCHBACK',
    'PICKUP'
]

def create_class_directories(base_path='data'):
    """
    Train, Validation ve Test setleri için klasörler oluştur
    """
    sets = ['train', 'validation', 'test']
    
    for set_type in sets:
        set_path = Path(base_path) / set_type
        set_path.mkdir(parents=True, exist_ok=True)
        
        for class_name in CLASSES:
            class_path = set_path / class_name
            class_path.mkdir(parents=True, exist_ok=True)
            print(f"✓ Klasör oluşturuldu: {class_path}")

def count_images_in_directory(data_path='data'):
    """
    Her sınıf için görüntü sayısını say
    """
    print("\n📊 Veri Seti İstatistikleri:")
    print("-" * 60)
    
    total_images = 0
    set_stats = {}
    
    for set_type in ['train', 'validation', 'test']:
        set_path = Path(data_path) / set_type
        set_total = 0
        
        print(f"\n{set_type.upper()}:")
        for class_name in CLASSES:
            class_path = set_path / class_name
            if class_path.exists():
                image_files = list(class_path.glob('*.[jJ][pP][gG]')) + \
                             list(class_path.glob('*.[pP][nN][gG]')) + \
                             list(class_path.glob('*.[jJ][pP][eE][gG]'))
                count = len(image_files)
                set_total += count
                print(f"  {class_name}: {count} görüntü")
        
        set_stats[set_type] = set_total
        total_images += set_total
    
    print("\n" + "=" * 60)
    print(f"TOPLAM GÖRÜNTÜ SAYISI: {total_images}")
    print(f"  - Train: {set_stats.get('train', 0)}")
    print(f"  - Validation: {set_stats.get('validation', 0)}")
    print(f"  - Test: {set_stats.get('test', 0)}")
    print("=" * 60)
    
    # Uyarı: Veri seti dengesini kontrol et
    if total_images < 1600:
        print("\n⚠️  UYARI: Çok az görüntü! Minimum 2000+ görüntü hedefleyin (sınıf başına 250+)")
    
    return set_stats

def save_dataset_config(config_path='dataset_config.json'):
    """
    Veri seti konfigürasyonunu kaydet
    """
    config = {
        'classes': CLASSES,
        'num_classes': len(CLASSES),
        'train_val_test_split': '70-15-15',
        'image_size': 224,
        'preprocessing': {
            'normalize': True,
            'augmentation': True
        }
    }
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Konfigürasyon kaydedildi: {config_path}")

def main():
    """Ana fonksiyon"""
    print("=" * 60)
    print("🚗 ARABA GÖVDE TİPİ SINIFLANDIRMASI - VERİ HAZIRLAMA")
    print("=" * 60)
    
    # Klasörleri oluştur
    print("\n1️⃣  Sınıf klasörleri oluşturuluyor...")
    create_class_directories()
    
    # İstatistikleri kontrol et
    print("\n2️⃣  Veri seti istatistikleri kontrol ediliyor...")
    count_images_in_directory()
    
    # Konfigürasyonu kaydet
    print("\n3️⃣  Konfigürasyon hazırlanıyor...")
    save_dataset_config()
    
    print("\n" + "=" * 60)
    print("✅ HAZIRLIK TAMAMLANDI!")
    print("=" * 60)
    print("\n📝 SONRAKI ADIMLAR:")
    print("1. Kaggle'dan görüntüleri indirin")
    print("2. İlgili sınıf klasörlerine yerleştirin")
    print("3. Data exploration notebook'u çalıştırın")
    print("4. Ön işleme ve augmentation yapın")
    print("5. Modeli eğitin\n")

if __name__ == "__main__":
    main()
