"""
Araba Gövde Sınıflandırması - Veri Ön İşleme (Preprocessing)
Image preprocessing, augmentation, histogram eşitleme, CLAHE, ve denoising
"""

import os
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# Konfigürasyon
CONFIG = {
    'IMAGE_SIZE': 224,
    'BATCH_SIZE': 32,
    'NORMALIZATION': 'imagenet',
    'SEED': 42
}

class ImagePreprocessor:
    """
    Görüntü ön işleme sınıfı
    - Histogram eşitleme (HE)
    - CLAHE (Contrast Limited Adaptive Histogram Equalization)
    - Bilateral filtering (gürültü giderme)
    - Normalizasyon
    """
    
    def __init__(self, image_size=224, normalization='imagenet', 
                 apply_histogram_eq=True, apply_clahe=True, apply_bilateral_filter=True):
        """
        Preprocessor'u başlat
        
        Args:
            image_size: Hedef görüntü boyutu (square)
            normalization: Normalizasyon yöntemi ('imagenet', 'standard', None)
            apply_histogram_eq: Histogram eşitleme uygula (contrast iyileştirme)
            apply_clahe: CLAHE uygula (adaptif contrast iyileştirme)
            apply_bilateral_filter: Bilateral filter uygula (gürültü giderme)
        """
        self.image_size = image_size
        self.normalization = normalization
        self.apply_histogram_eq = apply_histogram_eq
        self.apply_clahe = apply_clahe
        self.apply_bilateral_filter = apply_bilateral_filter

    def preprocess(self, image_path):
        """
        Görüntüyü ön işle, boyutlandır ve normalize et
        
        Ön işleme adımları:
        1. Görüntüyü yükle (OpenCV - BGR format)
        2. RGB'ye dönüştür
        3. Bilateral filtering (gürültü giderme)
        4. Histogram eşitleme + CLAHE (contrast iyileştirme)
        5. Boyutu değiştir (224x224)
        6. Normalizasyon
        
        Args:
            image_path: Görüntü dosyasının yolu
            
        Returns:
            Ön işlenmiş görüntü (numpy array)
        """
        try:
            # 1. Görüntüyü yükle (OpenCV ile - BGR format)
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"Görüntü yüklenemedi: {image_path}")
            
            # 2. RGB'ye dönüştür
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 3. Bilateral filtering (gürültü giderme - kenarları koruyor)
            if self.apply_bilateral_filter:
                img = cv2.bilateralFilter(img, 9, 75, 75)
            
            # 4. Histogram eşitleme + CLAHE (adaptif kontrastı iyileştirme)
            if self.apply_histogram_eq or self.apply_clahe:
                # LAB renk uzayına dönüştür (L=Light/Luminance)
                lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                
                if self.apply_clahe:
                    # CLAHE: Adaptif histogram eşitleme
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                    l = clahe.apply(l)
                elif self.apply_histogram_eq:
                    # Basit histogram eşitleme
                    l = cv2.equalizeHist(l)
                
                # LAB'ye geri dönüştür
                lab = cv2.merge([l, a, b])
                img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            # 5. Boyutu değiştir (224x224, LANCZOS interpolasyon)
            img = cv2.resize(img, (self.image_size, self.image_size), 
                           interpolation=cv2.INTER_LANCZOS4)
            
            # 6. Float32'ye dönüştür
            img_array = np.array(img, dtype=np.float32)
            
            # Normalizasyon
            if self.normalization == 'imagenet':
                # ImageNet normalizasyonu (ResNet için)
                IMAGENET_MEAN = np.array([103.939, 116.779, 123.68])
                img_array = img_array[..., ::-1]  # RGB -> BGR
                img_array -= IMAGENET_MEAN
            elif self.normalization == 'standard':
                # Basit normalizasyon (0-1)
                img_array = img_array / 255.0
            # else: normalizasyon yok
            
            return img_array
            
        except Exception as e:
            print(f"Hata ({image_path}): {e}")
            return None
    
    def create_train_augmenter(self):
        """
        Eğitim seti için data augmentation generator oluştur
        
        Augmentation teknikleri:
        - Rotation: Görüntüyü 20 derece döndür
        - Zoom: %20 zoom
        - Shift: Görüntüyü x/y ekseninde %20 kaydır
        - Horizontal flip: Sol-sağ çevirme
        - Brightness: Parlaklık değişimi
        """
        augmenter = ImageDataGenerator(
            rotation_range=20,           # 0-20 derece arası döndür
            width_shift_range=0.2,       # Genişliğin %20'si kadar kaydır
            height_shift_range=0.2,      # Yüksekliğin %20'si kadar kaydır
            shear_range=0.2,             # Makaslama transformasyonu
            zoom_range=0.2,              # %20 zoom aralığı
            horizontal_flip=True,        # Yatay çevirme
            brightness_range=[0.8, 1.2], # Parlaklık değişimi
            fill_mode='nearest',         # Eksik piksel doldurmayöntemi
            rescale=1./255.               # 0-1 arasında normalize et
        )
        return augmenter
    
    def create_val_augmenter(self):
        """
        Validasyon/Test seti için augmentation (minimal)
        
        Validasyon verisinde sadece basit normalizasyon yapılır,
        augmentation yapılmaz (veri kaybını önlemek için)
        """
        augmenter = ImageDataGenerator(
            rescale=1./255.  # Sadece normalizasyon
        )
        return augmenter

def create_data_generators(data_dir='data', batch_size=32, image_size=224):
    """
    Train ve Validation için data generator'lar oluştur
    
    Args:
        data_dir: Veri klasörü
        batch_size: Batch boyutu
        image_size: Görüntü boyutu
    
    Returns:
        train_generator, val_generator, test_generator
    """
    
    # Preprocessor'ü başlat
    preprocessor = ImagePreprocessor(image_size=image_size)
    
    # Augmenter'lar
    train_augmenter = preprocessor.create_train_augmenter()
    val_augmenter = preprocessor.create_val_augmenter()
    
    print("=" * 70)
    print("📊 DATA GENERATOR OLUŞTURULUYOR")
    print("=" * 70)
    
    # Train generator
    train_generator = train_augmenter.flow_from_directory(
        directory=f'{data_dir}/train',
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical',
        seed=42,
        shuffle=True
    )
    
    print(f"\n✓ Train Generator Oluşturuldu")
    print(f"  Sınıflar: {train_generator.class_indices}")
    print(f"  Toplam görüntü: {train_generator.samples}")
    print(f"  Batch boyutu: {batch_size}")
    
    # Validation generator
    val_generator = val_augmenter.flow_from_directory(
        directory=f'{data_dir}/validation',
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical',
        seed=42,
        shuffle=False
    )
    
    print(f"\n✓ Validation Generator Oluşturuldu")
    print(f"  Toplam görüntü: {val_generator.samples}")
    
    # Test generator (validation ile aynı - augmentation yok)
    test_generator = val_augmenter.flow_from_directory(
        directory=f'{data_dir}/test',
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical',
        seed=42,
        shuffle=False
    )
    
    print(f"\n✓ Test Generator Oluşturuldu")
    print(f"  Toplam görüntü: {test_generator.samples}")
    
    return train_generator, val_generator, test_generator

def visualize_augmentation(data_dir='data', num_samples=3):
    """
    Veri augmentation'ın etkisini görselleştir
    
    Args:
        data_dir: Veri klasörü
        num_samples: Görselleştirilecek örnek sayısı
    """
    print("\n" + "=" * 70)
    print("🎨 AUGMENTATION VİZÜALİZASYONU")
    print("=" * 70)
    
    preprocessor = ImagePreprocessor(image_size=224)
    augmenter = preprocessor.create_train_augmenter()
    
    # İlk sınıftan örnek görüntü yükle
    train_path = Path(data_dir) / 'train'
    first_class = list(train_path.iterdir())[0]
    sample_img_path = list(first_class.glob('*.jpg'))[0]
    
    print(f"\nSınıf: {first_class.name}")
    print(f"Görüntü: {sample_img_path.name}")
    
    # Orijinal görüntüyü yükle
    img = load_img(sample_img_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    
    # Augmentation'ı uygula
    img_array = np.expand_dims(img_array, axis=0)
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle(f'Veri Augmentation Örneği - {first_class.name}', fontsize=14, fontweight='bold')
    
    # Orijinal
    axes[0, 0].imshow(img_array[0])
    axes[0, 0].set_title('Orijinal')
    axes[0, 0].axis('off')
    
    # Augmented versiyonlar
    for i in range(1, 6):
        augmented = augmenter.random_transform(img_array[0])
        ax_idx = divmod(i, 3)
        axes[ax_idx].imshow(augmented)
        axes[ax_idx].set_title(f'Augmented {i}')
        axes[ax_idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('reports/augmentation_examples.png', dpi=300, bbox_inches='tight')
    print("\n✓ Augmentation görselleştirmesi kaydedildi: reports/augmentation_examples.png")
    plt.close()

def main():
    """Ana fonksiyon"""
    print("\n" + "=" * 70)
    print("🔧 VERİ ÖN İŞLEME (PREPROCESSING)")
    print("=" * 70)
    
    # Data generator'ları oluştur
    train_gen, val_gen, test_gen = create_data_generators(
        data_dir='data',
        batch_size=CONFIG['BATCH_SIZE'],
        image_size=CONFIG['IMAGE_SIZE']
    )
    
    # Augmentation'ı görselleştir
    visualize_augmentation()
    
    # Sınıf bilgileri
    print("\n" + "=" * 70)
    print("📋 SINIF HARITASI")
    print("=" * 70)
    for class_name, class_idx in train_gen.class_indices.items():
        print(f"  {class_idx}: {class_name}")
    
    print("\n" + "=" * 70)
    print("✅ ÖN İŞLEME TAMAMLANDI!")
    print("=" * 70)
    print("\nKonfigürasyon:")
    print(f"  Görüntü Boyutu: {CONFIG['IMAGE_SIZE']}x{CONFIG['IMAGE_SIZE']}")
    print(f"  Batch Boyutu: {CONFIG['BATCH_SIZE']}")
    print(f"  Normalizasyon: {CONFIG['NORMALIZATION']}")
    
    return train_gen, val_gen, test_gen

if __name__ == "__main__":
    main()
