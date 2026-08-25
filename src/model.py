"""
Araba Gövde Sınıflandırması - Model Mimarisi
Model architecture (EfficientNet, ResNet50) ile Transfer Learning
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0, ResNet50, EfficientNetB3
from tensorflow.keras.optimizers import Adam
import json
from pathlib import Path

# 8 sınıf
NUM_CLASSES = 8
CLASSES = ['SUV', 'VAN', 'SEDAN', 'HATCHBACK', 'PICKUP', 'ACIK_TEKERLEK', 'STATION_WAGON', 'MICRO']

class ModelBuilder:
    """
    Çeşitli model mimarilerini oluşturan sınıf
    """
    
    def __init__(self, num_classes=6, input_shape=(224, 224, 3)):
        """
        Model builder'ı başlat
        
        Args:
            num_classes: Sınıf sayısı
            input_shape: Giriş görüntü boyutu
        """
        self.num_classes = num_classes
        self.input_shape = input_shape
    
    def build_efficientnet_b0(self):
        """
        EfficientNet-B0 tabanlı model
        
        Neden EfficientNet?
        - Çok verimli (parameter sayısı az, model boyutu küçük)
        - İyi accuracy (ResNet'ten daha iyi)
        - Hızlı inference (sunum için uygun)
        - Model boyutu < 95 MB (proje gereksinimi)
        
        Returns:
            Compiled model
        """
        print("\n" + "=" * 70)
        print("🧠 MODEL OLUŞTURULUYOR: EfficientNet-B0")
        print("=" * 70)
        
        # Önceden eğitilmiş EfficientNet-B0 yükle (ImageNet ağırlıkları)
        base_model = EfficientNetB0(
            weights='imagenet',  # Önceden eğitilmiş ağırlıklar
            input_shape=self.input_shape,
            include_top=False     # Son FC katmanlarını hariç tut
        )
        
        # Transfer learning: Temel modeli dondurmaya başla (fine-tuning için)
        base_model.trainable = False
        
        # Yeni model oluştur
        model = models.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),
            
            # Preprocessing (ImageNet normalizasyonu)
            layers.Rescaling(1./127.5, offset=-1),
            
            # Önceden eğitilmiş EfficientNet
            base_model,
            
            # Global Average Pooling
            layers.GlobalAveragePooling2D(),
            
            # Dense katmanlar (classifier)
            layers.Dense(256, activation='relu', name='dense_1'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(128, activation='relu', name='dense_2'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax', name='output')
        ])
        
        return model, base_model
    
    def build_efficientnet_b3(self):
        """
        EfficientNet-B3 tabanlı model (daha büyük, daha iyi accuracy)
        
        Neden B3?
        - B0'dan daha iyi accuracy
        - Hala makul boyutta model
        - Daha yavaş training ama daha iyi sonuçlar
        
        Returns:
            Compiled model
        """
        print("\n" + "=" * 70)
        print("🧠 MODEL OLUŞTURULUYOR: EfficientNet-B3")
        print("=" * 70)
        
        base_model = EfficientNetB3(
            weights='imagenet',
            input_shape=self.input_shape,
            include_top=False
        )
        
        base_model.trainable = False
        
        model = models.Sequential([
            layers.Input(shape=self.input_shape),
            layers.Rescaling(1./127.5, offset=-1),
            base_model,
            layers.GlobalAveragePooling2D(),
            
            layers.Dense(512, activation='relu', name='dense_1'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            
            layers.Dense(256, activation='relu', name='dense_2'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(128, activation='relu', name='dense_3'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(self.num_classes, activation='softmax', name='output')
        ])
        
        return model, base_model
    
    def build_resnet50(self):
        """
        ResNet50 tabanlı model
        
        Neden ResNet?
        - Çok araştırılmış, stabil
        - İyi accuracy
        - Kütüphane desteği iyi
        - Ama EfficientNet'ten daha ağır
        
        Returns:
            Compiled model
        """
        print("\n" + "=" * 70)
        print("🧠 MODEL OLUŞTURULUYOR: ResNet50")
        print("=" * 70)
        
        base_model = ResNet50(
            weights='imagenet',
            input_shape=self.input_shape,
            include_top=False
        )
        
        base_model.trainable = False
        
        model = models.Sequential([
            layers.Input(shape=self.input_shape),
            layers.Rescaling(1./127.5, offset=-1),
            base_model,
            layers.GlobalAveragePooling2D(),
            
            layers.Dense(512, activation='relu', name='dense_1'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            
            layers.Dense(256, activation='relu', name='dense_2'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(128, activation='relu', name='dense_3'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(self.num_classes, activation='softmax', name='output')
        ])
        
        return model, base_model
    
    def compile_model(self, model, learning_rate=0.001):
        """
        Model'i compile et
        
        Args:
            model: Keras model
            learning_rate: Öğrenme hızı
        
        Returns:
            Compiled model
        """
        optimizer = Adam(learning_rate=learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',  # Çoklu sınıf sınıflandırması
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        )
        
        return model
    
    def enable_fine_tuning(self, base_model, num_layers_to_unfreeze=30):
        """
        Fine-tuning için temel modelin son katmanlarını unfreeze et
        
        Args:
            base_model: Temel model
            num_layers_to_unfreeze: Açılacak katman sayısı
        """
        # Tüm katmanları dondur
        base_model.trainable = True
        
        # Son n katmanı aç
        for layer in base_model.layers[:-num_layers_to_unfreeze]:
            layer.trainable = False
        
        print(f"\n✓ Fine-tuning aktivasyonu: Son {num_layers_to_unfreeze} katman eğitilebilir")
        
        # Düşük learning rate ile compile et (fine-tuning için)
        return self.compile_model(base_model, learning_rate=1e-5)

def get_callbacks():
    """
    Eğitim sırasında kullanılacak callbacks oluştur
    """
    callbacks = [
        # Early Stopping: Val loss artmaya başlarsa dur
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            verbose=1,
            restore_best_weights=True,
            mode='min'
        ),
        
        # Learning Rate Reduction: Val loss plateau'ya girerse learning rate düşür
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            verbose=1,
            min_lr=1e-7,
            mode='min'
        ),
        
        # Model Checkpoint: En iyi model'i kaydet
        tf.keras.callbacks.ModelCheckpoint(
            filepath='models/best_model.h5',
            monitor='val_accuracy',
            verbose=1,
            save_best_only=True,
            mode='max'
        ),
        
        # TensorBoard: Training'i görselleştir
        tf.keras.callbacks.TensorBoard(
            log_dir='logs/',
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        )
    ]
    
    return callbacks

def print_model_info(model):
    """
    Model hakkında bilgi yazdır
    """
    print("\n" + "=" * 70)
    print("📋 MODEL BİLGİSİ")
    print("=" * 70)
    
    # Model özeti
    model.summary()
    
    # Parametre sayısı
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    
    print(f"\n📊 PARAMETER İSTATİSTİKLERİ:")
    print(f"  Toplam: {total_params:,}")
    print(f"  Eğitilebilir: {trainable_params:,}")
    print(f"  Dondurulmuş: {non_trainable_params:,}")
    
    # Model boyutu tahmini
    model_size_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per float32
    print(f"\n💾 TAHMINI MODEL BOYUTU: {model_size_mb:.2f} MB")
    
    if model_size_mb > 95:
        print(f"  ⚠️  UYARI: Model 95 MB'ı aşıyor! Proje gereksinimini karşılamıyor")
    else:
        print(f"  ✓ Gereksinim içinde (< 95 MB)")

def main():
    """Ana fonksiyon - Model oluştur"""
    print("\n" + "=" * 70)
    print("🚀 MODEL MİMARİSİ OLUŞTURULUYOR")
    print("=" * 70)
    
    # Model builder oluştur
    builder = ModelBuilder(num_classes=NUM_CLASSES)
    
    # EfficientNet-B0 model oluştur (önerilir: hızlı ve etkili)
    model, base_model = builder.build_efficientnet_b0()
    
    # Model'i compile et
    model = builder.compile_model(model, learning_rate=0.001)
    
    # Model bilgisini yazdır
    print_model_info(model)
    
    # Model konfigürasyonunu kaydet
    config = {
        'model_type': 'EfficientNet-B0',
        'num_classes': NUM_CLASSES,
        'classes': CLASSES,
        'input_shape': builder.input_shape,
        'batch_size': 32,
        'learning_rate': 0.001,
        'epochs': 50,
        'early_stopping_patience': 10,
    }
    
    with open('models/model_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n✓ Model konfigürasyonu kaydedildi: models/model_config.json")
    
    print("\n" + "=" * 70)
    print("✅ MODEL HAZIR!")
    print("=" * 70)
    print("\nSonraki adım: Model eğitimini başlat")
    
    return model, base_model

if __name__ == "__main__":
    model, base_model = main()
