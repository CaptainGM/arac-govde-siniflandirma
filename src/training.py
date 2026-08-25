"""
Araba Gövde Sınıflandırması - Model Eğitimi
Training pipeline: veri yükleme, eğitim, ve performans takibi
"""

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import sys

# Proje modüllerini import et (src klasöründe olmak gerekir)
# from preprocessing import create_data_generators
# from model import ModelBuilder, get_callbacks

class TrainingConfig:
    """Eğitim konfigürasyonu"""
    
    def __init__(self):
        self.image_size = 224
        self.batch_size = 32
        self.epochs = 50
        self.learning_rate = 0.001
        self.early_stopping_patience = 10
        self.val_split = 0.15
        self.seed = 42
        self.model_type = 'EfficientNet-B0'

class ModelTrainer:
    """
    Model eğitim sınıfı
    """
    
    def __init__(self, config=None):
        """
        Trainer'ı başlat
        
        Args:
            config: TrainingConfig nesnesi
        """
        self.config = config or TrainingConfig()
        self.history = None
        self.model = None
        
        print("\n" + "=" * 70)
        print("🎓 MODEL EĞİTİM KÖŞESİ")
        print("=" * 70)
        print(f"Model: {self.config.model_type}")
        print(f"Görüntü Boyutu: {self.config.image_size}x{self.config.image_size}")
        print(f"Batch Boyutu: {self.config.batch_size}")
        print(f"Learning Rate: {self.config.learning_rate}")
        print(f"Epochs: {self.config.epochs}")
        print(f"Early Stopping Patience: {self.config.early_stopping_patience}")
    
    def create_data_generators(self, data_dir='data'):
        """
        Veri generator'ları oluştur
        
        Args:
            data_dir: Veri klasörü
        
        Returns:
            train_gen, val_gen, test_gen
        """
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        
        print("\n" + "=" * 70)
        print("📂 VERİ GENERATOR'LARI OLUŞTURULUYOR")
        print("=" * 70)
        
        # Eğitim verisi augmentation (genelleme yeteneğini artır)
        train_aug = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            rescale=1./255.
        )
        
        # Validasyon/Test verisi (augmentation yok)
        val_aug = ImageDataGenerator(rescale=1./255.)
        
        # Train generator
        train_gen = train_aug.flow_from_directory(
            f'{data_dir}/train',
            target_size=(self.config.image_size, self.config.image_size),
            batch_size=self.config.batch_size,
            class_mode='categorical',
            seed=self.config.seed,
            shuffle=True
        )
        
        print(f"\n✓ Train Generator")
        print(f"  Sınıflar: {train_gen.class_indices}")
        print(f"  Toplam: {train_gen.samples}")
        print(f"  Batch: {train_gen.samples // self.config.batch_size}")
        
        # Validation generator
        val_gen = val_aug.flow_from_directory(
            f'{data_dir}/validation',
            target_size=(self.config.image_size, self.config.image_size),
            batch_size=self.config.batch_size,
            class_mode='categorical',
            seed=self.config.seed,
            shuffle=False
        )
        
        print(f"\n✓ Validation Generator")
        print(f"  Toplam: {val_gen.samples}")
        print(f"  Batch: {val_gen.samples // self.config.batch_size}")
        
        # Test generator
        test_gen = val_aug.flow_from_directory(
            f'{data_dir}/test',
            target_size=(self.config.image_size, self.config.image_size),
            batch_size=self.config.batch_size,
            class_mode='categorical',
            seed=self.config.seed,
            shuffle=False
        )
        
        print(f"\n✓ Test Generator")
        print(f"  Toplam: {test_gen.samples}")
        
        return train_gen, val_gen, test_gen
    
    def build_model(self):
        """
        Model oluştur
        """
        from tensorflow.keras.applications import EfficientNetB0
        from tensorflow.keras import layers, models
        from tensorflow.keras.optimizers import Adam
        
        print("\n" + "=" * 70)
        print("🧠 MODEL OLUŞTURULUYOR")
        print("=" * 70)
        
        num_classes = 6  # Şimdilik 6 sınıf
        input_shape = (self.config.image_size, self.config.image_size, 3)
        
        # Önceden eğitilmiş EfficientNet-B0
        base_model = EfficientNetB0(
            weights='imagenet',
            input_shape=input_shape,
            include_top=False
        )
        
        # Transfer learning
        base_model.trainable = False
        
        # Yeni model
        model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Rescaling(1./127.5, offset=-1),
            base_model,
            layers.GlobalAveragePooling2D(),
            
            layers.Dense(256, activation='relu', name='dense_1'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(128, activation='relu', name='dense_2'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(num_classes, activation='softmax', name='output')
        ])
        
        # Compile
        optimizer = Adam(learning_rate=self.config.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Model özeti
        print("\n✓ Model oluşturuldu")
        total_params = model.count_params()
        model_size_mb = total_params * 4 / (1024 * 1024)
        print(f"  Total Parameters: {total_params:,}")
        print(f"  Estimated Size: {model_size_mb:.2f} MB")
        
        self.model = model
        return model
    
    def get_callbacks(self):
        """
        Eğitim callback'lerini oluştur
        """
        callbacks = [
            # Early Stopping
            EarlyStopping(
                monitor='val_loss',
                patience=self.config.early_stopping_patience,
                verbose=1,
                restore_best_weights=True,
                mode='min'
            ),
            
            # Learning Rate Reduction
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                verbose=1,
                min_lr=1e-7,
                mode='min'
            ),
            
            # Model Checkpoint
            ModelCheckpoint(
                filepath='models/best_model.h5',
                monitor='val_accuracy',
                verbose=1,
                save_best_only=True,
                mode='max'
            ),
            
            # Custom callback: Training loss/accuracy
            tf.keras.callbacks.LambdaCallback(
                on_epoch_end=self._on_epoch_end
            )
        ]
        
        return callbacks
    
    def _on_epoch_end(self, epoch, logs=None):
        """Her epoch'tan sonra çalışacak callback"""
        if epoch % 5 == 0:
            print(f"\n[Epoch {epoch+1}] Train Loss: {logs['loss']:.4f}, Val Loss: {logs['val_loss']:.4f}")
    
    def train(self, train_gen, val_gen):
        """
        Modeli eğit
        
        Args:
            train_gen: Eğitim data generator'ı
            val_gen: Validasyon data generator'ı
        """
        print("\n" + "=" * 70)
        print("🚀 EĞİTİM BAŞLANILIYOR")
        print("=" * 70)
        
        # Callbacks
        callbacks = self.get_callbacks()
        
        # Eğitim
        history = self.model.fit(
            train_gen,
            epochs=self.config.epochs,
            steps_per_epoch=train_gen.samples // self.config.batch_size,
            validation_data=val_gen,
            validation_steps=val_gen.samples // self.config.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        self.history = history
        
        print("\n" + "=" * 70)
        print("✅ EĞİTİM TAMAMLANDI!")
        print("=" * 70)
        
        return history
    
    def save_model(self, model_path='models/car_classifier_model.h5'):
        """
        Model'i kaydet
        
        Args:
            model_path: Kayıt yolu
        """
        self.model.save(model_path)
        print(f"\n✓ Model kaydedildi: {model_path}")
        
        # Training history'yi de kaydet
        history_dict = {
            'loss': self.history.history['loss'],
            'accuracy': self.history.history['accuracy'],
            'val_loss': self.history.history['val_loss'],
            'val_accuracy': self.history.history['val_accuracy']
        }
        
        with open('reports/training_history.json', 'w') as f:
            # Float değerleri string'e dönüştür (JSON uyumluluğu için)
            json_history = {k: [float(v) for v in vs] for k, vs in history_dict.items()}
            json.dump(json_history, f, indent=2)
        
        print(f"✓ Training history kaydedildi: reports/training_history.json")
    
    def plot_training_history(self):
        """
        Training history grafiklerini çiz
        """
        if self.history is None:
            print("⚠️  Eğitim verisi yok!")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss
        axes[0].plot(self.history.history['loss'], label='Train Loss', linewidth=2)
        axes[0].plot(self.history.history['val_loss'], label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training & Validation Loss')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Accuracy
        axes[1].plot(self.history.history['accuracy'], label='Train Accuracy', linewidth=2)
        axes[1].plot(self.history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training & Validation Accuracy')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('reports/training_curves.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Training grafiği kaydedildi: reports/training_curves.png")
        plt.close()

def main():
    """Ana fonksiyon - Eğitimi çalıştır"""
    
    # Konfigürasyon
    config = TrainingConfig()
    
    # Trainer oluştur
    trainer = ModelTrainer(config)
    
    # Data generator'ları oluştur
    train_gen, val_gen, test_gen = trainer.create_data_generators('data')
    
    # Model oluştur
    trainer.build_model()
    
    # Model'i eğit (BU KODUN ÇALIŞMASI BİLGİSAYARDA BİRKAÇ SAAT SÜREBİLİR)
    print("\n⏱️  UYARI: Eğitim birkaç saat sürebilir!")
    print("    GPU kullanıyorsanız hızlı olacak, CPU'da yavaş olacak.")
    
    history = trainer.train(train_gen, val_gen)
    
    # Model'i kaydet
    trainer.save_model()
    
    # Grafikleri çiz
    trainer.plot_training_history()
    
    print("\n" + "=" * 70)
    print("✅ TÜM EĞİTİM SÜRECİ TAMAMLANDI!")
    print("=" * 70)
    print("\nSonraki adımlar:")
    print("1. Model performansını test setinde ölçün")
    print("2. Confusion matrix oluşturun")
    print("3. Web arayüzünü geliştirin")
    print("4. IEEE raporu yazın")

if __name__ == "__main__":
    main()
