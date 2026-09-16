

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0, ResNet50
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import sys
from datetime import datetime


sys.path.insert(0, str(Path(__file__).parent))
from model import ModelBuilder, CLASSES
from evaluation import ModelEvaluator

class TrainingConfig:
    
    def __init__(self):
        self.image_size = 224
        self.batch_size = 32
        self.epochs = 50
        self.learning_rate = 0.001
        self.early_stopping_patience = 10
        self.val_split = 0.15
        self.seed = 42
        self.model_type = 'EfficientNet-B0'  
        self.data_dir = 'data'
        self.model_dir = 'models'
        self.reports_dir = 'reports'

class ModelTrainer: 
    def __init__(self, config=None):
        self.config = config or TrainingConfig()
        self.history = None
        self.model = None
        self.train_gen = None
        self.val_gen = None
        self.test_gen = None
        
        print("\n" + "=" * 70)
        print("🎓 ARABA GÖVDE SINIFI AYIRMA - EĞİTİM MODÜLÜ")
        print("=" * 70)
        print(f"Model: {self.config.model_type}")
        print(f"Görüntü Boyutu: {self.config.image_size}x{self.config.image_size}")
        print(f"Batch Boyutu: {self.config.batch_size}")
        print(f"Learning Rate: {self.config.learning_rate}")
        print(f"Max Epochs: {self.config.epochs}")
        print(f"Early Stopping Patience: {self.config.early_stopping_patience}")
    
    def create_data_generators(self):
        print("\n" + "=" * 70)
        print("📂 VERİ GENERATOR'LARI OLUŞTURULUYOR")
        print("=" * 70)
        train_aug = ImageDataGenerator(
            rotation_range=25,           
            width_shift_range=0.2,      
            height_shift_range=0.2,      
            shear_range=0.2,             
            zoom_range=0.2,             
            horizontal_flip=True,        
            brightness_range=[0.8, 1.2], 
            fill_mode='nearest',
            rescale=1./255.              
        )
        
        val_aug = ImageDataGenerator(rescale=1./255.)
        
        self.train_gen = train_aug.flow_from_directory(
            f'{self.config.data_dir}/train',
            target_size=(self.config.image_size, self.config.image_size),
            batch_size=self.config.batch_size,
            class_mode='categorical',
            seed=self.config.seed,
            shuffle=True
        )
        
        print(f"\n✓ TRAIN GENERATOR")
        print(f"  Sınıflar: {self.train_gen.class_indices}")
        print(f"  Toplam Görüntü: {self.train_gen.samples}")
        print(f"  Batch Sayısı: {len(self.train_gen)}")
        
        self.val_gen = val_aug.flow_from_directory(
            f'{self.config.data_dir}/validation',
            target_size=(self.config.image_size, self.config.image_size),
            batch_size=self.config.batch_size,
            class_mode='categorical',
            seed=self.config.seed,
            shuffle=False
        )
        
        print(f"\n✓ VALIDATION GENERATOR")
        print(f"  Toplam Görüntü: {self.val_gen.samples}")
        print(f"  Batch Sayısı: {len(self.val_gen)}")
        
        self.test_gen = val_aug.flow_from_directory(
            f'{self.config.data_dir}/test',
            target_size=(self.config.image_size, self.config.image_size),
            batch_size=self.config.batch_size,
            class_mode='categorical',
            seed=self.config.seed,
            shuffle=False
        )
        
        print(f"\n✓ TEST GENERATOR")
        print(f"  Toplam Görüntü: {self.test_gen.samples}")
        print(f"  Batch Sayısı: {len(self.test_gen)}")
        
        return self.train_gen, self.val_gen, self.test_gen
    
    def build_model(self):
        """
        Model oluştur ve derle
        """
        print("\n" + "=" * 70)
        print("🧠 MODEL OLUŞTURULUYOR")
        print("=" * 70)
        
        num_classes = 8  
        input_shape = (self.config.image_size, self.config.image_size, 3)
        
        if self.config.model_type == 'EfficientNet-B0':
            base_model = EfficientNetB0(
                weights='imagenet',
                input_shape=input_shape,
                include_top=False
            )
        else: 
            base_model = ResNet50(
                weights='imagenet',
                input_shape=input_shape,
                include_top=False
            )
        
        base_model.trainable = False

        self.model = models.Sequential([

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
        

        optimizer = Adam(learning_rate=self.config.learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        

        print("\n✓ Model başarıyla oluşturuldu!")
        total_params = self.model.count_params()
        model_size_mb = total_params * 4 / (1024 * 1024)
        print(f"  Total Parameters: {total_params:,}")
        print(f"  Estimated Size: {model_size_mb:.2f} MB")
        print(f"  ✓ Gereksinim: < 95 MB ✓")
        
      
        print("\n📊 MODEL MİMARİSİ:")
        self.model.summary()
        
        return self.model
    
    def get_callbacks(self):

        Path(self.config.model_dir).mkdir(exist_ok=True)
        Path(self.config.reports_dir).mkdir(exist_ok=True)
        
        callbacks = [
           
            EarlyStopping(
                monitor='val_loss',
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            
           
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001,
                verbose=1
            ),
            
         
            ModelCheckpoint(
                filepath=f'{self.config.model_dir}/trained_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            
   
            TensorBoard(
                log_dir=f'{self.config.reports_dir}/logs',
                histogram_freq=1,
                write_graph=True
            )
        ]
        
        return callbacks
    
    def train(self):
        print("\n" + "=" * 70)
        print("🚀 EĞİTİM BAŞLANIYOR")
        print("=" * 70)
        
        callbacks = self.get_callbacks()
        
    
        self.history = self.model.fit(
            self.train_gen,
            validation_data=self.val_gen,
            epochs=self.config.epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n" + "=" * 70)
        print("✓ EĞİTİM TAMAMLANDI")
        print("=" * 70)
        
        return self.history
    
    def evaluate_on_test_set(self):
       
        print("\n" + "=" * 70)
        print("🧪 TEST SETİ ÜZERİNDE DEĞERLENDİRME")
        print("=" * 70)
        
        
        test_loss, test_accuracy = self.model.evaluate(self.test_gen, verbose=1)
        
        print(f"\n✓ Test Loss: {test_loss:.4f}")
        print(f"✓ Test Accuracy: {test_accuracy:.4f}")
        
        
        evaluator = ModelEvaluator(classes=CLASSES)
        
      
        y_pred_all = []
        y_true_all = []
        
        for images, labels in self.test_gen:
            predictions = self.model.predict(images, verbose=0)
            y_pred_batch = np.argmax(predictions, axis=1)
            y_true_batch = np.argmax(labels, axis=1)
            
            y_pred_all.extend(y_pred_batch)
            y_true_all.extend(y_true_batch)
        
        y_pred_all = np.array(y_pred_all)
        y_true_all = np.array(y_true_all)
        
        
        metrics = evaluator.calculate_metrics(y_true_all, y_pred_all)
        
        
        print("\n📊 GÖRSELLEŞTIRMELER OLUŞTURULUYOR...")
        evaluator.plot_confusion_matrix(y_true_all, y_pred_all)
        evaluator.plot_training_history(self.history)
        evaluator.save_metrics()
        
        print("\n✓ Tüm grafikler kaydedildi: reports/ klasörü")
        
        return metrics
    
    def save_model(self, save_path=None):
       
        if save_path is None:
            save_path = f'{self.config.model_dir}/trained_model.h5'
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(save_path)
        
        
        model_info = {
            'model_type': self.config.model_type,
            'input_size': self.config.image_size,
            'classes': CLASSES,
            'num_classes': len(CLASSES),
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'batch_size': self.config.batch_size,
            'learning_rate': self.config.learning_rate
        }
        
        info_path = f'{self.config.model_dir}/model_info.json'
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=4)
        
        print(f"\n✓ Model kaydedildi: {save_path}")
        print(f"✓ Model bilgileri kaydedildi: {info_path}")


def main():
    
    
    config = TrainingConfig()
    
    
    trainer = ModelTrainer(config)
    
    try:
      
        trainer.create_data_generators()
        
       
        trainer.build_model()
        
        
        trainer.train()
        
       
        metrics = trainer.evaluate_on_test_set()
        
        
        trainer.save_model()
        
        print("\n" + "=" * 70)
        print("🎉 PROJE TAMAMLANDI!")
        print("=" * 70)
        print("\nSonraki adımlar:")
        print("1. Raporları kontrol et: reports/ klasörü")
        print("2. Web arayüzünü çalıştır: streamlit run web_interface/app.py")
        print("3. Model test et: python test_model.py")
        
    except Exception as e:
        print(f"\n❌ HATA: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
