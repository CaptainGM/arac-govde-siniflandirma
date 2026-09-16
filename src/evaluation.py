"""
Araba Gövde Sınıflandırması - Model Değerlendirmesi
Metrik hesapları: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import json
from pathlib import Path
import tensorflow as tf

CLASSES = ['SUV', 'VAN', 'SEDAN', 'HATCHBACK', 'PICKUP', 'ACIK_TEKERLEK', 'STATION_WAGON', 'MICRO']

class ModelEvaluator:
    """
    Model değerlendirme sınıfı
    Metrikleri hesapla ve görselleştir
    """
    
    def __init__(self, classes=CLASSES):
        """
        Evaluator'ı başlat
        
        Args:
            classes: Sınıf isimleri
        """
        self.classes = classes
        self.num_classes = len(classes)
        self.metrics = {}
        
    def calculate_metrics(self, y_true, y_pred, y_pred_proba=None):
        """
        Tüm metrikleri hesapla
        
        Args:
            y_true: Gerçek sınıflar
            y_pred: Tahmin edilen sınıflar
            y_pred_proba: Tahmin olasılıkları (opsiyonel)
        
        Returns:
            metrics dict
        """
        print("\n" + "=" * 70)
        print("📊 MODELİN DEĞERLENDİRİLMESİ")
        print("=" * 70)
        
        # Accuracy
        accuracy = accuracy_score(y_true, y_pred)
        self.metrics['accuracy'] = float(accuracy)
        print(f"✓ Accuracy: {accuracy:.4f}")
        
        # Precision (Macro ve Weighted)
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        
        self.metrics['precision_macro'] = float(precision_macro)
        self.metrics['precision_weighted'] = float(precision_weighted)
        self.metrics['precision_per_class'] = {
            self.classes[i]: float(precision_per_class[i]) 
            for i in range(len(self.classes))
        }
        print(f"✓ Precision (Macro): {precision_macro:.4f}")
        print(f"✓ Precision (Weighted): {precision_weighted:.4f}")
        
        # Recall (Macro ve Weighted)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        
        self.metrics['recall_macro'] = float(recall_macro)
        self.metrics['recall_weighted'] = float(recall_weighted)
        self.metrics['recall_per_class'] = {
            self.classes[i]: float(recall_per_class[i]) 
            for i in range(len(self.classes))
        }
        print(f"✓ Recall (Macro): {recall_macro:.4f}")
        print(f"✓ Recall (Weighted): {recall_weighted:.4f}")
        
        # F1-Score (Macro ve Weighted) - PRIMARY METRIC
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        self.metrics['f1_macro'] = float(f1_macro)
        self.metrics['f1_weighted'] = float(f1_weighted)
        self.metrics['f1_per_class'] = {
            self.classes[i]: float(f1_per_class[i]) 
            for i in range(len(self.classes))
        }
        print(f"\n🌟 F1-Score (Macro): {f1_macro:.4f} [PRIMARY METRIC]")
        print(f"🌟 F1-Score (Weighted): {f1_weighted:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(self.num_classes))
        self.metrics['confusion_matrix'] = cm.tolist()
        
        print("\n" + "=" * 70)
        print("📋 SINIFLARa AIT DETAYLI METRİKLER")
        print("=" * 70)
        for i, class_name in enumerate(self.classes):
            print(f"\n{class_name}:")
            print(f"  - Precision: {precision_per_class[i]:.4f}")
            print(f"  - Recall: {recall_per_class[i]:.4f}")
            print(f"  - F1-Score: {f1_per_class[i]:.4f}")
        
        return self.metrics
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path='reports/confusion_matrix.png'):
        """
        Normalize edilmiş confusion matrix'i çiz
        
        Args:
            y_true: Gerçek sınıflar
            y_pred: Tahmin edilen sınıflar
            save_path: Kaydedilecek dosya yolu
        """
        # Confusion matrix hesapla
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(self.num_classes))
        
        # Normalize et (0-1 aralığına)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Çiz
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=self.classes,
            yticklabels=self.classes,
            cbar_kws={'label': 'Normalized Frequency'}
        )
        plt.ylabel('Gerçek Sınıf')
        plt.xlabel('Tahmin Edilen Sınıf')
        plt.title('Normalized Confusion Matrix (8x8)')
        plt.tight_layout()
        
        # Kaydet
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion Matrix kaydedildi: {save_path}")
        plt.close()
        
        return cm_normalized
    
    def plot_training_history(self, history, save_dir='reports'):
        """
        Training ve Validation Loss & Accuracy grafiklerini çiz
        
        Args:
            history: Keras training history
            save_dir: Kaydedilecek klasör
        """
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # Loss grafiği
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
        plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training & Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Accuracy grafiği
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training & Validation Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Kaydet
        loss_acc_path = f'{save_dir}/training_history.png'
        plt.savefig(loss_acc_path, dpi=300, bbox_inches='tight')
        print(f"✓ Training History grafiği kaydedildi: {loss_acc_path}")
        plt.close()
    
    def save_metrics(self, save_path='reports/metrics.json'):
        """
        Metrikleri JSON dosyasına kaydet
        
        Args:
            save_path: Kaydedilecek dosya yolu
        """
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        print(f"✓ Metrikler kaydedildi: {save_path}")


def evaluate_model_on_test_set(model, test_generator, evaluator, steps=None):
    """
    Modeli test seti üzerinde değerlendir
    
    Args:
        model: Eğitilmiş Keras model
        test_generator: Test veri generator'ı
        evaluator: ModelEvaluator nesnesi
        steps: Evaluasyon adımları
    """
    print("\n" + "=" * 70)
    print("🧪 TEST SETİ ÜZERİNDE DEĞERLENDIRME")
    print("=" * 70)
    
    # Tahminler yap
    y_pred_all = []
    y_true_all = []
    
    for images, labels in test_generator:
        # Batch tahminleri
        predictions = model.predict(images, verbose=0)
        y_pred_batch = np.argmax(predictions, axis=1)
        y_true_batch = np.argmax(labels, axis=1) if len(labels.shape) > 1 else labels
        
        y_pred_all.extend(y_pred_batch)
        y_true_all.extend(y_true_batch)
        
        if steps and len(y_true_all) >= steps * test_generator.batch_size:
            break
    
    y_pred_all = np.array(y_pred_all)
    y_true_all = np.array(y_true_all)
    
    # Metrikleri hesapla
    metrics = evaluator.calculate_metrics(y_true_all, y_pred_all)
    
    # Görselleştirmeler
    evaluator.plot_confusion_matrix(y_true_all, y_pred_all)
    
    # Metrikleri kaydet
    evaluator.save_metrics()
    
    return metrics
