# 🚗 AutoVision: Araç Gövde Tipi Sınıflandırma ve XAI Teşhis Kokpiti

[![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![DirectML](https://img.shields.io/badge/DirectML-AMD_Radeon_GPU-ED1C24?style=for-the-badge&logo=amd&logoColor=white)](https://github.com/microsoft/DirectML)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.66-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Accuracy](https://img.shields.io/badge/Test_Doğruluk-%2588.23-00d2ff?style=for-the-badge)](#-başarım-metrikleri)
[![Macro F1](https://img.shields.io/badge/Macro_F1-0.9076-10b981?style=for-the-badge)](#-başarım-metrikleri)

Araç fotoğraflarından gövde tipini (**SUV, Sedan, Hatchback, Station Wagon, Pick-Up, Van, Micro, Açık Tekerlek**) yüksek doğrulukla sınıflandıran, **Grad-CAM (Açıklanabilir Yapay Zeka - XAI)** ısı haritası ve **4x Test-Time Augmentation (TTA)** destekli modern bir bilgisayarlı görü (computer vision) sistemidir.

---

## 📸 Uygulama Arayüzü ve Paneller

### 1. Gövde Tipi Tahmini ve Canlı Grad-CAM XAI Isı Haritası
Model sadece tahmin üretmekle kalmaz; aracın gövdesini sınıflandırırken **aracın tam olarak neresine baktığını** (kasa çizgisi, bagaj eğimi, tavan yüksekliği vb.) canlı ısı haritası katmanıyla görselleştirir.

![Gövde Tipi Tahmini ve Grad-CAM Isı Haritası](tahmin-yap-sekmesi.png)

### 2. Genişletilmiş Karışıklık Matrisi ve Telemetri Paneli
Modelin 8 sınıf arasındaki eşleşme doğruluğunu, hangi araçların hangi sınıflarla karıştığını ve F1/Precision/Recall dağılımlarını gösteren interaktif analiz paneli.

![Karışıklık Matrisi ve Performans Grafikleri](grafikler-sekmesi.png)

---

## 🌟 Temel Yetenekler

* **Derin Öğrenme Omurgası:** ResNet-50 mimarisi üzerine özel tasarlanmış tam bağlantılı sınıflandırma katmanı (`2048 → 1024 → 512 → 8 Sınıf`, BatchNorm1d + ReLU + Dropout).
* **Açıklanabilir Yapay Zeka (Grad-CAM XAI):** ResNet-50 son evrişim katmanından (`layer4[-1]`) türetilen ısı haritaları sayesinde modelin kararları doğrulanabilir ve şeffaftır.
* **4x Test-Time Augmentation (TTA):** Tahmin anında yatay simetri ve kenar kırpmalarıyla çoklu varyant ortalaması alarak sınıflandırma kararlılığı sağlar.
* **Letterboxing (En-Boy Oranı Koruma):** Araç fotoğraflarının en-boy oranını bozmadan 224×224 boyutunda kareye tamamlar.
* **Donanım Hızlandırması:** Windows üzerinde **AMD Radeon DirectML (`torch-directml`)**, **NVIDIA CUDA** ve **Çok Çekirdekli CPU** otomatik algılama desteği (~50 ms çıkarım gecikmesi).
* **4 Farklı Girdi Modu:**
  1. 📁 Bilgisayardan tekli / çoklu dosya sürükle-bırak
  2. 🌐 Doğrudan web görsel bağlantısı (URL) ile test
  3. 📸 Web kamerasından anlık canlı araç çekimi
  4. 🎲 Test veri setinden tek tıkla rastgele araç çekme simülasyonu

---

## 📊 Başarım Metrikleri (Test Seti)

Model, eğitim sırasında hiç görmediği **1.437 adet bağımsız test görseli** üzerinde değerlendirilmiştir:

### Genel Performans
| Metrik | Değer |
|---|:---:|
| **Test Doğruluğu (Accuracy)** | **%88.23** |
| **F1-Skor (Macro Average)** | **0.9076** |
| **F1-Skor (Weighted Average)** | **0.8827** |
| **Kesinlik (Precision - Macro)** | **0.9100** |
| **Duyarlılık (Recall - Macro)** | **0.9066** |
| **Ortalama Çıkarım Gecikmesi** | **~54 ms** (AMD DirectML GPU) |

### Sınıf Bazında Metrik Dağılımı
| Sınıf | F1-Skor | Kesinlik (Precision) | Duyarlılık (Recall) |
|---|:---:|:---:|:---:|
| **VAN** | **0.9910** | 0.9910 | 0.9910 |
| **AÇIK TEKERLEK** | **0.9869** | 1.0000 | 0.9741 |
| **MICRO** | **0.9722** | 0.9722 | 0.9722 |
| **PICK-UP** | **0.9688** | 0.9627 | 0.9748 |
| **SUV** | **0.8943** | 0.8943 | 0.8943 |
| **SEDAN** | **0.8510** | 0.8404 | 0.8618 |
| **STATION WAGON** | **0.8249** | 0.8818 | 0.7749 |
| **HATCHBACK** | **0.7722** | 0.7379 | 0.8097 |

---

## 🚀 Kurulum ve Çalıştırma

### 1. Model Ağırlıklarını Çekme ve Bağımlılıklar
```bash
# Git LFS kurulu değilse https://git-lfs.com adresinden kurun
git lfs pull

# Gerekli kütüphaneleri yükleyin
pip install -r requirements.txt
```

### 2. Arayüzü Başlatma
```bash
# Streamlit kokpitini başlatın:
python -m streamlit run app.py
```
*(Windows kullanıcıları doğrudan `RUN.bat` dosyasına çift tıklayarak ortamı ve arayüzü tek tıkla başlatabilir.)*

Tarayıcınızda otomatik olarak **`http://localhost:8501`** adresi açılacaktır.

---

## 🛠️ Yardımcı Araçlar

* `evaluate_model.py`: Test seti üzerinde güncel modelin tüm metriklerini ve karışıklık matrisini hesaplar.
* `audit_mislabeled_images.py`: Eğitim setinde yüksek güvenle farklı sınıfa benzeyen şüpheli etiketli görselleri tespit eder.
* `fix_labels_from_audit.py`: Tespit edilen etiket uyumsuzluklarını analiz eder.
* `show_confused.py`: En çok birbirine karışan sınıfları (Hatchback/Sedan/SW) ayrı klasöre toplayarak görsel denetim sağlar.
* `prepare_submission.py`: Modeli FP16 formatına sıkıştırarak teslimat paketini hazırlar.
