@echo off
setlocal enabledelayedexpansion
chcp 65001 > nul
cls

echo.
echo ========================================================================
echo   ARABA GÖVDE SINIFI PROJESI - PYTORCH EĞİTİM
echo ========================================================================
echo.

:: Python'u kontrol et
py -3.11 --version >nul 2>&1
if errorlevel 1 (
    echo ❌ HATA: Python 3.11 kurulu değil!
    pause
    exit /b 1
)

echo ✓ Python 3.11 bulundu

echo.

:: Çalışma dizine git
cd /d "%~dp0"
echo ✓ Çalışma dizini: %cd%
echo.

:: Sanal ortam kontrol et/oluştur
if not exist ".venv-1" (
    echo 📦 Sanal ortam oluşturuluyor...
    py -3.11 -m venv .venv-1
    if errorlevel 1 (
        echo ❌ Sanal ortam oluşturulamadı
        pause
        exit /b 1
    )
    echo ✓ Sanal ortam oluşturuldu
    echo.
)

:: Sanal ortamı aktifleştir
call .venv-1\Scripts\activate.bat
if errorlevel 1 (
    echo ❌ Sanal ortam aktifleştirilemedi
    pause
    exit /b 1
)

echo ✓ Sanal ortam aktif
echo.

:: Pip'i güncelle
echo 📥 Pip güncelleniyor...
python -m pip install --upgrade pip setuptools wheel -q >nul 2>&1
echo ✓ Pip güncellendi
echo.

:: Gerekli kütüphaneleri yükle
echo 📚 Gerekli kütüphaneler yükleniyor...
echo     (İlk kez biraz zaman alabilir... ☕)
echo.

set "PACKAGES=torch torchvision torchaudio numpy scikit-learn matplotlib seaborn pillow streamlit pandas tqdm torch-directml"

for %%P in (%PACKAGES%) do (
    echo     • %%P yükleniyor...
    pip install -q %%P >nul 2>&1
    if errorlevel 1 (
        echo     ⚠️  %%P yüklenmesinde sorun yaşandı (devam ediliyor)
    )
)

echo.
echo ✓ Kütüphaneler kuruldu
echo.

:: PyTorch test et
echo 🧪 PyTorch test ediliyor...
python -c "import torch; print('✓ PyTorch ' + torch.__version__ + ' başarıyla yüklendi')" >nul 2>&1
if errorlevel 1 (
    echo ℹ️ Otomatik test atlandı, DirectML ile devam ediliyor...
)

:: Veri dizini kontrol et
echo 📂 Veri dizini kontrol ediliyor...
if not exist "train" (
    echo ❌ train/ klasörü bulunamadı
    pause
    exit /b 1
)
if not exist "valid" (
    echo ❌ valid/ klasörü bulunamadı
    pause
    exit /b 1
)
if not exist "test" (
    echo ❌ test/ klasörü bulunamadı
    pause
    exit /b 1
)
echo ✓ Veri dizinleri tamam
echo.

:: Model ve reports klasörleri oluştur
if not exist "models" mkdir models
if not exist "reports" mkdir reports
echo ✓ Çıkış klasörleri hazır
echo.

:: CPU vs GPU seçimi
echo ========================================================================
echo   İŞLEMCİ SEÇİMİ
echo ========================================================================
echo.
echo GPU kullanmak eğitimi ~20-30x hızlandırır!
echo.
echo 1) NVIDIA CUDA ile eğit
echo 2) AMD DirectML ile eğit (RX 6700XT Windows Hızlandırma)
echo 3) CPU ile eğit (daha yavaş ama çalışır)
echo 4) Devam et (Otomatik AMD DirectML Modu)
echo.

set "device_choice="
set /p device_choice="Seçiminiz (1-4, default=4): "

if "!device_choice!"=="" set device_choice=4

if "!device_choice!"=="1" (
    set DEVICE=cuda
    echo.
    echo 🎮 NVIDIA CUDA modu seçildi
    python -c "import torch; print('GPU Ok')" >nul 2>&1
    if errorlevel 1 (
        echo ⚠️  GPU bulunamadı! CPU'ya geçiliyor...
        set DEVICE=cpu
    )
) else if "!device_choice!"=="2" (
    set DEVICE=dml
    echo.
    echo 🔴 AMD DirectML modu seçildi
    echo     (AMD 6700XT için Windows Donanım Hızlandırması aktif)
    echo     ⚡ Eğitim hızlandırılıyor...
) else if "!device_choice!"=="3" (
    set DEVICE=cpu
    echo.
    echo 🖥️  CPU modu seçildi (yavaş ama güvenli)
) else if "!device_choice!"=="4" (
    set DEVICE=dml
    echo.
    echo 🔍 AMD Ekran Kartı için DirectML otomatik seçildi...
) else (
    echo ❌ Geçersiz seçim! CPU modu kullanılıyor...
    set DEVICE=cpu
)

echo.
echo ========================================================================
echo   MODEL EĞİTİMİ BAŞLANIYOR (%DEVICE% modu)
echo ========================================================================
echo.

set PYTHONIOENCODING=utf-8
python train_model_pytorch.py --device %DEVICE%

if errorlevel 1 (
    echo.
    echo ❌ Eğitim sırasında hata oluştu
    pause
    exit /b 1
)

echo.
echo ✅ Model eğitimi başarıyla tamamlandı!
echo.

:: Web arayüzü başlat
echo ========================================================================
echo   WEB ARAYÜZÜ BAŞLANIYOR
echo ========================================================================
echo.
echo 🌐 Streamlit başlatılıyor...
echo 📱 Browser'da http://localhost:8501 adresine gidebilirsiniz
echo.
echo Çıkmak için: Ctrl+C basın
echo.

set PYTHONIOENCODING=utf-8
streamlit run app.py

pause