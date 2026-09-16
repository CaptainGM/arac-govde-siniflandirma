"""
🚗 ARABA GÖVDE SINIFI TAHMINI - WEB ARAYÜZÜ (MODERNIZE)
Streamlit Interface - Drag&Drop, Yan Yana Layout, Profesyonel Tasarım
"""

import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import json
from pathlib import Path
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import time
from class_config import get_canonical_classes
from transforms_config import build_eval_transform, resolve_preprocess, PREPROCESS_STRETCH, PREPROCESS_LETTERBOX

# ============================================================================
# SAYFA AYARLARI (RESPONSIVE)
# ============================================================================

st.set_page_config(
    page_title="🚗 Araba Sınıflandırıcı AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"Get Help": None, "Report a bug": None, "About": None}
)

# ============================================================================
# MODERNIZE TASARIM STİLİ
# ============================================================================

st.markdown("""
<style>
    /* GENEL */
    .main { 
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        color: #2c3e50;
    }
    
    /* BAŞLIK */
    .title-main {
        font-size: 3.5em;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
        text-align: center;
    }
    
    /* SUBTITLE */
    .subtitle {
        font-size: 1.1em;
        color: #555;
        text-align: center;
        margin-bottom: 30px;
    }
    
    /* KONTEYNER BOXLAR */
    .container-box {
        background: white;
        padding: 30px;
        border-radius: 20px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        backdrop-filter: blur(10px);
    }
    
    /* TAHMİN SONUCU KUTUSU */
    .prediction-result-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 40px;
        border-radius: 20px;
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.4);
        text-align: center;
    }
    
    .prediction-class-text {
        font-size: 3.5em;
        font-weight: 900;
        margin: 20px 0;
    }
    
    .confidence-box {
        background: rgba(255,255,255,0.2);
        padding: 20px;
        border-radius: 15px;
        margin: 20px 0;
    }
    
    .confidence-percent {
        font-size: 2.8em;
        font-weight: 900;
    }
    
    /* METRIC CARD */
    .metric-card {
        background: white;
        padding: 25px;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }
    
    /* DÜĞME */
    .stButton > button {
        width: 100%;
        padding: 12px;
        font-size: 1.2em;
        font-weight: 700;
        border-radius: 12px;
        border: none;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5);
    }
    
    /* DOSYA YÜKLEYICI */
    .stFileUploader {
        border: 2px dashed #667eea;
        border-radius: 15px;
        padding: 20px;
    }
    
    /* TAB */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 12px 24px;
        border-radius: 10px;
        font-weight: 600;
    }
    
    /* INFO BOX */
    .info-box {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        border-left: 4px solid #667eea;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MODEL YÜKLE (CACHE)
# ============================================================================

def get_inference_device():
    """Egitimle uyumlu: AMD DirectML varsa onu kullan."""
    try:
        import torch_directml
        return torch_directml.device()
    except ImportError:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_state_dict_safe(model_path, map_location):
    """best_model.pt guvenli yukleme (DirectML kayitlari weights_only=True ile acilmaz)."""
    try:
        return torch.load(model_path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(model_path, map_location=map_location)


@st.cache_resource
def load_model_and_info():
    """Modeli ve bilgilerini yükle"""
    try:
        # Model bilgileri
        with open('models/model_info.json', 'r') as f:
            model_info = json.load(f)
        model_info = normalize_model_info(model_info)
        
        # Model mimarisi
        device = get_inference_device()
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Custom head
        num_classes = model_info.get('num_classes', len(model_info.get('classes', [])))
        model.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )
        
        # Weights yükle (8 sınıf kontrolü)
        state = load_state_dict_safe('models/best_model.pt', map_location=device)
        out_dim = state.get('fc.8.weight', state.get('fc.weight', None))
        if out_dim is not None and out_dim.shape[0] != num_classes:
            raise ValueError(
                f"Kayıtlı model {out_dim.shape[0]} sınıf için eğitilmiş; "
                f"uygulama {num_classes} sınıf bekliyor. "
                "Lütfen `python train_model_pytorch.py` ile 8 sınıf modeli yeniden eğitin."
            )
        model.load_state_dict(state)
        model = model.to(device)
        model.eval()
        
        return model, model_info, device
    except Exception as e:
        st.error(f"❌ Model yüklenemedi: {e}")
        return None, None, None


def normalize_model_info(model_info):
    """Normalize metadata keys and enforce 8-class contract."""
    canonical_classes = get_canonical_classes()
    classes = model_info.get("classes", canonical_classes)
    was_normalized = False
    if classes != canonical_classes:
        classes = canonical_classes
        was_normalized = True

    model_info["classes"] = classes
    model_info["num_classes"] = len(classes)

    # Backward compatibility for older keys
    if "test_precision" not in model_info:
        model_info["test_precision"] = model_info.get("test_precision_macro", 0.0)
        was_normalized = True
    if "test_recall" not in model_info:
        model_info["test_recall"] = model_info.get("test_recall_macro", 0.0)
        was_normalized = True

    if "per_class_metrics" not in model_info:
        f1_map = model_info.get("per_class_f1", {})
        precision_map = model_info.get("per_class_precision", {})
        recall_map = model_info.get("per_class_recall", {})
        model_info["per_class_metrics"] = {
            cls: {
                "F1-Score": float(f1_map.get(cls, 0.0)),
                "Precision": float(precision_map.get(cls, 0.0)),
                "Recall": float(recall_map.get(cls, 0.0)),
            }
            for cls in classes
        }
        was_normalized = True

    model_info["metadata_normalized"] = was_normalized

    if "preprocess" not in model_info:
        model_info["preprocess"] = PREPROCESS_STRETCH

    return model_info

# ============================================================================
# TAHMİN FONKSİYONU
# ============================================================================

def predict_image(model, img_tensor, classes, device):
    """Görüntüyü TTA (Test-Time Augmentation) ile tahmin et — 5 varyant ortalaması"""
    start_time = time.perf_counter()
    
    with torch.no_grad():
        # Orijinal tahmin
        base = img_tensor.unsqueeze(0).to(device)
        
        # TTA varyantları oluştur
        variants = [base]
        
        # 1) Yatay flip
        variants.append(torch.flip(base, dims=[3]))
        
        # 2-3) Hafif crop (sol ve sağ) — sedan/hatchback/SW arka profil farkını yakalar
        _, _, h, w = base.shape
        crop_px = int(w * 0.08)  # %8 kırp
        if crop_px > 0 and w > crop_px * 2:
            left_crop = torch.nn.functional.interpolate(
                base[:, :, :, crop_px:], size=(h, w), mode='bilinear', align_corners=False
            )
            right_crop = torch.nn.functional.interpolate(
                base[:, :, :, :-crop_px], size=(h, w), mode='bilinear', align_corners=False
            )
            variants.append(left_crop)
            variants.append(right_crop)
        
        # 4) Üstten hafif crop — tavan hattı farkını vurgular
        top_crop_px = int(h * 0.06)
        if top_crop_px > 0 and h > top_crop_px * 2:
            top_crop = torch.nn.functional.interpolate(
                base[:, :, top_crop_px:, :], size=(h, w), mode='bilinear', align_corners=False
            )
            variants.append(top_crop)
        
        # Tüm varyantları modelden geçir ve ortala
        all_probs = []
        for v in variants:
            out = model(v)
            prob = torch.softmax(out, dim=1)[0].cpu().numpy()
            all_probs.append(prob)
        
        # Ortalama olasılık
        probabilities = np.mean(all_probs, axis=0)
        predicted_idx = np.argmax(probabilities)
    
    inference_time = time.perf_counter() - start_time
    
    return {
        'predicted_class': classes[predicted_idx],
        'predicted_idx': predicted_idx,
        'confidence': float(probabilities[predicted_idx]),
        'probabilities': probabilities,
        'classes': classes,
        'inference_time': inference_time,
        'tta_variants': len(variants),
    }

# ============================================================================
# MODEL YÜKLE
# ============================================================================

model, model_info, device = load_model_and_info()

if model is None:
    st.error("❌ Model yüklenemedi. Önce eğitin: `python train_model_pytorch.py --device rocm`")
    st.stop()

classes = model_info['classes']
num_classes = model_info['num_classes']
eval_preprocess = resolve_preprocess(model_info)
eval_transform = build_eval_transform(eval_preprocess)
if model_info.get("metadata_normalized", False):
    st.warning("Model metadata 8 sınıf standardından farklıydı, uygulama içinde otomatik normalize edildi.")

# ============================================================================
# HEADER
# ============================================================================

st.markdown(f"""
<div class="title-main">🚗 Araba Gövde Sınıflandırıcı</div>
<div class="subtitle">AI Tabanlı Görüntü Analizi • {num_classes} Sınıf • ResNet50 Transfer Learning</div>
""", unsafe_allow_html=True)

# Üst metrikler
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("📊 F1-Score", f"{model_info.get('test_f1_macro', 0):.4f}", delta=None)
with col2:
    st.metric("✓ Accuracy", f"{model_info.get('test_accuracy', 0):.4f}", delta=None)
with col3:
    st.metric("🎯 Precision", f"{model_info.get('test_precision', 0):.4f}", delta=None)
with col4:
    st.metric("📈 Recall", f"{model_info.get('test_recall', 0):.4f}", delta=None)

st.divider()

# ============================================================================
# TAB OLUŞTUR
# ============================================================================

tab_home, tab_predict, tab_metrics, tab_graphs, tab_info = st.tabs(
    ["🚀 Başla", "🎯 Tahmin Yap", "📊 Metrikler", "📈 Grafikler", "ℹ️ Bilgi"]
)

# ============================================================================
# TAB 0: BAŞLA (INTRO)
# ============================================================================

with tab_home:
    col_home1, col_home2 = st.columns([2, 1])
    
    with col_home1:
        st.markdown("""
        <div class="info-box">
        <h2>🎯 Nasıl Kullanılır?</h2>
        
        1. **🎯 Tahmin Yap** sekmesine git
        2. Bir veya daha fazla araba resmi seç (drag&drop veya tıkla!)
        3. **Tahmin Yap** butonuna tıkla
        4. Sonuçları ve güven oranını gör
        
        **💡 İpucu:** Birden fazla resim seçebilirsin ve hepsini aynı anda analiz edebilirsin!
        </div>
        """, unsafe_allow_html=True)
    
    with col_home2:
        st.markdown(f"""
        <div class="metric-card" style="color: #2c3e50;">
        <h3 style="color: #667eea; margin: 0 0 15px 0;">⚙️ Model İstatistikleri</h3>
        <p style="margin: 8px 0; color: #333;"><strong>Mimari:</strong> ResNet50</p>
        <p style="margin: 8px 0; color: #333;"><strong>Eğitim:</strong> Transfer Learning</p>
        <p style="margin: 8px 0; color: #333;"><strong>Sınıf Sayısı:</strong> {num_classes}</p>
        <p style="margin: 8px 0; color: #667eea; font-weight: 700;"><strong>F1-Score:</strong> {model_info.get('test_f1_macro', 0):.3f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    st.markdown("### ✨ Desteklenen Sınıflar")
    cols = st.columns(min(4, num_classes))
    for i, cls in enumerate(classes):
        with cols[i % len(cols)]:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; text-align: center; padding: 30px 15px; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.08);">
            <h4 style="color: white; margin: 0; font-size: 1.3em; font-weight: 700;">{i+1}. {cls}</h4>
            </div>
            """, unsafe_allow_html=True)

# ============================================================================
# TAB 1: TAHMİN YAP (DRAG&DROP + YAN YANA)
# ============================================================================

with tab_predict:
    st.subheader("📤 Görüntü Yükle ve Analiz Et")
    
    # Dosya yükleyici
    st.markdown("**📥 Resim Seç** (JPG, PNG, BMP - Birden fazla seçebilirsin!)")
    uploaded_files = st.file_uploader(
        "Dosya seç veya sürükle-bırak yap",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        accept_multiple_files=True,
        key="file_uploader"
    )
    
    if len(uploaded_files) > 0:
        st.markdown(f"**✓ {len(uploaded_files)} resim seçildi**")
        
        # Tahmin butonu
        if st.button("▶️ Hepsi için Tahmin Yap", type="primary", use_container_width=True):
            transform = eval_transform
            
            st.session_state.predictions = []
            
            with st.spinner(f"🤔 Model {len(uploaded_files)} resim analiz ediyor..."):
                for file in uploaded_files:
                    img = Image.open(file).convert('RGB')
                    img_tensor = transform(img)
                    result = predict_image(model, img_tensor, classes, device)
                    result['image'] = img
                    result['filename'] = file.name
                    st.session_state.predictions.append(result)
        
        # Sonuçları göster
        if 'predictions' in st.session_state and len(st.session_state.predictions) > 0:
            st.markdown("---")
            st.markdown(f"### 📊 Sonuçlar ({len(st.session_state.predictions)} resim)")
            
            # Her resim için yan yana layout
            for idx, pred in enumerate(st.session_state.predictions):
                st.markdown(f"#### 📸 Resim {idx + 1}: {pred['filename']}")
                
                col_img, col_result = st.columns([1, 1.2])
                
                # SOL: Resim
                with col_img:
                    st.image(pred['image'], caption="Yüklenen Görüntü")
                
                # SAĞ: Tahmin Sonucu
                with col_result:
                    # Tahmin kutusu
                    st.markdown(f"""
                    <div class="prediction-result-box">
                        <div>TAHMIN SONUCU</div>
                        <div class="prediction-class-text">{pred['predicted_class']}</div>
                        <div class="confidence-box">
                            <div>Güven Oranı</div>
                            <div class="confidence-percent">{pred['confidence']:.1%}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Güven göstergesi (progress bar)
                    st.progress(pred['confidence'])
                    
                    # Model hızı
                    inference_ms = pred.get('inference_time', 0) * 1000
                    st.markdown(f"⚡ **Tahmin Hızı:** `{inference_ms:.1f} ms` ({pred.get('inference_time', 0):.3f}s)")
                    
                    # Tüm sınıfların olasılıkları
                    st.markdown("**Tüm Sınıflar İçin Tahmin Olasılıkları:**")
                    
                    df_probs = pd.DataFrame({
                        'Sınıf': pred['classes'],
                        'Olasılık': pred['probabilities']
                    }).sort_values('Olasılık', ascending=False)
                    
                    fig = go.Figure(data=[
                        go.Bar(x=df_probs['Olasılık'], y=df_probs['Sınıf'],
                               orientation='h', marker_color='#667eea',
                               text=[f"{v:.1%}" for v in df_probs['Olasılık']],
                               textposition='auto')
                    ])
                    fig.update_layout(
                        title_text="Sınıf Olasılıkları",
                        xaxis_title="Olasılık",
                        yaxis_title="Sınıf",
                        height=400,
                        margin=dict(l=100),
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                st.divider()
            
            # Batch işleme istatistikleri
            if 'predictions' in st.session_state and len(st.session_state.predictions) > 0:
                total_time = sum(p.get('inference_time', 0) for p in st.session_state.predictions)
                avg_time = total_time / len(st.session_state.predictions)
                
                col_stat1, col_stat2, col_stat3 = st.columns(3)
                with col_stat1:
                    st.metric("🏃 Toplam Hız", f"{total_time:.2f}s")
                with col_stat2:
                    st.metric("⏱️ Ortalama Hız/Resim", f"{avg_time*1000:.1f}ms")
                with col_stat3:
                    st.metric("📊 İşlem Hızı", f"{1/avg_time:.1f} resim/s")

# ============================================================================
# TAB 2: METRİKLER
# ============================================================================

with tab_metrics:
    st.subheader("📊 Model Performans Metrikleri")
    
    # Ana metrikler
    col1, col2, col3, col4 = st.columns(4)
    
    metrics_data = [
        ("📊 F1-Score", model_info.get('test_f1_macro', 0), col1),
        ("✓ Accuracy", model_info.get('test_accuracy', 0), col2),
        ("🎯 Precision", model_info.get('test_precision', 0), col3),
        ("📈 Recall", model_info.get('test_recall', 0), col4),
    ]
    
    for label, value, col in metrics_data:
        with col:
            st.markdown(f"""
            <div style="background: white; padding: 25px; border-radius: 15px; border-left: 5px solid #667eea; box-shadow: 0 4px 15px rgba(0,0,0,0.08);">
            <h3 style="color: #667eea; margin: 0 0 15px 0;">{label}</h3>
            <h2 style="color: #764ba2; font-size: 2.5em; margin: 0; font-weight: 700;">{value:.4f}</h2>
            </div>
            """, unsafe_allow_html=True)
    
    st.divider()
    
    # Averaging Methods (Macro vs Weighted)
    st.markdown("### 📈 Ortalama Metrikleri (Averaging Methods)")
    col_avg1, col_avg2 = st.columns(2)
    
    with col_avg1:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 25px; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
        <h4 style="color: white; margin: 0 0 10px 0;">📊 Macro Average</h4>
        <p style="font-size: 0.95em; margin: 5px 0; opacity: 0.9;">F1-Score: <strong>{model_info.get('test_f1_macro', 0):.4f}</strong></p>
        <p style="font-size: 0.95em; margin: 5px 0; opacity: 0.9;">Precision: <strong>{model_info.get('test_precision', 0):.4f}</strong></p>
        <p style="font-size: 0.95em; margin: 5px 0; opacity: 0.9;">Recall: <strong>{model_info.get('test_recall', 0):.4f}</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_avg2:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; padding: 25px; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
        <h4 style="color: white; margin: 0 0 10px 0;">⚖️ Weighted Average</h4>
        <p style="font-size: 0.95em; margin: 5px 0; opacity: 0.9;">F1-Score: <strong>{model_info.get('test_f1_weighted', 0):.4f}</strong></p>
        <p style="font-size: 0.95em; margin: 5px 0; opacity: 0.9;">Accuracy: <strong>{model_info.get('test_accuracy', 0):.4f}</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Numeric veriler (grafik için)
    per_class_numeric = []
    for cls in classes:
        metrics = model_info['per_class_metrics'].get(cls, {})
        per_class_numeric.append({
            'Sınıf': cls,
            'F1-Score': metrics.get('F1-Score', 0),
            'Precision': metrics.get('Precision', 0),
            'Recall': metrics.get('Recall', 0),
        })
    
    df_numeric = pd.DataFrame(per_class_numeric)
    
    # Formatted veriler (tablo gösterimi için)
    per_class_data = []
    for cls in classes:
        metrics = model_info['per_class_metrics'].get(cls, {})
        per_class_data.append({
            'Sınıf': cls,
            'F1-Score': f"{metrics.get('F1-Score', 0):.4f}",
            'Precision': f"{metrics.get('Precision', 0):.4f}",
            'Recall': f"{metrics.get('Recall', 0):.4f}",
        })
    
    df_metrics = pd.DataFrame(per_class_data)
    
    # Tablo (formatted)
    st.dataframe(df_metrics, use_container_width=True, hide_index=True)
    
    # Grafik: F1 vs Precision vs Recall (numeric)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df_numeric['Sınıf'], y=df_numeric['F1-Score'], name='F1-Score', marker_color='#667eea'))
    fig.add_trace(go.Bar(x=df_numeric['Sınıf'], y=df_numeric['Precision'], name='Precision', marker_color='#764ba2'))
    fig.add_trace(go.Bar(x=df_numeric['Sınıf'], y=df_numeric['Recall'], name='Recall', marker_color='#f093fb'))
    
    fig.update_layout(
        title_text="Sınıf Başına Performans Karşılaştırması",
        barmode='group',
        height=500,
        xaxis_title="Sınıf",
        yaxis_title="Skor",
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 3: GRAFİKLER
# ============================================================================

with tab_graphs:
    st.subheader("📈 Eğitim Grafikleri ve Hata Matrisi")
    
    # Grafikleri yükle
    try:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Training & Validation Loss/Accuracy")
            if Path('reports/training_history.png').exists():
                st.image(Image.open('reports/training_history.png'))
            else:
                st.warning("⚠️ training_history.png bulunamadı")
        
        with col2:
            st.markdown("### Normalized Confusion Matrix")
            if Path('reports/confusion_matrix.png').exists():
                st.image(Image.open('reports/confusion_matrix.png'))
            else:
                st.warning("⚠️ confusion_matrix.png bulunamadı")
    
    except Exception as e:
        st.error(f"❌ Grafik yüklenirken hata: {e}")

# ============================================================================
# TAB 4: BİLGİ
# ============================================================================

with tab_info:
    st.subheader("ℹ️ Proje Bilgileri")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style="background: white; padding: 25px; border-radius: 15px; border-left: 5px solid #667eea; box-shadow: 0 4px 15px rgba(0,0,0,0.08);">
        <h3 style="color: #667eea; margin: 0 0 15px 0;">🧠 Model Mimarisi</h3>
        <p style="color: #333; margin: 5px 0;"><strong>Temel Model:</strong> ResNet50</p>
        <p style="color: #333; margin: 5px 0;"><strong>Önceden Eğitim:</strong> ImageNet</p>
        <p style="color: #333; margin: 5px 0;"><strong>Transfer Learning:</strong> Evet</p>
        <p style="color: #333; margin: 5px 0;"><strong>Eğitim Stratejisi:</strong> Fine-tuning</p>
        <p style="color: #333; margin: 5px 0;"><strong>Toplam Parametre:</strong> 26.1M</p>
        <hr style="border: 1px solid #eee; margin: 15px 0;">
        <p style="color: #667eea; font-weight: 700; margin: 10px 0 5px 0;">FC Layer Yapısı:</p>
        <p style="color: #333; margin: 3px 0;">• 2048 → 1024 (BatchNorm + ReLU + Dropout)</p>
        <p style="color: #333; margin: 3px 0;">• 1024 → 512 (BatchNorm + ReLU + Dropout)</p>
        <p style="color: #333; margin: 3px 0;">• 512 → {num_classes} (Output)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background: white; padding: 25px; border-radius: 15px; border-left: 5px solid #764ba2; box-shadow: 0 4px 15px rgba(0,0,0,0.08);">
        <h3 style="color: #764ba2; margin: 0 0 15px 0;">⚙️ Eğitim Ayarları</h3>
        <p style="color: #333; margin: 5px 0;"><strong>Input Size:</strong> 224×224</p>
        <p style="color: #333; margin: 5px 0;"><strong>Batch Size:</strong> 16</p>
        <p style="color: #333; margin: 5px 0;"><strong>Epochs:</strong> 68</p>
        <p style="color: #333; margin: 5px 0;"><strong>Learning Rate:</strong> 1e-5 / 1e-4</p>
        <p style="color: #333; margin: 5px 0;"><strong>Optimizer:</strong> SGD w/ Momentum</p>
        <p style="color: #333; margin: 5px 0;"><strong>Loss Function:</strong> CrossEntropyLoss</p>
        <p style="color: #333; margin: 5px 0;"><strong>Scheduler:</strong> LinearLR + ReduceLROnPlateau</p>
        <p style="color: #333; margin: 5px 0;"><strong>Early Stopping:</strong> Patience 15</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    st.markdown(f"""
    <div style="background: white; border-left: 5px solid #667eea; padding: 30px; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.08); margin: 20px 0;">
    <h3 style="color: #667eea; margin: 0 0 20px 0; font-size: 1.3em;">🔧 Ön İşleme (Preprocessing)</h3>
    
    <div style="background: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 15px;">
    <p style="color: #2c3e50; font-weight: 700; margin: 0 0 10px 0; font-size: 1.05em;">📊 Eğitim Seti Augmentasyonu:</p>
    <p style="color: #333; margin: 3px 0; padding-left: 15px;">• <strong>Rotation:</strong> ±45°</p>
    <p style="color: #333; margin: 3px 0; padding-left: 15px;">• <strong>Scale:</strong> 0.85-1.15</p>
    <p style="color: #333; margin: 3px 0; padding-left: 15px;">• <strong>Affine Translation:</strong> ±15%</p>
    <p style="color: #333; margin: 3px 0; padding-left: 15px;">• <strong>ColorJitter</strong></p>
    <p style="color: #333; margin: 3px 0; padding-left: 15px;">• <strong>RandomCrop</strong></p>
    <p style="color: #333; margin: 3px 0; padding-left: 15px;">• <strong>RandomErasing</strong></p>
    </div>
    
    <div style="background: #f8f9fa; padding: 15px; border-radius: 10px;">
    <p style="color: #2c3e50; font-weight: 700; margin: 0 0 10px 0; font-size: 1.05em;">🎨 Görüntü Ön İşlemesi (tahmin = eğitim ile aynı):</p>
    <p style="color: #333; margin: 3px 0; padding-left: 15px;">• <strong>Ön işleme:</strong> model ile aynı mod ({eval_preprocess})</p>
    <p style="color: #333; margin: 3px 0; padding-left: 15px;">• <strong>ImageNet Normalizasyonu</strong> (mean/std)</p>
    <p style="color: #333; margin: 3px 0; padding-left: 15px;">• Eğitimde ek: flip, hafif renk/döndürme</p>
    </div>
    </div>
    """, unsafe_allow_html=True)

st.divider()
st.markdown("<p style='text-align: center; color: #999;'>🚗 Araba Gövde Sınıflandırması • AI Tabanlı Analiz</p>", unsafe_allow_html=True)
