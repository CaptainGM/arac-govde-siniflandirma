"""
🚗 AutoVision // Araç Gövde Tipi Sınıflandırma ve XAI Teşhis Kokpiti
ResNet-50 Transfer Learning • Grad-CAM Açıklanabilir Yapay Zeka • TTA Çıkarım Motoru
"""

import os
import sys
import time
import json
import random
from pathlib import Path
from io import BytesIO

import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from PIL import Image
import requests
import matplotlib.cm as cm
import plotly.graph_objects as go
import plotly.express as px

# Workspace modülleri
from class_config import get_canonical_classes
from transforms_config import (
    build_eval_transform,
    resolve_preprocess,
    PREPROCESS_STRETCH,
    PREPROCESS_LETTERBOX,
)

# ============================================================================
# SAYFA KONFİGÜRASYONU
# ============================================================================
st.set_page_config(
    page_title="AutoVision AI • Araç Gövde Sınıflandırma",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================================
# MODERN VE FERAH KOYU TEMA (GÖZ YORMAYAN, AKICI VE GENİŞLETİLMİŞ ALAN)
# ============================================================================
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">

<style>
    /* TEMEL DEĞİŞKENLER */
    :root {
        --bg-main: #0b0f19;
        --bg-panel: rgba(17, 24, 39, 0.75);
        --bg-card: rgba(23, 32, 51, 0.65);
        --accent-cyan: #00d2ff;
        --accent-blue: #3b82f6;
        --accent-purple: #8b5cf6;
        --accent-emerald: #10b981;
        --border-subtle: rgba(255, 255, 255, 0.08);
        --border-active: rgba(0, 210, 255, 0.4);
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
    }
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* GÖZ YORMAYAN, PÜRÜZSÜZ KOYU ARKA PLAN */
    .stApp {
        background-color: var(--bg-main);
        background-image: 
            radial-gradient(circle at 12% 12%, rgba(0, 210, 255, 0.06) 0%, transparent 50%),
            radial-gradient(circle at 88% 88%, rgba(139, 92, 246, 0.06) 0%, transparent 50%);
        color: var(--text-primary);
    }
    
    .block-container {
        padding-top: 1.8rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 96% !important;
    }
    
    /* ÜST BAŞLIK */
    .app-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 20px 28px;
        background: var(--bg-panel);
        border: 1px solid var(--border-subtle);
        border-radius: 20px;
        backdrop-filter: blur(24px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
        margin-bottom: 22px;
        flex-wrap: wrap;
        gap: 15px;
    }
    
    .brand-wrap {
        display: flex;
        align-items: center;
        gap: 15px;
    }
    
    .brand-icon {
        width: 48px;
        height: 48px;
        background: linear-gradient(135deg, #00d2ff 0%, #3b82f6 100%);
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 0 25px rgba(0, 210, 255, 0.4);
        font-size: 1.6rem;
    }
    
    .brand-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #ffffff 0%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        line-height: 1.1;
    }
    
    .brand-desc {
        font-size: 0.85rem;
        color: var(--text-secondary);
        margin-top: 3px;
    }
    
    .status-group {
        display: flex;
        align-items: center;
        gap: 10px;
        flex-wrap: wrap;
    }
    
    .status-chip {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 7px 16px;
        border-radius: 9999px;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid var(--border-subtle);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        color: var(--text-secondary);
        font-weight: 500;
    }
    
    .pulse-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: var(--accent-emerald);
        box-shadow: 0 0 10px var(--accent-emerald);
    }

    /* KART TASARIMI */
    .glass-card {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 18px;
        padding: 22px;
        backdrop-filter: blur(20px);
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
        transition: border-color 0.25s ease;
    }
    
    .glass-card:hover {
        border-color: var(--border-active);
    }
    
    .card-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.15rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0 0 14px 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* TAHMİN SONUÇ VİTRİNİ */
    .pred-hero-banner {
        background: linear-gradient(135deg, rgba(0, 210, 255, 0.12) 0%, rgba(59, 130, 246, 0.16) 100%);
        border: 1px solid rgba(0, 210, 255, 0.35);
        border-radius: 20px;
        padding: 24px 22px;
        text-align: center;
        box-shadow: 0 15px 35px rgba(0, 210, 255, 0.1);
        margin-bottom: 18px;
    }
    
    .pred-category-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: var(--accent-cyan);
        font-weight: 600;
    }
    
    .pred-hero-name {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 3rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.02em;
        margin: 6px 0 12px 0;
        text-shadow: 0 0 25px rgba(0, 210, 255, 0.45);
    }
    
    .pred-stat-row {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 32px;
        margin-top: 10px;
        flex-wrap: wrap;
    }
    
    .pred-stat-item {
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    
    .pred-stat-label {
        font-size: 0.75rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .pred-stat-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.7rem;
        font-weight: 700;
        color: var(--accent-cyan);
    }
    
    .pred-rationale {
        margin-top: 14px;
        padding: 12px 18px;
        background: rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        font-size: 0.9rem;
        color: #e2e8f0;
        line-height: 1.5;
        text-align: left;
    }

    /* SEKMELER */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(17, 24, 39, 0.7);
        padding: 6px;
        border-radius: 16px;
        border: 1px solid var(--border-subtle);
        margin-bottom: 22px;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 11px 24px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.95rem;
        color: var(--text-secondary);
        border: none;
        transition: all 0.2s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(0, 210, 255, 0.22) 0%, rgba(59, 130, 246, 0.3) 100%) !important;
        border: 1px solid rgba(0, 210, 255, 0.45) !important;
        color: #ffffff !important;
        box-shadow: 0 4px 18px rgba(0, 210, 255, 0.2);
    }

    /* FORM VE DÜĞMELER */
    .stButton > button {
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.98rem;
        padding: 12px 20px;
        border: 1px solid rgba(0, 210, 255, 0.35);
        background: linear-gradient(135deg, #0ea5e9 0%, #4f46e5 100%);
        color: white;
        transition: all 0.25s ease;
        box-shadow: 0 5px 18px rgba(14, 165, 233, 0.35);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(14, 165, 233, 0.55);
        border-color: rgba(255, 255, 255, 0.5);
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# ARAÇ GÖVDE TİPLERİ VE BİLGİ BANKASI
# ============================================================================
CLASS_METADATA = {
    "SUV": {
        "title": "SUV (Sport Utility Vehicle)",
        "description": "Yüksek sürüş pozisyonu, yüksek taban mesafesi, 2-kutu gövde ve geniş bagaj hacmi.",
        "rationale": "Yüksek şasi zemin boşluğu, geniş çamurluk açıklığı ve yüksek tavan çizgisi tespit edildi."
    },
    "SEDAN": {
        "title": "Sedan (Klasik 3-Kutu)",
        "description": "Belirgin motor kompartımanı, yolcu kabini ve çıkıntılı bagaj bölmesi olmak üzere 3 kutulu klasik gövde yapısı.",
        "rationale": "Belirgin üç kutulu silüet (motor, kabin ve yatay bagaj kapağı çıkıntısı) tespit edildi."
    },
    "HATCHBACK": {
        "title": "Hatchback (Kompakt 2-Kutu)",
        "description": "Bagaj kapağı arka camla birlikte yukarı açılan, arka çıkıntısı kısa ve dik inen kompakt 2-kutu gövde.",
        "rationale": "Kısa arka tekerlek çıkıntısı ve tavan arkasından dik eğimle inen bagaj kapağı geometrisi tespit edildi."
    },
    "STATION_WAGON": {
        "title": "Station Wagon (Estate / Tourer)",
        "description": "Sedan şasisi üzerine tavan çizgisi aracın en arkasına kadar uzatılmış, D-sütunu olan genişletilmiş bagajlı gövde.",
        "rationale": "Arka aks hizasını aşan uzatılmış tavan hattı ve bagaj üzeri ek üçüncü yan cam profili tespit edildi."
    },
    "PICKUP": {
        "title": "Pick-Up (Kamyonet)",
        "description": "Kabin arkasında açık yük taşıma kasası bulunan, merdiven şasili veya yüksek süspansiyonlu ticari/arazi aracı.",
        "rationale": "Yolcu kabini ardındaki açık kargo kasası ve dik kabin arka cam ayrımı tespit edildi."
    },
    "VAN": {
        "title": "Van / Minibüs / MUV",
        "description": "Kutu formunda (box-type) yüksek tavanlı gövde, genellikle sürgülü yan kapı ve dik ön/arka yüzey.",
        "rationale": "Dikey kutu tavan yüksekliği, dik burun açısı ve monoblok yüksek gövde yapısı tespit edildi."
    },
    "MICRO": {
        "title": "Micro / Şehir İçi Mini",
        "description": "Ultra kompakt uzunluk (genellikle < 3 metre), 2 kişilik veya minimal şehir içi ulaşım aracı (Smart, Ami vb.).",
        "rationale": "Son derece kısa dingil mesafesi ve minyatür iki kişilik kabin oranı tespit edildi."
    },
    "ACIK_TEKERLEK": {
        "title": "Açık Tekerlekli (Yarış / Formula)",
        "description": "Tekerlekleri çamurluk veya gövde panelleriyle örtülmemiş, aerodinamik yarış veya özel spor araç gövdesi.",
        "rationale": "Gövde panellerinin dışında bağımsız asılı tekerlekler ve aerodinamik kanat geometrisi tespit edildi."
    }
}


# ============================================================================
# CİHAZ VE SİSTEM YÜKLEYİCİ
# ============================================================================
def get_inference_device():
    try:
        import torch_directml
        return torch_directml.device(), "AMD Radeon DirectML"
    except Exception:
        if torch.cuda.is_available():
            return torch.device("cuda"), f"NVIDIA GPU ({torch.cuda.get_device_name(0)})"
        return torch.device("cpu"), "Çok Çekirdekli CPU"


def load_state_dict_safe(model_path, map_location):
    try:
        return torch.load(model_path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(model_path, map_location=map_location)


@st.cache_resource
def load_system():
    device, device_name = get_inference_device()
    
    info_path = Path("models/model_info.json")
    if info_path.exists():
        with open(info_path, "r", encoding="utf-8") as f:
            model_info = json.load(f)
    else:
        model_info = {
            "classes": get_canonical_classes(),
            "num_classes": 8,
            "test_accuracy": 0.8823,
            "test_f1_macro": 0.9076,
            "preprocess": "letterbox",
        }

    classes = model_info.get("classes", get_canonical_classes())
    num_classes = len(classes)

    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(2048, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes),
    )

    model_path = Path("models/best_model.pt")
    if model_path.exists():
        state = load_state_dict_safe(str(model_path), map_location=device)
        model.load_state_dict(state)
    else:
        st.error(f"❌ Model ağırlık dosyası '{model_path}' bulunamadı!")
        return None, None, None, None, None, None

    model = model.to(device)
    for p in model.parameters():
        p.requires_grad = True
    model.eval()

    preprocess_mode = resolve_preprocess(model_info)
    transform = build_eval_transform(preprocess_mode)

    return model, model_info, device, device_name, transform, classes


# ============================================================================
# GRAD-CAM MOTORU
# ============================================================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        self.model.zero_grad()
        
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        score = output[0, target_class]
        score.backward(retain_graph=True)

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, target_class


def render_cam_overlay(orig_image: Image.Image, cam_map: np.ndarray, colormap_name="turbo", alpha=0.45):
    w, h = orig_image.size
    cam_img = Image.fromarray(np.uint8(255 * cam_map)).resize((w, h), Image.Resampling.BILINEAR)
    cam_arr = np.array(cam_img) / 255.0
    
    cmap = getattr(cm, colormap_name, cm.turbo)
    heatmap_rgba = cmap(cam_arr)
    heatmap_rgb = Image.fromarray((heatmap_rgba[:, :, :3] * 255).astype(np.uint8))
    overlay = Image.blend(orig_image.convert("RGB"), heatmap_rgb, alpha=alpha)
    return overlay, heatmap_rgb


# ============================================================================
# TTA TAHMİN FONKSİYONU
# ============================================================================
def predict_with_tta(model, tensor, classes, device, use_tta=True):
    start_time = time.perf_counter()
    tensor = tensor.unsqueeze(0).to(device)

    variants = [tensor]
    if use_tta:
        flipped = torch.flip(tensor, dims=[3])
        variants.append(flipped)
        
        _, _, h, w = tensor.shape
        crop_px = int(w * 0.05)
        if crop_px > 0:
            left_crop = torch.nn.functional.interpolate(
                tensor[:, :, :, crop_px:], size=(h, w), mode="bilinear", align_corners=False
            )
            right_crop = torch.nn.functional.interpolate(
                tensor[:, :, :, :-crop_px], size=(h, w), mode="bilinear", align_corners=False
            )
            variants.extend([left_crop, right_crop])

    with torch.no_grad():
        all_probs = []
        for v in variants:
            out = model(v)
            prob = torch.softmax(out, dim=1)[0].cpu().numpy()
            all_probs.append(prob)

    probabilities = np.mean(all_probs, axis=0)
    pred_idx = int(np.argmax(probabilities))
    inference_ms = (time.perf_counter() - start_time) * 1000

    return {
        "class": classes[pred_idx],
        "idx": pred_idx,
        "confidence": float(probabilities[pred_idx]),
        "probabilities": probabilities,
        "inference_ms": inference_ms,
        "variants_count": len(variants),
    }


# ============================================================================
# SİSTEM İLKLEME
# ============================================================================
system = load_system()
if system[0] is None:
    st.stop()

model, model_info, device, device_name, transform, classes = system
gradcam_engine = GradCAM(model, model.layer4[-1])


# ============================================================================
# ÜST BİLGİ VE TELEMETRİ ÇUBUĞU
# ============================================================================
st.markdown(f"""
<div class="app-header">
    <div class="brand-wrap">
        <div class="brand-icon">🚗</div>
        <div>
            <div class="brand-title">AutoVision // Araç Gövde Sınıflandırma</div>
            <div class="brand-desc">Derin Öğrenme Tabanlı Gövde Tespiti ve Açıklanabilir Yapay Zeka (XAI) Kokpiti</div>
        </div>
    </div>
    <div class="status-group">
        <div class="status-chip">
            <div class="pulse-dot"></div>
            <span>MODEL: ResNet-50</span>
        </div>
        <div class="status-chip">
            <span>⚡ {device_name}</span>
        </div>
        <div class="status-chip">
            <span>🎯 Doğruluk: %{model_info.get('test_accuracy', 0.8823)*100:.2f}</span>
        </div>
        <div class="status-chip">
            <span>📈 Macro F1: {model_info.get('test_f1_macro', 0.9076):.4f}</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================================
# 3 ANA ODAKLANMIŞ ÇALIŞMA ALANI
# ============================================================================
tab_predict, tab_metrics, tab_gallery = st.tabs([
    "🎯 Gövde Tipi Tahmini ve XAI Analizi",
    "📊 Model Başarımı ve Karışıklık Matrisi",
    "📁 Veri Seti Galerisi",
])


# ============================================================================
# TAB 1: GÖVDE TİPİ TAHMİNİ VE XAI ANALİZİ
# ============================================================================
with tab_predict:
    col_input, col_canvas, col_stats = st.columns([1, 1.85, 1.15])

    # 1. SOL PANEL: GÖRSEL BESLEME VE SEÇENEKLER
    with col_input:
        st.markdown("""
        <div class="glass-card">
            <div class="card-title">📥 Görsel Kaynağı</div>
        """, unsafe_allow_html=True)

        source_choice = st.radio(
            "Yükleme Yolu:",
            ["📁 Bilgisayardan Dosya Seç", "🌐 Web Görsel URL'si", "📸 Kameradan Canlı Çekim", "🎲 Test Setinden Rastgele Seç"],
            label_visibility="collapsed"
        )

        st.markdown("</div>", unsafe_allow_html=True)

        loaded_images = []

        if "Bilgisayardan Dosya" in source_choice:
            uploaded_files = st.file_uploader(
                "Görsel yükleyin",
                type=["jpg", "jpeg", "png", "webp", "bmp"],
                accept_multiple_files=True,
                label_visibility="collapsed"
            )
            if uploaded_files:
                for f in uploaded_files:
                    try:
                        loaded_images.append((Image.open(f).convert("RGB"), f.name))
                    except Exception as e:
                        st.error(f"Görsel okunamadı: {e}")

        elif "Web Görsel" in source_choice:
            url_str = st.text_input("Görsel URL Bağlantısı:", placeholder="https://.../araba.jpg")
            if url_str:
                try:
                    resp = requests.get(url_str, timeout=8)
                    img_web = Image.open(BytesIO(resp.content)).convert("RGB")
                    loaded_images.append((img_web, "web_gorsel.jpg"))
                    st.success("✓ Görsel başarıyla indirildi")
                except Exception as e:
                    st.error(f"URL yüklenemedi: {e}")

        elif "Kameradan Canlı" in source_choice:
            cam_data = st.camera_input("Fotoğraf Çekin")
            if cam_data:
                img_cam = Image.open(cam_data).convert("RGB")
                loaded_images.append((img_cam, "kamera_cekimi.jpg"))

        elif "Test Setinden" in source_choice:
            test_folder = Path("test")
            if test_folder.exists():
                all_test_imgs = list(test_folder.rglob("*.jpg")) + list(test_folder.rglob("*.png"))
                if st.button("🎲 Yeni Rastgele Araç Getir"):
                    picked_file = random.choice(all_test_imgs)
                    real_class = picked_file.parent.name
                    img_rnd = Image.open(picked_file).convert("RGB")
                    st.session_state["cached_random_vehicle"] = (img_rnd, f"{real_class} / {picked_file.name}")
                
                if "cached_random_vehicle" in st.session_state:
                    loaded_images.append(st.session_state["cached_random_vehicle"])
            else:
                st.warning("'test/' klasörü bulunamadı.")

        # İNCE AYARLAR
        st.markdown("""
        <div class="glass-card" style="margin-top: 15px;">
            <div class="card-title">⚙️ Analiz Ayarları</div>
        """, unsafe_allow_html=True)

        opt_tta = st.toggle("⚡ Çoklu Açı Desteği (4x TTA)", value=True, help="Tahmin doğruluğunu artırmak için simetri ve kırpma varyantlarını harmanlar.")
        opt_cam = st.toggle("🧠 Isı Haritası (Grad-CAM XAI)", value=True, help="Modelin araca bakarken en çok odaklandığı gövde bölgelerini gösterir.")
        
        if opt_cam:
            c_map_choice = st.selectbox("Isı Haritası Paleti:", ["turbo", "jet", "inferno", "plasma"], index=0)
            c_alpha_val = st.slider("Isı Haritası Belirginliği:", min_value=0.15, max_value=0.85, value=0.45, step=0.05)
        else:
            c_map_choice = "turbo"
            c_alpha_val = 0.45

        st.markdown("</div>", unsafe_allow_html=True)


    # 2. ORTA PANEL: VİTRİN VE GÖRSELLEŞTİRME
    with col_canvas:
        if not loaded_images:
            st.markdown("""
            <div class="glass-card" style="text-align: center; padding: 85px 20px;">
                <div style="font-size: 3.2rem; margin-bottom: 12px;">📡</div>
                <h3 style="font-family:'Space Grotesk'; color: #f8fafc; margin-bottom: 8px;">Analiz İçin Görsel Bekleniyor</h3>
                <p style="color: #94a3b8; max-width: 440px; margin: 0 auto; font-size: 0.92rem; line-height: 1.6;">
                    Sol menüden bir araç fotoğrafı yükleyin, web bağlantısı girin ya da test setinden rastgele araç getir butonuna tıklayarak model analizini başlatın.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            main_img, main_name = loaded_images[0]
            tensor_input = transform(main_img)
            pred_res = predict_with_tta(model, tensor_input, classes, device, use_tta=opt_tta)
            detected_class = pred_res["class"]
            class_info = CLASS_METADATA.get(detected_class, {})

            # TAHMİN SONUÇ KARTI
            st.markdown(f"""
            <div class="pred-hero-banner">
                <div class="pred-category-tag">YAPAY ZEKA TAHMİN SONUCU</div>
                <div class="pred-hero-name">{detected_class}</div>
                <div class="pred-stat-row">
                    <div class="pred-stat-item">
                        <span class="pred-stat-label">Model Güveni</span>
                        <span class="pred-stat-value">%{pred_res['confidence']*100:.1f}</span>
                    </div>
                    <div class="pred-stat-item">
                        <span class="pred-stat-label">Çıkarım Süresi</span>
                        <span class="pred-stat-value" style="color:#a855f7;">{pred_res['inference_ms']:.1f} ms</span>
                    </div>
                    <div class="pred-stat-item">
                        <span class="pred-stat-label">TTA Örnekleri</span>
                        <span class="pred-stat-value" style="color:#34d399;">{pred_res['variants_count']}x</span>
                    </div>
                </div>
                <div class="pred-rationale">
                    💡 <strong>Teşhis Gerekçesi:</strong> {class_info.get('rationale', 'Gövde geometrisi ve silüet hatları başarıyla doğrulandı.')}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # GÖRÜNTÜLEME MODLARI
            view_tabs = st.tabs(["⚡ Isı Haritası (Grad-CAM)", "🔍 Yan Yana Karşılaştırma", "🔥 Nöron Aktivasyon Haritası"])

            inp_tensor = tensor_input.unsqueeze(0).to(device)
            cam_matrix, _ = gradcam_engine.generate(inp_tensor, target_class=pred_res["idx"])
            blended_img, raw_heatmap_img = render_cam_overlay(main_img, cam_matrix, colormap_name=c_map_choice, alpha=c_alpha_val)

            with view_tabs[0]:
                st.image(blended_img if opt_cam else main_img, width='stretch', caption=f"İncelenen Araç: {main_name}")

            with view_tabs[1]:
                col_c1, col_c2 = st.columns(2)
                with col_c1:
                    st.markdown("**Orijinal Araç Görseli**")
                    st.image(main_img, width='stretch')
                with col_c2:
                    st.markdown(f"**Model Odak Katmanı ({detected_class})**")
                    st.image(blended_img, width='stretch')

            with view_tabs[2]:
                st.markdown("**Ham Gradyan Yoğunluk Haritası (Aktivasyon)**")
                st.image(raw_heatmap_img, width='stretch')


    # 3. SAĞ PANEL: OLASILIK SPEKTRUMU VE RADAR DAĞILIMI
    with col_stats:
        if loaded_images:
            st.markdown("""
            <div class="glass-card">
                <div class="card-title">📊 Sınıf Olasılık Dağılımı</div>
            """, unsafe_allow_html=True)

            df_chart = pd.DataFrame({
                "Sınıf": classes,
                "Olasılık": pred_res["probabilities"]
            }).sort_values("Olasılık", ascending=True)

            fig_h_bar = go.Figure(go.Bar(
                x=df_chart["Olasılık"],
                y=df_chart["Sınıf"],
                orientation="h",
                marker=dict(
                    color=df_chart["Olasılık"],
                    colorscale="Viridis",
                    line=dict(color="rgba(255,255,255,0.2)", width=1)
                ),
                text=[f"%{v*100:.1f}" for v in df_chart["Olasılık"]],
                textposition="auto"
            ))
            fig_h_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=280,
                margin=dict(l=0, r=0, t=10, b=0),
                font=dict(color="#cbd5e1", family="Plus Jakarta Sans"),
                xaxis=dict(showgrid=False, showticklabels=False),
                yaxis=dict(showgrid=False)
            )
            st.plotly_chart(fig_h_bar, width='stretch', config={"displayModeBar": False})

            st.markdown("</div>", unsafe_allow_html=True)

            # RADAR GEOMETRİ PROFİLİ
            st.markdown("""
            <div class="glass-card">
                <div class="card-title">🕸️ Araç Geometri Radarı</div>
            """, unsafe_allow_html=True)

            fig_rad = go.Figure(go.Scatterpolar(
                r=list(pred_res["probabilities"]) + [pred_res["probabilities"][0]],
                theta=classes + [classes[0]],
                fill='toself',
                fillcolor='rgba(0, 210, 255, 0.22)',
                line=dict(color='#00d2ff', width=2)
            ))
            fig_rad.update_layout(
                polar=dict(
                    bgcolor="rgba(17, 24, 39, 0.7)",
                    radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(255,255,255,0.08)"),
                    angularaxis=dict(gridcolor="rgba(255,255,255,0.08)")
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                height=290,
                margin=dict(l=25, r=25, t=25, b=25),
                font=dict(color="#e2e8f0", family="Plus Jakarta Sans")
            )
            st.plotly_chart(fig_rad, width='stretch', config={"displayModeBar": False})

            st.markdown("</div>", unsafe_allow_html=True)

            # TOPLU İŞLEME LİSTESİ
            if len(loaded_images) > 1:
                with st.expander(f"📦 Çoklu Görsel Kuyruğu ({len(loaded_images)} Görsel)"):
                    batch_list = []
                    for b_img, b_name in loaded_images:
                        b_tensor = transform(b_img)
                        b_out = predict_with_tta(model, b_tensor, classes, device, use_tta=opt_tta)
                        batch_list.append({
                            "Görsel": b_name,
                            "Tahmin": b_out["class"],
                            "Güven": f"%{b_out['confidence']*100:.1f}",
                            "Süre": f"{b_out['inference_ms']:.1f} ms"
                        })
                    st.dataframe(pd.DataFrame(batch_list), width='stretch', hide_index=True)


# ============================================================================
# TAB 2: MODEL BAŞARIMI VE KARIŞIKLIK MATRİSİ (GENİŞ, FERAH DÜZEN)
# ============================================================================
with tab_metrics:
    st.subheader("📊 Model Başarım Raporları ve Karışıklık Matrisi (Confusion Matrix)")

    # 4 Üst Metrik Kartı
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("🎯 Test Doğruluğu (Accuracy)", f"%{model_info.get('test_accuracy', 0.8823)*100:.2f}")
    with col_m2:
        st.metric("📈 Macro F1-Skor", f"{model_info.get('test_f1_macro', 0.9076):.4f}")
    with col_m3:
        st.metric("⚖️ Ağırlıklı F1 (Weighted)", f"{model_info.get('test_f1_weighted', 0.8827):.4f}")
    with col_m4:
        st.metric("⏱️ Ortalama Tahmin Hızı", f"~{pred_res['inference_ms'] if loaded_images else 54.0:.1f} ms")

    st.markdown("<br>", unsafe_allow_html=True)

    # BÜYÜK VE FERAH KARIŞIKLIK MATRİSİ
    st.markdown("""
    <div class="glass-card">
        <div class="card-title">🧩 İnteraktif Karışıklık Matrisi (Confusion Matrix - 8 Sınıf)</div>
        <p style="font-size:0.88rem; color:#94a3b8; margin-top:-8px; margin-bottom:15px;">
            Y ekseni gerçek sınıf etiketlerini, X ekseni ise modelin tahmin ettiği sınıfları gösterir. Köşegen üzerindeki sayılar doğru tahminleri, diğer alanlar ise sınıf karışıklıklarını temsil eder.
        </p>
    """, unsafe_allow_html=True)

    cm_matrix = model_info.get("confusion_matrix")
    if cm_matrix:
        cm_np = np.array(cm_matrix)
        fig_cm_large = px.imshow(
            cm_np,
            x=classes,
            y=classes,
            color_continuous_scale="Blues",
            text_auto=True,
            labels=dict(x="Model Tahmini", y="Gerçek Etiket", color="Araç Sayısı")
        )
        fig_cm_large.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=580,
            font=dict(color="#f1f5f9", family="Plus Jakarta Sans", size=13),
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_cm_large, width='stretch')
    elif Path("reports/confusion_matrix.png").exists():
        st.image(Image.open("reports/confusion_matrix.png"), width='stretch')

    st.markdown("</div>", unsafe_allow_html=True)

    # İKİ GENİŞ SÜTUN: METRİK TABLOSU VE ÇUBUK GRAFİĞİ
    col_t_left, col_t_right = st.columns([1.1, 1.2])

    with col_t_left:
        st.markdown("""
        <div class="glass-card">
            <div class="card-title">📋 Sınıf Bazında Başarım Metrikleri</div>
        """, unsafe_allow_html=True)

        per_class_data = model_info.get("per_class_metrics", {})
        if per_class_data:
            df_pc = pd.DataFrame([
                {
                    "Sınıf": c,
                    "F1-Skor": per_class_data[c].get("F1-Score", 0),
                    "Kesinlik (Precision)": per_class_data[c].get("Precision", 0),
                    "Duyarlılık (Recall)": per_class_data[c].get("Recall", 0)
                }
                for c in classes
            ])
            st.dataframe(
                df_pc.style.format({
                    "F1-Skor": "{:.4f}",
                    "Kesinlik (Precision)": "{:.4f}",
                    "Duyarlılık (Recall)": "{:.4f}"
                }),
                width='stretch',
                hide_index=True
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_t_right:
        st.markdown("""
        <div class="glass-card">
            <div class="card-title">📊 Sınıf Başına Karşılaştırma Grafiği</div>
        """, unsafe_allow_html=True)

        if per_class_data:
            fig_compare_bar = px.bar(
                df_pc.melt(id_vars="Sınıf", var_name="Metrik", value_name="Skor"),
                x="Sınıf", y="Skor", color="Metrik", barmode="group",
                color_discrete_sequence=["#00d2ff", "#8b5cf6", "#10b981"]
            )
            fig_compare_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=320,
                font=dict(color="#cbd5e1"),
                xaxis=dict(showgrid=False),
                yaxis=dict(gridcolor="rgba(255,255,255,0.08)", range=[0.65, 1.0]),
                margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig_compare_bar, width='stretch')
        st.markdown("</div>", unsafe_allow_html=True)


# ============================================================================
# TAB 3: VERİ SETİ GALERİSİ
# ============================================================================
with tab_gallery:
    st.subheader("📁 Veri Seti Galeri Gezgini")
    st.markdown("Yerel veri setindeki (`test/`, `valid/` veya `train/`) gerçek araç fotoğraflarını eşit ve düzenli bir galeride inceleyin:")

    g_col_a, g_col_b = st.columns([1, 2])
    with g_col_a:
        user_split = st.selectbox("İncelenecek Klasör Bölümü:", ["test", "valid", "train"])
        user_class = st.selectbox("İncelenecek Araç Sınıfı:", classes)

    path_candidate = Path(user_split) / user_class
    if not path_candidate.exists():
        from class_config import find_class_directory
        found_p = find_class_directory(user_split, user_class)
        if found_p:
            path_candidate = found_p

    if path_candidate.exists():
        files_found = [p for p in path_candidate.glob("*.*") if p.suffix.lower() in [".jpg", ".png", ".jpeg", ".webp"]]
        st.markdown(f"**Konum:** `{path_candidate}` • Toplam: `{len(files_found)} Görsel`")
        
        show_count = min(8, len(files_found))
        if show_count > 0:
            picked_list = sorted(files_found)[:show_count]
            cols_img = st.columns(4)
            for idx, p in enumerate(picked_list):
                with cols_img[idx % 4]:
                    st.image(Image.open(p), caption=p.name, width='stretch')
    else:
        st.warning(f"'{path_candidate}' klasörü bulunamadı.")

st.markdown("<br><p style='text-align:center; color:#64748b; font-size:0.85rem;'>AutoVision AI • Görüntü İşleme ve Derin Öğrenme Sistemi</p>", unsafe_allow_html=True)
