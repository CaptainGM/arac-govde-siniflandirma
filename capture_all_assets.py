import sys
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

sys.stdout.reconfigure(encoding='utf-8')

OUTPUT_DIR = Path("assets/screenshots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        # Geniş masaüstü görünümü (1600 x 1000)
        page = browser.new_page(viewport={"width": 1600, "height": 1000}, device_scale_factor=1.5)
        
        print("1. Opening http://localhost:8501 ...")
        page.goto("http://localhost:8501", wait_until="networkidle", timeout=60000)
        time.sleep(3)

        # 1. SUV Tahmin ve Kokpit Genel Görünümü
        print("Capturing 01_dashboard_overview.png ...")
        page.screenshot(path=str(OUTPUT_DIR / "01_dashboard_overview.png"))

        # 2. Yan Yana Karşılaştırma Sekmesi
        print("Clicking Yan Yana Karşılaştırma ...")
        try:
            side_tab = page.locator('div[data-testid="stTab"]:has-text("Yan Yana Karşılaştırma")').first
            if side_tab.count() > 0:
                side_tab.click()
                time.sleep(1.5)
                page.screenshot(path=str(OUTPUT_DIR / "02_xai_comparison_sidebyside.png"))
                
                # Isı Haritası sekmesine geri dön
                cam_tab = page.locator('div[data-testid="stTab"]:has-text("Isı Haritası (Grad-CAM)")').first
                if cam_tab.count() > 0:
                    cam_tab.click()
                    time.sleep(1)
        except Exception as e:
            print("Error capturing sidebyside:", e)

        # 3. Hazır Araçları Tek Tek Test Etme
        archetypes = [
            ("Sedan", "03_inference_sedan.png"),
            ("Pick-Up", "04_inference_pickup.png"),
            ("Açık Tekerlek", "05_inference_openwheel.png"),
            ("Van", "06_inference_van.png"),
            ("Hatchback", "07_inference_hatchback.png"),
            ("Station Wagon", "08_inference_stationwagon.png"),
            ("Micro", "09_inference_micro.png"),
        ]

        for label_sub, filename in archetypes:
            print(f"Selecting {label_sub} -> {filename} ...")
            try:
                sbox = page.locator('div[data-testid="stSelectbox"]').first
                sbox.click()
                time.sleep(0.8)
                opt = page.locator(f'li[role="option"]:has-text("{label_sub}"), div[role="option"]:has-text("{label_sub}")').first
                if opt.count() > 0:
                    opt.click()
                    time.sleep(2.5)
                    page.screenshot(path=str(OUTPUT_DIR / filename))
                else:
                    print(f"Option not found: {label_sub}")
            except Exception as e:
                print(f"Error selecting {label_sub}:", e)

        # 4. Tab 2: Karışıklık Matrisi ve Başarım
        print("Switching to Tab 2: Model Başarımı ...")
        tab2 = page.locator('div[data-testid="stTab"]:has-text("Model Başarımı ve Karışıklık Matrisi")').first
        tab2.click()
        time.sleep(2.5)
        page.screenshot(path=str(OUTPUT_DIR / "10_confusion_matrix_dashboard.png"))

        # 5. Tab 3: 8 Araç Gövde Tipi Rehberi
        print("Switching to Tab 3: Gövde Tipi Rehberi ...")
        tab3 = page.locator('div[data-testid="stTab"]:has-text("8 Araç Gövde Tipi Rehberi")').first
        tab3.click()
        time.sleep(2.5)
        page.screenshot(path=str(OUTPUT_DIR / "11_vehicle_body_guide.png"))

        # 6. Tab 4: Veri Seti Galeri Gezgini
        print("Switching to Tab 4: Veri Seti Galeri Gezgini ...")
        tab4 = page.locator('div[data-testid="stTab"]:has-text("Veri Seti Galeri Gezgini")').first
        tab4.click()
        time.sleep(3)
        page.screenshot(path=str(OUTPUT_DIR / "12_dataset_gallery_explorer.png"))

        # 7. Tab 5: Sistem Mimarisi ve Rapor
        print("Switching to Tab 5: Sistem Mimarisi ve Rapor ...")
        tab5 = page.locator('div[data-testid="stTab"]:has-text("Sistem Mimarisi ve Rapor")').first
        tab5.click()
        time.sleep(2.5)
        page.screenshot(path=str(OUTPUT_DIR / "13_system_architecture_report.png"))

        browser.close()
        print("ALL 13 HIGH-RES APPLICATION SCREENSHOTS CAPTURED SUCCESSFULLY!")

if __name__ == "__main__":
    run()
