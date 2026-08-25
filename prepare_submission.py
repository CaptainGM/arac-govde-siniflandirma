

import os
import shutil
import zipfile
from pathlib import Path
import torch

def compress_model_to_fp16(src_model_path, dst_model_path):
    print("📦 Model yükleniyor ve FP16 (float16) formatına dönüştürülüyor...")
    try:
        state_dict = torch.load(src_model_path, map_location="cpu")
    except Exception as e:
        print(f"❌ Model yüklenirken hata oluştu: {e}")
        return False
        
    
    fp16_state_dict = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            fp16_state_dict[k] = v.half()
        else:
            fp16_state_dict[k] = v
            
    try:
        torch.save(fp16_state_dict, dst_model_path)
        size_mb = os.path.getsize(dst_model_path) / (1024 * 1024)
        print(f"✅ Sıkıştırılmış FP16 Model Başarıyla Kaydedildi: {dst_model_path} ({size_mb:.2f} MB)")
        return True
    except Exception as e:
        print(f"❌ Model kaydedilirken hata oluştu: {e}")
        return False

def main():
    print("=" * 80)
    print("🎓 YAZLAB 2 PROJE 3 - OTOMATİK TESLİMAT HAZIRLAMA ARACI")
    print("=" * 80)
    
    
    print("\nLütfen teslimat klasörü ismi için öğrenci numaralarını girin.")
    print("Örnek: Tek kişi için '220201001', çift kişi için '220201001_220201002'")
    
    student_input = input("Klasör İsmi (örn. 220201001_220201002): ").strip()
    if not student_input:
        print("⚠️ Giriş yapılmadı!")
        folder_name = "teslimat"
    else:
        folder_name = student_input.replace("/", "").replace("\\", "")

    project_root = Path(__file__).parent.resolve()
    target_dir = project_root / folder_name
    
    
    if target_dir.exists():
        shutil.rmtree(target_dir)
        
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📂 Teslimat klasörü oluşturuldu: {target_dir}")
    
    
    src_model = project_root / "models" / "best_model.pt"
    if not src_model.exists():
        src_model = project_root / "best_model.pt"
        
    if not src_model.exists():
        print(f"❌ Hata: Kaynak model bulunamadı! 'models/best_model.pt' dosyasının var olduğundan emin olun.")
        return
        
    dst_model = target_dir / "best_model.pt"
    success = compress_model_to_fp16(src_model, dst_model)
    if not success:
        return
        
    
    src_script = project_root / "PredictionScript.txt"
    if not src_script.exists():
        print("❌ Hata: 'PredictionScript.txt' bulunamadı!")
        return
        
    dst_script = target_dir / "PredictionScript.txt"
    shutil.copy2(src_script, dst_script)
    print(f"📝 'PredictionScript.txt' başarıyla kopyalandı.")
    
   
    zip_filename = project_root / f"{folder_name}.zip"
    if zip_filename.exists():
        zip_filename.unlink()
        
    print(f"\n🗜️ Klasör zip arşivine dönüştürülüyor...")
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in target_dir.iterdir():
            if file.is_file():
                zipf.write(file, arcname=Path(folder_name) / file.name)
                
    zip_size_mb = os.path.getsize(zip_filename) / (1024 * 1024)
    print(f"✅ Zip arşivi başarıyla oluşturuldu: {zip_filename.name}")
    print(f"📊 Toplam Zipli Dosya Boyutu: {zip_size_mb:.2f} MB")
    
    
    print("\n" + "=" * 80)
    print("🔍 TESLİMAT UYGUNLUK RAPORU")
    print("=" * 80)
    
    warnings = []
    
    
    if zip_size_mb > 95.0:
        warnings.append(f"❌ Hata: Zipli dosya boyutu {zip_size_mb:.2f} MB! (Maksimum limit 95 MB'ı aşıyor!)")
    else:
        print(f"  ✓ Zipli Dosya Boyutu: {zip_size_mb:.2f} MB (95 MB sınırının altında, ÇOK İYİ!)")
        
    
    has_script = (target_dir / "PredictionScript.txt").exists()
    has_model = (target_dir / "best_model.pt").exists()
    
    if has_script:
        print("  ✓ PredictionScript.txt mevcut.")
    else:
        warnings.append("❌ Hata: PredictionScript.txt bulunamadı!")
        
    if has_model:
        print("  ✓ best_model.pt mevcut.")
    else:
        warnings.append("❌ Hata: best_model.pt bulunamadı!")
        
    
    print(f"  ✓ Klasör ismi: {folder_name}")
    
    if warnings:
        print("\n⚠️ DIKKAT! Aşağıdaki hataları düzeltmeniz gerekiyor:")
        for w in warnings:
            print(f"  {w}")
    else:
        print("\n🎉 TEBRİKLER! Klasörünüz ve Zip dosyanız teslimat kurallarına %100 UYGUNDUR!")
        print(f"👉 Yüklemeniz gereken nihai dosya: '{zip_filename.name}'")
        
    print("=" * 80)

if __name__ == "__main__":
    main()
