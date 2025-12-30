import os
import cv2
import pytesseract
import numpy as np
import shutil
import json
from image_processor import normalize_id_card, segment_fields_from_blobs
# Zeka modülümüzü dahil ediyoruz
from id_parser import parse_id_card_data

# --- AYARLAR ---
INPUT_FOLDER = "input"
OUTPUT_ROOT = "test_outputs"

# --- YARDIMCI FONKSİYON: OCR ÖN İŞLEME ---
def preprocess_for_ocr(img):
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    padded = cv2.copyMakeBorder(binary, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    return padded

def run_batch_test():
    if not os.path.exists(INPUT_FOLDER):
        print(f"HATA: '{INPUT_FOLDER}' klasörü yok!")
        return

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(valid_extensions)]

    if not files:
        print(f"UYARI: '{INPUT_FOLDER}' boş.")
        return

    print(f"Toplam {len(files)} adet resim test edilecek.\n" + "-"*40)

    for filename in files:
        print(f"\n>> İşleniyor: {filename}")
        
        input_path = os.path.join(INPUT_FOLDER, filename)
        folder_name = os.path.splitext(filename)[0]
        current_output_dir = os.path.join(OUTPUT_ROOT, folder_name)
        
        if os.path.exists(current_output_dir):
            shutil.rmtree(current_output_dir)
        os.makedirs(current_output_dir)

        status_suffix = ""

        try:
            # 1. Normalizasyon
            normalized_image = normalize_id_card(input_path, output_dir=current_output_dir)

            if normalized_image is not None:
                # 2. Segmentasyon
                field_images = segment_fields_from_blobs(normalized_image, output_dir=current_output_dir)

                if field_images:
                    # 3. OCR ve Parsing
                    results_txt_path = os.path.join(current_output_dir, "ocr_results.txt")
                    json_path = os.path.join(current_output_dir, "final_data.json")
                    
                    raw_ocr_lines = []

                    with open(results_txt_path, "w", encoding="utf-8") as f:
                        f.write(f"Dosya: {filename}\n")
                        f.write("-" * 30 + "\n")
                        print(f"   Bulunan Blok Sayısı: {len(field_images)}")
                        
                        for i, field_img in enumerate(field_images):
                            processed_img = preprocess_for_ocr(field_img)
                            cv2.imwrite(os.path.join(current_output_dir, f"final_{i:02d}.png"), processed_img)
                            
                            custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist="ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ0123456789./- "'
                            text = pytesseract.image_to_string(processed_img, config=custom_config)
                            cleaned_text = text.replace('\n', ' ').strip()
                            
                            if len(cleaned_text) > 1:
                                raw_ocr_lines.append(cleaned_text)
                            
                            f.write(f"Alan {i}: {cleaned_text}\n")
                    
                    # --- AŞAMA 4: PARSING VE EKRANA BASMA (GÜNCELLENDİ) ---
                    parsed_data = parse_id_card_data(raw_ocr_lines)
                    
                    # Dosyaya kaydet
                    with open(json_path, "w", encoding="utf-8") as json_file:
                        json.dump(parsed_data, json_file, indent=4, ensure_ascii=False)
                    
                    # --- TERMİNALE BASILAN KISIM ---
                    print("\n   [SONUÇ RAPORU]")
                    print(json.dumps(parsed_data, indent=4, ensure_ascii=False))
                    print("   ----------------------------------------")

                else:
                    print("   UYARI: Segmentasyon boş döndü.")
                    status_suffix = "_FAIL_SEG"
            else:
                print("   UYARI: Normalizasyon başarısız.")
                status_suffix = "_FAIL_NORM"

        except Exception as e:
            print(f"!!! KRİTİK HATA: {str(e)}")
            status_suffix = "_CRASH"

        if status_suffix != "":
            new_dir_name = current_output_dir + status_suffix
            if os.path.exists(new_dir_name):
                shutil.rmtree(new_dir_name)
            os.rename(current_output_dir, new_dir_name)
            print(f"   -> Klasör işaretlendi: {os.path.basename(new_dir_name)}")

    print("\n" + "="*40)
    print("Tüm testler tamamlandı.")

if __name__ == "__main__":
    run_batch_test()