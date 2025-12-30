import os
import pytesseract
import cv2
import numpy as np
import json
from image_processor import normalize_id_card, segment_fields_from_blobs
# Yeni oluşturduğumuz parser dosyasını import ediyoruz
from id_parser import parse_id_card_data

# --- GÖRÜNTÜ İYİLEŞTİRME FONKSİYONU ---
def preprocess_for_ocr(img):
    """
    OCR başarısını artırmak için resmi büyütür, netleştirir ve çerçeve ekler.
    """
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    padded = cv2.copyMakeBorder(binary, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    return padded

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    
    INPUT_IMAGE_PATH = "kimlik.jpeg" 
    OUTPUT_DIRECTORY = "output_lines" 

    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)

    print(f"Processing '{INPUT_IMAGE_PATH}'...")

    # --- Step 1: Normalization ---
    normalized_image = normalize_id_card(INPUT_IMAGE_PATH, OUTPUT_DIRECTORY)

    if normalized_image is not None:
        print("Step 1: Normalization successful.")
        
        # --- Step 2: Segmentation ---
        field_images = segment_fields_from_blobs(normalized_image, OUTPUT_DIRECTORY)

        if not field_images:
            print("Hata: Hiçbir metin bloğu bulunamadı!")
        else:
            print(f"Step 2: Segmentation successful. Found {len(field_images)} potential fields.")
            
            # --- Step 3: OCR & Parsing ---
            print("\n--- OCR Okuma İşlemi Başlıyor ---")
            
            # Tüm ham metinleri bu listede toplayacağız
            raw_ocr_lines = []
            
            for i, field_img in enumerate(field_images):
                # 1. OCR için iyileştir
                processed_img = preprocess_for_ocr(field_img)
                
                # Görüntüyü kaydet
                output_path = os.path.join(OUTPUT_DIRECTORY, f"field_{i+1:02d}.png")
                cv2.imwrite(output_path, processed_img)
                
                # 2. OCR Oku
                custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist="ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ0123456789./- "'
                text = pytesseract.image_to_string(processed_img, config=custom_config)
                
                # Temizlik
                cleaned_text = text.replace('\n', ' ').strip()
                
                # Boş veya çok kısa gürültüleri listeye alma
                if len(cleaned_text) > 1:
                    raw_ocr_lines.append(cleaned_text)
                    print(f"Raw Field {i+1}: {cleaned_text}")
            
            # --- Step 4: Logic Parsing (Rafine Etme) ---
            print("\n" + "="*40)
            print("KİMLİK VERİSİ AYRIŞTIRILIYOR...")
            print("="*40)
            
            # Ham listeyi parser'a gönderiyoruz
            final_data = parse_id_card_data(raw_ocr_lines)
            
            # Sonucu güzel bir JSON formatında yazdırıyoruz
            print(json.dumps(final_data, indent=4, ensure_ascii=False))
            
            # İstersen sonucu bir dosyaya da kaydedebilirsin
            with open(os.path.join(OUTPUT_DIRECTORY, "final_data.json"), "w", encoding="utf-8") as f:
                json.dump(final_data, f, indent=4, ensure_ascii=False)

    else:
        print("Step 1: Normalization failed.")