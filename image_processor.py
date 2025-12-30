import cv2
import numpy as np
import os
import pytesseract # OSD için gerekli
import re          # Regex ile açı değerini çekmek için

# -----------------------------------------------------------------------------
# PART I: PERSPECTIVE CORRECTION
# -----------------------------------------------------------------------------

def order_points(pts):
    """
    Sorts 4 points (coordinates) into a consistent order:
    top-left, top-right, bottom-right, bottom-left.
    This is crucial for the getPerspectiveTransform function.
    """
    # Initialize a 4x2 array to store the sorted points
    rect = np.zeros((4, 2), dtype="float32")

    # The top-left point will have the smallest (x+y) sum.
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    # The bottom-right point will have the largest (x+y) sum.
    rect[2] = pts[np.argmax(s)]

    # The top-right point will have the smallest (x-y) difference.
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    # The bottom-left point will have the largest (x-y) difference.
    rect[3] = pts[np.argmax(diff)]

    # Return the sorted coordinates
    return rect

def fix_orientation(image):
    """
    Resmin yönünü bulmak için 'Deneme-Yanılma' yöntemini kullanır.
    Resmi 0, 90, 180, 270 derece açılarda tarar.
    'TURKIYE', 'REPUBLIC' gibi anahtar kelimeleri okuyabildiği açıyı doğru kabul eder.
    """
    
    # Kimlik kartında kesinlikle olması gereken kelimeler
    keywords = ["TURKIYE", "CUMHURIYETI", "REPUBLIC", "TURKEY", "IDENTITY", "KART", "SOYADI"]
    
    print("   [Smart-Rotate] Doğru açı aranıyor...")

    # Orijinal resmi bozmamak için kopyala
    current_img = image.copy()
    
    # Maksimum 4 tur (0, 90, 180, 270 derece)
    for angle in [0, 90, 180, 270]:
        try:
            # Hızlı OCR taraması (PSM 11: Sparse Text - Hızlıdır)
            # Sadece büyük harfleri ve boşlukları alarak hızlandırıyoruz
            ocr_data = pytesseract.image_to_string(current_img, config='--psm 11').upper()
            
            # Anahtar kelimelerden HERHANGİ BİRİ var mı?
            for keyword in keywords:
                if keyword in ocr_data:
                    if angle == 0:
                        print(f"      -> Yön zaten doğru (0°). İşlem yapılmadı.")
                    else:
                        print(f"      -> {angle}° çevrilince '{keyword}' okundu. Yön düzeltildi! ✅")
                    return current_img

            # Eğer kelime bulunamadıysa, bir sonraki tur için 90 derece çevir
            # (Saat yönünde)
            current_img = cv2.rotate(current_img, cv2.ROTATE_90_CLOCKWISE)
            
        except Exception as e:
            print(f"      -> Hata oluştu: {e}")
            continue

    # Eğer 4 turda da hiçbir şey okunamazsa (Resim çok bulanıksa vs.)
    # Yapacak bir şey yok, orijinali (veya en son hali) döndür.
    print("   [Smart-Rotate] Uyarı: Hiçbir açıda anlamlı kelime okunamadı. Orijinal varsayılıyor.")
    return image

def normalize_id_card(image_path, output_dir="output_lines"):
    """
    Finds the ID card in an image, corrects its perspective,
    and returns a top-down, "scanned" view of the card.
    """
    print(f"Starting normalization for: {image_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read the image from the specified path
    image = cv2.imread(image_path)
    if image is None: 
        print(f"Error: Could not read image from {image_path}")
        return None

    # Save original
    cv2.imwrite(os.path.join(output_dir, "norm_01_original.png"), image)

    # Keep a copy of the original image for the final warp
    orig = image.copy()
    
    # Resize the image for faster processing. 500px height is a good trade-off.
    # Keep the ratio to scale coordinates back later.
    target_height = 600.0
    ratio = image.shape[0] / target_height
    image = cv2.resize(image, (int(image.shape[1] / ratio), int(target_height)))


    # Save resized
    cv2.imwrite(os.path.join(output_dir, "norm_02_resized.png"), image)

    # --- Pre-processing ---
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(output_dir, "norm_03_gray.png"), gray)

    # 1. Bilateral Filter: "Gürültüsüz yüzey, keskin kenar" (Smooth surface, sharp edges).
    # GaussianBlur yerine Bilateral Filter kullanıyoruz. Bu filtre, kimlik kartının
    # düz/beyaz yüzeyini pürüzsüzleştirirken kenarlarını keskin tutar.
    # d=9: Piksel komşuluğu çapı.
    # sigmaColor=75: Renk uzayındaki filtre standart sapması (büyük değer = uzak renkler birbirine karışır).
    # sigmaSpace=75: Koordinat uzayındaki filtre standart sapması.
    # Bilateral yerine GaussianBlur kullanıyoruz.
    # (7, 7) kernel boyutu masadaki ince detayları (damarları) öldürür.
    filtered = cv2.GaussianBlur(gray, (7, 7), 0)
    cv2.imwrite(os.path.join(output_dir, "norm_04_gaussian.png"), filtered)


    screenCnt = None
    contours = []

    # --- KENAR TESPİTİ (Canny) ---
    # Arkadaşının kodundaki mantık: Önce kenarları bul, sonra birleştir.
    
    # 1. Canny ile kenarları bul (30-150 arası iyi bir aralıktır)
    edged = cv2.Canny(filtered, 30, 150)
    
    # 2. Dilation (Genişletme): Kopuk çizgileri birleştirmek için kritik adım.
    # Kartın kenarı ışık yüzünden kopuk görünüyorsa burası tamireder.
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edged = cv2.dilate(edged, dilate_kernel, iterations=1)
    
    # ... Önceki kodlar aynı (Canny ve Dilation kısmı) ...
    
    cv2.imwrite(os.path.join(output_dir, "norm_05_canny_dilated.png"), edged)

    # --- KONTUR BULMA ve FİLTRELEME (GÜNCELLENDİ) ---
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Arama derinliğini artır: İlk 10 kontura bak (Kart bazen 1. sırada çıkmayabilir)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    found_contour = False
    screenCnt = None
    
    print(f"Toplam {len(contours)} aday kontur inceleniyor...")

    for i, c in enumerate(contours):
        # Çevre uzunluğu
        peri = cv2.arcLength(c, True)
        # Köşe sayısını azalt (Approximate)
        # 0.02 bazen çok kaba kalabilir, 0.015 deneyelim daha hassas olsun
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        # Bounding Box al
        (x, y, w_box, h_box) = cv2.boundingRect(approx)
        aspect_ratio = w_box / float(h_box)
        
        # DEBUG: Görelim bakalım ne bulmuş da reddetmiş?
        print(f"Kontur #{i}: Köşe={len(approx)}, Oran={aspect_ratio:.2f}, Alan={cv2.contourArea(c)}")

        # --- KRİTİK DÜZELTME ---
        # Sadece 4 ve üzeri köşe olması yeterli.
        # Aspect Ratio aralığını GENİŞLETİYORUZ. 
        # Çünkü kart yamuksa bounding box kare olur (oran 1.0'a yaklaşır).
        # 0.8 ile 2.5 arası diyerek neredeyse her türlü dikdörtgeni kabul edelim.
        if len(approx) >= 4:
            if 0.8 < aspect_ratio < 2.5: # Aralığı genişlettik!
                # EĞER TAM 4 KÖŞE DEĞİLSE (Örn: 8 köşe), DİKDÖRTGENE ZORLA
                if len(approx) == 4:
                    # Zaten 4 köşe ise direkt al, sorun yok.
                    screenCnt = approx
                else:
                    # Köşeler yuvarlatılmışsa (8, 10 köşe vs.),
                    # Şekli içine alan en küçük dönük dikdörtgeni (Rotated Rect) bul.
                    rect = cv2.minAreaRect(c)
                    box = cv2.boxPoints(rect)
                    
                    # approx formatına uydurmak için reshape yapıyoruz: (4, 1, 2)
                    screenCnt = np.int32(box).reshape(-1, 1, 2)

                found_contour = True
                print(f"--> KABUL EDİLDİ! (Oran: {aspect_ratio:.2f}, Final Köşe: 4)")
                break

            else:
                print("--> Reddedildi (Oran uyumsuz)")
        else:
             print("--> Reddedildi (Köşe sayısı yetersiz)")

    if not found_contour:
        print("Uygun formatta kimlik kartı konturu bulunamadı.")
        # Fallback (Acil Durum): Eğer hiçbiri uymazsa, en büyük 4 köşeli olanı almayı deneyebilirsin.
        # Şimdilik None dönüyor.
        return None

    # ... Buradan sonrası (Perspective Transform) aynı kalacak ...

    # --- Apply Perspective Transform ---
    # Scale the contour points back to the original image size
    # and sort them using our helper function.
    ordered_points = order_points(screenCnt.reshape(4, 2) * ratio)
    
    (tl, tr, br, bl) = ordered_points
    
    # Calculate the width of the new, normalized image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    # Calculate the height of the new, normalized image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # Define the destination points for the new image (a perfect rectangle)
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # Calculate the perspective transform matrix M
    M = cv2.getPerspectiveTransform(ordered_points, dst)
    # Apply the transform to the *original, unresized* image
    warped = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))
    
    warped = fix_orientation(warped)    
    
    # Save final warped image
    cv2.imwrite(os.path.join(output_dir, "norm_08_warped.png"), warped)
    
    print("Normalization successful.")
    # Return the top-down, warped image
    return warped

# -----------------------------------------------------------------------------
# PART II: FIELD SEGMENTATION (WITH PRECISE MASKING)
# -----------------------------------------------------------------------------
def segment_fields_from_blobs(normalized_image, output_dir):
    """
    Stabil Versiyon: Yatay Dilation (30, 1) kullanarak satırları birbirine
    yapıştırmadan kelime gruplarını ayırır.
    """
    print("Segmentasyon (Yatay Ayrıştırma Modu) başlatılıyor...")
    
    h, w = normalized_image.shape[:2]
    
    # 1. Griye Çevir ve Threshold
    gray = cv2.cvtColor(normalized_image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(output_dir, "seg_01_gray.png"), gray) # <--- KAYIT

    # 2. Adaptive Threshold (Arka plan desenlerini temizler)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 19, 25)
    cv2.imwrite(os.path.join(output_dir, "seg_02_binary.png"), binary) # <--- KAYIT

    masked = binary.copy()
    
    # --- MASKELEME (Gürültü Bölgelerini Kapat) ---
    # Bu maskeler veriye dokunmaz, sadece fotoğraf, bayrak ve başlığı siler.
    
    # A. Başlık Maskesi (Türkiye Cumhuriyeti yazısı) - %16
    cv2.rectangle(masked, (0, 0), (w, int(h * 0.16)), (0), -1)
    
    # B. Fotoğraf Maskesi (Sol Taraf)
    # TC'nin altından başlar (0.32), en alta kadar iner.
    photo_x_end = int(w * 0.29)
    photo_y_start = int(h * 0.32) 
    cv2.rectangle(masked, (0, photo_y_start), (photo_x_end, h), (0), -1)
    
    # C. Bayrak Maskesi (Sağ Taraf)
    # Cinsiyet yazısına dokunmadan (0.60) biter.
    flag_x_start = int(w * 0.66) 
    flag_y_start = int(h * 0.15)
    flag_y_end = int(h * 0.60)
    cv2.rectangle(masked, (flag_x_start, flag_y_start), (w, flag_y_end), (0), -1)

    cv2.imwrite(os.path.join(output_dir, "seg_03_masked.png"), masked) # <--- KAYIT

    # --- MORFOLOJİ (KRİTİK DÜZELTME) ---
    
    # 1. Erosion: Çok ince gürültü noktalarını koparır
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    eroded = cv2.erode(masked, kernel_erode, iterations=1)
    cv2.imwrite(os.path.join(output_dir, "seg_04_eroded.png"), eroded) # <--- KAYIT

    # 2. Dilation: Harfleri birleştirir AMA satırları yapıştırmaz
    # (30, 1) -> Yatayda çok bağla, Dikeyde hiç bağlama.
    # Bu sayede "Soyadı" üst kutuda, "ATABEY" alt kutuda kalır.
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1)) 
    dilated = cv2.dilate(eroded, kernel_dilate, iterations=2)
    cv2.imwrite(os.path.join(output_dir, "seg_05_dilated.png"), dilated) # <--- KAYIT

    # --- KONTUR BULMA ---
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_boxes = []
    
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        
        # Filtreleme
        if bw < 10 or bh < 8: continue # Çok küçük gürültü
        if bw > w * 0.9: continue # Aşırı büyük hatalar
        
        # Oran Kontrolü (Dikey çizgi gürültülerini at)
        aspect = bw / float(bh)
        if aspect < 0.2: continue 

        valid_boxes.append((x, y, bw, bh))

    # --- AKILLI SIRALAMA (SATIR + SÜTUN) ---
    # 1. Önce Y koordinatına göre kabaca sırala
    valid_boxes.sort(key=lambda b: b[1])
    
    sorted_boxes = []
    if valid_boxes:
        current_row = [valid_boxes[0]]
        row_threshold = 10 # 10 piksel dikey yakınlık varsa aynı satır say
        
        for i in range(1, len(valid_boxes)):
            prev_box = current_row[-1]
            curr_box = valid_boxes[i]
            
            # Y koordinatları yakınsa aynı satıra ekle
            if abs(curr_box[1] - prev_box[1]) < row_threshold:
                current_row.append(curr_box)
            else:
                # Satır bitti, bu satırı X'e göre (Soldan Sağa) sırala
                current_row.sort(key=lambda b: b[0])
                sorted_boxes.extend(current_row)
                current_row = [curr_box]
        
        # Son grubu ekle
        current_row.sort(key=lambda b: b[0])
        sorted_boxes.extend(current_row)
    
    # --- KESME VE DÖNDÜRME ---
    field_images = []
    debug_img = normalized_image.copy()
    
    for i, (x, y, bw, bh) in enumerate(sorted_boxes):
        # ROI (Region of Interest) Kesme
        # Padding eklemeden saf halini kesiyoruz (Padding'i main.py'de ekleyeceğiz)
        roi = normalized_image[y:y+bh, x:x+bw]
        
        field_images.append(roi)
        
        # Görsel Debug
        color = (0, 255, 0)
        # Eğer yükseklik çok küçükse (Etiket olma ihtimali yüksek) rengi farklı yap
        if bh < 18: color = (0, 255, 255) # Sarı (Muhtemel Etiket)
        
        cv2.rectangle(debug_img, (x, y), (x+bw, y+bh), color, 1)
        # Sıra numarasını yaz
        cv2.putText(debug_img, f"{i}", (x, y-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

    cv2.imwrite(os.path.join(output_dir, "seg_06_final_boxes.png"), debug_img) # <--- KAYIT
    print(f"Segmentasyon tamamlandı. {len(field_images)} blok bulundu.")
    
    return field_images