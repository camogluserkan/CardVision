import cv2
import numpy as np
import os

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
    ratio = 1 # image.shape[0] / 500.0
    image = orig #cv2.resize(image, (int(image.shape[1] / ratio), 500))
    
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
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    cv2.imwrite(os.path.join(output_dir, "norm_04_bilateral.png"), filtered)

    screenCnt = None
    contours = []

    # --- YÖNTEM 1: Parlaklık Tabanlı (Thresholding) ---
    # "Kimlik beyazdır" varsayımı.
    # Otsu thresholding ile otomatik olarak "Açık renkli ön plan" ve "Koyu arka plan" ayrımı yapıyoruz.
    print("Trying Method 1: Thresholding (Whiteness detection)...")
    ret, thresh = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Check for failure case: Image is mostly white (background merged)
    total_pixels = image.shape[0] * image.shape[1]
    white_pixels = cv2.countNonZero(thresh)
    white_ratio = white_pixels / total_pixels
    
    if white_ratio > 0.80:
        print(f"Warning: Thresholding result is too white ({white_ratio:.2f}). Background is likely light. Skipping Method 1.")
        # Method 1 skipped, screenCnt remains None
    else:
        # Threshold sonucunda oluşan gürültüleri temizle (Opening/Closing)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        # Closing: Kartın içindeki delikleri (yazılar vb.) kapatır, bütün bir beyaz blok yapar.
        thresh_cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel) 
        # Opening: Kartın etrafındaki küçük beyaz gürültüleri siler.
        thresh_cleaned = cv2.morphologyEx(thresh_cleaned, cv2.MORPH_OPEN, kernel) 
        cv2.imwrite(os.path.join(output_dir, "norm_05_threshold.png"), thresh_cleaned)

        # Kontur ara (Threshold üzerinden)
        contours, _ = cv2.findContours(thresh_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        
        for c in contours:
            area = cv2.contourArea(c)
            # Alan kontrolü: %10'dan küçükse gürültü, %95'ten büyükse çerçeve/arka plan
            if area < (total_pixels * 0.10) or area > (total_pixels * 0.95):
                continue

            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                screenCnt = approx
                print("ID card found using THRESHOLDING (Method 1).")
                break

    # --- YÖNTEM 2: Kenar Tabanlı (Keskinleştirme + Morfolojik Gradyan + Opening/Closing) ---
    # Eğer Threshold yöntemi başarısız olursa, sert geçişleri bulup gürültüyü temizliyoruz.
    if screenCnt is None:
        print("Method 1 failed or skipped. Trying Method 2: Sharpening + Morph Gradient + Open/Close...")
        
        # 1. CLAHE ile kontrastı artır
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(filtered)
        
        # 2. Keskinleştirme (Sharpening)
        # Kenarları daha belirgin hale getirir.
        sharpen_kernel = np.array([[-1, -1, -1],
                                   [-1,  9, -1],
                                   [-1, -1, -1]])
        enhanced = cv2.filter2D(enhanced, -1, sharpen_kernel)
        cv2.imwrite(os.path.join(output_dir, "norm_04_sharpened_m2.png"), enhanced)
        
        # 3. Canny Kenar Tespiti + Dilation (Genişletme)
        # Önceki "Gradient + Open/Close" yöntemi görüntüyü bozduğu için,
        # daha temiz çizgiler üreten Canny algoritmasına geçiyoruz.
        # "enhanced" (güçlendirilmiş) görüntü üzerinde Canny çok daha iyi sonuç verir.
        
        # Canny: Kenarları ince çizgiler halinde bulur.
        # 30 ve 150 eşik değerleri, hem zayıf hem güçlü kenarları yakalamak için seçildi.
        edged = cv2.Canny(enhanced, 30, 150)
        cv2.imwrite(os.path.join(output_dir, "norm_05_m2_canny.png"), edged)
        
        # Dilation: Canny'nin bulduğu çizgilerdeki ufak kopuklukları birleştirir.
        # (3,3) boyutunda küçük bir kernel ile kenarları hafifçe kalınlaştırıp bağlıyoruz.
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edged = cv2.dilate(edged, dilate_kernel, iterations=1)
        cv2.imwrite(os.path.join(output_dir, "norm_05_m2_dilated.png"), edged)
        
        cv2.imwrite(os.path.join(output_dir, "norm_05_m2_processed.png"), edged)

        # --- Find the Card Contour ---
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        
        for c in contours:
            area = cv2.contourArea(c)
            if area < (total_pixels * 0.10) or area > (total_pixels * 0.95):
                continue

            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            
            if len(approx) == 4:
                screenCnt = approx
                print("ID card contour found using Method 2 (Sharpen+Gradient+Open/Close).")
                break
    
    # If still None (no contours found at all), fail
    if screenCnt is None:
        print("Error: No suitable contour found.")
        return None    # Debug: Draw the selected contour
    debug_selected = image.copy()
    cv2.drawContours(debug_selected, [screenCnt], -1, (0, 0, 255), 3)
    cv2.imwrite(os.path.join(output_dir, "norm_07_selected_contour.png"), debug_selected)

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
    Segments the normalized ID card into individual fields of interest (like name, TC no, etc.)
    using morphological operations and ROI masking.
    """
    print("Starting segmentation with precise photo masking...")
    
    # --- Step 1: Binarization ---
    # Convert the normalized color image to grayscale
    gray = cv2.cvtColor(normalized_image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(output_dir, "seg_01_gray.png"), gray)

    # Invert the grayscale image (text becomes white, background becomes black)
    # inverted_gray = cv2.bitwise_not(gray) # GEREK YOK
    # cv2.imwrite(os.path.join(output_dir, "seg_02_inverted.png"), inverted_gray)

    # Apply Adaptive Thresholding
    # THRESH_BINARY_INV: Açık renk zemin -> Siyah, Koyu renk yazı -> Beyaz yapar.
    # Bu sayede tam istediğimiz "Siyah zemin üzerinde Beyaz yazılar" görüntüsünü elde ederiz.
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 10)
    
    # Save the binary image for debugging
    cv2.imwrite(os.path.join(output_dir, "seg_03_binary.png"), binary)

    # --- Step 2: ROI Masking (Your Idea) ---
    # This step blacks out non-text regions (photo, flag) *before* dilation
    # to prevent them from merging with text blobs.
    binary_masked = binary.copy()
    height, width = binary_masked.shape[:2]

    # Define coordinates for the Photo area (based on percentages)
    # Fotoğraf maskesini biraz daralttık (0.35 -> 0.28) ki yanındaki yazıları (TC, Soyad) kapatmasın.
    photo_x_start = int(width * 0.02) 
    photo_x_end = int(width * 0.31)
    photo_y_start = int(height * 0.35)
    photo_y_end = int(height * 0.93) # Alt kısmı da biraz kıstık
    # Draw a black rectangle over the photo area
    cv2.rectangle(binary_masked, (photo_x_start, photo_y_start), (photo_x_end, photo_y_end), (0), thickness=cv2.FILLED)
    
    # Define coordinates for the Flag (Ay-Yıldız) area (User's addition)
    flag_x_start = int(width * 0.65) # Estimated start X
    flag_x_end = int(width * 0.98)   # Estimated end X
    flag_y_start = int(height * 0.17) # Estimated start Y
    flag_y_end = int(height * 0.50)   # Estimated end Y
    
    # Draw a black rectangle over the flag area
    cv2.rectangle(binary_masked, (flag_x_start, flag_y_start), (flag_x_end, flag_y_end), (0), thickness=cv2.FILLED)

    # Save the masked binary image for debugging
    cv2.imwrite(os.path.join(output_dir, "seg_04_masked.png"), binary_masked)
    
    # --- Step 2.5: Salt-and-Pepper Noise Removal (Morphological Opening) ---
    # Çok küçük beyaz noktaları (gürültü) temizlemek için "Opening" işlemi yapıyoruz.
    # Opening = Erosion (Aşındırma) + Dilation (Genişletme)
    # (2,2) boyutunda çok küçük bir kernel kullanıyoruz ki yazılar zarar görmesin.
    noise_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary_masked = cv2.morphologyEx(binary_masked, cv2.MORPH_OPEN, noise_kernel)
    cv2.imwrite(os.path.join(output_dir, "seg_04a_denoised.png"), binary_masked)

    # --- Step 2.6: Closing (Morphological Closing) ---
    # Harflerin içindeki küçük siyah boşlukları veya kopuklukları birleştirmek için "Closing" yapıyoruz.
    # Closing = Dilation (Genişletme) + Erosion (Aşındırma)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary_masked = cv2.morphologyEx(binary_masked, cv2.MORPH_CLOSE, close_kernel)
    cv2.imwrite(os.path.join(output_dir, "seg_04b_closed.png"), binary_masked)

    # --- Step 3: Dilation ---
    # Create a rectangular kernel (15 wide, 3 high)
    # This will connect letters into words, and nearby words into fields.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    # Apply dilation to the *masked* image
    dilated_words = cv2.dilate(binary_masked, kernel, iterations=2) 
    # Save the dilated "blobs" image for debugging
    cv2.imwrite(os.path.join(output_dir, "seg_05_dilated.png"), dilated_words)

    # --- Step 4: Contour Extraction ---
    # Find the contours of all the white blobs in the dilated image
    contours, _ = cv2.findContours(dilated_words, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Debug: Draw all contours
    debug_contours = normalized_image.copy()
    cv2.drawContours(debug_contours, contours, -1, (0, 0, 255), 1)
    cv2.imwrite(os.path.join(output_dir, "seg_06_contours.png"), debug_contours)

    field_images = []
    # Create a copy of the original normalized image to draw debug boxes on
    output_with_boxes = normalized_image.copy()
    
    # Loop over every contour (blob) found
    for contour in contours:
        # Get the bounding box (x, y, width, height) of the blob
        x, y, w, h = cv2.boundingRect(contour)
        
        # --- Step 5: Filtering ---
        # Filter out blobs that are too small (noise) or too large
        if w > 30 and h > 15 and w < normalized_image.shape[1] * 0.8:
            # Add a small padding around the box to ensure no text is cut off
            padding = 5
            start_y, start_x = max(0, y - padding), max(0, x - padding)
            end_y, end_x = y + h + padding, x + w + padding
            
            # Crop the field from the *original normalized color image*
            field_crop = normalized_image[start_y:end_y, start_x:end_x]
            
            if field_crop.size > 0:
                # Add the cropped image to our list
                field_images.append(field_crop)
                # Draw a green box on the debug image
                cv2.rectangle(output_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Save the final debug image with green boxes
    cv2.imwrite(os.path.join(output_dir, "seg_07_final_boxes.png"), output_with_boxes)
    # Return the list of cropped field images
    return field_images