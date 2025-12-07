import os
import pytesseract
import cv2  # cv2.imwrite için gerekli
import numpy as np # image_processor için gerekli
# Kendi yazdığımız modülden fonksiyonları import et
from image_processor import normalize_id_card, segment_fields_from_blobs

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    
    # --- Configuration ---
    INPUT_IMAGE_PATH = "sample_tc.jpeg" # Test image file
    OUTPUT_DIRECTORY = "output_lines" # Directory for all debug/output images
    # ---------------------

    # Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    print(f"Processing '{INPUT_IMAGE_PATH}'...")

    # --- Step 1: Normalization ---
    # Call the function from our image_processor.py file
    normalized_image = normalize_id_card(INPUT_IMAGE_PATH)

    # Check if normalization was successful
    if normalized_image is not None:
        print("Step 1: Normalization successful.")
        # Save the normalized image
        cv2.imwrite(os.path.join(OUTPUT_DIRECTORY, "00_normalized.png"), normalized_image)
        
        # --- Step 2: Segmentation ---
        # Call the segmentation function from image_processor.py
        field_images = segment_fields_from_blobs(normalized_image, OUTPUT_DIRECTORY)

        if not field_images:
            print("Step 2: Segmentation failed or no fields were found.")
        else:
            print(f"Step 2: Segmentation successful. Found {len(field_images)} logical fields.")
            
            # --- Step 3: OCR ---
            print("\n--- OCR RESULTS ---")
            
            try: 
                # Loop through each cropped field image
                for i, field_img in enumerate(field_images):
                    # --- Pre-OCR Enhancement ---
                    
                    # 1. Upscale: 3x using Lanczos4 (Best quality for upscaling)
                    # Lanczos4 is slower but produces sharper edges than Cubic
                    field_img = cv2.resize(field_img, None, fx=3, fy=3, interpolation=cv2.INTER_LANCZOS4)
                    
                    # 2. Convert to Grayscale
                    gray_field = cv2.cvtColor(field_img, cv2.COLOR_BGR2GRAY)
                    
                    # 3. Binarization (Black & White)
                    # Using Otsu's thresholding automatically finds the best separation value
                    _, binary_field = cv2.threshold(gray_field, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                    # --- ADVANCED CLEANING (Yöntem 1: Morphological Opening) ---
                    # Morfolojik işlemler için "Beyaz Yazı, Siyah Arka Plan" formatına geçiyoruz.
                    inverted_binary = cv2.bitwise_not(binary_field)
                    
                    # 1. Morphological Opening (Çizgi Temizleme)
                    # İnce çizgileri ve gürültüyü yok eder, kalın harfleri korur.
                    # 2x2 kernel çizgileri silmek için ideal, harfleri bozmaz.
                    kernel = np.ones((2, 2), np.uint8)
                    cleaned_binary = cv2.morphologyEx(inverted_binary, cv2.MORPH_OPEN, kernel)
                    
                    # 2. Erosion (Harf İnceltme/Ayırma)
                    # Harfler birbirine yapışıyorsa (örn: M ve O), aralarını açmak için hafifçe inceltiyoruz.
                    # Önceki kodda yanlışlıkla 'erode' siyah yazıya uygulanmıştı (kalınlaştırmıştı),
                    # şimdi beyaz yazıya uyguluyoruz (doğru şekilde inceltiyor).
                    cleaned_binary = cv2.erode(cleaned_binary, kernel, iterations=1)

                    # Tekrar Siyah Yazı / Beyaz Arka Plan formatına dönüyoruz (Tesseract için)
                    final_processed = cv2.bitwise_not(cleaned_binary)

                    # Define the output path for the individual field
                    output_path = os.path.join(OUTPUT_DIRECTORY, f"field_{i+1:02d}.png")
                    # Save the processed image (ready for OCR)
                    cv2.imwrite(output_path, final_processed)
                    
                    # --- Run Tesseract OCR ---
                    # -l tur: Use the Turkish language model
                    # --psm 6: Assume a single uniform block of text (handles multiple lines better)
                    config = '--psm 6 -l tur'
                    text = pytesseract.image_to_string(final_processed, config=config)
                    
                    # Clean up the output text (remove newlines)
                    cleaned_text = text.replace('\n', ' ').strip()
                    print(f"Field {i+1} ({output_path}): {cleaned_text}")
            
            except pytesseract.TesseractNotFoundError: # More specific error
                 print("\n--- OCR STEP FAILED ---")
                 print("ERROR: 'tesseract' executable not found.")
                 print("Please install Tesseract OCR on your system.")
                 print("On Linux: sudo apt-get install tesseract-ocr tesseract-ocr-tur")
                 print("---------------------------")
            except Exception as e: # Catch other potential errors
                 print(f"\nAn error occurred during OCR: {e}")
            
            print("----------------------")
            print(f"All fields saved to '{OUTPUT_DIRECTORY}'.")
            # ---------------------------------
            
    else:
        # This runs if normalize_id_card returned None
        print("Step 1: Normalization failed. Stopping process.")