import os
import pytesseract
import cv2  # cv2.imwrite için gerekli
# Kendi yazdığımız modülden fonksiyonları import et
from image_processor import normalize_id_card, segment_fields_from_blobs

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    
    # --- Configuration ---
    INPUT_IMAGE_PATH = "kimlik.jpeg" # Test image file
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
                    # Define the output path for the individual field
                    output_path = os.path.join(OUTPUT_DIRECTORY, f"field_{i+1:02d}.png")
                    # Save the field image
                    cv2.imwrite(output_path, field_img)
                    
                    # --- Run Tesseract OCR ---
                    # -l tur: Use the Turkish language model
                    # --psm 7: Assume a single uniform line of text
                    config = '--psm 7 -l tur'
                    text = pytesseract.image_to_string(field_img, config=config)
                    
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