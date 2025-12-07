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

def normalize_id_card(image_path):
    """
    Finds the ID card in an image, corrects its perspective,
    and returns a top-down, "scanned" view of the card.
    """
    print(f"Starting normalization for: {image_path}")
    
    # Read the image from the specified path
    image = cv2.imread(image_path)
    if image is None: 
        print(f"Error: Could not read image from {image_path}")
        return None

    # Keep a copy of the original image for the final warp
    orig = image.copy()
    
    # Resize the image for faster processing. 500px height is a good trade-off.
    # Keep the ratio to scale coordinates back later.
    ratio = image.shape[0] / 500.0
    image = cv2.resize(image, (int(image.shape[1] / ratio), 500))

    # --- Pre-processing for Edge Detection ---
    # Convert to grayscale (edge detection works on intensity)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to remove high-frequency noise (helps Canny)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Apply Canny edge detection to find edges
    edged = cv2.Canny(blurred, 75, 200)

    # --- Find the Card Contour ---
    # Find all contours, but only keep the 5 largest ones
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    
    screenCnt = None

    # Loop over the sorted contours
    for c in contours:
        # Calculate the perimeter of the contour
        peri = cv2.arcLength(c, True)
        # Approximate the contour shape with fewer vertices
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        # The ID card contour should have 4 vertices (it's a quadrilateral)
        if len(approx) == 4:
            screenCnt = approx
            print("ID card contour found (4 vertices detected).")
            break
    
    # If no 4-vertex contour was found, normalization fails
    if screenCnt is None:
        print("Automatic ID card contour not found (4 vertices not detected).")
        return None

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
    # Invert the grayscale image (text becomes white, background becomes black)
    inverted_gray = cv2.bitwise_not(gray)
    # Apply a fixed binary threshold (115) to get a clean black & white image
    _, binary = cv2.threshold(inverted_gray, 115, 255, cv2.THRESH_BINARY)
    # Save the binary image for debugging
    cv2.imwrite(os.path.join(output_dir, "01_binary_image.png"), binary)

    # --- Step 2: ROI Masking (Your Idea) ---
    # This step blacks out non-text regions (photo, flag) *before* dilation
    # to prevent them from merging with text blobs.
    binary_masked = binary.copy()
    height, width = binary_masked.shape[:2]

    # Define coordinates for the Photo area (based on percentages)
    photo_x_start = int(width * 0.00) # User's correction from 0.0
    photo_x_end = int(width * 0.35)
    photo_y_start = int(height * 0.30)
    photo_y_end = int(height * 1)
    # Draw a black rectangle over the photo area
    cv2.rectangle(binary_masked, (photo_x_start, photo_y_start), (photo_x_end, photo_y_end), (0), thickness=cv2.FILLED)
    
    # Define coordinates for the Flag (Ay-Yıldız) area (User's addition)
    flag_x_start = int(width * 0.65) # Estimated start X
    flag_x_end = int(width * 0.98)   # Estimated end X
    flag_y_start = int(height * 0.17) # Estimated start Y
    flag_y_end = int(height * 0.50)   # Estimated end Y
    
    # Draw a black rectangle over the flag area
    cv2.rectangle(binary_masked, (flag_x_start, flag_y_start), (flag_x_end, flag_y_end), (0), thickness=cv2.FILLED)

    # Define coordinates for the Ghost Photo (Small photo on right)
    ghost_x_start = int(width * 0.80) 
    ghost_x_end = int(width * 0.98)
    ghost_y_start = int(height * 0.55)
    ghost_y_end = int(height * 0.75)
    
    # Draw a black rectangle over the ghost photo area
    cv2.rectangle(binary_masked, (ghost_x_start, ghost_y_start), (ghost_x_end, ghost_y_end), (0), thickness=cv2.FILLED)

    # Define coordinates for the Signature (Bottom right)
    sig_x_start = int(width * 0.80)
    sig_x_end = int(width * 0.98)
    sig_y_start = int(height * 0.80)
    sig_y_end = int(height * 0.98)

    # Draw a black rectangle over the signature area
    cv2.rectangle(binary_masked, (sig_x_start, sig_y_start), (sig_x_end, sig_y_end), (0), thickness=cv2.FILLED)


    # Save the masked binary image for debugging
    cv2.imwrite(os.path.join(output_dir, "01a_binary_masked.png"), binary_masked)
    
    # --- Step 3: Dilation ---
    # Create a rectangular kernel (15 wide, 3 high)
    # This will connect letters into words, and nearby words into fields.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    # Apply dilation to the *masked* image
    dilated_words = cv2.dilate(binary_masked, kernel, iterations=2) 
    # Save the dilated "blobs" image for debugging
    cv2.imwrite(os.path.join(output_dir, "02_word_blobs.png"), dilated_words)

    # --- Step 4: Contour Extraction ---
    # Find the contours of all the white blobs in the dilated image
    contours, _ = cv2.findContours(dilated_words, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    field_images = []
    # Create a copy of the original normalized image to draw debug boxes on
    output_with_boxes = normalized_image.copy()
    
    # Loop over every contour (blob) found
    for contour in contours:
        # Get the bounding box (x, y, width, height) of the blob
        x, y, w, h = cv2.boundingRect(contour)
        
        # --- Step 5: Filtering ---
        # Filter out blobs that are too small (noise) or too large
        if w > 25 and h > 10 and w < normalized_image.shape[1] * 0.8:
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
    cv2.imwrite(os.path.join(output_dir, "03_final_field_boxes.png"), output_with_boxes)
    # Return the list of cropped field images
    return field_images