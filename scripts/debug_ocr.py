"""
Debug script to analyze OCR and table extraction issues.
"""
import cv2
import pytesseract
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.utils import preprocess_image_for_ocr, normalize_input_image
from config import TESSERACT_CMD, TESSDATA_DIR

pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
os.environ["TESSDATA_PREFIX"] = TESSDATA_DIR

def analyze_image(image_path):
    """Analyze an image and show OCR results."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {image_path}")
    print(f"{'='*60}")
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load {image_path}")
        return
    
    print(f"Image shape: {img.shape}")
    
    # Try OCR without preprocessing
    print("\n--- OCR WITHOUT preprocessing ---")
    try:
        text_raw = pytesseract.image_to_string(img, lang='eng+fra+ara', config='--oem 1 --psm 6')
        print(f"Raw OCR result ({len(text_raw)} chars):")
        print(text_raw[:500] if len(text_raw) > 500 else text_raw)
    except Exception as e:
        print(f"Error: {e}")
    
    # Try OCR WITH preprocessing
    print("\n--- OCR WITH preprocessing ---")
    try:
        processed = preprocess_image_for_ocr(img)
        cv2.imwrite("debug_processed.png", processed)
        print(f"Saved processed image to debug_processed.png")
        
        text_processed = pytesseract.image_to_string(processed, lang='eng+fra+ara', config='--oem 1 --psm 6')
        print(f"Processed OCR result ({len(text_processed)} chars):")
        print(text_processed[:500] if len(text_processed) > 500 else text_processed)
    except Exception as e:
        print(f"Error: {e}")
    
    # Get word-level data
    print("\n--- Word-level data ---")
    try:
        data = pytesseract.image_to_data(img, lang='eng', config='--oem 1 --psm 6', output_type=pytesseract.Output.DICT)
        words = [(text, conf) for text, conf in zip(data['text'], data['conf']) if text.strip() and int(conf) > 30]
        print(f"Words detected (conf > 30): {len(words)}")
        for word, conf in words[:20]:
            print(f"  '{word}' (conf: {conf})")
    except Exception as e:
        print(f"Error: {e}")


def analyze_table(image_path):
    """Analyze table structure in an image."""
    print(f"\n{'='*60}")
    print(f"Table Analysis: {image_path}")
    print(f"{'='*60}")
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load {image_path}")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # Detect horizontal and vertical lines
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, img.shape[1] // 20), 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(1, img.shape[0] // 20)))
    
    h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel)
    v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel)
    
    # Find contours
    table_mask = cv2.add(h_lines, v_lines)
    contours, _ = cv2.findContours(table_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    cell_contours = [c for c in contours if cv2.contourArea(c) > 100]
    print(f"Detected {len(cell_contours)} potential cells")
    
    # Save visualization
    vis = img.copy()
    cv2.drawContours(vis, cell_contours, -1, (0, 255, 0), 2)
    cv2.imwrite("debug_table_cells.png", vis)
    print("Saved debug_table_cells.png")
    
    # OCR each cell
    print("\n--- Cell OCR ---")
    for i, cnt in enumerate(cell_contours[:10]):  # First 10 cells
        x, y, w, h = cv2.boundingRect(cnt)
        cell_crop = img[y:y+h, x:x+w]
        
        if cell_crop.size > 0 and w > 10 and h > 10:
            try:
                text = pytesseract.image_to_string(cell_crop, lang='eng', config='--oem 1 --psm 6').strip()
                print(f"Cell {i} ({x},{y},{w}x{h}): '{text[:50]}'" if text else f"Cell {i}: (empty)")
            except:
                pass


if __name__ == "__main__":
    # Analyze text crops
    for i in range(3):
        crop_path = f"output/text_crops/text_{i}.png"
        if os.path.exists(crop_path):
            analyze_image(crop_path)
    
    # Analyze the original image for table
    img_path = "data/document_images/3.jpeg"
    if os.path.exists(img_path):
        analyze_table(img_path)
