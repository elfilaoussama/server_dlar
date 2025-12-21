"""
Image processing utilities for document layout analysis.
"""
import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageOps


def normalize_input_image(image_path):
    """
    Loads and normalizes the input image from all critical aspects:
    1. Corrects Orientation (using EXIF).
    2. Converts to standard BGR format (OpenCV).
    3. Ensures 3 channels.
    """
    try:
        # Use PIL to handle EXIF orientation automatically
        pil_img = Image.open(image_path)
        pil_img = ImageOps.exif_transpose(pil_img)
        
        # Convert to RGB then BGR for OpenCV
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
            
        img_np = np.array(pil_img)
        # RGB to BGR
        image = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        print(f"    [Normalization] Image loaded and normalized: {image.shape}")
        return image
    except Exception as e:
        print(f"    [Normalization] Failed to load/normalize with PIL: {e}. Falling back to cv2.imread.")
        return cv2.imread(image_path)


def preprocess_image_for_ocr(image, force_heavy=False):
    """
    Preprocesses an image crop for optimal OCR accuracy.
    
    Uses SMART preprocessing:
    - For clean, high-quality images: minimal processing (grayscale + padding)
    - For low-quality/noisy images: full preprocessing pipeline
    
    Args:
        image: Input image (BGR or grayscale)
        force_heavy: If True, always use heavy preprocessing
    
    Returns:
        Processed grayscale image ready for OCR
    """
    # 1. Convert to Grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 2. Check if image needs heavy preprocessing
    needs_heavy_preprocessing = force_heavy or _needs_preprocessing(gray)
    
    if not needs_heavy_preprocessing:
        # Clean image: minimal preprocessing
        # Just add padding for better OCR boundary detection
        pad = 10
        processed = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
        return processed
    
    # Heavy preprocessing for degraded images
    return _heavy_preprocess(gray)


def _needs_preprocessing(gray_image):
    """
    Determines if an image needs heavy preprocessing based on quality metrics.
    
    Returns True if image is:
    - Very small (needs upscaling)
    - Low contrast
    - Very noisy
    """
    h, w = gray_image.shape[:2]
    
    # Check 1: Very small images need upscaling
    if h < 50 or w < 100:
        return True
    
    # Check 2: Low contrast detection
    # Calculate standard deviation of pixel values
    std_dev = np.std(gray_image)
    if std_dev < 40:  # Low contrast threshold
        return True
    
    # Check 3: Check if text is visible (not washed out)
    # A good document image should have both very dark (text) and very light (background) pixels
    dark_pixels = np.sum(gray_image < 80) / gray_image.size
    light_pixels = np.sum(gray_image > 200) / gray_image.size
    
    # If less than 5% dark pixels, text might be too light
    if dark_pixels < 0.05:
        return True
    
    # If no clear background (< 30% light pixels), might need contrast enhancement
    if light_pixels < 0.30:
        return True
    
    return False


def _heavy_preprocess(gray):
    """
    Full preprocessing pipeline for degraded/noisy images.
    """
    h, w = gray.shape[:2]
    
    # 1. Upscale small images only
    if h < 100 or w < 200:
        scale_factor = max(2.0, 100 / h, 200 / w)
        scale_factor = min(scale_factor, 3.0)  # Cap at 3x
        gray = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    
    # 2. Contrast enhancement using CLAHE (adaptive histogram equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # 3. Light denoising (preserve edges)
    denoised = cv2.bilateralFilter(enhanced, 5, 50, 50)
    
    # 4. Binarization using adaptive threshold (better for varying lighting)
    binarized = cv2.adaptiveThreshold(
        denoised, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        11, 2
    )
    
    # 5. Light morphological closing to connect broken characters
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binarized, cv2.MORPH_CLOSE, kernel)
    
    # 6. Add padding
    pad = 15
    processed = cv2.copyMakeBorder(cleaned, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    
    return processed


def preprocess_for_table_ocr(image):
    """
    Specialized preprocessing for table cell OCR.
    Tables often have borders that can interfere with OCR.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 1. Upscale small cells
    h, w = gray.shape[:2]
    if h < 30 or w < 30:
        scale = max(2.0, 30 / min(h, w))
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    # 2. Invert if needed (white text on dark background)
    mean_val = np.mean(gray)
    if mean_val < 128:
        gray = cv2.bitwise_not(gray)
    
    # 3. Binarize
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 4. Remove thin lines (table borders)
    # Horizontal lines
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, w // 10), 1))
    h_lines = cv2.morphologyEx(cv2.bitwise_not(binary), cv2.MORPH_OPEN, h_kernel)
    
    # Vertical lines
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(1, h // 10)))
    v_lines = cv2.morphologyEx(cv2.bitwise_not(binary), cv2.MORPH_OPEN, v_kernel)
    
    # Remove lines from image
    lines_mask = cv2.add(h_lines, v_lines)
    result = cv2.add(binary, lines_mask)
    
    # 5. Padding
    pad = 5
    result = cv2.copyMakeBorder(result, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    
    return result
