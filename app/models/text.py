"""
Text object model for document layout analysis.
Optimized: Uses single OCR call for all text analysis.
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from .base import PageElement, BoundingBox


@dataclass
class TextObject(PageElement):
    """Represents a text block with OCR and formatting information."""
    rawText: str = ""
    fontName: str = "Arial"
    fontSize: float = 12.0
    alignment: str = "left"
    isBold: bool = False
    isItalic: bool = False
    lineSpacing: float = 1.0
    fontColor: str = "000000"  # Hex
    isEmptyOCR: bool = False  # True if OCR returned no text
    _pixelLineHeight: float = field(default=0.0, repr=False)

    def analyze_content(self, page_image):
        """
        Analyze the text content from the full page image.
        OPTIMIZED: Uses single OCR call for text, metrics, alignment, and color.
        """
        import cv2
        import pytesseract
        import numpy as np
        from app.utils import preprocess_image_for_ocr
        
        # Crop the text region
        x, y, w, h = int(self.bbox.x), int(self.bbox.y), int(self.bbox.width), int(self.bbox.height)
        
        # Ensure bounds
        y1, y2 = max(0, y), min(page_image.shape[0], y+h)
        x1, x2 = max(0, x), min(page_image.shape[1], x+w)
        
        if y2 <= y1 or x2 <= x1:
            return

        crop = page_image[y1:y2, x1:x2]
        
        # Perform OCR only if text is missing (Post-Merge Analysis)
        if not self.rawText.strip():
            # Add padding for better OCR
            pad = 3
            if y1 > pad and x1 > pad and y2 < page_image.shape[0]-pad and x2 < page_image.shape[1]-pad:
                crop_padded = page_image[y1-pad:y2+pad, x1-pad:x2+pad]
            else:
                crop_padded = crop
            
            # SINGLE OCR call - get all data at once
            custom_config = '--oem 1 --psm 6'  # PSM 6 for block of text
            languages = 'eng+fra+ara'
            
            try:
                # Try raw image first (works better for clean images)
                ocr_data = pytesseract.image_to_data(
                    crop_padded, lang=languages, config=custom_config,
                    output_type=pytesseract.Output.DICT
                )
                
                # Extract text from OCR data
                self.rawText = self._extract_text_from_data(ocr_data)
                
                # If raw OCR failed, try with preprocessing
                if not self.rawText.strip():
                    processed_crop = preprocess_image_for_ocr(crop_padded)
                    ocr_data = pytesseract.image_to_data(
                        processed_crop, lang=languages, config=custom_config,
                        output_type=pytesseract.Output.DICT
                    )
                    self.rawText = self._extract_text_from_data(ocr_data)
                
                # Clean text: Remove single newlines, preserve double newlines
                self.rawText = self.rawText.replace('\n\n', '<PARA_BREAK>').replace('\n', ' ').replace('<PARA_BREAK>', '\n\n')
                
                # Mark as empty OCR if no text was recognized
                if not self.rawText.strip():
                    self.isEmptyOCR = True
                    print(f"    -> Text region at ({int(self.bbox.x)}, {int(self.bbox.y)}) has no OCR output")
                    return
                
                # REUSE ocr_data for all metrics (no additional OCR calls!)
                self._pixelLineHeight = self._measure_font_metrics_from_data(ocr_data)
                self.alignment = self._detect_alignment_from_data(ocr_data, crop_padded.shape[1])
                
            except Exception as e:
                print(f"    [WARNING] OCR failed: {e}")
                self.isEmptyOCR = True
                return
        
        # Color detection still needs the original crop (not preprocessed)
        self.fontColor = self._detect_color(crop)

    def _extract_text_from_data(self, ocr_data: Dict) -> str:
        """Extract text from OCR data dictionary (mimics image_to_string behavior)."""
        n_boxes = len(ocr_data['text'])
        words = []
        current_line = -1
        
        for i in range(n_boxes):
            text = ocr_data['text'][i]
            line_num = ocr_data['line_num'][i]
            
            # Include any text (don't filter by confidence - image_to_string doesn't)
            if text.strip():
                # Add newline when line changes
                if current_line != -1 and line_num != current_line:
                    words.append('\n')
                current_line = line_num
                words.append(text)
        
        return ' '.join(words).replace(' \n ', '\n').strip()

    def _measure_font_metrics_from_data(self, ocr_data: Dict) -> float:
        """Measures median word height from pre-computed OCR data."""
        import numpy as np
        
        word_heights = []
        n_boxes = len(ocr_data['text'])
        
        for i in range(n_boxes):
            text = ocr_data['text'][i]
            conf = int(ocr_data['conf'][i])
            
            if conf > 60 and text.strip():
                word_heights.append(ocr_data['height'][i])
        
        if word_heights:
            return float(np.median(word_heights))
        return 0.0

    def _detect_alignment_from_data(self, ocr_data: Dict, image_width: int) -> str:
        """Analyzes text alignment from pre-computed OCR data."""
        import numpy as np
        from collections import defaultdict
        
        n_boxes = len(ocr_data['text'])
        lines = defaultdict(list)
        line_threshold = 10
        
        # Group words into lines
        for i in range(n_boxes):
            if int(ocr_data['conf'][i]) > 60 and ocr_data['text'][i].strip():
                y = ocr_data['top'][i]
                found_line = False
                for line_y in lines.keys():
                    if abs(y - line_y) < line_threshold:
                        lines[line_y].append(i)
                        found_line = True
                        break
                if not found_line:
                    lines[y].append(i)

        if len(lines) < 2:
            return self.alignment

        sorted_y = sorted(lines.keys())
        lines_to_analyze = sorted_y[1:-1] if len(sorted_y) >= 4 else sorted_y

        left_coords = []
        right_coords = []
        center_coords = []
        
        for y in lines_to_analyze:
            indices = lines[y]
            line_start = min(ocr_data['left'][i] for i in indices)
            line_end = max(ocr_data['left'][i] + ocr_data['width'][i] for i in indices)
            
            left_coords.append(line_start)
            right_coords.append(line_end)
            center_coords.append((line_start + line_end) / 2)

        if not left_coords:
            return self.alignment

        left_std = np.std(left_coords)
        right_std = np.std(right_coords)
        center_std = np.std(center_coords)
        
        consistency_threshold = max(5.0, image_width * 0.01)
        
        if left_std < consistency_threshold and right_std < consistency_threshold:
            return "justify"
        elif left_std < right_std and left_std < center_std:
            return "left"
        elif right_std < left_std and right_std < center_std:
            return "right"
        elif center_std < left_std and center_std < right_std:
            return "center"
        
        return self.alignment

    def _detect_color(self, image_crop) -> str:
        """Analyzes text color using simple thresholding (no OCR call)."""
        import cv2
        import numpy as np
        
        try:
            # Convert to RGB
            if len(image_crop.shape) == 3:
                rgb_img = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
            else:
                return self.fontColor
            
            # Simple approach: find dark pixels (text) and get their color
            gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            text_pixels = rgb_img[mask == 255]
            
            if len(text_pixels) > 0:
                avg_color = np.mean(text_pixels, axis=0).astype(int)
                return "{:02x}{:02x}{:02x}".format(avg_color[0], avg_color[1], avg_color[2])
            
            return self.fontColor
            
        except Exception:
            return self.fontColor

    def update_font_size(self, dpi: int):
        """
        Updates font size using the Gold-Standard formula.
        font_size_pt = (h_char_px_page / dpi) * 72 * 1.33
        """
        if self._pixelLineHeight > 0:
            h_char_pt = (self._pixelLineHeight / dpi) * 72
            size = h_char_pt * 1.43
            
            # Snap to Common Sizes
            common_sizes = [8, 9, 10, 11, 12, 14, 16, 18, 24, 30, 36, 48, 60, 72]
            best_size = min(common_sizes, key=lambda x: abs(x - size))
            
            if abs(best_size - size) < 2.0:
                self.fontSize = float(best_size)
            else:
                self.fontSize = round(size, 1)
