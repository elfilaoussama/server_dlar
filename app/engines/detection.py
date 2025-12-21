"""
Detection Engine for document layout analysis using YOLO.
"""
import cv2
import numpy as np
import os
import pytesseract
from ultralytics import YOLO
from typing import List, Tuple, Dict

from app.models import (
    PageElement, TextObject, TableObject, TableRow, TableCell,
    BoundingBox, ImageObject, Page, Document, SeparatorObject
)
from app.utils import normalize_input_image, preprocess_image_for_ocr
from app.engines.table_recognition import TableRecognitionEngine
from config import TESSERACT_CMD

# Ensure pytesseract can find the executable
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD 

class DetectionEngine:
    def __init__(self, model_path="yolov10m_doclaynet.onnx", use_table_transformer=True):
        print(f"DetectionEngine: Loading YOLO model from {model_path}...")
        try:
            self.model = YOLO(model_path, task='detect')
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            print("Using default 'yolov8n.pt' (Note: This is COCO trained, not DocLayNet!)")
            self.model = YOLO("yolov8n.pt")
        
        # Initialize Table Transformer for better table structure recognition
        self.table_transformer = None
        if use_table_transformer:
            try:
                self.table_transformer = TableRecognitionEngine()
            except Exception as e:
                print(f"Warning: Failed to load Table Transformer: {e}")
                print("Falling back to basic table detection.")
            
        # DocLayNet Class Mapping (Alphabetical as per model.names)
        self.class_map = {
            0: "Caption",
            1: "Footnote",
            2: "Formula",
            3: "List-item",
            4: "Page-footer",
            5: "Page-header",
            6: "Picture",
            7: "Section-header",
            8: "Table",
            9: "Text",
            10: "Title"
        }
            
    def _detect_separators(self, image: np.ndarray) -> List[SeparatorObject]:
        """
        Detects horizontal separator lines using computer vision.
        """
        separators = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding to handle varying lighting/shadows
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 2)
        
        # Create a horizontal kernel
        # Width should be significant (e.g., 1/30 of image width) to filter out text underlines
        h_kernel_len = max(20, image.shape[1] // 30)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_len, 1))
        
        # Morphological opening to isolate horizontal lines
        detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        
        # Dilate a bit to merge broken segments
        detected_lines = cv2.dilate(detected_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1)), iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Filter based on dimensions
            # Must be wide enough (e.g., > 10% of page width)
            # Must be thin enough (e.g., < 2% of page height)
            # Aspect ratio must be high
            
            if w > image.shape[1] * 0.1 and h < image.shape[0] * 0.02 and w / h > 10:
                # Create SeparatorObject
                sep = SeparatorObject(
                    id=f"sep_{x}_{y}",
                    bbox=BoundingBox(float(x), float(y), float(w), float(h)),
                    label="Separator",
                    orientation="horizontal",
                    thickness=h,
                    color="000000" # Default black
                )
                separators.append(sep)
                
        print(f"DetectionEngine: Detected {len(separators)} separators.")
        return separators
            


    def detectLayout(self, image_path: str, output_dir: str = ".") -> Tuple[Document, str, np.ndarray]:
        """
        Detect layout elements in an image.
        OPTIMIZED: Returns image object for reuse in subsequent pipeline steps.
        
        Returns:
            Tuple of (Document, image_path, image_array)
        """
        import os
        
        image = normalize_input_image(image_path)
        
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")
            
        print(f"DetectionEngine: Detecting layout in {image_path} using YOLOv8...")
        
        # Run inference
        results = self.model(image)
        
        elements: List[PageElement] = []
        
        # Process results
        # YOLOv8 results object contains boxes, scores, classes
        result = results[0]
        boxes = result.boxes
        
        print(f"PipelineController: Detected {len(boxes)} raw elements.")
        
        # Collect image candidates for merging
        image_candidates = []
        
        for box in boxes:
            # Get coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = self.class_map.get(cls, "Unknown")
            
            bbox = BoundingBox(x=x1, y=y1, width=x2-x1, height=y2-y1)
            
            # Create element based on label
            if label in ["Text", "Title", "Section-header", "Caption", "Footnote", "List-item"]:
                x, y, w, h = int(x1), int(y1), int(x2-x1), int(y2-y1)
                
                # Crop the region with PADDING for better OCR
                pad = 10
                
                # Apply padding, ensuring we don't go out of bounds
                x_p = max(0, x - pad)
                y_p = max(0, y - pad)
                w_p = min(image.shape[1] - x_p, w + 2*pad)
                h_p = min(image.shape[0] - y_p, h + 2*pad)
                
                crop_padded = image[y_p:y_p+h_p, x_p:x_p+w_p]
                
                # Create TextObject (pass padded crop for analysis, but original bbox for layout)
                text_obj = self._read_text_content(crop_padded, bbox, label)
                text_obj.label = label
                elements.append(text_obj)
                
            elif label == "Table":
                # Create TableObject
                # Crop for table analysis (Tables usually don't need padding for structure, 
                # but cells might. We handle cell padding inside _analyze_table_structure)
                x, y, w, h = int(x1), int(y1), int(x2-x1), int(y2-y1)
                crop = image[max(0, y):min(image.shape[0], y+h), max(0, x):min(image.shape[1], x+w)]
                
                table_obj = self._analyze_table_structure(crop, bbox, image)
                table_obj.label = label
                elements.append(table_obj)
                
            elif label in ["Picture", "Figure", "Image", "Page-header", "Page-footer"]:
                # Force Header/Footer to full width to ensure they contain all text
                # User requested: "images in header should not be merged"
                # So we disable full width for Page-header to prevent it from swallowing logos.
                if label == "Page-footer":
                    bbox.x = 0
                    bbox.width = image.shape[1]
                
                # Collect for merging
                image_candidates.append(bbox)

        # Merge intersecting image boxes
        merged_image_boxes = self._merge_intersecting_boxes(image_candidates)
        
        # Check for overlap with text elements BEFORE processing images
        # If an image overlaps significantly with a text box, we assume it contains text.
        image_has_text = [False] * len(merged_image_boxes)
        
        text_elements = [e for e in elements if isinstance(e, TextObject)]
        
        for i, img_bbox in enumerate(merged_image_boxes):
            img_x2 = img_bbox.x + img_bbox.width
            img_y2 = img_bbox.y + img_bbox.height
            
            for text_obj in text_elements:
                txt_bbox = text_obj.bbox
                txt_x2 = txt_bbox.x + txt_bbox.width
                txt_y2 = txt_bbox.y + txt_bbox.height
                
                # Check intersection
                x_left = max(img_bbox.x, txt_bbox.x)
                y_top = max(img_bbox.y, txt_bbox.y)
                x_right = min(img_x2, txt_x2)
                y_bottom = min(img_y2, txt_y2)
                
                if x_right > x_left and y_bottom > y_top:
                    # Intersection exists
                    intersection_area = (x_right - x_left) * (y_bottom - y_top)
                    text_area = txt_bbox.width * txt_bbox.height
                    
                    # If significant part of text is inside image (e.g., > 20%)
                    if intersection_area > 0.2 * text_area:
                        image_has_text[i] = True
                        break
        
        # Create ImageObjects from merged boxes
        for i, bbox in enumerate(merged_image_boxes):
            img_path = os.path.join(output_dir, "images", f"image_{int(bbox.y)}_{i}.png")
            os.makedirs(os.path.dirname(img_path), exist_ok=True)
            
            # Save crop with 5px padding (User Request)
            pad = 5
            x, y, w, h = int(bbox.x), int(bbox.y), int(bbox.width), int(bbox.height)
            
            # Apply padding
            y1, y2 = max(0, y - pad), min(image.shape[0], y + h + pad)
            x1, x2 = max(0, x - pad), min(image.shape[1], x + w + pad)
            
            crop = image[y1:y2, x1:x2]
            
            has_text = image_has_text[i]
            
            if crop.size > 0:
                if has_text:
                    print(f"DetectionEngine: Image {i} overlaps with text. Skipping OCR/rembg.")
                    cv2.imwrite(img_path, crop)
                else:
                    # If it's a logo or icon, removing background is better.
                    try:
                        # TESSDATA_PREFIX is set in main.py
                        custom_config = f'--oem 1 --psm 3'
                        languages = 'eng+fra+ara'
                        
                        processed_crop = preprocess_image_for_ocr(crop)
                        
                        text_content = pytesseract.image_to_string(processed_crop, lang=languages, config=custom_config).strip()
                    except:
                        text_content = ""
                    
                    if len(text_content) < 5: # Threshold for "no text"
                        # Skip rembg for small images (e.g. < 150px) to prevent degradation
                        if crop.shape[0] < 150 or crop.shape[1] < 150:
                            print(f"DetectionEngine: Image {i} is small ({crop.shape[1]}x{crop.shape[0]}). Skipping rembg.")
                            cv2.imwrite(img_path, crop)
                        else:
                            try:
                                from rembg import remove
                                # rembg expects input as bytes or PIL image or numpy array
                                # It returns a numpy array (if input is numpy) with alpha channel
                                print(f"DetectionEngine: Removing background for image {i} (Text len={len(text_content)})...")
                                crop_nobg = remove(
                                    crop
                                )
                                cv2.imwrite(img_path, crop_nobg)
                            except ImportError:
                                print("DetectionEngine: rembg not installed. Saving original.")
                                cv2.imwrite(img_path, crop)
                            except Exception as e:
                                print(f"DetectionEngine: Background removal failed: {e}. Saving original.")
                                cv2.imwrite(img_path, crop)

                    else:
                        print(f"DetectionEngine: Keeping background for image {i} (Text detected: '{text_content[:10]}...').")
                        cv2.imwrite(img_path, crop)
            
            img_obj = ImageObject(
                id=f"image_{int(bbox.y)}_{i}",
                bbox=bbox,
                imagePath=img_path,
                label="Image", # Merged images are generic
                hasTextContent=has_text
            )
            elements.append(img_obj)
            
        # Detect Separators (Horizontal Lines)
        separators = self._detect_separators(image)
        elements.extend(separators)
        
        # Try to detect DPI from image metadata
        from PIL import Image as PILImage
        pil_img = PILImage.open(image_path)
        detected_dpi = 0
        if 'dpi' in pil_img.info:
            try:
                detected_dpi = int(pil_img.info['dpi'][0])
                print(f"DetectionEngine: Found DPI in metadata: {detected_dpi}")
            except:
                pass
        
        # Create Page
        page = Page(
            pageNumber=1,
            width=float(image.shape[1]),
            height=float(image.shape[0]),
            originalDPI=detected_dpi, # 0 if not found, will trigger inference
            elements=elements
        )
        
        # Fill dimensions (x, y, w, h)
        # Assuming page starts at 0,0
        page.fill_dimensions(0.0, 0.0, float(image.shape[1]), float(image.shape[0]))
        
        # Detect and propagate DPI - MOVED to LayoutEngine after post-merge analysis
        # page.detect_and_propagate_dpi()
        
        # Create Document
        doc = Document(docID="doc_1", pages=[page])
        
        return doc, image_path, image

    def _merge_intersecting_boxes(self, boxes: List[BoundingBox]) -> List[BoundingBox]:
        """
        Merge bounding boxes that intersect.
        """
        if not boxes:
            return []
            
        # Convert to list of [x1, y1, x2, y2] for easier processing
        # We use a simple iterative merge strategy
        merged_boxes = []
        
        # Sort by Y to optimize
        sorted_boxes = sorted(boxes, key=lambda b: b.y)
        
        while sorted_boxes:
            current = sorted_boxes.pop(0)
            
            # Check for intersection with remaining boxes
            # If intersect, merge and keep checking
            has_merged = True
            while has_merged:
                has_merged = False
                i = 0
                while i < len(sorted_boxes):
                    other = sorted_boxes[i]
                    
                    # Check intersection
                    # Box A: current.x, current.y, current.width, current.height
                    # Box B: other.x, other.y, other.width, other.height
                    
                    x1_a, y1_a = current.x, current.y
                    x2_a, y2_a = current.x + current.width, current.y + current.height
                    
                    x1_b, y1_b = other.x, other.y
                    x2_b, y2_b = other.x + other.width, other.y + other.height
                    
                    # Intersection check
                    x_left = max(x1_a, x1_b)
                    y_top = max(y1_a, y1_b)
                    x_right = min(x2_a, x2_b)
                    y_bottom = min(y2_a, y2_b)
                    
                    if x_right > x_left and y_bottom > y_top:
                        # Intersecting! Merge into current
                        new_x1 = min(x1_a, x1_b)
                        new_y1 = min(y1_a, y1_b)
                        new_x2 = max(x2_a, x2_b)
                        new_y2 = max(y2_a, y2_b)
                        
                        current.x = new_x1
                        current.y = new_y1
                        current.width = new_x2 - new_x1
                        current.height = new_y2 - new_y1
                        
                        # Remove other from list
                        sorted_boxes.pop(i)
                        has_merged = True
                        # Don't increment i, as we removed element
                    else:
                        i += 1
            
            merged_boxes.append(current)
            
        return merged_boxes

    def _analyze_table_structure(self, crop, bbox: BoundingBox, full_image=None) -> TableObject:
        """
        Analyze table structure using Table Transformer if available, 
        otherwise fall back to basic morphological detection.
        """
        print("    -> Analyzing table structure...")
        
        # Try Table Transformer first
        if self.table_transformer is not None:
            try:
                return self._analyze_with_transformer(crop, bbox)
            except Exception as e:
                print(f"    [WARNING] Table Transformer failed: {e}. Falling back to basic detection.")
        
        # Fallback to basic morphological detection
        return self._analyze_basic(crop, bbox)
    
    def _analyze_with_transformer(self, crop, bbox: BoundingBox) -> TableObject:
        """Use Table Transformer for structured table extraction."""
        print("    -> Using Table Transformer for cell detection...")
        
        # Get structure from Table Transformer
        structure = self.table_transformer.recognize_from_cv2(crop, confidence_threshold=0.5)
        
        rows_detected = structure["rows"]
        columns_detected = structure["columns"]
        cells_detected = structure["cells"]
        
        print(f"    -> Table Transformer: {len(rows_detected)} rows, {len(columns_detected)} cols, {len(cells_detected)} cells")
        
        table = TableObject(id=f"table_{int(bbox.y)}", bbox=bbox)
        
        if not cells_detected:
            # Fallback if no cells detected
            print("    -> No cells detected by Transformer, using basic method...")
            return self._analyze_basic(crop, bbox)
        
        # Group cells by row
        from collections import defaultdict
        rows_dict = defaultdict(list)
        
        for cell in cells_detected:
            rows_dict[cell["row_idx"]].append(cell)
        
        # Sort rows by index
        row_indices = sorted(rows_dict.keys())
        
        for row_idx in row_indices:
            row_cells = rows_dict[row_idx]
            row_cells.sort(key=lambda c: c["col_idx"])  # Sort by column
            
            tr = TableRow(rowIndex=row_idx)
            
            for cell in row_cells:
                cell_box = cell["bbox"]  # [x1, y1, x2, y2]
                x1, y1, x2, y2 = [int(v) for v in cell_box]
                
                # Convert to absolute coordinates
                abs_x = bbox.x + x1
                abs_y = bbox.y + y1
                width = x2 - x1
                height = y2 - y1
                
                cell_bbox = BoundingBox(x=abs_x, y=abs_y, width=width, height=height)
                
                tc = TableCell(
                    colIndex=cell["col_idx"], 
                    rowSpan=1, colSpan=1,
                    hasVisibleBorder=True, 
                    backgroundColor="FFFFFF", 
                    verticalAlign="top",
                    bbox=cell_bbox
                )
                
                # Extract text from cell with expanded bounds
                if y2 > y1 and x2 > x1:
                    # Expand crop by 5px on all sides to capture edge text
                    pad_expand = 5
                    crop_y1 = max(0, y1 - pad_expand)
                    crop_y2 = min(crop.shape[0], y2 + pad_expand)
                    crop_x1 = max(0, x1 - pad_expand)
                    crop_x2 = min(crop.shape[1], x2 + pad_expand)
                    
                    cell_crop = crop[crop_y1:crop_y2, crop_x1:crop_x2]
                    
                    if cell_crop.size > 0:
                        text_obj = self._extract_cell_text(cell_crop, cell_bbox)
                        if text_obj and text_obj.rawText.strip():
                            tc.content.append(text_obj)
                
                tr.cells.append(tc)
            
            table.rows.append(tr)
        
        table.rowCount = len(row_indices)
        table.colCount = len(columns_detected)
        
        return table
    
    def _extract_cell_text(self, cell_crop, cell_bbox: BoundingBox) -> TextObject:
        """Extract text from a table cell using OCR with edge noise removal."""
        try:
            # Convert to grayscale if needed
            if len(cell_crop.shape) == 3:
                gray = cv2.cvtColor(cell_crop, cv2.COLOR_BGR2GRAY)
            else:
                gray = cell_crop.copy()
            
            h, w = gray.shape[:2]
            
            # Skip very small cells
            if h < 5 or w < 5:
                return None
            
            # Upscale small cells for better OCR
            if h < 30 or w < 30:
                scale = max(2.0, 30 / min(h, w))
                gray = cv2.resize(gray, None, fx=scale, fy=scale, 
                                  interpolation=cv2.INTER_CUBIC)
            
            # Edge noise removal using flood fill from borders
            h, w = gray.shape[:2]
            mask = np.zeros((h + 2, w + 2), np.uint8)
            edge_cleaned = gray.copy()
            flood_fill_value = 255
            threshold = 30
            
            # Fill dark areas connected to edges (noise/borders)
            for x in range(0, w, 5):
                if edge_cleaned[0, x] < 200:
                    cv2.floodFill(edge_cleaned, mask, (x, 0), flood_fill_value, threshold, threshold)
                if edge_cleaned[h-1, x] < 200:
                    cv2.floodFill(edge_cleaned, mask, (x, h-1), flood_fill_value, threshold, threshold)
            
            for y in range(0, h, 5):
                if edge_cleaned[y, 0] < 200:
                    cv2.floodFill(edge_cleaned, mask, (0, y), flood_fill_value, threshold, threshold)
                if edge_cleaned[y, w-1] < 200:
                    cv2.floodFill(edge_cleaned, mask, (w-1, y), flood_fill_value, threshold, threshold)
            
            # Add 5px white padding
            pad = 5
            cell_crop_padded = cv2.copyMakeBorder(
                edge_cleaned, pad, pad, pad, pad, 
                cv2.BORDER_CONSTANT, value=255
            )
            
            # OCR
            custom_config = '--oem 1 --psm 6'
            languages = 'eng+fra+ara'
            
            text = pytesseract.image_to_string(
                cell_crop_padded, lang=languages, config=custom_config
            ).strip()
            
            # Clean text - remove extra whitespace and newlines
            text = ' '.join(text.split())
            
            if not text:
                return None
            
            return TextObject(
                id=f"cell_text_{int(cell_bbox.y)}_{int(cell_bbox.x)}",
                bbox=cell_bbox,
                rawText=text,
                fontSize=9.0
            )
        except Exception as e:
            print(f"    [WARNING] Cell OCR failed: {e}")
            return None
    
    def _analyze_basic(self, crop, bbox: BoundingBox) -> TableObject:
        """Basic morphological table structure detection (fallback)."""
        # Grayscale and threshold
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 2)
                                       
        # Define kernels
        h_kernel_len = max(1, crop.shape[1] // 20)
        v_kernel_len = max(1, crop.shape[0] // 20)
        
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_len, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_len))
        
        h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel)
        v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel)
        
        # Combine
        table_mask = cv2.add(h_lines, v_lines)
        
        # Find contours
        contours, _ = cv2.findContours(table_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        cell_contours = [c for c in contours if cv2.contourArea(c) > 100]
        
        table = TableObject(id=f"table_{int(bbox.y)}", bbox=bbox)
        
        if not cell_contours:
            # Fallback: 1x1 table with full table OCR
            row = TableRow(rowIndex=0)
            cell = TableCell(colIndex=0, rowSpan=1, colSpan=1, hasVisibleBorder=False, 
                             backgroundColor="FFFFFF", verticalAlign="top", bbox=bbox)
            
            # Try to extract text from entire table
            text_obj = self._extract_cell_text(crop, bbox)
            if text_obj and text_obj.rawText.strip():
                cell.content.append(text_obj)
            
            row.cells.append(cell)
            table.rows.append(row)
            return table
            
        # Sort contours
        bounding_boxes = [cv2.boundingRect(c) for c in cell_contours]
        bounding_boxes.sort(key=lambda b: b[1])  # Sort by Y
        
        rows = []
        current_row_y = -1
        row_threshold = 10
        current_row_cells = []
        
        for (x, y, w, h) in bounding_boxes:
            if current_row_y == -1:
                current_row_y = y
                current_row_cells.append((x, y, w, h))
            elif abs(y - current_row_y) < row_threshold:
                current_row_cells.append((x, y, w, h))
            else:
                rows.append(current_row_cells)
                current_row_cells = [(x, y, w, h)]
                current_row_y = y
        if current_row_cells:
            rows.append(current_row_cells)
            
        for r_idx, row_data in enumerate(rows):
            row_data.sort(key=lambda b: b[0])  # Sort by X
            tr = TableRow(rowIndex=r_idx)
            for c_idx, (x, y, w, h) in enumerate(row_data):
                # Convert to absolute coordinates
                abs_x = bbox.x + x
                abs_y = bbox.y + y
                cell_bbox = BoundingBox(x=abs_x, y=abs_y, width=w, height=h)
                
                tc = TableCell(
                    colIndex=c_idx, rowSpan=1, colSpan=1, 
                    hasVisibleBorder=True, backgroundColor="FFFFFF", verticalAlign="top",
                    bbox=cell_bbox
                )
                
                # Extract text content from the cell
                y1, y2 = max(0, y), min(crop.shape[0], y+h)
                x1, x2 = max(0, x), min(crop.shape[1], x+w)
                
                if y2 > y1 and x2 > x1:
                    cell_crop = crop[y1:y2, x1:x2]
                    text_obj = self._extract_cell_text(cell_crop, cell_bbox)
                    if text_obj and text_obj.rawText.strip():
                        tc.content.append(text_obj)

                tr.cells.append(tc)
            table.rows.append(tr)
            
        table.rowCount = len(rows)
        table.colCount = max([len(r.cells) for r in table.rows]) if table.rows else 0
        
        return table

    def _read_text_content(self, crop, bbox: BoundingBox, label: str, config: str = '') -> TextObject:
        # DEFER OCR: We will perform OCR and analysis AFTER merging bounding boxes.
        # This ensures we analyze complete paragraphs/blocks for better font size estimation.
        
        # Heuristics for style based on label
        is_bold = (label == "Title" or label == "Section-header")
        font_size = 14 if label == "Title" else 12
        if label == "Section-header": font_size = 13
        if label == "Caption": font_size = 10
        
        # Create text object with EMPTY text
        text_obj = TextObject(
            id=f"text_{int(bbox.y)}",
            bbox=bbox,
            rawText="", # Will be populated later
            fontSize=font_size,
            isBold=is_bold
        )
        
        if config:
            # Immediate OCR (e.g. for Table Cells)
            try:
                import pytesseract
                # Use languages from config or default
                # We assume config string contains everything needed or we use default langs
                # Actually, _analyze_table_structure passes config with --tessdata-dir etc.
                # But it doesn't pass 'lang' argument to this method.
                # We should probably pass 'lang' or just hardcode 'eng+fra+ara' here as well.
                languages = 'eng+fra+ara'
                
                processed_crop = preprocess_image_for_ocr(crop)
                
                raw_text = pytesseract.image_to_string(processed_crop, lang=languages, config=config).strip()
                
                # Clean text: Remove single newlines, preserve double newlines
                # 1. Replace double newlines with placeholder
                # 2. Replace single newlines with space
                # 3. Restore double newlines
                cleaned_text = raw_text.replace('\n\n', '<PARA_BREAK>').replace('\n', ' ').replace('<PARA_BREAK>', '\n\n')
                text_obj.rawText = cleaned_text
            except Exception as e:
                print(f"    [WARNING] Immediate OCR failed: {e}")
        
        return text_obj


class LayoutEngine:
    def reconcileHierarchy(self, elements: List[PageElement], image = None, output_dir: str = None, known_dpi: int = 0) -> List[PageElement]:
        """
        Reconciles layout hierarchy and performs post-merge analysis.
        OPTIMIZED: Accepts image object directly instead of reading from disk.
        """
        print("LayoutEngine: Reconciling hierarchy...")
        
        # Step 0: Filter overlaps
        elements = self._filter_overlaps(elements)
        
        # Step 1: Merge close texts (needs image for crop analysis)
        elements = self._merge_close_texts(elements, image)
        
        # Step 1.5: Save merged text crops
        if image is not None and output_dir:
            self._save_merged_crops(elements, image, output_dir)
            
        # Step 2: Batch Analysis (Post-Merge)
        # Now that boxes are merged, we perform OCR and analysis on the full blocks.
        if image is not None:
            print("LayoutEngine: Performing post-merge text analysis...")
            for element in elements:
                if isinstance(element, TextObject):
                    element.analyze_content(image)
            
            # Convert empty OCR text elements to images
            empty_ocr_elements = [e for e in elements if isinstance(e, TextObject) and e.isEmptyOCR]
            if empty_ocr_elements:
                print(f"LayoutEngine: Converting {len(empty_ocr_elements)} empty OCR text regions to images...")
                for i, text_elem in enumerate(empty_ocr_elements):
                    # Remove from elements
                    elements.remove(text_elem)
                    
                    # Create image from the text region
                    bbox = text_elem.bbox
                    x, y, w, h = int(bbox.x), int(bbox.y), int(bbox.width), int(bbox.height)
                    
                    # Crop with padding
                    pad = 5
                    y1, y2 = max(0, y - pad), min(image.shape[0], y + h + pad)
                    x1, x2 = max(0, x - pad), min(image.shape[1], x + w + pad)
                    
                    crop = image[y1:y2, x1:x2]
                    
                    if crop.size > 0:
                        # Save as image
                        img_path = os.path.join(output_dir, "images", f"text_as_image_{int(bbox.y)}_{i}.png")
                        os.makedirs(os.path.dirname(img_path), exist_ok=True)
                        cv2.imwrite(img_path, crop)
                        
                        # Create ImageObject to replace the text
                        img_obj = ImageObject(
                            id=f"text_image_{int(bbox.y)}_{i}",
                            bbox=bbox,
                            imagePath=img_path
                        )
                        elements.append(img_obj)
                    
            # Step 3: Detect and Propagate DPI (Global Font Size Correction)
            # If known_dpi is provided (from metadata), use it. Otherwise, infer it.
            
            final_dpi = known_dpi
            
            if final_dpi > 0:
                print(f"LayoutEngine: Using Metadata DPI = {final_dpi}")
            else:
                # Infer DPI using Gold-Standard logic
                import numpy as np
                line_heights = [e._pixelLineHeight for e in elements if isinstance(e, TextObject) and e._pixelLineHeight > 0]
                
                if line_heights:
                    median_height = np.median(line_heights)
                    possible_dpis = [72, 96, 150, 200, 300]
                    final_dpi = 96 # Default fallback
                    
                    for test_dpi in possible_dpis:
                        # test_size = (median_height / test_dpi) * 72 * 1.33
                        test_size = (median_height / test_dpi) * 72 * 1.33
                        if 9 <= test_size <= 14:
                            final_dpi = test_dpi
                            break
                    print(f"LayoutEngine: Inferred DPI = {final_dpi} (Median H={median_height:.1f}px)")
            
            # Apply DPI to all text elements
            if final_dpi > 0:
                for element in elements:
                    if isinstance(element, TextObject):
                        element.update_font_size(final_dpi)
        
        tables = [e for e in elements if isinstance(e, TableObject)]
        texts = [e for e in elements if isinstance(e, TextObject)]
        images = [e for e in elements if isinstance(e, ImageObject)]
        
        assigned_text_ids = set()

        for table in tables:
            for row in table.rows:
                for cell in row.cells:
                    if not cell.bbox:
                        continue
                        
                    for text in texts:
                        if text.id in assigned_text_ids:
                            continue
                        
                        cx = text.bbox.getCenterX()
                        cy = text.bbox.getCenterY()
                        
                        if (cell.bbox.x <= cx <= cell.bbox.x + cell.bbox.width) and \
                           (cell.bbox.y <= cy <= cell.bbox.y + cell.bbox.height):
                            
                            cell.content.append(text)
                            text.parentID = table.id
                            assigned_text_ids.add(text.id)

        # Filter out texts that are inside images (e.g. logos with text)
        for img in images:
            for text in texts:
                if text.id in assigned_text_ids:
                    continue
                
                # Check if text is inside image
                cx = text.bbox.getCenterX()
                cy = text.bbox.getCenterY()
                
                if (img.bbox.x <= cx <= img.bbox.x + img.bbox.width) and \
                   (img.bbox.y <= cy <= img.bbox.y + img.bbox.height):
                    print(f"  -> Ignoring text '{text.rawText[:20]}...' as it is inside an Image/Logo.")
                    assigned_text_ids.add(text.id) # Mark as assigned so it's not added to final list

        final_elements = []
        final_elements.extend(tables)
        for text in texts:
            if text.id not in assigned_text_ids:
                final_elements.append(text)
        final_elements.extend(images)
        
        return self.sortReadingOrder(final_elements)

    def sortReadingOrder(self, elements: List[PageElement]) -> List[PageElement]:
        return sorted(elements, key=lambda e: (e.bbox.y, e.bbox.x))

    def _filter_overlaps(self, elements: List[PageElement]) -> List[PageElement]:
        from collections import defaultdict
        type_groups = defaultdict(list)
        for e in elements:
            type_groups[type(e)].append(e)
            
        final_elements = []
        
        for elem_type, group in type_groups.items():
            group.sort(key=lambda e: e.bbox.width, reverse=True)
            
            kept_elements = []
            removed_ids = set()
            
            for i, elem_a in enumerate(group):
                if elem_a.id in removed_ids:
                    continue
                    
                kept_elements.append(elem_a)
                
                area_a = elem_a.bbox.width * elem_a.bbox.height
                
                for j in range(i + 1, len(group)):
                    elem_b = group[j]
                    if elem_b.id in removed_ids:
                        continue
                        
                    x_left = max(elem_a.bbox.x, elem_b.bbox.x)
                    y_top = max(elem_a.bbox.y, elem_b.bbox.y)
                    x_right = min(elem_a.bbox.x + elem_a.bbox.width, elem_b.bbox.x + elem_b.bbox.width)
                    y_bottom = min(elem_a.bbox.y + elem_a.bbox.height, elem_b.bbox.y + elem_b.bbox.height)
                    
                    if x_right > x_left and y_bottom > y_top:
                        intersection_area = (x_right - x_left) * (y_bottom - y_top)
                        area_b = elem_b.bbox.width * elem_b.bbox.height
                        
                        overlap_ratio = intersection_area / min(area_a, area_b)
                        
                        if overlap_ratio > 0.8:
                            removed_ids.add(elem_b.id)
            
            final_elements.extend(kept_elements)
                        
        return final_elements

    def _merge_close_texts(self, elements: List[PageElement], image = None) -> List[PageElement]:
        """Merge vertically adjacent text blocks. OPTIMIZED: Uses passed image object."""
        texts = [e for e in elements if isinstance(e, TextObject)]
        others = [e for e in elements if not isinstance(e, TextObject)]
        
        # Image is now passed directly, no need to read from disk
        
        if not texts:
            return elements
            
        texts.sort(key=lambda e: (e.bbox.y, e.bbox.x))
        
        merged_texts = []
        current_text = texts[0]
        
        for next_text in texts[1:]:
            distance = next_text.bbox.y - (current_text.bbox.y + current_text.bbox.height)
            x_diff = abs(current_text.bbox.x - next_text.bbox.x)
            width_diff = abs(current_text.bbox.width - next_text.bbox.width)
            
            is_close_vertically = -10 < distance < 25
            is_similar_width = width_diff < 50
            is_aligned_horizontally = x_diff < 50
            
            should_merge = is_close_vertically and (is_aligned_horizontally or is_similar_width)
            
            if should_merge:
                new_x = min(current_text.bbox.x, next_text.bbox.x)
                new_y = min(current_text.bbox.y, next_text.bbox.y)
                new_w = max(current_text.bbox.x + current_text.bbox.width, next_text.bbox.x + next_text.bbox.width) - new_x
                new_h = max(current_text.bbox.y + current_text.bbox.height, next_text.bbox.y + next_text.bbox.height) - new_y
                
                current_text.bbox.x = new_x
                current_text.bbox.y = new_y
                current_text.bbox.width = new_w
                current_text.bbox.height = new_h
                
                # Don't concatenate text yet, as it's empty. Just merge boxes.
                # print(f"  -> Merged text block (Dist: {distance:.1f}, W-Diff: {width_diff:.1f})")
            else:
                merged_texts.append(current_text)
                current_text = next_text
                
        merged_texts.append(current_text)
        
        return others + merged_texts

    def _save_merged_crops(self, elements: List[PageElement], image, output_dir: str):
        """Save merged text crops. OPTIMIZED: Uses passed image object."""
        import os
        
        if image is None:
            return
            
        merged_crops_dir = os.path.join(output_dir, "text_crops")
        os.makedirs(merged_crops_dir, exist_ok=True)
        
        text_count = 0
        for element in elements:
            if isinstance(element, TextObject):
                x1 = int(element.bbox.x)
                y1 = int(element.bbox.y)
                x2 = int(element.bbox.x + element.bbox.width)
                y2 = int(element.bbox.y + element.bbox.height)
                
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(image.shape[1], x2)
                y2 = min(image.shape[0], y2)
                
                crop = image[y1:y2, x1:x2]
                if crop.size > 0:
                    crop_filename = f"text_{text_count}.png"
                    crop_path = os.path.join(merged_crops_dir, crop_filename)
                    cv2.imwrite(crop_path, crop)
                    text_count += 1
        
        print(f"  -> Saved {text_count} text crops to {merged_crops_dir}")
