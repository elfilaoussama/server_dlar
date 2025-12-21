"""
Table Structure Recognition using Microsoft Table Transformer.
Uses HuggingFace transformers library.
"""
import torch
from PIL import Image
from transformers import AutoImageProcessor, TableTransformerForObjectDetection
from typing import List, Dict, Tuple
import cv2
import numpy as np


class TableRecognitionEngine:
    """
    Recognizes table structure (rows, columns, cells) using Microsoft's Table Transformer.
    Model: microsoft/table-transformer-structure-recognition
    """
    
    # Label mapping for table-transformer-structure-recognition
    # 0: table, 1: table column, 2: table row, 3: table column header, 
    # 4: table projected row header, 5: table spanning cell
    LABEL_MAP = {
        0: "table",
        1: "table column",
        2: "table row", 
        3: "table column header",
        4: "table projected row header",
        5: "table spanning cell"
    }
    
    def __init__(self, model_name: str = "microsoft/table-transformer-structure-recognition"):
        """
        Initialize the Table Transformer model.
        
        Args:
            model_name: HuggingFace model name/path
        """
        print(f"TableRecognitionEngine: Loading model '{model_name}'...")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = TableTransformerForObjectDetection.from_pretrained(model_name)
        self.model.eval()
        
        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"TableRecognitionEngine: Model loaded on {self.device}")
    
    def recognize_structure(self, table_crop: Image.Image, confidence_threshold: float = 0.5) -> Dict:
        """
        Recognize table structure from a cropped table image.
        
        Args:
            table_crop: PIL Image of the cropped table region
            confidence_threshold: Minimum confidence for detection
            
        Returns:
            Dictionary with keys:
            - rows: List of row bounding boxes [x1, y1, x2, y2]
            - columns: List of column bounding boxes
            - cells: List of cell bounding boxes (computed from row/column intersections)
            - headers: List of header bounding boxes
        """
        # Prepare inputs
        inputs = self.processor(images=table_crop, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process outputs
        target_sizes = torch.tensor([table_crop.size[::-1]]).to(self.device)  # (height, width)
        results = self.processor.post_process_object_detection(
            outputs, 
            threshold=confidence_threshold, 
            target_sizes=target_sizes
        )[0]
        
        # Organize results by type
        rows = []
        columns = []
        headers = []
        spanning_cells = []
        
        for box, label, score in zip(
            results["boxes"].cpu().numpy(),
            results["labels"].cpu().numpy(), 
            results["scores"].cpu().numpy()
        ):
            box_list = box.tolist()  # [x1, y1, x2, y2]
            
            if label == 2:  # table row
                rows.append({"bbox": box_list, "score": float(score)})
            elif label == 1:  # table column
                columns.append({"bbox": box_list, "score": float(score)})
            elif label == 3:  # table column header
                headers.append({"bbox": box_list, "score": float(score)})
            elif label == 5:  # spanning cell
                spanning_cells.append({"bbox": box_list, "score": float(score)})
        
        # Sort rows by Y coordinate (top to bottom)
        rows.sort(key=lambda r: r["bbox"][1])
        
        # Sort columns by X coordinate (left to right)
        columns.sort(key=lambda c: c["bbox"][0])
        
        # Compute cells from row/column intersections
        cells = self._compute_cells(rows, columns, table_crop.size)
        
        print(f"TableRecognitionEngine: Found {len(rows)} rows, {len(columns)} columns, {len(cells)} cells")
        
        return {
            "rows": rows,
            "columns": columns,
            "cells": cells,
            "headers": headers,
            "spanning_cells": spanning_cells
        }
    
    def _compute_cells(self, rows: List[Dict], columns: List[Dict], image_size: Tuple[int, int]) -> List[Dict]:
        """
        Compute cell bounding boxes from row and column intersections.
        
        Args:
            rows: List of row detections with bbox
            columns: List of column detections with bbox
            image_size: (width, height) of the image
            
        Returns:
            List of cell dictionaries with bbox and row/col indices
        """
        cells = []
        
        for row_idx, row in enumerate(rows):
            for col_idx, col in enumerate(columns):
                # Cell bbox is intersection of row and column
                x1 = max(row["bbox"][0], col["bbox"][0])
                y1 = max(row["bbox"][1], col["bbox"][1])
                x2 = min(row["bbox"][2], col["bbox"][2])
                y2 = min(row["bbox"][3], col["bbox"][3])
                
                # Only add if intersection is valid
                if x2 > x1 and y2 > y1:
                    cells.append({
                        "bbox": [x1, y1, x2, y2],
                        "row_idx": row_idx,
                        "col_idx": col_idx
                    })
        
        return cells
    
    def recognize_from_cv2(self, cv2_image: np.ndarray, confidence_threshold: float = 0.5) -> Dict:
        """
        Convenience method to recognize structure from a cv2/numpy image.
        
        Args:
            cv2_image: OpenCV image (BGR format)
            confidence_threshold: Minimum confidence for detection
            
        Returns:
            Same as recognize_structure()
        """
        # Convert BGR to RGB and then to PIL
        rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        return self.recognize_structure(pil_image, confidence_threshold)
