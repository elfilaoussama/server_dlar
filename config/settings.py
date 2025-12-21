"""
Centralized configuration settings for the Document Layout Analysis project.
"""
import os

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
TESSDATA_DIR = os.path.join(PROJECT_ROOT, "tessdata")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# Model paths
YOLO_MODEL_PATH = os.path.join(MODEL_DIR, "yolov10m_doclaynet.onnx")
TABLE_TRANSFORMER_MODEL = "microsoft/table-transformer-structure-recognition"

# Tesseract settings
TESSERACT_CMD = os.getenv('TESSERACT_CMD', '/usr/bin/tesseract')
OCR_LANGUAGES = "eng+fra+ara"
OCR_CONFIDENCE_THRESHOLD = 50

# Detection settings
DETECTION_CONFIDENCE_THRESHOLD = 0.25
TABLE_STRUCTURE_CONFIDENCE = 0.5
IOU_THRESHOLD = 0.45

# Image processing settings
IMAGE_PADDING = 5  # Padding for image crops
OCR_UPSCALE_FACTOR = 3.0
OCR_TARGET_TEXT_HEIGHT = 32  # Target text height in pixels for OCR

# Output settings
TARGET_WIDTH_INCHES = 8.5
DEFAULT_DPI = 72

# Class labels for YOLO DocLayNet model
DOCLAYNET_CLASSES = {
    0: 'Caption',
    1: 'Footnote', 
    2: 'Formula',
    3: 'List-item',
    4: 'Page-footer',
    5: 'Page-header',
    6: 'Picture',
    7: 'Section-header',
    8: 'Table',
    9: 'Text',
    10: 'Title'
}
