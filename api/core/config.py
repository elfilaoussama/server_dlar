"""
Application configuration using environment variables.
Uses simple module-level configuration with dynaconf-style approach.
"""
import os
from pathlib import Path
from functools import lru_cache
from typing import Optional

# Base paths
ROOT_DIR = Path(__file__).parent.parent.parent
LOGS_DIR = ROOT_DIR / "logs"
OUTPUT_DIR = ROOT_DIR / "output"
MODELS_DIR = ROOT_DIR / "models"
TESSDATA_DIR = ROOT_DIR / "tessdata"

# Ensure directories exist
LOGS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


def _get_env(key: str, default: str = "") -> str:
    """Get environment variable with default."""
    return os.getenv(key, default)


def _get_bool(key: str, default: bool = False) -> bool:
    """Get boolean environment variable."""
    return os.getenv(key, str(default)).lower() in ("true", "1", "yes")


def _get_int(key: str, default: int = 0) -> int:
    """Get integer environment variable."""
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


# =============================================================================
# API Configuration
# =============================================================================
APP_NAME = _get_env("APP_NAME", "Document Layout Analysis API")
APP_VERSION = _get_env("APP_VERSION", "1.0.0")
DEBUG = _get_bool("DEBUG", False)
HOST = _get_env("HOST", "0.0.0.0")
PORT = _get_int("PORT", 8000)

# =============================================================================
# Model Configuration
# =============================================================================
MODEL_PATH = _get_env("MODEL_PATH", str(MODELS_DIR / "yolov10m_doclaynet.onnx"))
USE_TABLE_TRANSFORMER = _get_bool("USE_TABLE_TRANSFORMER", True)
MODEL_CONFIDENCE_THRESHOLD = float(_get_env("MODEL_CONFIDENCE", "0.5"))

# =============================================================================
# Processing Configuration
# =============================================================================
MAX_WORKERS = _get_int("MAX_WORKERS", 4)
MAX_FILE_SIZE_MB = _get_int("MAX_FILE_SIZE_MB", 50)
ALLOWED_EXTENSIONS = _get_env("ALLOWED_EXTENSIONS", "jpg,jpeg,png,pdf").split(",")

# =============================================================================
# OCR Configuration
# =============================================================================
TESSERACT_CMD = _get_env("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
TESSDATA_PREFIX = _get_env("TESSDATA_PREFIX", str(TESSDATA_DIR))
OCR_LANGUAGES = _get_env("OCR_LANGUAGES", "eng+fra+ara")
OCR_DPI = _get_int("OCR_DPI", 300)

# =============================================================================
# Logging Configuration
# =============================================================================
LOG_LEVEL = _get_env("LOG_LEVEL", "INFO")
LOG_FORMAT = _get_env("LOG_FORMAT", "json")  # json or text
LOG_FILE_ENABLED = _get_bool("LOG_FILE_ENABLED", True)
LOG_FILE_PATH = _get_env("LOG_FILE_PATH", str(LOGS_DIR / "app.log"))
LOG_MAX_BYTES = _get_int("LOG_MAX_BYTES", 10 * 1024 * 1024)  # 10MB
LOG_BACKUP_COUNT = _get_int("LOG_BACKUP_COUNT", 5)

# =============================================================================
# Build/Deploy Information
# =============================================================================
GIT_COMMIT = _get_env("GIT_COMMIT", "development")
BUILD_DATE = _get_env("BUILD_DATE", "unknown")
ENVIRONMENT = _get_env("ENVIRONMENT", "development")


def get_config_dict() -> dict:
    """Get all configuration as dictionary (for debugging)."""
    return {
        "app_name": APP_NAME,
        "app_version": APP_VERSION,
        "debug": DEBUG,
        "environment": ENVIRONMENT,
        "model_path": MODEL_PATH,
        "log_level": LOG_LEVEL,
        "max_workers": MAX_WORKERS,
    }
