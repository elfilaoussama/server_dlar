"""
Pydantic schemas for API request/response validation.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime


class OutputFormat(str, Enum):
    DOCX = "docx"
    HTML = "html"
    XML = "xml"
    JSON = "json"
    ALL = "all"


class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ProcessingOptions(BaseModel):
    output_formats: List[OutputFormat] = Field(default=[OutputFormat.DOCX])
    ocr_languages: str = Field(default="eng+fra+ara")
    detect_tables: bool = Field(default=True)
    remove_backgrounds: bool = Field(default=True)
    dpi: int = Field(default=96, ge=72, le=600)


class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float


class DocumentElement(BaseModel):
    id: str
    type: str
    bbox: BoundingBox
    content: Optional[str] = None
    font_size: Optional[float] = None
    alignment: Optional[str] = None
    color: Optional[str] = None


class ProcessingResult(BaseModel):
    request_id: str
    status: ProcessingStatus
    processing_time_ms: float
    page_count: int = 1
    elements_detected: int
    elements: List[DocumentElement] = []
    output_files: Dict[str, str] = Field(default_factory=dict)
    errors: List[str] = []


class BatchProcessingResult(BaseModel):
    request_id: str
    total_documents: int
    successful: int
    failed: int
    processing_time_ms: float
    results: List[ProcessingResult]


# Operational Schemas
class HealthResponse(BaseModel):
    status: str = "healthy"
    timestamp: datetime


class ReadinessResponse(BaseModel):
    status: str
    model_loaded: bool
    tesseract_available: bool
    details: Dict[str, Any] = {}


class InfoResponse(BaseModel):
    app_name: str
    version: str
    git_commit: str
    build_date: str
    model_version: str
    python_version: str


class MetricsResponse(BaseModel):
    requests_total: int
    requests_success: int
    requests_failed: int
    avg_processing_time_ms: float
    uptime_seconds: float


class ErrorResponse(BaseModel):
    error: str
    detail: str
    request_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
