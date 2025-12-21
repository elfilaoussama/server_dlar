"""
Operational endpoints: /health, /ready, /metrics, /info
Essential for Kubernetes and monitoring.
"""
import sys
import time
import shutil
from datetime import datetime
from fastapi import APIRouter, Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

from api.schemas import HealthResponse, ReadinessResponse, InfoResponse, MetricsResponse
from api.core import config, get_logger
from api.services import get_processor

router = APIRouter(tags=["Operations"])
logger = get_logger("dla.operations")

# Prometheus metrics
REQUEST_COUNT = Counter("dla_requests_total", "Total requests", ["method", "endpoint", "status"])
REQUEST_LATENCY = Histogram("dla_request_latency_seconds", "Request latency", ["endpoint"])
DOCUMENTS_PROCESSED = Counter("dla_documents_processed_total", "Documents processed", ["status"])
MODEL_LOADED = Gauge("dla_model_loaded", "Model loaded status")

START_TIME = time.time()
_metrics = {"requests_total": 0, "requests_success": 0, "requests_failed": 0, "total_processing_time_ms": 0.0}


def record_request(success: bool, processing_time_ms: float = 0):
    _metrics["requests_total"] += 1
    if success:
        _metrics["requests_success"] += 1
        DOCUMENTS_PROCESSED.labels(status="success").inc()
    else:
        _metrics["requests_failed"] += 1
        DOCUMENTS_PROCESSED.labels(status="failed").inc()
    _metrics["total_processing_time_ms"] += processing_time_ms


@router.get("/health", response_model=HealthResponse, summary="Liveness Check")
async def health_check() -> HealthResponse:
    """Liveness probe - always returns 200 if process is running."""
    logger.debug("Health check called")
    return HealthResponse(status="healthy", timestamp=datetime.utcnow())


@router.get("/ready", response_model=ReadinessResponse, summary="Readiness Check")
async def readiness_check() -> ReadinessResponse:
    """Readiness probe - returns 200 only when models are loaded."""
    processor = await get_processor()
    model_loaded = processor.is_ready
    tesseract_available = shutil.which("tesseract") is not None
    
    MODEL_LOADED.set(1 if model_loaded else 0)
    status = "ready" if (model_loaded and tesseract_available) else "not_ready"
    
    logger.info(f"Readiness check: {status}", extra={"extra_data": {"model_loaded": model_loaded}})
    
    return ReadinessResponse(
        status=status,
        model_loaded=model_loaded,
        tesseract_available=tesseract_available,
        details={"model_path": config.MODEL_PATH, "uptime_seconds": time.time() - START_TIME}
    )


@router.get("/metrics", summary="Prometheus Metrics")
async def prometheus_metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.get("/metrics/json", response_model=MetricsResponse, summary="JSON Metrics")
async def json_metrics() -> MetricsResponse:
    avg = _metrics["total_processing_time_ms"] / max(_metrics["requests_total"], 1)
    return MetricsResponse(
        requests_total=_metrics["requests_total"],
        requests_success=_metrics["requests_success"],
        requests_failed=_metrics["requests_failed"],
        avg_processing_time_ms=avg,
        uptime_seconds=time.time() - START_TIME
    )


@router.get("/info", response_model=InfoResponse, summary="API Metadata")
async def api_info() -> InfoResponse:
    return InfoResponse(
        app_name=config.APP_NAME,
        version=config.APP_VERSION,
        git_commit=config.GIT_COMMIT,
        build_date=config.BUILD_DATE,
        model_version="YOLOv10m-DocLayNet",
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
