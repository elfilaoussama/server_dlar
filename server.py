"""
Document Layout Analysis API - FastAPI Application

Production-ready API for document layout analysis and reconstruction.
"""
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Callable

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from api.core import config, init_logger, get_logger, generate_request_id, set_request_id
from api.routes import operations_router, documents_router
from api.services import initialize_processor, shutdown_processor


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: initialize on startup, cleanup on shutdown."""
    # Initialize logger first
    logger = init_logger()
    
    logger.info("=" * 60)
    logger.info(f"Starting {config.APP_NAME} v{config.APP_VERSION}")
    logger.info(f"Environment: {config.ENVIRONMENT}")
    logger.info(f"Debug: {config.DEBUG}")
    logger.info(f"Log level: {config.LOG_LEVEL}")
    logger.info(f"Log file: {config.LOG_FILE_PATH}")
    logger.info("=" * 60)
    
    # Load AI models
    logger.info("Loading AI models...")
    await initialize_processor()
    logger.info("✅ Models loaded - API ready")
    
    yield
    
    logger.info("Shutting down...")
    await shutdown_processor()
    logger.info("Shutdown complete")


app = FastAPI(
    title=config.APP_NAME,
    description="""
## Document Layout Analysis & Reconstruction API

AI-powered service that converts document images to editable formats.

### Endpoints
| Endpoint | Description |
|----------|-------------|
| `POST /v1/process/image` | Process single image |
| `POST /v1/process/images` | Process multiple images (parallel) |
| `POST /v1/process/pdf` | Process PDF document |
| `GET /health` | Liveness check |
| `GET /ready` | Readiness check |
| `GET /metrics` | Prometheus metrics |
| `GET /info` | API metadata |
    """,
    version=config.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_middleware(request: Request, call_next: Callable) -> Response:
    """Add correlation ID, timing, and logging to all requests."""
    request_id = request.headers.get("X-Request-ID", generate_request_id())
    set_request_id(request_id)
    
    logger = get_logger("dla.http")
    logger.debug(f"→ {request.method} {request.url.path}", extra={"extra_data": {"client": request.client.host if request.client else "unknown"}})
    
    start = datetime.utcnow()
    response = await call_next(request)
    duration = (datetime.utcnow() - start).total_seconds() * 1000
    
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Processing-Time-Ms"] = f"{duration:.1f}"
    
    logger.debug(f"← {response.status_code} ({duration:.1f}ms)", extra={"extra_data": {"status": response.status_code}})
    
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger = get_logger("dla")
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "detail": str(exc) if config.DEBUG else "Internal error",
            "request_id": request.headers.get("X-Request-ID"),
            "timestamp": datetime.utcnow().isoformat()
        }
    )


# Register routers
app.include_router(operations_router)
app.include_router(documents_router)


@app.get("/", tags=["Root"])
async def root():
    """API root with navigation links."""
    return {
        "name": config.APP_NAME,
        "version": config.APP_VERSION,
        "environment": config.ENVIRONMENT,
        "docs": "/docs",
        "health": "/health",
        "ready": "/ready",
        "info": "/info"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host=config.HOST, port=config.PORT, reload=config.DEBUG)
