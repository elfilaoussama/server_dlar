# Document Layout Analysis & Reconstruction

AI-powered document layout analysis and reconstruction system that converts document images to editable formats (DOCX, HTML, XML).

## Quick Start

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\Activate.ps1  # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy environment config
cp .env.example .env

# 4. Run API
uvicorn server:app --host 0.0.0.0 --port 8000
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/docs` | GET | Swagger UI documentation |
| `/health` | GET | Liveness check |
| `/ready` | GET | Readiness check (model loaded) |
| `/metrics` | GET | Prometheus metrics |
| `/info` | GET | API version and metadata |
| `/v1/process/image` | POST | Process single image |
| `/v1/process/images` | POST | Process multiple images (parallel) |
| `/v1/process/pdf` | POST | Process PDF document |

## Project Structure

```
├── server.py           # FastAPI application entry point
├── main.py             # CLI entry point
├── api/                # API layer
│   ├── core/           # Config, logging
│   ├── routes/         # HTTP endpoints
│   ├── schemas/        # Pydantic models
│   └── services/       # Business logic
├── app/                # Core application
│   ├── engines/        # Detection, Layout engines
│   ├── models/         # Document models
│   └── builders/       # DOCX, HTML, XML builders
├── models/             # AI model files (.onnx)
├── tessdata/           # Tesseract language data
├── logs/               # Application logs
└── output/             # Generated documents
```

## Configuration

Copy `.env.example` to `.env` and configure:

```bash
# Key settings
DEBUG=false
LOG_LEVEL=INFO
MODEL_PATH=models/yolov10m_doclaynet.onnx
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

## Requirements

- Python 3.10+
- Tesseract OCR
- CUDA (optional, for GPU acceleration)

## License

MIT
