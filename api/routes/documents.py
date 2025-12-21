"""
Document processing routes for images and PDFs.
"""
import os
import time
import tempfile
from typing import List
from fastapi import APIRouter, File, UploadFile, Form, HTTPException

from api.schemas import ProcessingOptions, ProcessingResult, BatchProcessingResult, ProcessingStatus, OutputFormat
from api.core import config, get_logger, generate_request_id, set_request_id
from api.services import get_processor
from api.routes.operations import record_request, REQUEST_COUNT, REQUEST_LATENCY

router = APIRouter(prefix="/v1", tags=["Document Processing"])
logger = get_logger("dla.documents")

ALLOWED_IMAGES = {".jpg", ".jpeg", ".png", ".tiff", ".bmp"}
ALLOWED_PDF = {".pdf"}


def validate_ext(filename: str, allowed: set) -> bool:
    return os.path.splitext(filename)[1].lower() in allowed


async def save_file(upload: UploadFile, dest: str) -> str:
    with open(dest, "wb") as f:
        f.write(await upload.read())
    return dest


@router.post("/process/image", response_model=ProcessingResult, summary="Process Single Image")
async def process_single_image(
    file: UploadFile = File(...),
    output_formats: str = Form("docx"),
    ocr_languages: str = Form("eng+fra+ara"),
    detect_tables: bool = Form(True),
    remove_backgrounds: bool = Form(True)
):
    """Upload and process a single document image."""
    request_id = generate_request_id()
    set_request_id(request_id)
    start = time.time()
    REQUEST_COUNT.labels(method="POST", endpoint="/v1/process/image", status="started").inc()
    
    logger.info(f"Processing image: {file.filename}", extra={"extra_data": {"request_id": request_id, "size": file.size}})
    
    if not validate_ext(file.filename, ALLOWED_IMAGES):
        logger.warning(f"Invalid file type: {file.filename}")
        raise HTTPException(400, f"Invalid file type. Allowed: {ALLOWED_IMAGES}")
    
    formats = [OutputFormat(f.strip()) for f in output_formats.split(",") if f.strip()]
    options = ProcessingOptions(output_formats=formats, ocr_languages=ocr_languages, detect_tables=detect_tables, remove_backgrounds=remove_backgrounds)
    
    temp_dir = tempfile.mkdtemp(prefix="dla_")
    try:
        input_path = os.path.join(temp_dir, file.filename)
        await save_file(file, input_path)
        
        processor = await get_processor()
        result = await processor.process_image(input_path, temp_dir, options, request_id)
        
        processing_time = (time.time() - start) * 1000
        record_request(result.status == ProcessingStatus.COMPLETED, processing_time)
        
        logger.info(f"Image processed: {result.status}", extra={"extra_data": {"request_id": request_id, "time_ms": processing_time}})
        
        REQUEST_COUNT.labels(method="POST", endpoint="/v1/process/image", status="success" if result.status == ProcessingStatus.COMPLETED else "failed").inc()
        REQUEST_LATENCY.labels(endpoint="/v1/process/image").observe(time.time() - start)
        
        return result
    except Exception as e:
        logger.exception(f"Processing failed: {e}")
        record_request(False)
        raise HTTPException(500, str(e))


@router.post("/process/images", response_model=BatchProcessingResult, summary="Process Multiple Images (Parallel)")
async def process_multiple_images(
    files: List[UploadFile] = File(...),
    output_formats: str = Form("docx"),
    ocr_languages: str = Form("eng+fra+ara")
):
    """Upload and process multiple images in parallel."""
    request_id = generate_request_id()
    set_request_id(request_id)
    start = time.time()
    
    logger.info(f"Batch processing {len(files)} images", extra={"extra_data": {"request_id": request_id}})
    
    for f in files:
        if not validate_ext(f.filename, ALLOWED_IMAGES):
            logger.warning(f"Invalid file in batch: {f.filename}")
            raise HTTPException(400, f"Invalid file: {f.filename}")
    
    formats = [OutputFormat(f.strip()) for f in output_formats.split(",") if f.strip()]
    options = ProcessingOptions(output_formats=formats, ocr_languages=ocr_languages)
    
    temp_dir = tempfile.mkdtemp(prefix="dla_batch_")
    try:
        paths = []
        for i, f in enumerate(files):
            path = os.path.join(temp_dir, f"doc_{i}_{f.filename}")
            await save_file(f, path)
            paths.append(path)
        
        processor = await get_processor()
        results = await processor.process_images_batch(paths, temp_dir, options, request_id)
        
        successful = sum(1 for r in results if r.status == ProcessingStatus.COMPLETED)
        processing_time = (time.time() - start) * 1000
        record_request(successful == len(results), processing_time)
        
        logger.info(f"Batch complete: {successful}/{len(results)} successful", extra={"extra_data": {"request_id": request_id, "time_ms": processing_time}})
        
        return BatchProcessingResult(
            request_id=request_id, total_documents=len(files),
            successful=successful, failed=len(results) - successful,
            processing_time_ms=processing_time, results=results
        )
    except Exception as e:
        logger.exception(f"Batch failed: {e}")
        raise HTTPException(500, str(e))


@router.post("/process/pdf", response_model=BatchProcessingResult, summary="Process PDF Document")
async def process_pdf(
    file: UploadFile = File(...),
    output_formats: str = Form("docx"),
    ocr_languages: str = Form("eng+fra+ara")
):
    """Upload and process a multi-page PDF document."""
    request_id = generate_request_id()
    set_request_id(request_id)
    start = time.time()
    
    logger.info(f"Processing PDF: {file.filename}", extra={"extra_data": {"request_id": request_id, "size": file.size}})
    
    if not validate_ext(file.filename, ALLOWED_PDF):
        logger.warning(f"Invalid PDF file: {file.filename}")
        raise HTTPException(400, "Invalid file type. Expected PDF.")
    
    formats = [OutputFormat(f.strip()) for f in output_formats.split(",") if f.strip()]
    options = ProcessingOptions(output_formats=formats, ocr_languages=ocr_languages)
    
    temp_dir = tempfile.mkdtemp(prefix="dla_pdf_")
    try:
        pdf_path = os.path.join(temp_dir, file.filename)
        await save_file(file, pdf_path)
        
        processor = await get_processor()
        results = await processor.process_pdf(pdf_path, temp_dir, options, request_id)
        
        successful = sum(1 for r in results if r.status == ProcessingStatus.COMPLETED)
        processing_time = (time.time() - start) * 1000
        record_request(successful == len(results), processing_time)
        
        logger.info(f"PDF complete: {len(results)} pages, {successful} successful", extra={"extra_data": {"request_id": request_id, "time_ms": processing_time}})
        
        return BatchProcessingResult(
            request_id=request_id, total_documents=len(results),
            successful=successful, failed=len(results) - successful,
            processing_time_ms=processing_time, results=results
        )
    except Exception as e:
        logger.exception(f"PDF failed: {e}")
        raise HTTPException(500, str(e))
