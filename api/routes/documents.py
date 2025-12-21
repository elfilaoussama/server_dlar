"""
Document processing routes for images and PDFs.
"""
import os
import time
import tempfile
import shutil
from typing import List
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import FileResponse

from api.schemas import ProcessingOptions, ProcessingResult, BatchProcessingResult, ProcessingStatus, OutputFormat
from api.core import config, get_logger, generate_request_id, set_request_id
from api.services import get_processor
from api.routes.operations import record_request, REQUEST_COUNT, REQUEST_LATENCY
from api.utils.docx_utils import merge_docx_files

router = APIRouter(prefix="/v1", tags=["Document Processing"])
logger = get_logger("dla.documents")

ALLOWED_IMAGES = {".jpg", ".jpeg", ".png", ".tiff", ".bmp"}
ALLOWED_PDF = {".pdf"}

# Store output files temporarily for download (in production, use blob storage)
_output_store = {}


def validate_ext(filename: str, allowed: set) -> bool:
    return os.path.splitext(filename)[1].lower() in allowed


async def save_file(upload: UploadFile, dest: str) -> str:
    with open(dest, "wb") as f:
        f.write(await upload.read())
    return dest


def generate_download_urls(request: Request, request_id: str, output_files: dict) -> dict:
    """Generate download URLs for output files."""
    base_url = str(request.base_url).rstrip("/")
    download_urls = {}
    
    for format_type, file_path in output_files.items():
        if os.path.exists(file_path):
            # Store file path for later download
            file_key = f"{request_id}_{format_type}"
            _output_store[file_key] = file_path
            download_urls[format_type] = f"{base_url}/v1/download/{file_key}"
    
    return download_urls


@router.get("/download/{file_key}", summary="Download Generated File")
async def download_file(file_key: str):
    """Download a generated output file."""
    if file_key not in _output_store:
        raise HTTPException(404, "File not found or expired")
    
    file_path = _output_store[file_key]
    if not os.path.exists(file_path):
        del _output_store[file_key]
        raise HTTPException(404, "File not found")
    
    # Determine filename based on format
    format_type = file_key.split("_")[-1]
    filename = f"document.{format_type}"
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/octet-stream"
    )


@router.post("/process/image", response_model=ProcessingResult, summary="Process Single Image")
async def process_single_image(
    request: Request,
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
        
        # Generate download URLs
        result.download_urls = generate_download_urls(request, request_id, result.output_files)
        
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
    request: Request,
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
        
        # Generate download URLs for each result
        for i, result in enumerate(results):
            result.download_urls = generate_download_urls(request, f"{request_id}_{i}", result.output_files)
        
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
    request: Request,
    file: UploadFile = File(...),
    output_formats: str = Form("docx"),
    ocr_languages: str = Form("eng+fra+ara")
):
    """Upload and process a multi-page PDF document. Returns merged DOCX."""
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
        
        # Merge all DOCX files into one document
        merged_download_urls = {}
        merged_doc_path = None
        
        if OutputFormat.DOCX in formats or OutputFormat.ALL in formats:
            # Collect all DOCX files in order
            docx_paths = []
            for i, result in enumerate(results):
                if result.status == ProcessingStatus.COMPLETED and "docx" in result.output_files:
                    docx_paths.append(result.output_files["docx"])
            
            if docx_paths:
                # Merge all pages into one DOCX
                merged_doc_path = os.path.join(temp_dir, "merged_document.docx")
                try:
                    merge_docx_files(docx_paths, merged_doc_path)
                    logger.info(f"Merged {len(docx_paths)} pages into {merged_doc_path}")
                    
                    # Generate download URL for merged document
                    merged_key = f"{request_id}_merged_docx"
                    _output_store[merged_key] = merged_doc_path
                    base_url = str(request.base_url).rstrip("/")
                    merged_download_urls["docx"] = f"{base_url}/v1/download/{merged_key}"
                except Exception as e:
                    logger.error(f"Failed to merge DOCX files: {e}")
        
        logger.info(f"PDF complete: {len(results)} pages, {successful} successful", extra={"extra_data": {"request_id": request_id, "time_ms": processing_time}})
        
        return BatchProcessingResult(
            request_id=request_id, total_documents=len(results),
            successful=successful, failed=len(results) - successful,
            processing_time_ms=processing_time, results=results,
            merged_document=merged_doc_path,
            download_urls=merged_download_urls
        )
    except Exception as e:
        logger.exception(f"PDF failed: {e}")
        raise HTTPException(500, str(e))
