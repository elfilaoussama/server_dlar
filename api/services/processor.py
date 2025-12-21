"""
Document processing service with parallel processing support.
"""
import os
import time
import asyncio
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor

from api.core import config, get_logger
from api.schemas import ProcessingOptions, ProcessingResult, ProcessingStatus, DocumentElement, BoundingBox, OutputFormat

logger = get_logger("dla.processor")


class DocumentProcessor:
    """Main service for document processing with AI models."""
    
    def __init__(self):
        self._model_loaded = False
        self._detection_engine = None
        self._layout_engine = None
        self._executor = ThreadPoolExecutor(max_workers=config.MAX_WORKERS)
        
    async def initialize(self) -> bool:
        """Load AI models during startup."""
        logger.info("Initializing DocumentProcessor...")
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self._executor, self._load_models)
            self._model_loaded = True
            logger.info("Models loaded successfully")
            return True
        except Exception as e:
            logger.exception(f"Failed to initialize: {e}")
            return False
    
    def _load_models(self):
        """Load AI models (runs in thread pool)."""
        from app.engines.detection import DetectionEngine, LayoutEngine
        logger.info(f"Loading detection model from: {config.MODEL_PATH}")
        self._detection_engine = DetectionEngine(
            model_path=config.MODEL_PATH,
            use_table_transformer=config.USE_TABLE_TRANSFORMER
        )
        self._layout_engine = LayoutEngine()
        logger.info("Detection and layout engines loaded")
    
    @property
    def is_ready(self) -> bool:
        return self._model_loaded and self._detection_engine is not None
    
    async def process_image(self, image_path: str, output_dir: str, options: ProcessingOptions, request_id: str) -> ProcessingResult:
        """Process a single document image."""
        start_time = time.time()
        errors = []
        elements = []
        output_files = {}
        
        logger.info(f"Processing image: {image_path}", extra={"extra_data": {"request_id": request_id}})
        
        try:
            loop = asyncio.get_event_loop()
            
            # Detection
            doc, processed_path, image = await loop.run_in_executor(
                self._executor, self._detection_engine.detectLayout, image_path, output_dir
            )
            
            logger.debug(f"Detection complete: {len(doc.pages[0].elements)} elements", extra={"extra_data": {"request_id": request_id}})
            
            # Layout reconciliation
            known_dpi = doc.pages[0].originalDPI if doc.pages else 0
            doc.pages[0].elements = await loop.run_in_executor(
                self._executor,
                lambda: self._layout_engine.reconcileHierarchy(doc.pages[0].elements, image=image, output_dir=output_dir, known_dpi=known_dpi)
            )
            
            logger.debug(f"Layout reconciliation complete: {len(doc.pages[0].elements)} final elements", extra={"extra_data": {"request_id": request_id}})
            
            # Convert elements
            for elem in doc.pages[0].elements:
                elements.append(self._convert_element(elem))
            
            # Generate outputs
            output_files = await self._generate_outputs(doc, output_dir, options, request_id)
            
            processing_time = (time.time() - start_time) * 1000
            logger.info(f"Image processed successfully in {processing_time:.1f}ms", extra={"extra_data": {"request_id": request_id, "elements": len(elements)}})
            
            return ProcessingResult(
                request_id=request_id, status=ProcessingStatus.COMPLETED,
                processing_time_ms=processing_time, page_count=len(doc.pages),
                elements_detected=len(elements), elements=elements,
                output_files=output_files, errors=errors
            )
        except Exception as e:
            logger.exception(f"Processing failed: {e}", extra={"extra_data": {"request_id": request_id}})
            return ProcessingResult(
                request_id=request_id, status=ProcessingStatus.FAILED,
                processing_time_ms=(time.time() - start_time) * 1000,
                page_count=0, elements_detected=0, errors=[str(e)]
            )
    
    async def process_images_batch(self, image_paths: List[str], output_base_dir: str, options: ProcessingOptions, request_id: str) -> List[ProcessingResult]:
        """Process multiple images in parallel."""
        logger.info(f"Batch processing {len(image_paths)} images", extra={"extra_data": {"request_id": request_id}})
        
        tasks = []
        for i, image_path in enumerate(image_paths):
            output_dir = os.path.join(output_base_dir, f"doc_{i}")
            os.makedirs(output_dir, exist_ok=True)
            task = self.process_image(image_path, output_dir, options, f"{request_id}_{i}")
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        processed = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch item {i} failed: {result}", extra={"extra_data": {"request_id": f"{request_id}_{i}"}})
                processed.append(ProcessingResult(
                    request_id=f"{request_id}_{i}", status=ProcessingStatus.FAILED,
                    processing_time_ms=0, page_count=0, elements_detected=0, errors=[str(result)]
                ))
            else:
                processed.append(result)
        
        successful = sum(1 for r in processed if r.status == ProcessingStatus.COMPLETED)
        logger.info(f"Batch complete: {successful}/{len(processed)} successful", extra={"extra_data": {"request_id": request_id}})
        
        return processed
    
    async def process_pdf(self, pdf_path: str, output_dir: str, options: ProcessingOptions, request_id: str) -> List[ProcessingResult]:
        """Process PDF by converting to images first."""
        logger.info(f"Processing PDF: {pdf_path}", extra={"extra_data": {"request_id": request_id}})
        try:
            loop = asyncio.get_event_loop()
            image_paths = await loop.run_in_executor(self._executor, self._pdf_to_images, pdf_path, output_dir)
            logger.info(f"PDF converted to {len(image_paths)} images", extra={"extra_data": {"request_id": request_id}})
            return await self.process_images_batch(image_paths, output_dir, options, request_id)
        except Exception as e:
            logger.exception(f"PDF processing failed: {e}", extra={"extra_data": {"request_id": request_id}})
            return [ProcessingResult(
                request_id=request_id, status=ProcessingStatus.FAILED,
                processing_time_ms=0, page_count=0, elements_detected=0, errors=[str(e)]
            )]
    
    def _pdf_to_images(self, pdf_path: str, output_dir: str) -> List[str]:
        """Convert PDF pages to images."""
        from pdf2image import convert_from_path
        images_dir = os.path.join(output_dir, "pdf_pages")
        os.makedirs(images_dir, exist_ok=True)
        images = convert_from_path(pdf_path, dpi=150)
        image_paths = []
        for i, image in enumerate(images):
            path = os.path.join(images_dir, f"page_{i+1}.png")
            image.save(path, "PNG")
            image_paths.append(path)
            logger.debug(f"PDF page {i+1} saved to {path}")
        return image_paths
    
    def _convert_element(self, elem) -> DocumentElement:
        """Convert internal element to API schema."""
        return DocumentElement(
            id=getattr(elem, 'id', 'unknown'),
            type=type(elem).__name__,
            bbox=BoundingBox(x=elem.bbox.x, y=elem.bbox.y, width=elem.bbox.width, height=elem.bbox.height),
            content=getattr(elem, 'rawText', None),
            font_size=getattr(elem, 'fontSize', None),
            alignment=getattr(elem, 'alignment', None),
            color=getattr(elem, 'fontColor', None)
        )
    
    async def _generate_outputs(self, doc, output_dir: str, options: ProcessingOptions, request_id: str) -> dict:
        """Generate output files."""
        output_files = {}
        loop = asyncio.get_event_loop()
        
        formats = options.output_formats
        if OutputFormat.ALL in formats:
            formats = [OutputFormat.DOCX, OutputFormat.HTML, OutputFormat.XML]
        
        for fmt in formats:
            try:
                logger.debug(f"Generating {fmt} output", extra={"extra_data": {"request_id": request_id}})
                if fmt == OutputFormat.DOCX:
                    from app.builders.docx_builder import DocxBuilder
                    path = os.path.join(output_dir, "output.docx")
                    await loop.run_in_executor(self._executor, DocxBuilder().build, doc, path)
                    output_files["docx"] = path
                elif fmt == OutputFormat.HTML:
                    from app.builders.html_builder import HtmlBuilder
                    path = os.path.join(output_dir, "output.html")
                    await loop.run_in_executor(self._executor, HtmlBuilder().build, doc, path)
                    output_files["html"] = path
                elif fmt == OutputFormat.XML:
                    from app.builders.docbook_builder import DocBookBuilder
                    path = os.path.join(output_dir, "output.xml")
                    await loop.run_in_executor(self._executor, DocBookBuilder().build, doc, path)
                    output_files["xml"] = path
                logger.debug(f"Generated {fmt} at {path}", extra={"extra_data": {"request_id": request_id}})
            except Exception as e:
                logger.error(f"Failed to generate {fmt}: {e}", extra={"extra_data": {"request_id": request_id}})
        return output_files
    
    async def shutdown(self):
        logger.info("Shutting down DocumentProcessor")
        self._executor.shutdown(wait=True)


# Global instance
_processor: Optional[DocumentProcessor] = None


async def get_processor() -> DocumentProcessor:
    global _processor
    if _processor is None:
        _processor = DocumentProcessor()
        await _processor.initialize()
    return _processor


async def initialize_processor():
    global _processor
    _processor = DocumentProcessor()
    await _processor.initialize()


async def shutdown_processor():
    global _processor
    if _processor:
        await _processor.shutdown()
        _processor = None
