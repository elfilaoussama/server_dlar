"""
Document Layout Analysis and Reconstruction - Main Entry Point

This is the main entry point for the document layout analysis pipeline.
It processes document images, detects layout elements using YOLO,
and reconstructs the document in various formats (DOCX, HTML, XML).

Usage:
    python main.py --input image.jpg --output output_dir
    python main.py -i document.png -o results
"""
import os
import sys
import argparse

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import TESSDATA_DIR, OUTPUT_DIR, YOLO_MODEL_PATH
from app.engines import DetectionEngine, LayoutEngine
from app.builders import DocxBuilder, HtmlBuilder, DocBookBuilder


class PipelineController:
    """Main controller for the document processing pipeline."""
    
    def __init__(self, model_path: str = None):
        """
        Initialize the pipeline controller.
        
        Args:
            model_path: Path to YOLO model. Uses default from config if not provided.
        """
        print("PipelineController: Initializing...")
        self.detection_engine = DetectionEngine(model_path=model_path or YOLO_MODEL_PATH)
        self.layout_engine = LayoutEngine()
        self.docx_builder = DocxBuilder()
        self.html_builder = HtmlBuilder()
        self.docbook_builder = DocBookBuilder()
        print("PipelineController: Ready.")
    
    def process_document(self, image_path: str, output_dir: str):
        """
        Process a document image through the full pipeline.
        
        Args:
            image_path: Path to the input document image
            output_dir: Directory for output files
        """
        # Verify input exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Input image not found: {image_path}")
        
        # Create output directories
        os.makedirs(os.path.join(output_dir, "docx"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "html"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "xml"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        
        # Step 1: Detection
        print("\n=== Step 1: Layout Detection ===")
        doc, processed_image_path, image = self.detection_engine.detectLayout(image_path, output_dir)
        
        # Step 2: Layout Reconciliation
        print("\n=== Step 2: Layout Reconciliation ===")
        known_dpi = doc.pages[0].originalDPI if doc.pages else 0
        doc.pages[0].elements = self.layout_engine.reconcileHierarchy(
            doc.pages[0].elements,
            image=image,  # Pass image object directly (no re-reading from disk)
            output_dir=output_dir,
            known_dpi=known_dpi
        )
        
        # Step 3: Build Outputs
        print("\n=== Step 3: Building Outputs ===")
        
        docx_path = os.path.join(output_dir, "docx", "reconstructed_document.docx")
        html_path = os.path.join(output_dir, "html", "reconstructed_document.html")
        xml_path = os.path.join(output_dir, "xml", "reconstructed_document.xml")
        
        self.docx_builder.build(doc, docx_path)
        self.html_builder.build(doc, html_path)
        self.docbook_builder.build(doc, xml_path)
        
        print(f"\n=== Pipeline Complete ===")
        print(f"DOCX: {docx_path}")
        print(f"HTML: {html_path}")
        print(f"XML: {xml_path}")
        
        return doc


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Document Layout Analysis and Reconstruction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py -i document.jpg -o output
    python main.py --input scan.png --output results --model custom_model.onnx
        """
    )
    
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Path to the input document image"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="output",
        help="Output directory (default: output)"
    )
    
    parser.add_argument(
        "-m", "--model",
        type=str,
        default=None,
        help="Path to YOLO model (default: models/yolov10m_doclaynet.onnx)"
    )
    
    parser.add_argument(
        "--tessdata",
        type=str,
        default=None,
        help="Path to Tesseract data directory (default: tessdata/)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Set TESSDATA_PREFIX for Tesseract
    tessdata = args.tessdata or TESSDATA_DIR
    os.environ["TESSDATA_PREFIX"] = tessdata
    print(f"Main: TESSDATA_PREFIX = {tessdata}")
    
    # Run pipeline
    controller = PipelineController(model_path=args.model)
    controller.process_document(args.input, args.output)


if __name__ == "__main__":
    main()
