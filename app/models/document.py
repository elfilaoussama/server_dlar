"""
Document and Page models for document layout analysis.
"""
from dataclasses import dataclass, field
from typing import List, Dict
import numpy as np

from .base import PageElement
from .text import TextObject
from .table import TableObject


@dataclass
class Page:
    """Represents a single page in a document."""
    pageNumber: int
    width: float
    height: float
    originalDPI: int
    elements: List[PageElement] = field(default_factory=list)
    dimensions: Dict[str, float] = field(default_factory=lambda: {'x': 0.0, 'y': 0.0, 'w': 0.0, 'h': 0.0})

    def fill_dimensions(self, x: float, y: float, w: float, h: float):
        """Fill the dimensions attribute."""
        self.dimensions = {'x': x, 'y': y, 'w': w, 'h': h}
        self.width = w
        self.height = h

    def detect_and_propagate_dpi(self):
        """
        Detects the page DPI based on text elements and updates their font sizes.
        """
        # Collect all measured line heights from text objects
        line_heights = []
        text_objects = []
        
        for element in self.elements:
            if isinstance(element, TextObject) and element._pixelLineHeight > 0:
                line_heights.append(element._pixelLineHeight)
                text_objects.append(element)
            # Also check table cells
            elif isinstance(element, TableObject):
                for row in element.rows:
                    for cell in row.cells:
                        for item in cell.content:
                            if isinstance(item, TextObject) and item._pixelLineHeight > 0:
                                line_heights.append(item._pixelLineHeight)
                                text_objects.append(item)
        
        if not line_heights:
            return

        # Use median line height of the document
        median_height = np.median(line_heights)
        
        # Adaptive DPI estimation
        possible_dpis = [72, 96, 150, 200, 300]
        best_dpi = 96  # Default
        
        for test_dpi in possible_dpis:
            test_size = (median_height / test_dpi) * 72 * 1.33
            
            # If it falls in typical body text range (9pt - 14pt)
            if 9 <= test_size <= 14:
                best_dpi = test_dpi
                break
        
        self.originalDPI = best_dpi
        print(f"Page: Detected DPI = {best_dpi} (based on median line height {median_height:.1f}px)")
        
        # Update all text objects with the detected DPI
        for text_obj in text_objects:
            text_obj.update_font_size(best_dpi)


@dataclass
class Document:
    """Represents a complete document with multiple pages."""
    docID: str
    pages: List[Page] = field(default_factory=list)

    def addPage(self, page: Page):
        """Add a page to the document."""
        self.pages.append(page)
