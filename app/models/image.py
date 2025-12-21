"""
Image object model for document layout analysis.
"""
from dataclasses import dataclass, field
from .base import PageElement


@dataclass
class ImageObject(PageElement):
    """Represents an image element in the document."""
    imagePath: str = ""
    format: str = "png"
    data: bytes = field(default_factory=bytes)
    hasTextContent: bool = False
