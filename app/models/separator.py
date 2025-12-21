"""
Separator object model for document layout analysis.
"""
from dataclasses import dataclass
from .base import PageElement


@dataclass
class SeparatorObject(PageElement):
    """Represents a horizontal or vertical separator line."""
    orientation: str = "horizontal"  # horizontal, vertical
    thickness: int = 2
    color: str = "000000"  # Hex
