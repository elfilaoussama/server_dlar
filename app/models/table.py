"""
Table-related models for document layout analysis.
"""
from dataclasses import dataclass, field
from typing import List, Optional
from .base import PageElement, BoundingBox


@dataclass
class TableCell:
    """Represents a single cell in a table."""
    colIndex: int
    rowSpan: int
    colSpan: int
    hasVisibleBorder: bool
    backgroundColor: str  # Hex
    verticalAlign: str
    content: List[PageElement] = field(default_factory=list)
    bbox: Optional[BoundingBox] = None


@dataclass
class TableRow:
    """Represents a row in a table."""
    rowIndex: int
    cells: List[TableCell] = field(default_factory=list)


@dataclass
class TableObject(PageElement):
    """Represents a table with rows and cells."""
    rowCount: int = 0
    colCount: int = 0
    rows: List[TableRow] = field(default_factory=list)
