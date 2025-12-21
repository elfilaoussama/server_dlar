"""
Base model classes for document layout analysis.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BoundingBox:
    """Represents a rectangular bounding box with position and dimensions."""
    x: float
    y: float
    width: float
    height: float

    def getCenterX(self) -> float:
        """Get the X coordinate of the center point."""
        return self.x + self.width / 2

    def getCenterY(self) -> float:
        """Get the Y coordinate of the center point."""
        return self.y + self.height / 2

    def calculateIoU(self, other: 'BoundingBox') -> float:
        """Calculate Intersection over Union with another bounding box."""
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x + self.width, other.x + other.width)
        y2 = min(self.y + self.height, other.y + other.height)

        intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
        
        box1_area = self.width * self.height
        box2_area = other.width * other.height
        
        union_area = box1_area + box2_area - intersection_area
        
        if union_area == 0:
            return 0.0
            
        return intersection_area / union_area

    def isContainedIn(self, other: 'BoundingBox') -> bool:
        """Check if this bounding box is completely contained within another."""
        return (self.x >= other.x and 
                self.y >= other.y and 
                (self.x + self.width) <= (other.x + other.width) and 
                (self.y + self.height) <= (other.y + other.height))


@dataclass
class PageElement:
    """Base class for all page elements."""
    id: str
    bbox: BoundingBox
    confidenceScore: float = 1.0
    parentID: Optional[str] = None
    label: str = "Unknown"
