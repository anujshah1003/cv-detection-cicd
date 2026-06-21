"""
Core object detection logic.

This module wraps the Ultralytics YOLOv8 model behind a small, testable
interface. Keeping this logic separate from the API layer (main.py) means
we can unit test detection behavior without spinning up a web server.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List

from ultralytics import YOLO


@dataclass
class Detection:
    """A single detected object."""
    class_name: str
    confidence: float
    box: List[float]  # [x1, y1, x2, y2]


class ObjectDetector:
    """Thin wrapper around a YOLO model for object detection."""

    def __init__(self, model_path: str = "yolov8n.pt", confidence_threshold: float = 0.25):
        """
        Args:
            model_path: Path or name of YOLO weights. 'yolov8n.pt' is the
                nano (smallest) pretrained model, auto-downloaded by
                ultralytics on first use.
            confidence_threshold: Minimum confidence to keep a detection.
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold

    def predict(self, image_path: str) -> List[Detection]:
        """
        Run object detection on an image.

        Args:
            image_path: Path to an image file.

        Returns:
            List of Detection objects, one per detected object above the
            confidence threshold.
        """
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        results = self.model.predict(
            source=image_path,
            conf=self.confidence_threshold,
            verbose=False,
        )

        detections: List[Detection] = []
        result = results[0]
        for box in result.boxes:
            cls_id = int(box.cls[0])
            class_name = result.names[cls_id]
            confidence = float(box.conf[0])
            coords = box.xyxy[0].tolist()
            detections.append(
                Detection(class_name=class_name, confidence=confidence, box=coords)
            )

        return detections
