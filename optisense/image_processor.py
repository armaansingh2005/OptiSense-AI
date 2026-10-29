"""Image decoding, object detection, and annotation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import time
from typing import Protocol
import warnings

import cv2
import numpy as np
from PIL import Image

from optisense.detector import Detection, DetectionSettings
from optisense.validators import MediaValidationError, validate_image_upload


class ImageDetector(Protocol):
    def detect(
        self,
        frame: np.ndarray,
        settings: DetectionSettings | None = None,
    ) -> list[Detection]: ...


@dataclass(frozen=True, slots=True)
class ImageProcessingResult:
    """Actual image inference output and elapsed inference time."""

    original_rgb: np.ndarray
    annotated_rgb: np.ndarray
    detections: tuple[Detection, ...]
    inference_ms: float

    def annotated_png(self) -> bytes:
        """Encode the annotated image as PNG bytes for preview/download."""
        output = BytesIO()
        Image.fromarray(self.annotated_rgb, mode="RGB").save(output, format="PNG")
        return output.getvalue()


def process_image(
    filename: str,
    data: bytes,
    detector: ImageDetector,
    settings: DetectionSettings | None = None,
    *,
    draw_boxes: bool = True,
    show_confidence: bool = True,
) -> ImageProcessingResult:
    """Validate, decode, infer on, and annotate one uploaded image."""
    validate_image_upload(filename, data)
    original_rgb = _decode_rgb(data)
    frame_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)

    started = time.perf_counter()
    detections = detector.detect(frame_bgr, settings)
    inference_ms = (time.perf_counter() - started) * 1000

    annotated_bgr = frame_bgr.copy()
    if draw_boxes:
        _draw_detections(annotated_bgr, detections, show_confidence)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    return ImageProcessingResult(
        original_rgb=original_rgb,
        annotated_rgb=annotated_rgb,
        detections=tuple(detections),
        inference_ms=inference_ms,
    )


def _decode_rgb(data: bytes) -> np.ndarray:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as image:
                return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise MediaValidationError("Image dimensions are too large to process safely.") from exc
    except (OSError, ValueError) as exc:
        raise MediaValidationError("The uploaded image could not be decoded.") from exc


def _draw_detections(
    image_bgr: np.ndarray,
    detections: list[Detection],
    show_confidence: bool,
) -> None:
    height, width = image_bgr.shape[:2]
    thickness = max(1, round(min(width, height) / 500))
    font_scale = max(0.45, min(width, height) / 1000)
    font_thickness = max(1, thickness)
    for detection in detections:
        x1 = max(0, min(width - 1, round(detection.x1)))
        y1 = max(0, min(height - 1, round(detection.y1)))
        x2 = max(0, min(width - 1, round(detection.x2)))
        y2 = max(0, min(height - 1, round(detection.y2)))
        color = (255, 145, 45)
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, thickness)

        label = detection.class_name
        if show_confidence:
            label = f"{label} {detection.confidence:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )
        label_top = max(0, y1 - text_height - baseline - 6)
        cv2.rectangle(
            image_bgr,
            (x1, label_top),
            (min(width - 1, x1 + text_width + 8), min(height - 1, label_top + text_height + baseline + 6)),
            color,
            thickness=cv2.FILLED,
        )
        cv2.putText(
            image_bgr,
            label,
            (x1 + 4, min(height - 1, label_top + text_height + 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA,
        )
