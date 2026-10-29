from __future__ import annotations

from io import BytesIO

from PIL import Image
import numpy as np
import pytest

from optisense.detector import Detection, DetectionSettings
from optisense.image_processor import process_image
from optisense.validators import MediaValidationError


def make_image_bytes() -> bytes:
    output = BytesIO()
    Image.new("RGB", (80, 60), color=(30, 60, 90)).save(output, format="PNG")
    return output.getvalue()


class FakeDetector:
    def __init__(self, detections: list[Detection] | None = None) -> None:
        self.detections = detections or []
        self.frame: np.ndarray | None = None
        self.settings: DetectionSettings | None = None

    def detect(
        self,
        frame: np.ndarray,
        settings: DetectionSettings | None = None,
    ) -> list[Detection]:
        self.frame = frame.copy()
        self.settings = settings
        return self.detections


def test_process_image_decodes_runs_inference_and_annotates() -> None:
    detection = Detection(
        class_id=0,
        class_name="person",
        confidence=0.91,
        x1=10,
        y1=10,
        x2=50,
        y2=50,
    )
    detector = FakeDetector([detection])
    settings = DetectionSettings(confidence=0.4, image_size=320)

    result = process_image(
        "sample.png",
        make_image_bytes(),
        detector,
        settings,
    )

    assert result.original_rgb.shape == (60, 80, 3)
    assert result.original_rgb[0, 0].tolist() == [30, 60, 90]
    assert detector.frame is not None
    assert detector.frame[0, 0].tolist() == [90, 60, 30]
    assert detector.settings == settings
    assert result.detections == (detection,)
    assert result.inference_ms >= 0
    assert not np.array_equal(result.original_rgb, result.annotated_rgb)
    assert result.annotated_png().startswith(b"\x89PNG\r\n\x1a\n")


def test_process_image_handles_empty_detections() -> None:
    result = process_image("sample.png", make_image_bytes(), FakeDetector())

    assert result.detections == ()
    assert np.array_equal(result.original_rgb, result.annotated_rgb)


def test_process_image_can_disable_annotation() -> None:
    detection = Detection(0, "person", 0.91, 10, 10, 50, 50)
    result = process_image(
        "sample.png",
        make_image_bytes(),
        FakeDetector([detection]),
        draw_boxes=False,
    )

    assert result.detections == (detection,)
    assert np.array_equal(result.original_rgb, result.annotated_rgb)


def test_process_image_rejects_invalid_upload_before_inference() -> None:
    detector = FakeDetector()

    with pytest.raises(MediaValidationError):
        process_image("bad.png", b"not image data", detector)

    assert detector.frame is None
