from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from optisense.detector import (
    DetectionResultError,
    DetectionSettings,
    Detector,
    InferenceError,
    ModelLoadError,
    normalize_results,
)


def make_result(
    coordinates: list[list[float]] | None = None,
    classes: list[float] | None = None,
    confidences: list[float] | None = None,
    names: dict[int, str] | None = None,
    track_ids: list[float] | None = None,
) -> SimpleNamespace:
    coordinates = coordinates or []
    classes = classes or []
    confidences = confidences or []
    boxes = SimpleNamespace(
        xyxy=np.asarray(coordinates, dtype=float).reshape((-1, 4)),
        cls=np.asarray(classes, dtype=float),
        conf=np.asarray(confidences, dtype=float),
        id=None if track_ids is None else np.asarray(track_ids, dtype=float),
    )
    return SimpleNamespace(boxes=boxes, names=names or {0: "person"})


class FakeModel:
    def __init__(self, results: object | None = None, error: Exception | None = None) -> None:
        self.results = results if results is not None else [make_result()]
        self.error = error
        self.calls: list[dict[str, object]] = []

    def predict(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.results


def test_detection_settings_validate_thresholds_resolution_and_classes() -> None:
    with pytest.raises(ValueError, match="confidence"):
        DetectionSettings(confidence=1.1)
    with pytest.raises(ValueError, match="iou"):
        DetectionSettings(iou=-0.1)
    with pytest.raises(ValueError, match="image_size"):
        DetectionSettings(image_size=16)
    with pytest.raises(ValueError, match="class_ids"):
        DetectionSettings(class_ids=(-1,))

    settings = DetectionSettings(class_ids=(0, 1, 0))
    assert settings.class_ids == (0, 1)


def test_normalize_results_returns_structured_detection() -> None:
    result = make_result([[1, 2, 30, 40]], [0], [0.87])

    detections = normalize_results([result], image_width=100, image_height=80)

    assert len(detections) == 1
    assert detections[0].to_dict() == {
        "class_id": 0,
        "class_name": "person",
        "confidence": 0.87,
        "x1": 1.0,
        "y1": 2.0,
        "x2": 30.0,
        "y2": 40.0,
        "frame_number": None,
        "track_id": None,
    }


def test_normalize_results_preserves_optional_track_id() -> None:
    detections = normalize_results(
        [make_result([[1, 2, 30, 40]], [0], [0.87], track_ids=[8])],
        image_width=100,
        image_height=80,
    )

    assert detections[0].track_id == 8


def test_normalize_results_handles_empty_predictions() -> None:
    assert normalize_results([make_result()], 100, 80) == []
    assert normalize_results([], 100, 80) == []


@pytest.mark.parametrize(
    ("coordinates", "confidence", "message"),
    [
        ([[-1, 2, 30, 40]], 0.8, "outside image bounds"),
        ([[1, 2, 30, 40]], 1.2, "confidence outside"),
        ([[30, 2, 1, 40]], 0.8, "outside image bounds"),
    ],
)
def test_normalize_results_rejects_invalid_detection_data(
    coordinates: list[list[float]], confidence: float, message: str
) -> None:
    result = make_result(coordinates, [0], [confidence])
    with pytest.raises(DetectionResultError, match=message):
        normalize_results([result], 100, 80)


def test_detector_loads_model_once_and_passes_settings() -> None:
    model = FakeModel([make_result([[1, 2, 30, 40]], [0], [0.9])])
    factory_calls: list[str] = []

    def factory(reference: str) -> FakeModel:
        factory_calls.append(reference)
        return model

    detector = Detector(device="cpu", model_factory=factory)
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    settings = DetectionSettings(confidence=0.4, iou=0.6, image_size=320, class_ids=(0, 2))

    detector.detect(image, settings)
    detector.detect(image, settings)

    assert factory_calls == ["yolo11n.pt"]
    assert detector.is_loaded
    assert detector.device == "cpu"
    assert model.calls[0]["conf"] == 0.4
    assert model.calls[0]["iou"] == 0.6
    assert model.calls[0]["imgsz"] == 320
    assert model.calls[0]["classes"] == [0, 2]


def test_detector_rejects_invalid_frame_before_loading() -> None:
    detector = Detector(device="cpu", model_factory=lambda _: FakeModel())

    with pytest.raises(ValueError, match="shape"):
        detector.detect(np.zeros((20, 20), dtype=np.uint8))

    assert not detector.is_loaded


def test_detector_surfaces_model_loading_error() -> None:
    detector = Detector(
        model_path=Path("missing-model.pt"),
        model_factory=lambda _: pytest.fail("factory must not run for missing weights"),
    )

    with pytest.raises(ModelLoadError, match="not found"):
        detector.load()


def test_detector_wraps_inference_error() -> None:
    model = FakeModel(error=RuntimeError("backend failed"))
    detector = Detector(device="cpu", model_factory=lambda _: model)

    with pytest.raises(InferenceError, match="backend failed"):
        detector.detect(np.zeros((20, 20, 3), dtype=np.uint8))
