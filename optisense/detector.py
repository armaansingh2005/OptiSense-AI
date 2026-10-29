"""YOLO inference service and normalized detection records."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from config import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_MODEL_NAME,
    MODEL_PATH,
)


class DetectionError(RuntimeError):
    """Base error for detector operations."""


class ModelLoadError(DetectionError):
    """Raised when the configured YOLO model cannot be loaded."""


class InferenceError(DetectionError):
    """Raised when model inference or result normalization fails."""


class DetectionResultError(InferenceError):
    """Raised when the model returns invalid detection data."""


@dataclass(frozen=True, slots=True)
class DetectionSettings:
    """Validated parameters passed to YOLO for one inference call."""

    confidence: float = DEFAULT_CONFIDENCE_THRESHOLD
    iou: float = DEFAULT_IOU_THRESHOLD
    image_size: int = DEFAULT_IMAGE_SIZE
    class_ids: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        confidence = _threshold(self.confidence, "confidence")
        iou = _threshold(self.iou, "iou")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be a finite number between 0 and 1.")
        if not 0.0 <= iou <= 1.0:
            raise ValueError("iou must be a finite number between 0 and 1.")
        if isinstance(self.image_size, bool) or not isinstance(self.image_size, int):
            raise ValueError("image_size must be a positive integer.")
        if self.image_size < 32:
            raise ValueError("image_size must be at least 32 pixels.")
        if self.class_ids is not None:
            if any(
                isinstance(class_id, bool)
                or not isinstance(class_id, int)
                or class_id < 0
                for class_id in self.class_ids
            ):
                raise ValueError("class_ids must contain non-negative integers.")
            object.__setattr__(self, "class_ids", tuple(dict.fromkeys(self.class_ids)))


@dataclass(frozen=True, slots=True)
class Detection:
    """A normalized object detection in image pixel coordinates."""

    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    frame_number: int | None = None
    track_id: int | None = None

    def to_dict(self) -> dict[str, int | float | str | None]:
        """Return a serialization-friendly representation."""
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "frame_number": self.frame_number,
            "track_id": self.track_id,
        }


ModelFactory = Callable[[str], Any]


class Detector:
    """Load and reuse one Ultralytics YOLO model for frame inference."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        device: str | None = None,
        model_factory: ModelFactory | None = None,
    ) -> None:
        configured_path = model_path if model_path is not None else MODEL_PATH
        self.model_path = Path(configured_path).expanduser() if configured_path is not None else None
        self.requested_device = device
        self._model_factory = model_factory
        self._model: Any | None = None
        self._device: str | None = None
        self._load_lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> str | None:
        return self._device

    @property
    def model_name(self) -> str:
        if self.model_path is not None:
            return self.model_path.name
        return DEFAULT_MODEL_NAME

    @property
    def class_names(self) -> dict[int, str]:
        """Return model class names when the model has been initialized."""
        if self._model is None:
            return {}
        names = getattr(self._model, "names", {})
        if isinstance(names, dict):
            return {int(class_id): str(name) for class_id, name in names.items()}
        if isinstance(names, (list, tuple)):
            return {class_id: str(name) for class_id, name in enumerate(names)}
        return {}

    def load(self) -> None:
        """Load the model once. Default weights are obtained by Ultralytics."""
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            if self.model_path is not None and not self.model_path.is_file():
                raise ModelLoadError(
                    f"Configured model weights were not found: {self.model_path}"
                )

            model_reference = str(self.model_path) if self.model_path is not None else self.model_name
            try:
                factory = self._model_factory or self._default_model_factory
                model = factory(model_reference)
                self._device = self._select_device()
                self._model = model
            except ModelLoadError:
                raise
            except Exception as exc:
                raise ModelLoadError(
                    f"Could not load model '{model_reference}': {exc}"
                ) from exc

    def detect(
        self,
        frame: np.ndarray,
        settings: DetectionSettings | None = None,
    ) -> list[Detection]:
        """Run inference on one BGR or RGB image array."""
        self._validate_frame(frame)
        parameters = settings or DetectionSettings()
        self.load()
        assert self._model is not None

        options: dict[str, Any] = {
            "conf": parameters.confidence,
            "iou": parameters.iou,
            "imgsz": parameters.image_size,
            "device": self._device,
            "verbose": False,
        }
        if parameters.class_ids is not None:
            options["classes"] = list(parameters.class_ids)

        try:
            results = self._model.predict(source=frame, **options)
        except Exception as exc:
            raise InferenceError(f"YOLO inference failed: {exc}") from exc
        try:
            return normalize_results(results, frame.shape[1], frame.shape[0])
        except DetectionResultError:
            raise
        except Exception as exc:
            raise DetectionResultError(f"Could not normalize YOLO results: {exc}") from exc

    @staticmethod
    def _validate_frame(frame: np.ndarray) -> None:
        if not isinstance(frame, np.ndarray):
            raise ValueError("frame must be a NumPy array.")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must have shape (height, width, 3).")
        if frame.shape[0] <= 0 or frame.shape[1] <= 0:
            raise ValueError("frame dimensions must be non-zero.")
        if frame.dtype != np.uint8:
            raise ValueError("frame must use uint8 pixel values.")

    @staticmethod
    def _default_model_factory(model_reference: str) -> Any:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ModelLoadError(
                "Ultralytics is not installed. Install project dependencies with "
                "'python -m pip install -r requirements.txt'."
            ) from exc
        return YOLO(model_reference)

    def _select_device(self) -> str:
        if self.requested_device is not None:
            if self.requested_device == "cpu":
                return "cpu"
            if self.requested_device != "cuda" and not self.requested_device.startswith("cuda:"):
                raise ModelLoadError("device must be 'cpu', 'cuda', or a CUDA device such as 'cuda:0'.")
            try:
                import torch
            except ImportError as exc:
                raise ModelLoadError(
                    f"Requested device '{self.requested_device}' requires PyTorch."
                ) from exc
            if self.requested_device.startswith("cuda") and not torch.cuda.is_available():
                raise ModelLoadError(
                    f"Requested device '{self.requested_device}' is unavailable."
                )
            if self.requested_device.startswith("cuda:"):
                try:
                    index = int(self.requested_device.split(":", 1)[1])
                except ValueError as exc:
                    raise ModelLoadError(
                        f"Invalid CUDA device '{self.requested_device}'."
                    ) from exc
                if index < 0 or index >= torch.cuda.device_count():
                    raise ModelLoadError(
                        f"Requested device '{self.requested_device}' is unavailable."
                    )
            return self.requested_device

        try:
            import torch
        except ImportError:
            return "cpu"
        return "cuda:0" if torch.cuda.is_available() else "cpu"


def normalize_results(
    results: Sequence[Any] | Any,
    image_width: int,
    image_height: int,
) -> list[Detection]:
    """Convert Ultralytics result objects to validated Detection records."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive.")
    if results is None:
        return []
    if not isinstance(results, (list, tuple)):
        results = [results]

    detections: list[Detection] = []
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        coordinates = _as_rows(getattr(boxes, "xyxy", None), width=4, field="xyxy")
        class_ids = _as_vector(getattr(boxes, "cls", None), field="cls")
        confidences = _as_vector(getattr(boxes, "conf", None), field="conf")
        track_ids_value = getattr(boxes, "id", None)
        track_ids = (
            _as_vector(track_ids_value, field="id")
            if track_ids_value is not None
            else [None] * len(coordinates)
        )
        names = getattr(result, "names", {})

        if not (len(coordinates) == len(class_ids) == len(confidences) == len(track_ids)):
            raise DetectionResultError("Model result fields have inconsistent lengths.")

        for coords, class_value, confidence_value, track_value in zip(
            coordinates, class_ids, confidences, track_ids
        ):
            class_id = _valid_class_id(class_value)
            confidence = _valid_confidence(confidence_value)
            x1, y1, x2, y2 = (_finite_number(value, "bounding-box coordinate") for value in coords)
            if x1 < 0 or y1 < 0 or x2 > image_width or y2 > image_height or x1 >= x2 or y1 >= y2:
                raise DetectionResultError("Model returned a bounding box outside image bounds.")
            class_name = _class_name(names, class_id)
            track_id = None if track_value is None else _valid_class_id(track_value)
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    track_id=track_id,
                )
            )
    return detections


def _as_list(value: Any, field: str) -> list[Any]:
    if value is None:
        raise DetectionResultError(f"Model result is missing '{field}'.")
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise DetectionResultError(f"Model result field '{field}' is not a sequence.")
    return list(value)


def _as_rows(value: Any, width: int, field: str) -> list[list[Any]]:
    rows = _as_list(value, field)
    if not rows:
        return []
    if not isinstance(rows[0], (list, tuple)):
        raise DetectionResultError(f"Model result field '{field}' has invalid dimensions.")
    normalized = [list(row) for row in rows]
    if any(len(row) != width for row in normalized):
        raise DetectionResultError(f"Model result field '{field}' must have {width} values per box.")
    return normalized


def _as_vector(value: Any, field: str) -> list[Any]:
    values = _as_list(value, field)
    if values and isinstance(values[0], (list, tuple)):
        if any(len(row) != 1 for row in values):
            raise DetectionResultError(f"Model result field '{field}' must be one-dimensional.")
        values = [row[0] for row in values]
    return values


def _finite_number(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise DetectionResultError(f"Model returned an invalid {field}.") from exc
    if not math.isfinite(number):
        raise DetectionResultError(f"Model returned a non-finite {field}.")
    return number


def _threshold(value: Any, field: str) -> float:
    try:
        return _finite_number(value, field)
    except DetectionResultError as exc:
        raise ValueError(f"{field} must be a finite number.") from exc


def _valid_class_id(value: Any) -> int:
    number = _finite_number(value, "class or tracking ID")
    if number < 0 or not number.is_integer():
        raise DetectionResultError("Model returned an invalid class or tracking ID.")
    return int(number)


def _valid_confidence(value: Any) -> float:
    confidence = _finite_number(value, "confidence")
    if not 0.0 <= confidence <= 1.0:
        raise DetectionResultError("Model returned confidence outside [0, 1].")
    return confidence


def _class_name(names: Any, class_id: int) -> str:
    try:
        name = names[class_id]
    except (KeyError, IndexError, TypeError) as exc:
        raise DetectionResultError(f"Model returned unknown class ID {class_id}.") from exc
    if not isinstance(name, str) or not name.strip():
        raise DetectionResultError(f"Model returned an invalid name for class ID {class_id}.")
    return name
