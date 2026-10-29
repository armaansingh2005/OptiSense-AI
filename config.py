"""Shared application configuration for OptiSense AI."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
_configured_model_path = os.getenv("OPTISENSE_MODEL_PATH", "").strip()
DEFAULT_MODEL_NAME = "yolo11n.pt"
if _configured_model_path:
    _model_path = Path(_configured_model_path).expanduser()
    MODEL_PATH = _model_path if _model_path.is_absolute() else PROJECT_ROOT / _model_path
else:
    MODEL_PATH = None
MAX_UPLOAD_SIZE_MB = int(os.getenv("OPTISENSE_MAX_UPLOAD_SIZE_MB", "200"))

DEFAULT_CONFIDENCE_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.45
DEFAULT_IMAGE_SIZE = 640
SUPPORTED_IMAGE_TYPES = ("jpg", "jpeg", "png", "webp")
SUPPORTED_VIDEO_TYPES = ("mp4", "avi", "mov", "mkv")

if MAX_UPLOAD_SIZE_MB <= 0:
    raise ValueError("OPTISENSE_MAX_UPLOAD_SIZE_MB must be a positive integer.")
