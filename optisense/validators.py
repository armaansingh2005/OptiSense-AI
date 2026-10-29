"""Validation helpers for uploaded image media."""

from __future__ import annotations

from io import BytesIO
import warnings

from PIL import Image, UnidentifiedImageError

from config import MAX_UPLOAD_SIZE_MB, SUPPORTED_IMAGE_TYPES

MAX_IMAGE_PIXELS = 40_000_000
MAX_UPLOAD_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
_SUPPORTED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP"}
_EXTENSION_FORMATS = {
    "jpg": "JPEG",
    "jpeg": "JPEG",
    "png": "PNG",
    "webp": "WEBP",
}


class MediaValidationError(ValueError):
    """Raised when uploaded media is invalid or exceeds configured limits."""


def validate_image_upload(filename: str, data: bytes) -> tuple[int, int]:
    """Validate extension, payload size, image format, and decoded dimensions."""
    if not isinstance(filename, str) or not filename.strip():
        raise MediaValidationError("Choose an image file with a valid filename.")
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if suffix not in SUPPORTED_IMAGE_TYPES:
        supported = ", ".join(f".{extension}" for extension in SUPPORTED_IMAGE_TYPES)
        raise MediaValidationError(f"Unsupported image type. Use {supported}.")
    if not isinstance(data, bytes) or not data:
        raise MediaValidationError("The uploaded image is empty or unreadable.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise MediaValidationError(
            f"Image exceeds the {MAX_UPLOAD_SIZE_MB} MB upload limit."
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as image:
                expected_format = _EXTENSION_FORMATS[suffix]
                if image.format not in _SUPPORTED_IMAGE_FORMATS:
                    raise MediaValidationError(
                        "The file contents do not match a supported image format."
                    )
                if image.format != expected_format:
                    raise MediaValidationError(
                        "The file extension does not match the image contents."
                    )
                width, height = image.size
                if width <= 0 or height <= 0:
                    raise MediaValidationError("Image dimensions must be positive.")
                if width * height > MAX_IMAGE_PIXELS:
                    raise MediaValidationError(
                        f"Image dimensions exceed the {MAX_IMAGE_PIXELS:,}-pixel limit."
                    )
                image.verify()
                return width, height
    except MediaValidationError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise MediaValidationError("Image dimensions are too large to process safely.") from exc
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise MediaValidationError("The uploaded file is not a valid supported image.") from exc
