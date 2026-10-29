from __future__ import annotations

from io import BytesIO

from PIL import Image
import pytest

from optisense import validators
from optisense.validators import MediaValidationError, validate_image_upload


def make_image_bytes(
    image_format: str = "PNG",
    size: tuple[int, int] = (12, 8),
) -> bytes:
    output = BytesIO()
    Image.new("RGB", size, color=(20, 40, 60)).save(output, format=image_format)
    return output.getvalue()


def test_validate_image_upload_accepts_supported_image() -> None:
    width, height = validate_image_upload("photo.png", make_image_bytes())

    assert (width, height) == (12, 8)


@pytest.mark.parametrize(
    ("filename", "payload", "message"),
    [
        ("photo.gif", make_image_bytes(), "Unsupported image type"),
        ("photo.jpg", make_image_bytes("PNG"), "extension does not match"),
        ("photo.png", b"not an image", "not a valid"),
        ("photo.png", b"", "empty or unreadable"),
    ],
)
def test_validate_image_upload_rejects_invalid_data(
    filename: str, payload: bytes, message: str
) -> None:
    with pytest.raises(MediaValidationError, match=message):
        validate_image_upload(filename, payload)


def test_validate_image_upload_enforces_upload_size(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(validators, "MAX_UPLOAD_BYTES", 4)

    with pytest.raises(MediaValidationError, match="upload limit"):
        validate_image_upload("photo.png", make_image_bytes())


def test_validate_image_upload_enforces_pixel_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(validators, "MAX_IMAGE_PIXELS", 10)

    with pytest.raises(MediaValidationError, match="pixel limit"):
        validate_image_upload("photo.png", make_image_bytes(size=(4, 4)))
