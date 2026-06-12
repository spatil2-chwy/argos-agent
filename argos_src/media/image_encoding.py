"""Image normalization and encoding helpers."""

from __future__ import annotations

import base64
import os
from io import BytesIO
from typing import Callable

import numpy as np
import requests
from PIL import Image as PILImage
from PIL.Image import Image


def preprocess_image(
    image: Image | str | bytes | np.ndarray,
    encoding_function: Callable[[bytes], str] = lambda b: base64.b64encode(b).decode(
        "utf-8"
    ),
) -> str:
    """Convert a PIL image, path/URL, bytes, or ndarray into base64 PNG text."""

    def _to_pil_from_ndarray(arr: np.ndarray) -> Image:
        normalized = arr
        if normalized.dtype in (np.float32, np.float64):
            normalized = np.clip(normalized, 0.0, 1.0)
            normalized = (normalized * 255.0).round().astype(np.uint8)
        return PILImage.fromarray(np.ascontiguousarray(normalized))

    def _ensure_pil(img: Image | str | bytes | np.ndarray) -> Image:
        if isinstance(img, Image):
            return img
        if isinstance(img, np.ndarray):
            return _to_pil_from_ndarray(img)
        if isinstance(img, str):
            if img.startswith(("http://", "https://")):
                response = requests.get(img, timeout=(5, 15))
                response.raise_for_status()
                return PILImage.open(BytesIO(response.content))
            file_path = img[len("file://") :] if img.startswith("file://") else img
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            return PILImage.open(file_path)
        if isinstance(img, bytes):
            return PILImage.open(BytesIO(img))
        raise TypeError(f"Unsupported image type: {type(img).__name__}")

    pil_image = _ensure_pil(image)
    with BytesIO() as buffer:
        pil_image.save(buffer, format="PNG")
        return encoding_function(buffer.getvalue())
