import base64
from io import BytesIO

import numpy as np
from PIL import Image

from argos_src.media.image_encoding import preprocess_image


def test_preprocess_image_encodes_pil_image_as_png_base64():
    encoded = preprocess_image(Image.new("RGB", (1, 1), (255, 0, 0)))

    assert base64.b64decode(encoded).startswith(b"\x89PNG")


def test_preprocess_image_accepts_float_ndarray():
    image = np.ones((1, 1, 3), dtype=np.float32)

    encoded = preprocess_image(image)

    assert base64.b64decode(encoded).startswith(b"\x89PNG")


def test_preprocess_image_accepts_image_bytes():
    buffer = BytesIO()
    Image.new("RGB", (1, 1), (0, 255, 0)).save(buffer, format="PNG")

    encoded = preprocess_image(buffer.getvalue())

    assert base64.b64decode(encoded).startswith(b"\x89PNG")
