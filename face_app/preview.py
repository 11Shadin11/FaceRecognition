"""Prepare a camera preview with Pillow's native BGR decoder."""
import numpy as np
from PIL import Image, ImageOps


def make_preview(frame, size):
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError("Для превью нужен 8-битный кадр BGR с тремя каналами.")
    height, width = frame.shape[:2]
    # Avoid the costly NumPy copy of a view with reversed channel strides.
    image = Image.frombytes("RGB", (width, height), frame.tobytes(), "raw", "BGR")
    return ImageOps.contain(image, size, method=Image.Resampling.BILINEAR)
