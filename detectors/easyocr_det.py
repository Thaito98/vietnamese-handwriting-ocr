from __future__ import annotations
import numpy as np

_reader = None


def _get_reader():
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(["vi", "en"], recognizer=False, verbose=False)
    return _reader


def detect(image_np: np.ndarray) -> list[tuple[int, int, int, int]]:
    reader = _get_reader()
    bounds = reader.detect(image_np)
    boxes: list[tuple[int, int, int, int]] = []
    if bounds and bounds[0] and bounds[0][0]:
        for b in bounds[0][0]:
            x1, x2, y1, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
            boxes.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
    return boxes
