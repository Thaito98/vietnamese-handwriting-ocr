from __future__ import annotations
import numpy as np

_ocr = None


def _get_ocr():
    global _ocr
    if _ocr is None:
        import os
        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
        from paddleocr import PaddleOCR
        _ocr = PaddleOCR(
            use_angle_cls=False,
            lang="en",
            use_gpu=False,
            show_log=False,
        )
    return _ocr


def detect(image_np: np.ndarray) -> list[tuple[int, int, int, int]]:
    ocr = _get_ocr()
    result = ocr.ocr(image_np, det=True, rec=False, cls=False)
    boxes: list[tuple[int, int, int, int]] = []

    if not result or not result[0]:
        return boxes

    for item in result[0]:
        pts = np.array(item, dtype=np.int32)
        x1, y1 = int(pts[:, 0].min()), int(pts[:, 1].min())
        x2, y2 = int(pts[:, 0].max()), int(pts[:, 1].max())
        boxes.append((x1, y1, x2, y2))

    return boxes
