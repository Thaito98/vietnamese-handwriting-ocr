import cv2
import numpy as np
from PIL import Image

# Chỉ xoay khi góc nghiêng trong khoảng này
_MIN_ANGLE_DEG = 3.0   # nhỏ hơn 3° → coi như thẳng, không xoay
_MAX_ANGLE_DEG = 45.0  # lớn hơn 45° → có thể bị nhầm, không xoay


def deskew(pil_img: Image.Image) -> tuple[Image.Image, float]:
    """
    Chỉnh nghiêng ảnh chữ viết tay.
    Trả về (ảnh đã xoay, góc xoay theo độ).
    Chỉ xoay khi _MIN_ANGLE_DEG <= |góc| <= _MAX_ANGLE_DEG.
    """
    img_gray = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2GRAY)

    _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    coords = np.column_stack(np.where(binary > 0))
    if len(coords) < 50:
        return pil_img, 0.0

    angle = cv2.minAreaRect(coords)[-1]

    # Quy đổi về góc nghiêng thực tế
    if angle < -45:
        angle = 90 + angle   # nghiêng trái
    else:
        angle = -angle       # nghiêng phải

    # Chỉ xoay khi góc trong khoảng hợp lệ
    if abs(angle) < _MIN_ANGLE_DEG or abs(angle) > _MAX_ANGLE_DEG:
        return pil_img, 0.0

    rotated = pil_img.rotate(angle, expand=True, fillcolor=(255, 255, 255))
    return rotated, angle