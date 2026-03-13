"""
Tách line-box thành word-box bằng vertical projection.
Dùng khi detector trả về cả dòng thay vì từng từ.
"""
from __future__ import annotations
import numpy as np
import cv2
from PIL import Image


def split_line_to_words(
    img_rgb: Image.Image,
    line_box: tuple[int, int, int, int],
    scale: int = 3,
    min_gap_px: int = 4,
    min_word_w_px: int = 15,
) -> list[tuple[int, int, int, int]]:
    """
    Tách một line-box thành các word-box bằng vertical projection.

    Parameters
    ----------
    img_rgb      : PIL Image RGB gốc (chưa upscale)
    line_box     : (x1, y1, x2, y2) của dòng chữ
    scale        : upscale nội bộ để tăng độ chính xác tách từ
    min_gap_px   : khoảng trắng tối thiểu giữa 2 từ (đơn vị: pixel gốc)
    min_word_w_px: chiều rộng tối thiểu của một từ (đơn vị: pixel gốc)

    Returns
    -------
    list of (x1, y1, x2, y2) – trả về [line_box] nếu không tách được
    """
    x1, y1, x2, y2 = line_box
    if x2 <= x1 or y2 <= y1:
        return [line_box]

    crop = img_rgb.crop((x1, y1, x2, y2)).convert("L")
    sw, sh = crop.width * scale, crop.height * scale
    crop_up = np.array(crop.resize((sw, sh), Image.LANCZOS))

    _, binary = cv2.threshold(
        crop_up, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    vproj = binary.sum(axis=0).astype(float)
    # Smooth để bỏ nhiễu nhỏ giữa ký tự
    k = max(1, scale * 2)
    vproj = np.convolve(vproj, np.ones(k) / k, mode="same")

    threshold = vproj.max() * 0.05
    if threshold == 0:
        return [line_box]

    # Phát hiện vùng chữ và khoảng trắng
    in_word = False
    gap_count = 0
    word_start = 0
    words: list[tuple[int, int, int, int]] = []

    for col in range(len(vproj)):
        is_text = vproj[col] > threshold
        if is_text and not in_word:
            word_start = col
            in_word = True
            gap_count = 0
        elif not is_text and in_word:
            gap_count += 1
            if gap_count >= min_gap_px * scale:
                ox1 = x1 + word_start // scale
                ox2 = x1 + (col - gap_count // 2) // scale
                if ox2 - ox1 >= min_word_w_px:
                    words.append((ox1, y1, ox2, y2))
                in_word = False
        elif is_text and in_word:
            gap_count = 0

    # Từ cuối dòng
    if in_word:
        ox1 = x1 + word_start // scale
        ox2 = x1 + len(vproj) // scale
        if ox2 - ox1 >= min_word_w_px:
            words.append((ox1, y1, ox2, y2))

    return words if words else [line_box]


def split_all_lines(
    img_rgb: Image.Image,
    line_boxes: list[tuple[int, int, int, int]],
    **kwargs,
) -> list[tuple[int, int, int, int]]:
    """
    Áp dụng split_line_to_words cho tất cả các line-box.
    Kết quả sắp xếp theo thứ tự đọc (trên→dưới, trái→phải).
    """
    all_words: list[tuple[int, int, int, int]] = []
    for lb in line_boxes:
        all_words.extend(split_line_to_words(img_rgb, lb, **kwargs))
    all_words.sort(key=lambda b: (b[1], b[0]))
    return all_words
