"""
OCR Web – Streamlit
Phát hiện chữ: EasyOCR | PaddleOCR
Nhận dạng    : TrOCR fine-tuned (VNOnDB) + ToneAttentionGate
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import io
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from transformers import GenerationConfig, TrOCRProcessor, VisionEncoderDecoderModel

from utils.deskew import deskew
from utils.word_split import split_all_lines

# ================================================================
# ĐƯỜNG DẪN MODEL
# ================================================================
MODEL_DIR = Path(__file__).parent / "models" / "best_model"

# Màu bounding box theo detector
COLORS = {
    "EasyOCR"  : "#FF4B4B",
    "PaddleOCR": "#1E88E5",
}


# ================================================================
# MODULE MỚI: ToneAttentionGate
# Squeeze-and-Excitation cho dấu tiếng Việt
# Khuếch đại vùng feature có dấu thanh → giảm nhầm ỗ→ố, ỏ→ó
# ================================================================
class ToneAttentionGate(nn.Module):
    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),  # 768 → 192
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim),  # 192 → 768
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x: (batch, 576, 768)
        return self.norm(x + x * self.gate(x))       # residual connection


# ================================================================
# TIỆN ÍCH
# ================================================================
def pil_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


# ================================================================
# LOAD MODEL — cache 1 lần duy nhất
# ================================================================
@st.cache_resource(show_spinner="Đang tải mô hình TrOCR…")
def load_trocr():
    if not MODEL_DIR.exists():
        st.error(
            f"Không tìm thấy thư mục model: {MODEL_DIR}\n"
            "Hãy tải best_model từ Drive về đặt vào models/best_model/"
        )
        st.stop()

    processor = TrOCRProcessor.from_pretrained(str(MODEL_DIR))
    model     = VisionEncoderDecoderModel.from_pretrained(str(MODEL_DIR))
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    model     = model.to(device)

    # Load ToneAttentionGate
    gate_path = MODEL_DIR / "tone_gate.pt"
    if gate_path.exists():
        tone_gate = ToneAttentionGate(hidden_dim=768).to(device)
        tone_gate.load_state_dict(
            torch.load(str(gate_path), map_location=device, weights_only=True)
        )
        tone_gate.eval()

        # Đăng ký hook — chèn gate vào giữa encoder và decoder
        def _encoder_hook(module, input, output):
            output.last_hidden_state = tone_gate(output.last_hidden_state)
            return output

        model.encoder.register_forward_hook(_encoder_hook)
        print("✓ ToneAttentionGate loaded!")
    else:
        st.warning(
            "Không tìm thấy tone_gate.pt trong models/best_model/ — "
            "chạy không có ToneAttentionGate"
        )

    model.eval()
    return processor, model, device


# ================================================================
# LOAD DETECTOR — cache theo tên
# ================================================================
@st.cache_resource(show_spinner="Đang tải detector…")
def load_detector(name: str):
    if name == "EasyOCR":
        import easyocr
        return easyocr.Reader(["vi", "en"], recognizer=False, verbose=False)
    elif name == "PaddleOCR":
        from paddleocr import PaddleOCR
        return PaddleOCR(
            use_angle_cls=False, lang="en",
            use_gpu=False, show_log=False
        )
    raise ValueError(f"Detector không hợp lệ: {name}")


# ================================================================
# PHÁT HIỆN VÙNG CHỮ
# ================================================================
def detect_boxes(
    detector_obj, detector_name: str, image_np: np.ndarray
) -> list[tuple[int, int, int, int]]:
    boxes = []

    if detector_name == "EasyOCR":
        bounds = detector_obj.detect(image_np)
        if bounds and bounds[0] and bounds[0][0]:
            for b in bounds[0][0]:
                x1, x2, y1, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                boxes.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))

    elif detector_name == "PaddleOCR":
        result = detector_obj.ocr(image_np, det=True, rec=False, cls=False)
        if result and result[0]:
            for item in result[0]:
                pts = np.array(item, dtype=np.int32)
                x1, y1 = int(pts[:, 0].min()), int(pts[:, 1].min())
                x2, y2 = int(pts[:, 0].max()), int(pts[:, 1].max())
                boxes.append((x1, y1, x2, y2))

    # Sắp xếp theo thứ tự đọc: trên→dưới, trái→phải
    if boxes:
        boxes = sorted(boxes, key=lambda b: (b[1], b[0]))

    return boxes


# ================================================================
# NHẬN DẠNG 1 CROP
# ================================================================
def recognize_crop(
    crop     : Image.Image,
    processor: TrOCRProcessor,
    model    : VisionEncoderDecoderModel,
    device   : str,
    gen_cfg  : GenerationConfig,
) -> str:
    pv = processor(crop.convert("RGB"), return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        ids = model.generate(pv, generation_config=gen_cfg)
    return processor.batch_decode(ids, skip_special_tokens=True)[0]


# ================================================================
# VẼ BOUNDING BOX
# ================================================================
def draw_boxes(
    image : Image.Image,
    boxes : list,
    labels: list,
    color : str,
) -> Image.Image:
    img  = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", max(12, image.height // 40))
    except Exception:
        font = ImageFont.load_default()

    for (x1, y1, x2, y2), label in zip(boxes, labels):
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        bbox_txt = draw.textbbox((x1, y1 - 2), label, font=font)
        draw.rectangle(bbox_txt, fill=color)
        draw.text((x1, y1 - 2), label, fill="white", font=font)

    return img


# ================================================================
# GIAO DIỆN STREAMLIT
# ================================================================
def main():
    st.set_page_config(
        page_title="OCR Chữ Viết Tay Tiếng Việt",
        layout="wide",
    )

    st.title(" Nhận Dạng Chữ Viết Tay Tiếng Việt")
    st.markdown(
        "Mô hình **TrOCR + ToneAttentionGate** fine-tuned trên **VNOnDB** (~110k mẫu). "
        "Chọn bộ phát hiện văn bản rồi tải ảnh lên."
    )

    # ── Sidebar ─────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙ Cấu hình")

        detector_name = st.selectbox(
            "Bộ phát hiện văn bản",
            ["EasyOCR", "PaddleOCR"],
            help=(
                "EasyOCR – ổn định, hỗ trợ tiếng Việt tốt\n"
                "PaddleOCR – nhanh, nhẹ"
            ),
        )

        st.divider()

        use_deskew    = st.toggle("Chỉnh nghiêng (deskew)", value=False)
        use_wordsplit = st.toggle(
            "Tách từng từ (word split)", value=True,
            help="Tự động tách line-box thành word-box."
        )

        st.divider()

        st.subheader("Padding vùng crop")
        pad_x   = st.slider("Ngang (px)",              0,  40,  20)
        pad_top = st.slider("Dọc phía trên (% height)", 0,  80,  60)
        pad_bot = st.slider("Dọc phía dưới (% height)", 0,  80,  60)

        st.divider()

        num_beams = st.slider(
            "Beam search (num_beams)", 1, 8, 4,
            help="Càng lớn càng chính xác nhưng chậm hơn"
        )

        st.divider()
        st.info(
            "**Cài thư viện:**\n"
            "```\n"
            "pip install streamlit transformers\n"
            "pip install torch Pillow opencv-python\n"
            "pip install easyocr\n"
            "pip install paddlepaddle==2.6.2 paddleocr==2.9.1\n"
            "```"
        )

    # ── Upload ──────────────────────────────────────────────────
    uploaded = st.file_uploader(
        " Tải ảnh lên (jpg / jpeg / png / bmp)",
        type=["jpg", "jpeg", "png", "bmp"],
    )

    if uploaded is None:
        st.markdown(
            """
            <div style='text-align:center; padding:60px; color:#888;'>
                <h3> Tải ảnh lên để bắt đầu nhận dạng</h3>
                <p>Hỗ trợ: JPG · PNG · BMP</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # ── Load models ─────────────────────────────────────────────
    processor, model, device = load_trocr()

    try:
        detector_obj = load_detector(detector_name)
    except Exception as e:
        st.error(
            f"Không thể tải detector **{detector_name}**.\n\n"
            f"Lỗi: `{e}`\n\n"
            "Hãy cài thư viện theo hướng dẫn trong sidebar."
        )
        return

    # ── Đọc ảnh ─────────────────────────────────────────────────
    image_bytes = uploaded.read()
    image       = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np    = np.array(image)
    h_img, w_img = image_np.shape[:2]

    st.image(pil_to_bytes(image), caption="Ảnh gốc", use_container_width=True)

    # ── Nút xử lý ───────────────────────────────────────────────
    if not st.button(" Nhận dạng", type="primary", use_container_width=True):
        return

    gen_cfg = GenerationConfig(
        max_new_tokens         = 32,
        num_beams              = num_beams,
        no_repeat_ngram_size   = 3,
        decoder_start_token_id = processor.tokenizer.cls_token_id,
        eos_token_id           = processor.tokenizer.sep_token_id,
        pad_token_id           = processor.tokenizer.pad_token_id,
    )

    # ── Phát hiện ───────────────────────────────────────────────
    with st.spinner(f"Đang phát hiện vùng chữ bằng {detector_name}…"):
        try:
            boxes = detect_boxes(detector_obj, detector_name, image_np)
        except Exception as e:
            st.error(f"Lỗi phát hiện: {e}")
            return

    if not boxes:
        st.warning("Không phát hiện được vùng chữ nào trong ảnh.")
        return

    line_count = len(boxes)

    # ── Tách từ ─────────────────────────────────────────────────
    if use_wordsplit:
        with st.spinner("Đang tách từng từ…"):
            boxes = split_all_lines(image, boxes)
        st.success(
            f" Phát hiện **{line_count}** dòng → tách thành "
            f"**{len(boxes)}** từ bằng **{detector_name}**"
        )
    else:
        st.success(
            f" Phát hiện **{line_count}** vùng chữ bằng **{detector_name}**"
        )

    # ── Nhận dạng từng vùng ─────────────────────────────────────
    results  = []
    progress = st.progress(0, text="Đang nhận dạng…")

    for i, (x1, y1, x2, y2) in enumerate(boxes):
        h_box = y2 - y1

        # Padding
        cx1 = max(0,      x1 - pad_x)
        cx2 = min(w_img,  x2 + pad_x)
        cy1 = max(0,      y1 - int(h_box * pad_top / 100))
        cy2 = min(h_img,  y2 + int(h_box * pad_bot / 100))

        crop  = image.crop((cx1, cy1, cx2, cy2))
        angle = 0.0

        if use_deskew:
            crop, angle = deskew(crop)

        text = recognize_crop(crop, processor, model, device, gen_cfg)
        results.append({
            "box"  : (x1, y1, x2, y2),
            "crop" : crop,
            "text" : text,
            "angle": angle,
        })
        progress.progress(
            (i + 1) / len(boxes),
            text=f"Đã xử lý {i+1}/{len(boxes)}"
        )

    progress.empty()

    # ── Hiển thị ảnh kết quả ────────────────────────────────────
    st.subheader(" Kết quả phát hiện & nhận dạng")
    annotated = draw_boxes(
        image,
        [r["box"]  for r in results],
        [r["text"] for r in results],
        COLORS.get(detector_name, "#FF4B4B"),
    )
    st.image(pil_to_bytes(annotated), caption="Ảnh kết quả",
             use_container_width=True)

    # ── Văn bản đầy đủ ──────────────────────────────────────────
    full_text = " ".join(r["text"] for r in results)
    st.subheader(" Văn bản nhận dạng được")
    st.text_area("Văn bản", full_text, height=120, label_visibility="collapsed")

    # ── Chi tiết từng vùng ──────────────────────────────────────
    st.subheader(" Chi tiết từng vùng")
    cols_per_row = 4
    for row_start in range(0, len(results), cols_per_row):
        row_items = results[row_start: row_start + cols_per_row]
        cols      = st.columns(len(row_items))
        for col, r in zip(cols, row_items):
            with col:
                st.image(pil_to_bytes(r["crop"]), use_container_width=True)
                angle_str = (
                    f"{r['angle']:+.1f} độ"
                    if abs(r["angle"]) >= 0.5 else "thẳng"
                )
                st.caption(
                    f'**"{r["text"]}"**'
                    + (f"\n_(xoay {angle_str})_" if use_deskew else "")
                )

    # ── Bảng tổng hợp ───────────────────────────────────────────
    with st.expander(" Bảng tổng hợp kết quả"):
        import pandas as pd
        df = pd.DataFrame([
            {
                "STT"              : i + 1,
                "Vùng (x1,y1,x2,y2)": r["box"],
                "Góc xoay (độ)"    : f"{r['angle']:+.1f}" if use_deskew else "—",
                "Văn bản nhận dạng": r["text"],
            }
            for i, r in enumerate(results)
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()