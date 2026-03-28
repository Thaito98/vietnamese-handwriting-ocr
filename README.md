# Vietnamese Handwriting OCR

A Streamlit web application for recognizing Vietnamese handwritten text using a two-stage OCR pipeline: text detection followed by recognition with a fine-tuned TrOCR model.

## Overview

The pipeline consists of two stages:

1. **Text Detection** — Locates text regions in an uploaded image using either EasyOCR or PaddleOCR.

2. **Text Recognition** — Reads each word crop using a `VisionEncoderDecoderModel` (ViT encoder + TrOCR decoder) fine-tuned on the VNOnDB Vietnamese handwriting dataset. A custom `ToneSpatialGate` module (Channel Attention + Spatial Attention) is injected as a forward hook on the encoder output to improve accuracy on Vietnamese tone marks (diacritics).

## Features

- Two detector options: EasyOCR and PaddleOCR
- Configurable beam search (1-8 beams)
- Adjustable padding around crops
- Annotated output image with labeled bounding boxes
- Combined text output, per-word crop grid, and summary DataFrame
- Built-in usage guide in the web interface

## Project Structure

```
ocr_web/
├── app.py                  # Main Streamlit application
├── requirements.txt
├── detectors/
│   ├── easyocr_det.py      # EasyOCR detection wrapper
│   └── paddle_det.py       # PaddleOCR detection wrapper
└── models/
    └── best_model/         # Fine-tuned TrOCR model files
        ├── config.json
        ├── generation_config.json
        ├── processor_config.json
        ├── tokenizer.json
        └── tokenizer_config.json
```

> **Note:** Model weight files (`.pt`, `.bin`, `.safetensors`) are not included in this repository due to file size. Download them separately and place them in `models/best_model/`.

## Requirements

- Python 3.8+
- CUDA-capable GPU recommended (CPU inference is supported but slow)

## Installation

```bash
git clone https://github.com/Thaito98/vietnamese-handwriting-ocr.git
cd vietnamese-handwriting-ocr

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## Model Weights

Place the following files in `models/best_model/`:

| File | Description |
|---|---|
| `pytorch_model.bin` or `model.safetensors` | TrOCR encoder-decoder weights |
| `tone_spatial.pt` | ToneSpatialGate weights |

## Usage

```bash
streamlit run app.py
```

Open the URL shown in the terminal (default `http://localhost:8501`), upload an image (JPG, PNG, BMP), configure the sidebar options, and click **Nhan dang**.

## Configuration

| Option | Description |
|---|---|
| Detector | EasyOCR or PaddleOCR |
| Padding H | Horizontal padding around each crop (px) |
| Padding V | Vertical padding as % of crop height |
| Beam search | Number of beams for decoding (1 = greedy, max 8) |

## License

MIT
