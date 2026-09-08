# Vietnamese Handwriting Recognition

Fine-tuning Microsoft's TrOCR model on the VNOnDB Vietnamese handwriting dataset, with a Streamlit web app for testing.

**Accuracy: 94.42% of words read completely correct. Character error rate 2.30%.** Measured on 11,064 samples never used during training.

<!-- Add demo screenshot: ![Demo](docs/demo.png) -->

---

## Table of contents

- [The problem](#the-problem)
- [Data](#data)
- [Model architecture](#model-architecture)
- [Training](#training)
- [Test results](#test-results)
- [Error analysis](#error-analysis)
- [Web application](#web-application)
- [Installation](#installation)
- [Project structure](#project-structure)
- [Limitations](#limitations)

---

## The problem

Reading Vietnamese handwriting is harder than English because of the tone mark system.

Vietnamese has 6 tones combined with vowel modifiers (ă, â, ê, ô, ơ, ư, đ), producing 216 distinct characters - over 4 times the English alphabet. In handwriting, a tone mark is just a few small strokes, often written off-position or touching the letter body.

Getting one mark wrong changes the word entirely:

| Correct word | If misread as | Consequence |
|---|---|---|
| `mà` | `má`, `mã`, `mả` | 4 completely different words |
| `tuổi` | `tuối`, `tuôi` | misspelling |

For this reason the project adds a dedicated metric to measure tone mark accuracy, alongside standard evaluation metrics.

---

## Data

The **VNOnDB** dataset (HANDS-VNOnDB) from Tokyo University of Agriculture and Technology, Japan. 200 Vietnamese writers copied sample texts from newspapers using a stylus on a tablet. The original data records pen stroke trajectories over time, later converted to images using the [vnondb-extractor](https://github.com/vndee/vnondb-extractor) tool.

This project uses the word-level data: each image contains one handwritten word, paired with a label file containing that word's text.

### Validation and cleaning

| Step | Result |
|---|---|
| Verify image and label pairs | 110,746 pairs, no missing files |
| Verify label content | No encoding errors, no empty labels, no invalid characters |
| Normalize tone mark encoding | The same accented character can be stored two different ways, so all labels were normalized to one form |
| Remove faulty images | Dropped 107 images that were too faint or completely blank, keeping 99.90% |

### Label analysis

| Metric | Value |
|---|---|
| Average word length | 3.33 characters |
| Longest word | 11 characters |
| Distinct characters | 147 |
| Distinct words | 3,511 |

The dataset contains **27 characters appearing fewer than 50 times**, mostly uppercase letters with tone marks. For example `Ầ`, `Ố`, `Ý` each appear exactly once across all 110,746 samples. This weakness was identified before training, and the later error analysis shows a consistent trend.

Also, only **11.2% of labels contain an uppercase letter**. This imbalance may be related to a specific error type, covered in [Error analysis](#error-analysis).

### Image analysis

| Metric | Value |
|---|---|
| Average size | 480 x 341 pixels |
| Average aspect ratio | 1.49 (wider than tall) |
| Format | RGBA, must be converted to RGB before feeding the model |
| Average brightness | 253/255 (white background) |

### Data split

Split 80% training, 10% validation, 10% test. The split keeps word length distribution even across all three sets, and was verified so no sample appears in two sets at once.

| Set | Samples | Average word length |
|---|---|---|
| Training | 88,511 | 3.328 characters |
| Validation | 11,064 | 3.328 characters |
| Test | 11,064 | 3.328 characters |

### Training data augmentation

To make the model robust to real photographs, each training image is randomly transformed before being fed to the model:

| Transformation | Simulates |
|---|---|
| Elastic stroke distortion | Uneven handwriting strokes |
| Slight rotation, shift, scale | Slanted writing, tilted photos |
| Gaussian noise | Camera sensor noise |
| Slight blur | Out-of-focus photos |
| Brightness and contrast shift | Varying lighting conditions |

Validation and test sets are left unmodified, keeping original images for fair evaluation.

---

## Model architecture

TrOCR has two parts: one reads the image, the other generates text.

```
Image 384x384 pixels
      |
      v
[ Image encoder - Vision Transformer ]
  12 layers, about 87 million parameters
  Splits the image into 576 patches of 16x16 pixels
  Converts each patch into a 768-dimensional vector
      |
      v  visual features
[ Text decoder - RoBERTa Decoder ]
  12 layers, about 247 million parameters
  Generates one token at a time (a token may be one or several characters)
  Each token depends on both the image and previously generated tokens
      |
      v
Output: "tuổi"
```

Starting checkpoint: `microsoft/trocr-base-handwritten` - already trained by Microsoft on English handwriting. This project continues training on Vietnamese data.

Total 333.9 million parameters (encoder 86.65 million + decoder 247.27 million), model file 1.28 GB. Counted directly from the weight file.

The decoder holds most of the parameters because it uses 1024-dimensional vectors (the encoder uses 768) and needs an output projection covering all 50,265 vocabulary tokens.

---

## Training

### Configuration

| Setting | Value |
|---|---|
| Base model | `microsoft/trocr-base-handwritten` |
| Epochs | 10 |
| Batch size | 64 |
| Learning rate | 0.00002 |
| Optimizer | AdamW |
| Environment | Google Colab, GPU, about 5 minutes per epoch |

### Progress across 10 epochs

| Epoch | Train Loss | Val CER |
|---|---|---|
| 1 | 0.6916 | 4.99% |
| 2 | 0.0871 | 3.88% |
| 3 | 0.0586 | 3.55% |
| 4 | 0.0419 | 3.20% |
| 5 | 0.0301 | 3.07% |
| 6 | 0.0218 | 2.87% |
| 7 | 0.0150 | 2.75% |
| 8 | 0.0097 | 2.67% |
| 9 | 0.0063 | 2.55% |
| **10** | **0.0040** | **2.47%** |

Loss dropped most sharply at epoch 2 (from 0.69 to 0.09), showing the base model already had a solid foundation and only needed adjustment for Vietnamese.

Validation error decreased steadily across all 10 epochs with no oscillation. Early stopping (set to wait 3 epochs without improvement) never triggered, meaning the model still had room to improve with more training.

---

## Test results

Evaluated on 11,064 samples never used for training or validation.

| Metric | Value | Meaning |
|---|---|---|
| Word Accuracy | **94.42%** | Out of 100 words, 94 are read completely correct |
| CER | **2.30%** | Out of 100 characters, about 2 are wrong |
| WER | **5.58%** | Out of 100 words, about 6 are wrong |
| Tone Accuracy | 97.51% | See the note on how this is computed below |
| Char F1 | 97.90% | Combined character-level accuracy |

Validation error decreased continuously across all 10 training epochs performed, with no sign of turning back up.

A caveat when comparing against the validation figure: the two measurements use different configurations. Validation was measured during training with fast decoding, while the test set was measured with more thorough decoding (trying 4 candidates and picking the best). The figures 2.47% and 2.30% are therefore not directly comparable.

### Note on the Tone Accuracy metric

This metric groups Vietnamese characters into 26 groups (for example the `a` group contains `a à á ả ã ạ`), then counts the fraction read correctly among characters belonging to those groups: 16,321 characters, 15,914 correct, 407 wrong.

The calculation has three limitations worth knowing before reading the 97.51% figure:

- **It includes unaccented vowels.** Of the 148 characters counted, 26 carry no tone mark (`a`, `e`, `i`, `o`, `u`, `y`...). So this is not pure tone mark accuracy but accuracy over characters belonging to groups that have accented variants.
- **It groups `d` and `đ` together.** But `đ` is not `d` with a tone mark; they are two distinct letters.
- **It compares position by position.** When the model reads too few or too many characters (106 samples), the comparison shifts out of alignment and may skip characters. Estimated error around 1%.

This figure is therefore best read as a relative indicator for comparing training runs, not an exact measure of tone mark reading ability.

---

## Error analysis

The model misread 617 out of 11,064 test samples (5.58%). This section classifies those errors and identifies the cause of each type.

First, false errors need to be ruled out: if a label and a prediction differ only in tone mark encoding or whitespace, that is a comparison artifact rather than a model error. After normalizing both, accuracy did not increase at all. So all 617 errors are genuine.

| Error type | Samples | Share | Ceiling if fully fixed |
|---|---|---|---|
| Wrong letter | 252 | 40.8% | 96.70% |
| Wrong uppercase / lowercase | 133 | 21.6% | 95.63% |
| Wrong tone mark | 124 | 20.1% | 95.54% |
| Missing or extra characters | 106 | 17.2% | 95.38% |
| Both tone mark and case wrong | 2 | 0.3% | - |

### Type 1: Wrong letter (252 samples, most common)

Real examples:

```
'lớn'   read as 'lên'        'răn'   read as 'văn'
'nhỏ'   read as 'nhơ'        'rõ'    read as 'lõ'
'chậm'  read as 'chận'       'hầu'   read as 'hần'
'lặng'  read as 'làng'       'cưa'   read as 'của'
```

**Cause:** the confused letters all have similar stroke shapes in handwriting. The group `r` - `v` - `l` differs only in stroke curvature and height. The group `m` - `n` - `u` differs in the number of arcs. When written quickly, these strokes overlap enough that a human reader could make the same mistake.

This is not a weakness specific to the model but a **limit on the information available in the image**. Distinguishing them requires context: knowing the neighboring word would make `lớn` more plausible than `lên`. But the model only sees one isolated word, with no context at all.

### Type 2: Wrong uppercase / lowercase (133 samples)

Most frequently confused pairs:

```
'C' read as 'c'   19 times      'T' read as 't'   18 times
'K' read as 'k'   14 times      'c' read as 'C'   13 times
'V' read as 'v'   10 times      's' read as 'S'    8 times
```

**First cause - missing information:** for letters like `C/c`, `K/k`, `V/v`, `S/s`, the uppercase and lowercase forms have **identical shapes** and differ only in size. When reading a full line, we distinguish them by comparing against the height of surrounding letters. But once a single word is cropped out of its line, that comparison is gone.

**Second cause - data imbalance:** only 11.2% of labels contain an uppercase letter. The model learned the statistical rule that "guessing lowercase is usually correct."

A signal suggesting the second cause: uppercase read as lowercase occurs 85 times, the reverse 48 times. Normalized by the number of samples at risk the gap is clearer: of 1,234 labels containing uppercase, 85 were read as lowercase (6.89%), while of 9,830 all-lowercase labels, 48 were read as containing uppercase (0.49%). A gap of roughly 14 times.

If missing height comparison were the only cause, errors would probably be more balanced between the two directions. However this remains **correlation, not causation** - the two phenomena co-occur but that does not prove one causes the other. Confirming it would require retraining with more balanced data and measuring again.

The second cause can be addressed by increasing the proportion of uppercase samples in the training data.

### Type 3: Wrong tone mark (124 samples)

Broken down by kind of mistake:

```
Mark dropped   (accented character read as unaccented)  : 28 times
Mark added     (unaccented character read as accented)  : 25 times
Mark confused  (one mark read as a different mark)      : 65 times
```

Most frequently confused mark pairs:

```
Hook above  read as acute      13 times
Dot below   read as no mark    11 times
Acute       read as grave      10 times
```

**First cause - marks are too small:** a tone mark occupies only a few pixels. The model splits the image into 16x16 pixel patches for processing, so a tone mark can fall entirely inside one patch, mixed together with the main letter strokes. The hook-above and acute marks are both short curved strokes above the letter, differing very little in shape.

**Second cause - insufficient data:** measuring the relationship between how often an accented character appears in the training data and how accurately it is read:

| Occurrences in training data | Characters | Average accuracy |
|---|---|---|
| Under 100 | 3 | 87.50% |
| 100 - 500 | 17 | 96.28% |
| 500 - 2,000 | 28 | 97.22% |
| Over 2,000 | 11 | 97.89% |

Over 10% difference. For example `ỡ` appears only 58 times, accuracy 62.5%. Meanwhile `à` appears 6,339 times, accuracy 98.6%.

This trend is consistent with the weakness noted in the [label analysis](#label-analysis) section, but the table has limits worth noting: the under-100 group contains only 3 characters, and most of the 27 characters appearing fewer than 50 times were excluded from the table because they occur too rarely in the test set to measure reliably. So the conclusion supported is that characters with fewer samples have noticeably lower accuracy, not a specific claim about each of those 27 characters.

The second cause can be addressed by adding data for rare characters. The first is harder and would require changing how the model processes images.

### Type 4: Missing or extra characters (106 samples)

Real examples:

```
'Trước' (5 chars)  read as 'Muốc' (4 chars)
'Camry' (5 chars)  read as 'Cam'  (3 chars)
'lặng'  (4 chars)  read as 'im'   (2 chars)
'vào'   (3 chars)  read as 'và'   (2 chars)
```

**Cause:** the model generates text auto-regressively, deciding on its own when a word ends. When strokes in the image are faint or run together, the model may stop early (missing characters) or keep generating (extra characters).

The case of `'lặng'` read as `'im'` reveals something further: the model does not read purely by shape but also draws on words it saw frequently during training. When an image is unclear, it tends to guess a common word rather than transcribe the actual strokes.

### Cause summary

Ordered by how addressable each cause is:

| Cause | Mainly affects | Addressable? |
|---|---|---|
| Data imbalance (11.2% uppercase) | Type 2 | Yes - increase uppercase sample ratio |
| Insufficient data for rare characters | Type 3 | Yes - add more data |
| Similar stroke shapes in handwriting | Type 1 | Hard - needs sentence context, unavailable at word level |
| Missing height comparison reference | Type 2 | Hard - needs a full line instead of one word |
| Tone marks smaller than processing patches | Type 3 | Hard - requires changing image processing |
| Faint images, strokes running together | Type 4 | Hard - limited by image quality |

The first three causes relate to data and can be improved. The last three are inherent limits of looking at a single isolated word.

---

## Web application

The model reads **one word** per image. The web app adds a step to locate and crop individual words, allowing testing with photographs containing multiple words.

```
Photograph
   |
   v
Locate words (EasyOCR or PaddleOCR)
   |
   v
Sort into reading order, crop each word with margin to preserve tone marks
   |
   v
TrOCR reads each word
   |
   v
Output: text + annotated image + detail table
```

Two Vietnamese-specific adjustments:

- **Vertical margin when cropping.** Detectors tend to crop tightly around the letter body, cutting off marks above (`ắ`, `ế`) and the dot below (`ậ`, `ợ`).
- **Separating ink from ruled paper.** The model was trained on clean white backgrounds with black ink, while photographs of real notebooks have blue ink and ruled lines. Handled by converting to HSV color space and separating by color saturation.

The app allows choosing the detector, adjusting detection sensitivity, crop margin, and decoding thoroughness.

---

## Installation

```bash
git clone https://github.com/Thaito98/vietnamese-handwriting-ocr.git
cd vietnamese-handwriting-ocr

# Large model files are managed with Git LFS
git lfs pull

python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux / macOS

pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501` in a browser, upload an image (JPG, PNG or BMP), and click **Nhận dạng** (Recognize).

**Requirements:** Python 3.9 or newer (tested on 3.10). The source uses type annotation syntax available only from Python 3.9, so it will not run on 3.8. An NVIDIA GPU makes it considerably faster; it runs on CPU but slowly.

If `models/best_model/model.safetensors` is only a few KB, Git LFS has not finished downloading. Run `git lfs pull` again.

---

## Project structure

```
ocr_web/
  TrOCR.ipynb                  Notebook for training and evaluating the model
  app.py                       Streamlit web app for testing
  requirements.txt             Required libraries

  models/best_model/           Fine-tuned model files (Git LFS)
    config.json                Model architecture configuration
    model.safetensors          Trained weights, 1.28 GB
    tokenizer.json             Text tokenizer
    processor_config.json      Input image processing configuration

  detectors/
    easyocr_det.py             EasyOCR detection wrapper
    paddle_det.py              PaddleOCR detection wrapper
```

---

## Limitations

| Limitation | Detail |
|---|---|
| Reads only one word per image | Multi-word images depend on an external detector to crop first. If cropping is wrong, the model cannot recover |
| 27 rare characters | Appear fewer than 50 times in training data, mostly uppercase with tone marks. Too few test samples to measure reliably, but the overall trend shows accuracy falling as sample count drops |
| Biased toward lowercase | Only 11.2% of training data contains uppercase, so the model often reads uppercase as lowercase |
| No spell correction step | Output is not checked against a Vietnamese dictionary |
| Training data from a tablet | Clean white background, even strokes. Photographs of real paper differ substantially, so real-world accuracy will be lower than the figures above |
| Tokenizer not optimized for Vietnamese | The model uses an English tokenizer, where an accented character such as `ạ` is split into 3 pieces |
| Large model | 334 million parameters, slow without a GPU |

---

## References

- Li et al. (2021). *TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models* - [arXiv:2109.10282](https://arxiv.org/abs/2109.10282)
- Dosovitskiy et al. (2020). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale* - [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
- Nguyen et al. (2018). *HANDS-VNOnDB: Vietnamese Online Handwriting Database* - Tokyo University of Agriculture and Technology
- [vnondb-extractor](https://github.com/vndee/vnondb-extractor) - tool for converting pen stroke data into images

---

## License

MIT
