# Real vs. AI-Generated Face Classification

> **Project Status: Discontinued / Archived — Open for Community Continuation**
>
> Active development of this project has been discontinued due to **training data insufficiency for modern generative AI models** (particularly **xAI Grok / FLUX** and **Ideogram**). The codebase, training pipeline, data-collection scripts, and the best-performing baseline checkpoint (`weights/Face_Detector_v1_Gold.pth`) are open-sourced under the MIT License so anyone interested can fork and continue the work.You can use this link to get the dataset I used:https://huggingface.co/datasets/Eren8705/face_classification_project

---

## Overview

This repository contains a binary image classification pipeline designed to distinguish **real human faces** (`class 1`) from **AI-generated / synthetic faces** (`class 0`).

The pipeline combines:
1. **MTCNN (`facenet-pytorch`)** for face detection and tight facial cropping (forcing the classifier to evaluate facial features rather than backgrounds or clothing).
2. **ResNet-18 (`torchvision`)** fine-tuned via transfer learning on nearly **99,000 MTCNN-cropped faces**.
3. **Streamlit** web application (`app.py`) for interactive single-image testing with side-by-side face cropping and confidence visualization.

---

## Datasets Used

Due to GitHub storage constraints (tens of gigabytes of image data), the raw and cropped datasets are not included in this repository. During development, the final cropped training/validation pool (`Data_cropped/`) reached **98,797 MTCNN-cropped face images** across two classes:

### 1. Real Human Faces (`real_faces` — 61,968 images)

| Source Dataset | Cropped Count | Notes & Purpose |
| --- | ---: | --- |
| **FFHQ (Flickr-Faces-HQ)** | 51,992 | Everyday, in-the-wild human portraits across diverse ages, lighting conditions, and backgrounds. |
| **CelebA-HQ (256x256)** | 9,976 | Extracted from a 10,000-image subset via `scripts/process_celeba.py`. Added specifically to fix domain shift where studio-lit or celebrity portraits were mistakenly flagged as AI-generated. |
| **Total Real Faces** | **61,968** | |

### 2. AI-Generated / Fake Faces (`fake_faces` — 36,829 images)

| Source / Generator | Cropped Count | Notes & Purpose |
| --- | ---: | --- |
| **GAN Face Collections (StyleGAN / ThisPersonDoesNotExist & Kaggle GAN sets)** | 24,514 | Baseline synthetic human portraits (`14,630` numbered images + `9,884` UUID-named GAN images). |
| **StyleGAN3** | 2,001 | Higher-fidelity GAN-generated faces (`styleGAN3_*.jpg`). |
| **Stable Diffusion (768px & 1024px)** | 5,987 | Downloaded from Kaggle (`mohannadaymansalah/stable-diffusion-dataaaaaaaaa`) via `scripts/download_dataset.py` and cropped with `scripts/process_new_fakes.py` (`2,987` @ 768px + `3,000` @ 1024px). |
| **Modern AI Multi-Face Collages (GPT-Image, Gemini, Grok/FLUX, Ideogram)** | 4,327 | Extracted from **44 custom multi-face collages** using multi-face MTCNN (`scripts/extract_collage_faces.py`) to expose the model to modern frontier generators. |
| **Total Fake Faces** | **36,829** | |

### Additional Data-Collection Scripts Included
The `scripts/` directory also preserves helper utilities written during data exploration:
- `scripts/download_gpt_grok.py`: Disk-safe streaming extractor (capped at 10,000 samples) for Kaggle's `shahzaibshazoo/detect-ai-generated-faces-high-quality-dataset`.
- `scripts/generate_hf_fakes.py`: Prompt-randomized synthetic face generator using HuggingFace Inference API (`black-forest-labs/FLUX.1-schnell`).

---

## Architecture & Training Pipeline

- **Face Extraction (`facenet-pytorch` MTCNN):** Crops tightly around detected faces (`prob >= 0.90` for bulk preprocessing) so the classifier does not overfit on background cues or watermarks.
- **Backbone (`model.py`):** Pretrained `ResNet18` (`ResNet18_Weights.DEFAULT`) with a 2-class linear classification head (`0 = Fake`, `1 = Real`).
- **Data Augmentation (`train.py`):**
  - Resize to `224x224`, random horizontal flip (`p=0.5`), random rotation (`-10` to `+10` degrees), and color jitter.
  - **Random Gaussian Blur (`p=0.25`, `kernel_size=5`)** to break the "sharpness shortcut" where sharp real photos and slightly softer images were separated superficially.
- **Imbalance Handling:**
  - Dynamic `CrossEntropyLoss` class weights computed automatically from folder counts (`[num_real / num_fake, 1.0]`, e.g., `1.68` Fake vs. `1.00` Real on the final 36,829 vs. 61,968 split).
  - Optional interactive **3x oversampling** (`WeightedRandomSampler`) for `collage_` samples so modern AI faces appear 3x more often per epoch.
- **Optimization:** Adam (`lr=0.001`), batch size `32`, 80/20 train/validation split, checkpoint resumption (`checkpoint.pth`).

---

## Evaluation & Benchmarks

Models were evaluated on a held-out 40-image out-of-distribution test set (`20 real` portraits including everyday and celebrity photos + `20 fake` portraits spanning Gemini, GPT, Grok, Ideogram, and ThisPersonDoesNotExist). MTCNN detected faces in **38 / 40** images.

### Checkpoint Comparison

| Metric | `Face_Detector_v1_Gold.pth` (Included Baseline) | Latest 3x-Oversampled `best_model.pth` |
| --- | ---: | ---: |
| **Accuracy** | **76.32%** | 76.32% |
| **Precision (Real)** | 70.83% | **72.73%** |
| **Recall (Real)** | **89.47%** | 84.21% |
| **F1 Score (Real)** | **0.7907** | 0.7805 |

### Per-Generator Fake Detection Breakdown (Latest Run)

| Generator | Detected / Total | Accuracy |
| --- | :---: | ---: |
| **Google Gemini** | 4 / 4 | **100.00%** |
| **ThisPersonDoesNotExist (StyleGAN)** | 5 / 6 | **83.33%** |
| **ChatGPT / GPT-Image** | 3 / 4 | **75.00%** |
| **Ideogram** | 1 / 2 | **50.00%** |
| **xAI Grok (FLUX)** | 0 / 3 | **0.00%** |

---

## Why This Project Was Discontinued

1. **Severe Data Insufficiency for Frontier Models:**
   While tens of thousands of legacy GAN images (StyleGAN/TPDNE) and Stable Diffusion samples are publicly available, gathering diverse, unwatermarked portraits from modern proprietary or flow-based generators (**Grok/FLUX**, **Ideogram**, **GPT-Image**) had to be done manually via multi-face collages under strict free-tier API/credit limits. Even after expanding to 4,327 collage faces, modern frontier generators represented only ~11% of the fake pool.
2. **Real-vs-Fake Realism Trade-off & Architectural Bottleneck:**
   Modern generators like FLUX/Grok produce skin textures and lighting nearly indistinguishable from real photography at RGB pixel level. Pushing the `ResNet-18` classifier harder on those samples (via class weighting and 3x oversampling) caused it to over-flag challenging real human portraits as fake (dropping Real Recall from **89.47%** to **84.21%**) without improving Grok detection (**0/3**).
3. **Storage & Diminishing Returns:**
   The local raw + cropped datasets grew to ~250,000+ images consuming tens of gigabytes of disk space while hitting an accuracy plateau around ~76% on modern OOD generators.

---

## Repository Structure

```text
face_classification_project/
├── app.py                        # Streamlit web UI for real-time inference
├── compare_models.py             # Head-to-head checkpoint comparison script
├── evaluate_model.py             # Detailed per-generator evaluation & confusion matrix
├── model.py                      # ResNet-18 FaceClassifier architecture
├── train.py                      # Training loop with class weights & 3x collage oversampling
├── requirements.txt              # Python dependencies
├── LICENSE                       # MIT License
├── weights/
│   └── Face_Detector_v1_Gold.pth # Best baseline checkpoint (~42.7 MiB)
└── scripts/
    ├── data_analysis.py          # Side-by-side dataset sample visualizer
    ├── download_dataset.py       # Kaggle Stable Diffusion dataset downloader
    ├── download_gpt_grok.py      # Kaggle high-quality AI faces downloader (10k cap)
    ├── extract_collage_faces.py  # Multi-face MTCNN collage extractor
    ├── generate_hf_fakes.py      # FLUX.1-schnell synthetic face generator (HF API)
    ├── noise_detection.py        # Block-entropy image noise analyzer
    ├── photocropper.py           # Bottom watermark cropping helper
    ├── preprocess_data.py        # Bulk MTCNN face cropper for Data/ -> Data_cropped/
    ├── process_celeba.py         # CelebA-HQ 256x256 MTCNN face extractor
    └── process_new_fakes.py      # Resolution-folder (512/768/1024) MTCNN face extractor
```

---

## Quickstart

### 1. Installation

```bash
git clone https://github.com/Eren386/face_classification_project.git
cd face_classification_project
pip install -r requirements.txt
```

*(For NVIDIA GPU acceleration, install the CUDA-enabled PyTorch wheel matching your driver from [pytorch.org](https://pytorch.org/get-started/locally/).)*

### 2. Run the Streamlit Web Demo (Ready Out of the Box)

The repository includes the baseline checkpoint at `weights/Face_Detector_v1_Gold.pth`:

```bash
streamlit run app.py
```

### 3. Prepare Your Own Dataset & Train

Organize your MTCNN-cropped training images into:

```text
Data_cropped/
├── fake_faces/
└── real_faces/
```

Use the helpers in `scripts/` (`preprocess_data.py`, `process_celeba.py`, `extract_collage_faces.py`, etc.) to extract faces from raw datasets, then run:

```bash
python train.py --data-dir Data_cropped --epochs 5 --batch-size 32
```

### 4. Evaluate & Compare Checkpoints

Organize test images into `test_data/fake/` and `test_data/real/`, then run:

```bash
python evaluate_model.py --model weights/Face_Detector_v1_Gold.pth --test-dir test_data
python compare_models.py --model-a weights/Face_Detector_v1_Gold.pth --model-b best_model.pth --test-dir test_data
```

---

## Ideas for Contributors Looking to Continue

If you want to pick up this project and break past the ~76% OOD ceiling:

1. **Scale Modern Generative Data:** Replace manual collage collection with large-scale open synthetic benchmarks (e.g., **GenImage**, **WildFake**, **DiffusionDB**, or bulk local **FLUX.1** / **SDXL** generation) to match the tens of thousands of real FFHQ/CelebA-HQ portraits.
2. **Upgrade the Backbone:** Swap `ResNet-18` in `model.py` for a higher-capacity architecture such as **ResNet-50**, **ConvNeXt-Small/Base**, **EfficientNet-B4**, or a **CLIP ViT** vision encoder (linear probe / LoRA fine-tuning).
3. **Add Frequency-Domain Signals:** Modern diffusion/flow artifacts are often invisible in RGB space but detectable in **FFT / DCT high-frequency spectra** or SRM noise residuals.
4. **Separate Validation Transforms:** In `train.py`, `random_split` currently wraps `full_dataset` with `train_transform` (including random blur/jitter) for both training and validation; refactor to wrap train and validation subsets with separate transforms for cleaner validation loss tracking.

---

## License

Released under the [MIT License](LICENSE).
