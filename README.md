# EO + SAR Binary Change Detection

## 1. Project title and description

This repository implements **pixel-wise binary change detection** for disaster / damage mapping using **aligned pre-event and post-event remote sensing**.

### EO / SAR binary change detection

- **Inputs:** For each location and time pair we use **EO (optical)** and **SAR** from **before** and **after** an event: **EO pre**, **EO post**, **SAR pre**, **SAR post**.
- **Output:** A dense **binary mask** per pixel: **`1` = change**, **`0` = no change**.
- **Labels:** Original masks are remapped as required: **`0` and `1` → no change (`0`)**, **`2` and `3` → change (`1`)**. If a split already stores **binary `0/1` masks**, the implementation **preserves** them (no destructive remap).

### Architecture

- **Backbone:** **[UNet++](https://arxiv.org/abs/1807.10165)** from **[segmentation_models_pytorch (SMP)](https://github.com/qubvel/segmentation_models.pytorch)** with **`resnet34`** encoder.
- **Decoder attention:** **`scse`** (squeeze-and-excitation style **channel + spatial** attention blocks in the decoder), configured via SMP as `decoder_attention_type="scse"`.
- **Head:** Single-channel **logits**; at inference **`sigmoid(logits)`** then **thresholding** (never threshold raw logits).

### Key ideas

1. **Early fusion (multi-channel stack):** Concatenate EO and SAR time steps so one network learns **joint** appearance and temporal cues.
2. **Difference fusion:** Append **`|EO_post − EO_pre|`** (3 channels) and **`|SAR_post − SAR_pre|`** (1 channel) so the model sees **explicit change magnitude** as well as raw inputs (**12 channels** when `dataset.concat_diff: true`).
3. **Class imbalance:** **BCE with logits + Dice**, with **`pos_weight`** on the positive class in the BCE term, to better handle **rare change pixels**.
4. **Threshold (frozen for submission):** After calibration on validation, the final decision threshold is **`training.threshold: 0.20`** with **`evaluation.tune_threshold_on_val: false`** so evaluation is reproducible. **TTA** (horizontal + vertical flip averaging of **sigmoid** probabilities) is enabled via **`evaluation.use_tta: true`** and implemented in **`eval.py`** — it is not encoded in the threshold number alone.
5. **Optional training extras:** `train.py` still supports **AdamW**, **ReduceLROnPlateau**, and **early stopping** if you add those keys back to a copy of the config; the **committed `config.yaml`** matches the **final reported run** without those extras.

---

## 2. Requirements

### Python version

- **Python 3.10** (recommended; **3.11** is also commonly used).

### Dependencies

Install everything from **`requirements.txt`**. Core packages include:

| Package | Role |
|---------|------|
| `torch`, `torchvision` | Training / inference |
| `segmentation-models-pytorch` | **UNet++** + **ResNet34** + **SCSE** decoder |
| `albumentations` | Augmentations (crop, flip, rotate) |
| `numpy`, `tifffile` | GeoTIFF I/O |
| `matplotlib` | Qualitative figures |
| `tqdm`, `PyYAML`, `scikit-learn` | Progress, config, metrics helpers |

Pinned versions are listed in **`requirements.txt`**. If the pinned **`torch`** wheel does not match your CUDA/CPU, install a compatible build from [pytorch.org](https://pytorch.org/get-started/locally/) and adjust pins if your course allows.

---

## 3. Environment setup

From the repository root (the folder that contains `train.py` and `config.yaml`):

```bash
python -m venv galaxeye
```

**Windows (PowerShell):**

```powershell
.\galaxeye\Scripts\activate
```

**Linux / macOS:**

```bash
source galaxeye/bin/activate
```

Then:

```bash
pip install -r requirements.txt
```

**Alternative (conda):**

```bash
conda create -n galaxeye python=3.10 -y
conda activate galaxeye
pip install -r requirements.txt
```

---

## 4. Dataset structure

Place data under **`./data/`** (paths in `config.yaml` are **relative to the directory containing `config.yaml`**). Each split must have **aligned** GeoTIFFs with the **same filenames** in all three folders.

```text
train/
  pre-event/
  post-event/
  target/

val/
  pre-event/
  post-event/
  target/

test/
  pre-event/
  post-event/
  target/
```

If your archive unpacks with an extra nesting (e.g. `train/train/pre-event/`), set `data.train_root` in **`config.yaml`** accordingly (e.g. `./data/train/train`).

**Do not commit** large rasters or `best_model.pt` to Git (see `.gitignore`).

---

## 5. Training command

```bash
python train.py
```

Optional custom config:

```bash
python train.py --config path/to/config.yaml
```

Training uses **`data.train_root`** and **`data.val_root` only** — the **test** split is never loaded during training.

---

## 6. Evaluation command

```bash
python eval.py
```

```bash
python eval.py --config path/to/config.yaml
```

Loads **`{training.output_dir}/best_model.pt`**, runs metrics on **validation** and **test**.

- If **`evaluation.tune_threshold_on_val: true`**, searches a threshold grid on val and writes **`best_threshold.json`**.
- **Official final setup:** **`tune_threshold_on_val: false`** and **`training.threshold: 0.20`** (frozen after you calibrated on val offline). **`evaluation.use_tta: true`** turns on flip TTA inside **`eval.py`** (average of **original + H-flip + V-flip** sigmoid probabilities).

Saves **≥5** qualitative PNGs under **`{training.output_dir}/visualizations/`**.

**Kaggle:** use **`config_kaggle.yaml`** (edit mount paths) or pass it explicitly:

```bash
python eval.py --config config_kaggle.yaml
```

## 7. Model weights (public link — required)

**`best_model.pt` is not stored in this repository** (too large for Git). You **must** host weights on a **public** service and paste the link below.

**Accepted options (pick one):**

- **Google Drive** — share “Anyone with the link” and paste URL here.  
- **Hugging Face Hub** — upload as a model file / repo and paste URL here.  
- **Kaggle** — Dataset or Model output with public access; paste URL here.

| File | Public download link |
|------|----------------------|
| **`best_model.pt`** | **REPLACE THIS:** [your public Drive / Hugging Face / Kaggle URL](https://example.com) |

After training, upload `models/checkpoints/best_model.pt` (or your Kaggle output path) and **replace** the placeholder above before submission.

---

## 8. Results table (final run)

**Reproducibility (read this):** The table below was produced with **`config.yaml`** exactly as committed:

- **Label remap:** fixed logic in `datasets/change_dataset.py` (4-class `0,1→0`, `2,3→1`; **binary `0/1` masks preserved**).
- **Threshold:** **`training.threshold = 0.20`** with **`evaluation.tune_threshold_on_val: false`** (frozen; no grid search at eval time).
- **TTA:** **`evaluation.use_tta: true`** — horizontal + vertical flip **probability averaging after `sigmoid`** in **`eval.py`** (not stored as a separate number in YAML).

Training/eval hyperparameters match that run: **20 epochs**, **`lr: 2e-5`**, **`crop_size: 512`**, **`batch_size: 2`**, **`accumulation_steps: 2`**, **`pos_weight: 5.0`**, **UNet++ / ResNet34 / SCSE / 12 channels**.

Metrics below are from the **best submission run** (e.g. **20 epochs** on Kaggle, **CUDA**).

| Split | IoU | F1 | Precision | Recall |
|-------|-----|-----|-----------|--------|
| **Val** | 0.243 | 0.391 | 0.530 | 0.309 |
| **Test** | 0.197 | 0.329 | 0.383 | 0.288 |

**Notes**

- **Threshold:** `0.20` (frozen in `config.yaml` as `training.threshold`).
- **TTA:** enabled via `evaluation.use_tta` + logic in `eval.py` (see above).

---

## 9. References and resources

1. **UNet++** — Zhou, Zongwei, et al. *“UNet++: A Nested U-Net Architecture for Medical Image Segmentation.”* MICCAI DLMIA workshop, 2018. [arXiv:1807.10165](https://arxiv.org/abs/1807.10165)  
2. **U-Net** (foundational encoder–decoder) — Ronneberger, Olaf, et al. *MICCAI* 2015. [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)  
3. **segmentation_models_pytorch (SMP)** — Qubvel et al. GitHub: [https://github.com/qubvel/segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch) (UNet++, encoders, pretrained weights API).  
4. **SCSE attention in SMP decoders** — Uses **Concurrent Spatial and Channel Squeeze & Excitation**-style blocks as `decoder_attention_type="scse"` (see SMP documentation / timm integration). Related reading: Roy et al., *“Recalibrating Fully Convolutional Networks…”* (Concurrent SE), ECCV 2018.  
5. **Dice / overlap losses for imbalance** — Sudre, Carole H., et al. *“Generalised Dice overlap as a deep learning loss function for highly unbalanced segmentations.”* DLMIA 2017.  
6. **PyTorch** — `BCEWithLogitsLoss` / `binary_cross_entropy_with_logits` for training; **`sigmoid`** at inference before thresholding.

---

## Repository layout (course checklist)

| Path | Role |
|------|------|
| `train.py` | Training |
| `eval.py` | Metrics + visualizations |
| `config.yaml` | Official final hyperparameters + relative data paths |
| `config_kaggle.yaml` | Example Kaggle absolute paths + `/kaggle/working/...` output (edit mounts) |
| `requirements.txt` | Dependencies |
| `datasets/` | Dataset + augmentations |
| `models/` | SMP model builder |
| `utils/` | Losses, metrics, path resolution |
| `.gitignore` | Excludes data, checkpoints, venv |

---

## Reproducibility

- **`training.seed`** in `config.yaml` seeds Python / NumPy / PyTorch in `train.py`.

---

## License / data use

Use only the **provided** train / val / test splits. Do **not** tune hyperparameters or thresholds using **test** labels.
