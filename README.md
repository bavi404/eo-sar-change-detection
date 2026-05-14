# EO + SAR Binary Change Detection

## Setup

```bash
pip install -r requirements.txt
```

**Train**

```bash
python train.py
```

**Evaluate**

```bash
python eval.py
```

---

## Dataset structure

Expected layout for each split (aligned filenames across folders):

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

Paths are set in `config.yaml` (`data.train_root`, etc.), relative to the config file unless absolute. Nested archives (e.g. `train/train/pre-event/`) are supported by pointing `train_root` at the inner split folder.

---

## Repository

Pixel-wise **binary change detection** from aligned **pre- and post-event EO (optical) and SAR** imagery. Output: **`1` = change**, **`0` = no change**.

**Labels:** `0` and `1` → no change; `2` and `3` → change. Binary `0/1` masks are detected and left unchanged (`datasets/change_dataset.py`).

**Model:** **UNet++** (segmentation_models_pytorch), **ResNet34** encoder, **SCSE** decoder attention, **12** input channels when `dataset.concat_diff: true` (EO/SAR stacks plus **|post − pre|** magnitude channels for EO and SAR).

**Training:** **BCE with logits + Dice** with **`pos_weight`** (`utils/losses.py`). **Inference:** `sigmoid(logits)` then threshold; **TTA** in `eval.py` (average of original, horizontal-flip, and vertical-flip probabilities after sigmoid).

**Configs:** `config.yaml` is the **final reported configuration** (20 epochs, `lr: 2e-5`, `crop_size: 512`, frozen **`training.threshold: 0.20`**, **`evaluation.tune_threshold_on_val: false`**, **`evaluation.use_tta: true`**). `config_kaggle.yaml` is a path template for Kaggle mounts and working output.

| Path | Role |
|------|------|
| `train.py` | Training |
| `eval.py` | Metrics, thresholding, TTA, qualitative figures |
| `config.yaml` | Final hyperparameters and data paths (relative to config file) |
| `config_kaggle.yaml` | Kaggle-style absolute paths (edit mounts) |
| `requirements.txt` | Python dependencies |
| `datasets/` | Dataset, augmentations, remap |
| `models/` | SMP UNet++ builder |
| `utils/` | Losses, metrics, path resolution |
| `sample_results/` | Example prediction PNGs from `eval.py` (`evaluation.sample_results_dir`) |
| `TIME_LOG.md` | Time and compute log |
| `.gitignore` | Excludes data, checkpoints, virtual environments |

---

## Submission requirements

| Deliverable | Where |
|-------------|--------|
| **Public GitHub repository** | This repo: `train.py`, `eval.py`, `datasets/`, `models/`, `utils/`, `requirements.txt`, `config.yaml`, `README.md` as required. |
| **Technical report (PDF)** | Not in the repo; submit per assignment. |
| **Time / resource log** | `TIME_LOG.md` (or duplicate in the report if required). |
| **Public model weights** | **`best_model.pt` is not in Git** (see `.gitignore`). Host on **Google Drive**, **Hugging Face Hub**, or **Kaggle** with public access and paste the URL below. |

### Public weights link

| File | Public download link |
|------|------------------------|
| `best_model.pt` | https://drive.google.com/file/d/1HZzqEIf_qKFSD01ixrj0bMvwyYcm0P6J/view?usp=sharing |

---

## Reported results (final run)

The **final reported configuration** follows correction of a **label preprocessing / remapping** issue (four-class vs already-binary masks), **threshold calibration on validation** (then frozen at **0.20**), and **TTA-enhanced inference** (horizontal + vertical flip averaging after sigmoid in `eval.py`).

Aligned with **`config.yaml`** as committed: **`tune_threshold_on_val: false`**, **`evaluation.use_tta: true`**, **20 epochs**, **`lr: 2e-5`**, **`crop_size: 512`**, **`batch_size: 2`**, **`accumulation_steps: 2`**, **`pos_weight: 5.0`**.

| Split | IoU | F1 | Precision | Recall |
|-------|-----|-----|-----------|--------|
| Val | 0.243 | 0.391 | 0.530 | 0.309 |
| Test | 0.197 | 0.329 | 0.383 | 0.288 |

---

## Qualitative results

Representative prediction examples are written to **`sample_results/`** when you run **`python eval.py`** (path set by **`evaluation.sample_results_dir`** in `config.yaml`). Commit exported PNGs there for the report or submission bundle if required.

---

## References

1. Zhou et al., **UNet++**, MICCAI DLMIA 2018 — [arXiv:1807.10165](https://arxiv.org/abs/1807.10165)  
2. Ronneberger et al., **U-Net**, MICCAI 2015 — [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)  
3. **segmentation_models_pytorch** — [GitHub](https://github.com/qubvel/segmentation_models.pytorch)  
4. Roy et al., **Concurrent SE** (related to SCSE-style blocks), ECCV 2018  
5. Sudre et al., **Generalised Dice** loss, DLMIA 2017  

---