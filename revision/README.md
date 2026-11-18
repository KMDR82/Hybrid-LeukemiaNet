# Revised evaluation protocol (PeerJ CS revision)

This folder replaces the original experiment scripts for the revision. Every
split is made at the **subject (patient) level**, and the held-out test
subjects are **never** used to choose an epoch, a checkpoint or a threshold.

| Reviewer request | Where it is handled |
| --- | --- |
| Subject-level splits, subject counts and IDs | `make_splits.py` -> `splits/*.csv`, `split_summary.json` |
| Subject-level 5-fold CV, no "best fold" selection | `train_source.py` (final model retrained on all 73 subjects, epoch count fixed from CV) |
| Validation data during adaptation, early stopping details | `adapt_target.py` (adapt-val subjects, val loss, patience 5, min_delta 0.001, max 30) |
| Independently trained CNN-only / ViT-only models | `models.py`, `train_source.py --arch cnn/vit` |
| Stream masking relabelled as secondary analysis | `adapt_target.py` (masking column) |
| Paired TTA comparison, McNemar, bootstrap CIs | `analyze.py`, `metrics.py` |
| Class-specific metrics (sens., spec., PPV, NPV) | `metrics.py` |
| Calibration (ECE, Brier, reliability diagram) | `metrics.py`, `figures.py` |
| Threshold selection from training data only | `train_source.py` (Youden J from source out-of-fold predictions) |
| Params, FLOPs, GPU memory, latency | `profile_compute.py` |
| Grad-CAM / attention maps | `explain.py` |
| Evidence for or against a colour shift between phases | `color_stats.py` |
| Deterministic TTA, fixed seeds | `data.py`, `common.py` |

## 1. Install (once)

**Windows** (Anaconda Prompt) or **Linux** (terminal):

```
conda create -n leuk python=3.11 -y
conda activate leuk
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r revision/requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())"
```

The last line must print `True` and your RTX card name.

## 2. Data

Download the Kaggle mirror of C-NMC 2019 (`andrewmvd/leukemia-classification`)
and unzip it. `--data-root` is the folder that **contains** `training_data` and
`validation_data` (usually `.../C-NMC_Leukemia`). The script checks that it finds
73 + 28 subjects and 10,661 + 1,867 images and stops otherwise.

## 3. Run everything (overnight)

From the repository root:

```
python revision/run_all.py --data-root "D:/data/C-NMC_Leukemia"
```

* Resume: if it stops (crash, power cut), run **the same command** again;
  finished folds, models and repeats are skipped.
* Out of GPU memory: add `--source-batch 16`.
* DataLoader errors on Windows: add `--workers 0`.
* Progress: `revision/results/run_all.log` and one log per stage.

## 4. What to send back

Zip `revision/splits` and `revision/results` **without** the `*.pt` weight files.
The weight files (`results/source/*/final_source_model.pt`,
`results/target/adapted_r0_*.pt`) should be kept for the Zenodo archive.
