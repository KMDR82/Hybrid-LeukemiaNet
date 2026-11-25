# Hybrid-LeukemiaNet: dual-stream CNN–Vision Transformer for ALL cell classification under subject-level evaluation

Code, subject-level split files, result tables and figures for the manuscript
**"A hybrid dual-stream CNN–Vision Transformer model for acute lymphoblastic leukemia cell classification under subject-level evaluation"**
(PeerJ Computer Science, CS-2026:03:136489, revised version).

Archived version (code, splits, per-image predictions, logs and model weights):
**https://doi.org/10.5281/zenodo.24528926**

## What the study does

A dual-stream model concatenates EfficientNet-B0 and ViT-B/16 features before a shared classifier. It is compared with EfficientNet-B0 and ViT-B/16 trained independently under the same protocol, on C-NMC 2019 with subject-level separation throughout:

* **Source phase:** subject-level 5-fold cross-validation on the training phase (73 subjects, 10,661 cells), then one final model on all 73 subjects.
* **Target phase:** the preliminary-test phase (28 subjects, 1,867 cells) is split five times into adaptation-train (4 subjects), adaptation-validation (2) and test (22) subjects. Models are evaluated without adaptation (zero-shot) and after limited adaptation, with and without test-time augmentation (TTA).
* Test subjects are never used to select epochs, checkpoints or thresholds.

## Repository layout

| Path | Contents |
| --- | --- |
| `revision/` | All code (see [`revision/README.md`](revision/README.md) for installation and use) |
| `revision/splits/` | Subject-level split files: image index of both phases, the five target-phase splits and a summary |
| `revision/results/tables/` | Result tables behind the manuscript (metrics per split, paired tests, calibration, compute cost, colour statistics) |
| `figures/` | Manuscript figures; `figures/drawio/` holds editable draw.io and SVG sources of Figures 1 and 2 |

Main scripts in `revision/`:

* `make_splits.py`: subject-level splits from the C-NMC file names
* `train_source.py`: source-phase cross-validation (`StratifiedGroupKFold`) and final source model
* `adapt_target.py`: target-phase adaptation with early stopping on adaptation-validation subjects
* `analyze.py`, `metrics.py`: class-specific metrics, subject-bootstrap CIs, McNemar tests, calibration
* `profile_compute.py`, `explain.py`, `color_stats.py`: compute cost, Grad-CAM/attention maps, colour comparison between phases
* `run_all.py`: runs the whole pipeline with one command; `run_variants.py`: pre-specified sensitivity analyses (Table S6)
* `figures.py`, `figures_drawio.py`, `render_figures.js`, `make_tex_tables.py`: figures and LaTeX tables

## Quick start

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python revision/run_all.py --data-root "/path/to/C-NMC_Leukemia"
```

`--data-root` is the folder that contains `training_data` and `validation_data` of the C-NMC 2019 dataset ([TCIA](https://doi.org/10.7937/tcia.2019.dc64i46r); Kaggle mirror `andrewmvd/leukemia-classification`). The dataset is not redistributed here.

## Figures

| File | Manuscript |
| --- | --- |
| `Figure1_protocol` | Figure 1: subject-level evaluation protocol |
| `Figure2_architecture` | Figure 2: dual-stream architecture |
| `Figure3_confusion.png` | Figure 3: confusion matrices of the adapted dual-stream model |
| `Figure4_backbones_and_masking.png` | Figure 4: independently trained backbones and stream masking |
| `Figure5_adaptation_curves.png` | Figure 5: adaptation loss and accuracy curves |
| `Figure6_reliability.png` | Figure 6: reliability diagram |
| `Figure7_explainability.png` | Figure 7: Grad-CAM and attention maps |
| `FigureS1_source_cv_curves.png` | Supplemental Figure S1: source-phase cross-validation curves |

## Earlier version

The first submission (image-level split) is preserved as release [v1.0.0](https://github.com/KMDR82/Hybrid-LeukemiaNet/tree/v1.0.0) and in its Zenodo record. Its results are superseded by this version.

## License

MIT (see `LICENSE`).
