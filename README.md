# A Hybrid Dual-Channel Deep Learning Model for Acute Lymphoblastic Leukemia Detection

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)

## Overview
This repository contains the official PyTorch implementation for the paper: **"A Hybrid Dual-Channel Deep Learning Model for Acute Lymphoblastic Leukemia Detection"**. 

The objective of this study is to classify Acute Lymphoblastic Leukemia (ALL) from microscopic blood smear images. Unlike many existing studies that suffer from "data leakage" due to internal validation splitting, this framework emphasizes **zero data leakage** by strictly evaluating on an unseen, independent **External Validation** cohort (1,494 images).

The proposed architecture uniquely fuses:
1. **EfficientNet-B0 (CNN):** To extract local morphological features and cellular boundaries.
2. **Vision Transformer (ViT-B_16):** To capture the global cellular context.
3. **Test-Time Augmentation (TTA):** A dynamic inference strategy to enhance robustness against staining artifacts and asymmetrical cell orientations.

## Repository Structure
The codebase is strictly modularized to ensure reproducibility and ease of integration:

* `model.py`: Defines the `DynamicLeukemiaNet`, a custom dual-stream architecture fusing CNN and ViT. It includes dynamic gating mechanisms to facilitate architectural ablation studies.
* `dataset.py`: Contains custom PyTorch Dataset classes (`StandardDataset`, `TTADataset`) and comprehensive data transformation pipelines designed for both standard evaluation and Test-Time Augmentation.
* `run_experiments.py`: The primary orchestration script. It executes a rigorous sequence of experimental configurations, logging detailed evaluation metrics (Loss, Accuracy, Precision, Recall, F1-Score, MCC) to CSV formats.
* `visualize.py`: An automated visualization script utilizing Matplotlib and Seaborn to generate publication-ready (300 DPI) figures, including dual-axis performance bar charts and Confusion Matrices.

## Data and Weight Preparation
Due to GitHub's file size limits, the pre-trained weights for the `DynamicLeukemiaNet` (approx. 350MB) and the dataset are hosted externally.

1. **Dataset:** Download the C-NMC 2019 dataset and place the validation data in `./data/C-NMC_test_prelim_phase_data`.
2. **Weights:** Download the pre-trained champion model weights from [https://drive.google.com/file/d/1_hXCYBV7vqRIHSCfNxF34MD7kycVTo87/view?usp=drive_link] and place the `.pth` file in the `./weights/` directory.



Ensure your paths in `run_experiments.py` point to these relative locations. For example:
```python
# Inside run_experiments.py
model_path = './weights/Exp6_Final_Model.pth' 
val_base_path = './data/C-NMC_test_prelim_phase_data'
csv_path = './data/C-NMC_test_prelim_phase_data_labels.csv'