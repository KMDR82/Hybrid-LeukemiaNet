# A Hybrid Dual-Channel Deep Learning Model for Acute Lymphoblastic Leukemia Detection

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)

## Overview
This repository contains the official PyTorch implementation for the paper: **"A Hybrid Dual-Channel Deep Learning Model for Acute Lymphoblastic Leukemia Detection"**. 

The objective of this study is to classify Acute Lymphoblastic Leukemia (ALL) from microscopic blood smear images. Unlike many existing studies that suffer from "data leakage" due to internal validation splitting, this framework emphasizes strict sample-level isolation by evaluating on an independent cross-domain (target-domain) cohort (1,494 images). It offers a highly efficient alternative to computationally heavy ensemble models by uniquely fusing:

1. **EfficientNet-B0 (CNN):** To extract local morphological features and cellular boundaries.
2. **Vision Transformer (ViT-B_16):** To capture the global cellular context.
3. **Test-Time Augmentation (TTA):** A dynamic inference strategy to enhance robustness against staining artifacts and asymmetrical cell orientations. (Note: The codebase supports evaluation both with and without TTA to ensure fair ablation comparisons against baseline models).

## Repository Structure
The codebase is strictly modularized to ensure high empirical reproducibility under controlled settings and ease of integration:

* `model.py`: Defines the `DynamicLeukemiaNet`, a custom dual-stream architecture fusing CNN and ViT. It includes dynamic gating mechanisms to facilitate architectural ablation studies.
* `dataset.py`: Contains custom PyTorch Dataset classes (`StandardDataset`, `TTADataset`) and comprehensive data transformation pipelines designed for both standard evaluation and Test-Time Augmentation.
* `run_experiments.py`: The primary orchestration script. It executes a rigorous sequence of experimental configurations, logging detailed evaluation metrics (Loss, Accuracy, Precision, Recall, F1-Score, MCC) to CSV formats.

### Visualization and Results Analysis
The repository includes an automated visualization script (`visualize.py`) utilizing Matplotlib and Seaborn to generate comprehensive, publication-ready (300 DPI) figures. The generated visual assets correspond to the figures presented in the manuscript:

* **Figure 1:** The proposed Hybrid Dual-Channel pipeline architecture (designed via Draw.io).
* **Figure 2:** A multi-metric radar chart providing a comparative analysis of the models across Accuracy, Precision, Recall, F1-Score, and Matthews Correlation Coefficient (MCC).
* **Figure 3:** Learning curves detailing the training and validation trajectories for both accuracy and loss over the experimental epochs.
* **Figure 4:** The confusion matrix illustrating the exact classification performance and error distribution on the independent target-domain cohort.
* **Figure 5:** A qualitative visual analysis grid demonstrating the predictive capabilities and feature alignment of the proposed Dual-Stream model on sample microscopic images.

## Data and Weight Preparation
Due to GitHub's file size limits, the pre-trained weights for the `DynamicLeukemiaNet` (approx. 350MB) and the dataset are hosted externally.

1. **Dataset:** Download the C-NMC 2019 dataset and place the validation data in `./data/C-NMC_test_prelim_phase_data`.
2. **Weights:** Download the pre-trained weights of the final proposed model (Exp6) from [https://drive.google.com/file/d/1_hXCYBV7vqRIHSCfNxF34MD7kycVTo87/view?usp=drive_link] and place the `.pth` file in the `./weights/` directory.



Ensure your paths in `run_experiments.py` point to these relative locations. For example:
```python
# Inside run_experiments.py
model_path = './weights/Exp6_Final_Model.pth' 
val_base_path = './data/C-NMC_test_prelim_phase_data'
csv_path = './data/C-NMC_test_prelim_phase_data_labels.csv'