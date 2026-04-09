# A Hybrid Dual-Channel Deep Learning Model for Acute Lymphoblastic Leukemia Detection

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)

## 1. Description
This repository contains the official codebase and implementation for the paper: **"A Hybrid Dual-Channel Deep Learning Model for Acute Lymphoblastic Leukemia Detection"**. 

The objective of this study is to classify Acute Lymphoblastic Leukemia (ALL) from microscopic blood smear images. Unlike many existing studies that suffer from "data leakage" due to internal validation splitting, this framework emphasizes strict sample-level isolation by evaluating on an independent cross-domain (target-domain) cohort. It offers a highly efficient alternative to computationally heavy ensemble models by fusing Convolutional Neural Networks (CNN) and Vision Transformers (ViT).

## 2. Dataset Information
The model was trained and evaluated using the publicly available **ISBI C-NMC (Classification of Normal versus Multiple Myeloma/Cancer) 2019 dataset**. 
* **Source Domain (Pre-training):** 10,661 images were used to teach the model foundational leukemia morphology.
* **Target Domain (Adaptation & Testing):** A completely separate cohort of 1,867 images was used to test cross-domain generalization. 20% of this was used for rapid adaptation (fine-tuning), and 80% (1,494 images) was kept strictly isolated for the final evaluation.

## 3. Code Information
The codebase is strictly modularized to ensure high empirical reproducibility under controlled settings:
* `model.py`: Defines the proposed dual-stream architecture (Hybrid-LeukemiaNet), fusing EfficientNet-B0 and ViT-B_16. It includes dynamic gating mechanisms for ablation studies.
* `dataset.py`: Contains custom PyTorch Dataset classes (`StandardDataset`, `TTADataset`) and data transformation pipelines.
* `run_experiments.py`: The primary orchestration script. It executes a rigorous sequence of experiments, logging detailed evaluation metrics (Loss, Accuracy, Precision, Recall, F1-Score, MCC).
* `visualize.py`: An automated script utilizing Matplotlib and Seaborn to generate publication-ready (300 DPI) figures (confusion matrices, learning curves, zero-shot radar charts, and qualitative prediction grids).

## 4. Requirements
The project is built using Python 3.12. The required libraries to run the code are listed below (and in the `requirements.txt` file):
* `torch` (>= 2.0.0)
* `torchvision` (>= 0.15.0)
* `numpy`
* `pandas`
* `scikit-learn`
* `matplotlib`
* `seaborn`
* `Pillow`

## 5. Usage Instructions
Follow these steps to replicate the study:

**Step 1: Data and Weight Preparation**
Due to file size limits, the pre-trained weights (approx. 350MB) and the dataset are hosted externally.
* Download the C-NMC 2019 dataset and place the validation data in: `./data/C-NMC_test_prelim_phase_data`
* Download the pre-trained weights (Exp6) from [Google Drive Link](https://drive.google.com/file/d/1_hXCYBV7vqRIHSCfNxF34MD7kycVTo87/view?usp=drive_link) and place the `.pth` file in the `./weights/` directory.

**Step 2: Install Dependencies**
```bash
pip install -r requirements.txt