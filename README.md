# Multimodal Skin Lesion Classification

This project implements a robust machine learning pipeline to classify skin lesions as **Benign** or **Malignant**. It leverages a multimodal approach, combining computer vision techniques for dermoscopic image analysis with tabular clinical metadata.

## 🚀 Overview

The project is split into two primary phases:

1. **Advanced Preprocessing**: Cleaning and segmenting raw dermoscopic images to isolate lesions.
2. **Multimodal Modeling**: Extracting handcrafted features and training a regularized XGBoost classifier optimized for imbalanced medical data.

---

## 🛠️ Pipeline Architecture

### 1. Image Preprocessing & Segmentation

Located in `preprocessing.ipynb`, this stage transforms raw images into high-quality inputs:

* **Hair Removal**: Adaptive black-hat morphological filtering to eliminate hair artifacts.
* **Color Correction**: Red-channel adjustment to normalize skin tones across different lighting conditions.
* **Lesion Segmentation**: Implementation of the **GrabCut algorithm** to generate binary masks and isolate the lesion of interest.
* **Data Balancing**: Strategic undersampling of the majority class (`nv`) and augmentation of minority classes to handle extreme class imbalance.

### 2. Feature Engineering

The model does not rely on raw pixels alone; instead, it extracts rich diagnostic features:

* **Texture**: Gray-Level Co-occurrence Matrix (GLCM) and Local Binary Patterns (LBP).
* **Shape**: Region-based properties (eccentricity, area, perimeter) from segmentation masks.
* **Color**: Statistical descriptors across RGB and HSV color spaces.
* **Metadata**: Integration of patient clinical data (age, sex, anatomical site).

### 3. Machine Learning Workflow

Located in `multimodal-skin-class-10.ipynb`:

* **Feature Selection**: High-dimensional features are filtered using **ANOVA F-scores**, **Mutual Information**, and **RFECV** (Recursive Feature Elimination).
* **Binary Mapping**: The original 7-class dataset is mapped into a binary classification problem:
* **Malignant**: Melanoma (MEL), Basal Cell Carcinoma (BCC), Actinic Keratoses (AKIEC).
* **Benign**: Nevus (NV), Benign Keratosis (BKL), Dermatofibroma (DF), Vascular Lesions (VASC).


* **Classification**: An **XGBoost** model is utilized with specialized parameters:
* `scale_pos_weight` to address remaining class imbalance.
* L1/L2 regularization to prevent overfitting.
* Threshold optimization using **Youden’s J Statistic** to maximize sensitivity.



---
## 📊 Results

| Split | AUC | Weighted F1 |
|------|------|-------------|
| Train | 0.9286 | 0.8145 |
| Validation | 0.8057 | 0.7128 |
| Test | 0.8579 | 0.7491 |


