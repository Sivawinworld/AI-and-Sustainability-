
---

## **Project Overview**

This project implements and compares multiple tabular deep learning models (MLP, TabularNN, TabNet) for multi-class air quality prediction. It includes:

- Baseline model training  
- Compression via **quantization** and **knowledge distillation**  
- Evaluation using accuracy, loss, recall, F1-score, and confusion matrix  
- Visualization of training curves and performance metrics  
- Reproducible reporting and sustainability alignment

---

## **Model Architecture**

### 🔹 Baseline Models
- **MLP**: Multi-layer perceptron with ReLU activations  
- **TabularNN**: Custom architecture optimized for tabular data  
- **TabNet**: Attention-based interpretable model

### 🔹 Compressed Models
- **Quantized MLP**: Post-training full integer quantization using TensorFlow Lite  
- **KD Compressed MLP**: Student model trained via knowledge distillation from baseline MLP

---

## **Getting Started**

### 1. Prerequisites

- Python 3.8+
- Required Libraries:  
  `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `tensorflow`, `pytorch-tabnet`, `tensorflow-model-optimization`

### 2. Data Preparation

- Load and preprocess the AIRDATASET CSV files  
- Convert to `.npy` format for efficient training  
- Normalize features and encode labels

### 3. Model Training

Navigate to `notebooks/training/` and run:

- `train_mlp.ipynb`  
- `train_tabularnn.ipynb`  
- `train_tabnet.ipynb`

Each notebook includes training, validation, and saving the model.

### 4. Model Compression

#### 🔸 Quantization
- Run `quantize_mlp.py` to apply full integer quantization  
- Save as `.tflite` model

#### 🔸 Knowledge Distillation
- Run `kd_train_student.py` to train a compressed student model  
- Save as `.h5` model

### 5. Evaluation

Navigate to `notebooks/evaluation/` and run:

- https://drive.google.com/file/d/1t7IDJY7vzsxZmAxZRorFL1gKPvtlONui/view?usp=sharing 
- https://drive.google.com/file/d/1Jj4c0O8K_dM-LN0ZEv78f5v5SI7Z_xKg/view?usp=sharing
- https://drive.google.com/file/d/1m8yXpPqkTBAOOCoTMY_ZViD5VIb1_BRh/view?usp=sharing

These notebooks generate:

- Accuracy, loss, recall, F1-score  
- Confusion matrix heatmaps  
- Training curves (accuracy/loss over epochs)

---

## **Saved Models and Results**

All trained models and evaluation outputs are saved in:

- `https://drive.google.com/drive/folders/1914mlH60VV8prCUSCHiikINQYT2iSskd?usp=sharing 


You can also access pre-trained models and result files from this [Google Drive link](https://drive.google.com/drive/folders/1EVwDeH4uYA7b9Cr0A9ucRMZq-ZJxSear?usp=sharing).

---

## **Sustainability Alignment**

This project contributes to **UNSDG Goal 11: Sustainable Cities and Communities** by enabling scalable, interpretable air quality monitoring using compressed AI models suitable for edge deployment.
