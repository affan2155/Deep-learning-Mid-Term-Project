# Deep Learning Midterm Project

Two deep learning models built with TensorFlow/Keras, each solving a real-world classification problem on a different dataset.

## Projects

### 1. DNN: Telco Customer Churn Prediction
Predicts whether a telecom customer will leave the service, so the company can target retention offers.

- **Dataset:** [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (7,043 customers, 21 columns: an ID, 19 input features and the churn label; about 26.5% of customers churned)
- **Approach:** Data cleaning, one-hot encoding (30 input features), feature scaling and class weighting for the imbalanced target. The model is a fully connected network (64-32-16 units) with BatchNorm, Dropout and L2 regularization, trained with early stopping.
- **Evaluation:** Accuracy, precision, recall, F1 and ROC-AUC on a held-out test set of 1,057 customers, compared against a logistic regression baseline.

**Results (test set):**

| Metric | DNN |
|---|---|
| Accuracy | 0.74 |
| Precision (churn class) | 0.51 |
| Recall (churn class) | 0.79 |
| F1-score (churn class) | 0.62 |
| ROC-AUC | 0.844 |

The DNN caught 221 of the 281 churners and missed 60, at the cost of 210 false alarms. Its ROC-AUC (0.844) was essentially tied with the logistic regression baseline (0.845), which is common on a small tabular dataset.

### 2. CNN: Brain Tumor MRI Classification
Classifies brain MRI scans into four classes (glioma, meningioma, pituitary tumor, no tumor) to support faster triage.

- **Dataset:** [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (7,023 images in four classes, with separate Training and Testing folders; the test set has 1,600 images, 400 per class)
- **Approach:** Images resized to 128x128 and normalized, with data augmentation (flip, rotation, zoom, contrast). A custom CNN with four convolutional blocks (32-64-128-256 filters), global average pooling and dropout. A MobileNetV2 transfer-learning model serves as a comparison.
- **Evaluation:** Test accuracy, per-class precision/recall/F1 and confusion matrix on the official Testing folder.

**Results (test set):**

| Model | Test accuracy |
|---|---|
| Custom CNN | 78.62% |
| MobileNetV2 transfer learning | 80.75% |

Per-class results for the custom CNN:

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| Glioma | 0.94 | 0.67 | 0.78 |
| Meningioma | 0.81 | 0.50 | 0.62 |
| No tumor | 0.73 | 0.99 | 0.84 |
| Pituitary | 0.75 | 0.98 | 0.84 |

The model recognizes no-tumor and pituitary scans best. Its main weakness is that many glioma and meningioma scans are predicted as no tumor or pituitary, so it is not reliable enough for real use.

## Repository Contents

| File | Description |
|---|---|
| `Deep_Learning_Midterm_Colab.ipynb` | Both models in one Colab-ready notebook |
| `DNN_Telco_Churn.ipynb` | Standalone DNN notebook |
| `CNN_Brain_Tumor_MRI.ipynb` | Standalone CNN notebook |
| `Project_Document.md` | Full project report |
| `images/` | Figures used in the report (confusion matrices, ROC curve, sample MRI images) |

## How to Run
1. Open the notebook in Google Colab.
2. For the CNN, set the runtime to GPU (Runtime > Change runtime type > T4 GPU).
3. Run the cells in order. The datasets download automatically via `kagglehub`; you may need a Kaggle API token.

## Tech Stack
Python, TensorFlow/Keras, scikit-learn, pandas, NumPy, Matplotlib, Seaborn

## Disclaimer
The brain tumor model is for educational purposes only and is not a medical diagnostic tool.
