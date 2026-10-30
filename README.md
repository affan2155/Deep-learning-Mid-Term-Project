# Deep Learning Midterm Project

Two deep learning models built with TensorFlow/Keras, each solving a real-world classification problem on a different dataset.

## Projects

### 1. DNN: Telco Customer Churn Prediction
Predicts whether a telecom customer will leave the service, so the company can target retention offers.
- **Dataset:** [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (7,043 customers, 20 features)
- **Approach:** Data cleaning, one-hot encoding, feature scaling and class weighting for the imbalanced target. The model is a fully connected network with BatchNorm, Dropout and L2 regularization.
- **Evaluation:** Accuracy, precision, recall, F1, ROC-AUC, compared against a logistic regression baseline.
- **Result:** XX% accuracy, 0.XX ROC-AUC

### 2. CNN: Brain Tumor MRI Classification
Classifies brain MRI scans into four classes (glioma, meningioma, pituitary tumor, no tumor) to support faster triage.
- **Dataset:** [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- **Approach:** Image resizing and normalization, data augmentation, and a custom CNN with four convolutional blocks. A MobileNetV2 transfer-learning model serves as a comparison.
- **Evaluation:** Test accuracy, per-class precision/recall/F1, confusion matrix.
- **Result:** XX% test accuracy

## Repository Contents
| File | Description |
|---|---|
| `Deep_Learning_Midterm_Colab.ipynb` | Both models in one Colab-ready notebook |
| `DNN_Telco_Churn.ipynb` | Standalone DNN notebook |
| `CNN_Brain_Tumor_MRI.ipynb` | Standalone CNN notebook |
| `Project_Document.pdf` | Full report |

## How to Run
1. Open the notebook in Google Colab.
2. For the CNN, set the runtime to GPU (Runtime > Change runtime type > T4 GPU).
3. Run the cells in order. The datasets download automatically via `kagglehub`; you may need a Kaggle API token.

## Tech Stack
Python, TensorFlow/Keras, scikit-learn, pandas, NumPy, Matplotlib, Seaborn

## Disclaimer
The brain tumor model is for educational purposes only and is not a medical diagnostic tool.
