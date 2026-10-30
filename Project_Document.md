# Deep Learning Midterm Project Report

**Student:** Affan Afzal Abbasi \[SP24-BBD-005\] \| **Course:** Deep Learning\[MGT388\]\| **Date:** \[30-10-2026\]

**Instructor:** \[Sir Saeed Ur Rehman\] \| **University:** \[COMSATS University Islamabad\] \|

## Project Overview

This project builds two deep learning models on two different datasets: a deep neural network (DNN) that predicts customer churn from tabular data, and a convolutional neural network (CNN) that classifies brain MRI images. Both models are implemented in Python with TensorFlow/Keras. The source code is in the notebook Deep_Learning_Midterm.ipynb (and the two standalone notebooks) in the GitHub repository: \[add repository link\].

# Part 1: DNN – Telco Customer Churn Prediction

## 1.1 Problem Definition

Customer churn is the loss of customers who stop using a company's service. In the telecommunications industry, where customers can switch providers easily, churn directly reduces revenue, and winning a new customer is generally more expensive than keeping an existing one. If a company can identify customers who are likely to leave, it can offer them targeted incentives such as discounts, plan upgrades or better support before they go.

The task is a binary classification problem: given a customer's demographic information, account details and subscribed services, predict whether the customer will churn (Yes) or stay (No). Because the goal is to find customers at risk, the model should be judged mainly on how many real churners it catches (recall) and on how well it ranks customers by risk (ROC-AUC), not on accuracy alone.

## 1.2 Dataset Description and Source Link

**Source:** Telco Customer Churn, Kaggle (uploaded by BlastChar), based on a sample dataset from IBM. **Link:** https://www.kaggle.com/datasets/blastchar/telco-customer-churn

The dataset contains 7,043 customer records and 21 columns: a customer identifier, 19 input features and the target column Churn. The features fall into three groups:

| Group               | Features                                                                                                                                |
|---------------------|-----------------------------------------------------------------------------------------------------------------------------------------|
| Demographics        | gender, SeniorCitizen, Partner, Dependents                                                                                              |
| Account information | tenure, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges                                                         |
| Subscribed services | PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies |

The target is imbalanced: about 26.5% of customers churned (1,869 of 7,043) and about 73.5% stayed. The column TotalCharges is stored as text and contains 11 blank values, all belonging to new customers with a tenure of 0 months.

## 1.3 Proposed Solution

The proposed solution is a fully connected feed-forward deep neural network (DNN) trained on the cleaned and encoded customer features, with a single sigmoid output that gives the probability of churn. To make the network reliable on this small tabular dataset, the solution combines several techniques:

- Class weighting, so the minority class (churners) is not ignored during training.

- Batch normalization, dropout and L2 regularization, to reduce overfitting.

- Early stopping and learning-rate reduction, so training stops when the validation loss stops improving.

- A logistic regression baseline, to check whether the DNN actually improves on a simple model.

## 1.4 Data Preprocessing

1.  **Remove the identifier:** customerID was dropped because it carries no predictive information.

2.  **Fix TotalCharges:** the column was converted from text to a number, and the 11 blank values were filled with 0 because they belong to customers with zero tenure.

3.  **Encode the target:** Churn was converted to 1 (Yes) and 0 (No).

4.  **Encode categorical features:** all text columns were one-hot encoded (with the first category dropped to avoid redundancy), which gives 30 numeric input features. \[Confirm the number against X_train.shape in the notebook.\]

5.  **Split the data:** a stratified split into 70% training, 15% validation and 15% test (about 4,930 / 1,056 / 1,057 customers) keeps the churn ratio the same in every subset.

6.  **Scale numeric features:** tenure, MonthlyCharges and TotalCharges were standardized. The scaler was fitted on the training set only and then applied to the validation and test sets, which prevents data leakage.

7.  **Handle imbalance:** balanced class weights were computed from the training labels and passed to the model during training.

## 1.5 Model Architecture

| Layer               | Details                                  |
|---------------------|------------------------------------------|
| Input               | 30 features                              |
| Dense 1             | 64 units, ReLU, L2 regularization (1e-4) |
| Batch Normalization |                                          |
| Dropout             | rate 0.3                                 |
| Dense 2             | 32 units, ReLU, L2 regularization (1e-4) |
| Batch Normalization |                                          |
| Dropout             | rate 0.3                                 |
| Dense 3             | 16 units, ReLU                           |
| Output              | 1 unit, sigmoid                          |

**Training configuration:**

| Setting                | Value                                               |
|------------------------|-----------------------------------------------------|
| Optimizer              | Adam, learning rate 0.001                           |
| Loss function          | Binary cross-entropy                                |
| Metrics                | Accuracy, AUC, precision, recall                    |
| Batch size             | 32                                                  |
| Maximum epochs         | 100                                                 |
| Early stopping         | monitor val_loss, patience 10, restore best weights |
| Learning-rate schedule | ReduceLROnPlateau, factor 0.5, patience 4           |
| Class weights          | balanced (computed from the training set)           |
| Total parameters       | \[copy from model.summary()\]                       |

## 1.6 Training Results and Evaluation

**Training behavior:** the model trained for \[number\] epochs before early stopping. The best validation loss was \[value\] at epoch \[number\].

\[Insert figure: training and validation loss and accuracy curves\]

\[Describe the curves: do training and validation curves stay close together, or does the validation loss rise while the training loss keeps falling (overfitting)?\]

**Test-set performance** (the 15% of data never used during training or tuning):

| Metric                  | DNN   |
|-------------------------|-------|
| Accuracy                | 0.74  |
| Precision (churn class) | 0.51  |
| Recall (churn class)    | 0.79  |
| F1-score (churn class)  | 0.62  |
| ROC-AUC                 | 0.844 |

<img src="images/dnn_confusion_matrix.png" width="520" alt="DNN confusion matrix on the test set" />

<img src="images/dnn_roc_curve.png" width="520" alt="DNN ROC curve on the test set" />

**Analysis:** On the test set of 1,057 customers (281 of whom actually churned), the DNN identified 221 of the churners (recall 0.79) and missed 60. It also raised 210 false alarms, which is why precision for the churn class is 0.51: about half of the customers it flagged were genuine churners. This trade-off comes from using class weights, which push the model to favor catching churners over avoiding false alarms. For a retention campaign this is a reasonable choice. A missed churner (false negative) means the company loses a customer and their future revenue, while a false alarm (false positive) only costs a discount or retention offer sent to a customer who would have stayed anyway. Overall accuracy was 0.74 and the F1-score for the churn class was 0.62, and the ROC-AUC of 0.844 shows the model ranks customers by churn risk well.

Compared with the logistic regression baseline, the DNN gave no meaningful improvement. The baseline reached a ROC-AUC of 0.845 against 0.844 for the DNN, a difference of about 0.0006, and \[add the baseline’s accuracy and churn recall: for example, “the baseline reached an accuracy of X and a churn recall of Y”\]. This is common on small tabular datasets of about 7,000 rows, where the relationships between features and churn are largely simple enough for a linear model to capture, so a deeper network has little extra structure to learn.

Regarding overfitting, \[add one sentence using your training curves: for example, “training and validation loss stayed close together and the training accuracy was X% against a validation accuracy of Y%, so the model did not overfit noticeably”\]. Dropout, batch normalization, L2 regularization and early stopping helped keep the model from overfitting.

## 1.7 Conclusion

**Limitations:** the dataset is small (about 7,000 customers) and comes from a single source, so gains over simpler models such as logistic regression may be modest, and the results may not transfer to other companies or regions.

**Future work:** tune the decision threshold to reach a target recall, try oversampling such as SMOTE, run a hyperparameter search, compare with gradient-boosted trees, and use SHAP values to explain which features drive churn.

# Part 2: CNN – Brain Tumor MRI Classification

## 2.1 Problem Definition

Brain tumors are serious conditions in which early and accurate diagnosis strongly affects treatment planning. MRI is the main imaging method used, but reading scans manually is time-consuming and depends on the radiologist's experience. An automated system that classifies MRI scans can act as a second opinion and help prioritize cases for review.

The task is a multi-class image classification problem: given a brain MRI image, classify it into one of four categories: glioma, meningioma, pituitary tumor, or no tumor. The system is intended only as an educational decision-support prototype and not as a medical diagnostic tool.

## 2.2 Dataset Description and Source Link

**Source:** Brain Tumor MRI Dataset, Kaggle (Masoud Nickparvar). **Link:** https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

The dataset contains 7,023 human brain MRI images in four classes: glioma, meningioma, no tumor and pituitary. It combines three public sources: figshare, the SARTAJ dataset and Br35H, and the no-tumor images come from Br35H. The images come in different sizes and are provided in separate Training and Testing folders, one subfolder per class.

<img src="images/mri_sample_images.png" width="620" alt="Sample brain MRI images from the dataset" />

## 2.3 Proposed Solution

The proposed solution is a convolutional neural network (CNN) built from scratch. Convolutional layers automatically learn visual features from the scans, from simple edges in early layers to more complex tumor-related textures and shapes in deeper layers. The final layers convert these features into probabilities for the four classes.

To improve generalization, the solution uses data augmentation, batch normalization and dropout. As an additional comparison, a transfer-learning model based on MobileNetV2 (pre-trained on ImageNet) is also trained, so the custom CNN can be compared with a well-known pre-trained architecture.

## 2.4 Data Preprocessing

1.  **Resize:** all images were resized to 128 x 128 pixels, since the original images have different sizes.

2.  **Normalize:** pixel values were rescaled from the range 0 to 255 to the range 0 to 1 (inside the model, using a Rescaling layer).

3.  **Labels:** class labels were one-hot encoded for categorical cross-entropy.

4.  **Train/validation split:** 80% of the Training folder was used for training and 20% for validation (random seed 42). The official Testing folder was kept untouched and used only for the final evaluation.

5.  **Data augmentation (training only):** random horizontal flip, random rotation (up to about 18 degrees), random zoom (10%) and random contrast (10%), which help the model handle variations in scan orientation and brightness.

6.  **Input pipeline:** images were cached and prefetched with tf.data to speed up training.

## 2.5 Model Architecture

| Layer        | Details                                                                |
|--------------|------------------------------------------------------------------------|
| Input        | 128 x 128 x 3 image                                                    |
| Augmentation | flip, rotation, zoom, contrast (active only during training)           |
| Rescaling    | divide pixel values by 255                                             |
| Conv block 1 | Conv2D 32 filters (3x3, same padding) + BatchNorm + ReLU + MaxPool 2x2 |
| Conv block 2 | Conv2D 64 filters + BatchNorm + ReLU + MaxPool 2x2                     |
| Conv block 3 | Conv2D 128 filters + BatchNorm + ReLU + MaxPool 2x2                    |
| Conv block 4 | Conv2D 256 filters + BatchNorm + ReLU + MaxPool 2x2                    |
| Global Average Pooling |                                                      |
| Dense        | 128 units, ReLU                                                        |
| Dropout      | rate 0.5                                                               |
| Output       | 4 units, softmax                                                       |

**Training configuration:**

| Setting                | Value                                              |
|------------------------|----------------------------------------------------|
| Optimizer              | Adam, learning rate 0.001                          |
| Loss function          | Categorical cross-entropy                          |
| Batch size             | 32                                                 |
| Maximum epochs         | 40                                                 |
| Early stopping         | monitor val_loss, patience 8, restore best weights |
| Learning-rate schedule | ReduceLROnPlateau, factor 0.5, patience 3          |
| Checkpoint             | best model saved by validation accuracy            |
| Total parameters       | \[copy from model.summary()\]                      |

**Transfer-learning comparison model:** MobileNetV2 pre-trained on ImageNet with its convolutional base frozen, followed by global average pooling, dropout (0.3) and a 4-unit softmax layer, trained for up to 10 epochs with early stopping (patience 3).

## 2.6 Training Results and Evaluation

**Training behavior:** the custom CNN trained for \[number\] epochs before early stopping. The best validation accuracy was \[value\]%.

\[Insert figure: training and validation loss and accuracy curves\]

\[Describe the curves: do they converge smoothly, and is there a large gap between training and validation accuracy?\]

**Test-set performance** (official Testing folder):

| Model                         | Test accuracy |
|-------------------------------|---------------|
| Custom CNN                    | 0.7862        |
| MobileNetV2 transfer learning | 0.8075        |

**Per-class results for the custom CNN:**

| Class      | Precision | Recall | F1-score |
|------------|-----------|--------|----------|
| Glioma     | 0.94      | 0.67   | 0.78     |
| Meningioma | 0.81      | 0.50   | 0.62     |
| No tumor   | 0.73      | 0.99   | 0.84     |
| Pituitary  | 0.75      | 0.98   | 0.84     |

<img src="images/cnn_confusion_matrix.png" width="520" alt="Custom CNN confusion matrix on the test set" />

<img src="images/cnn_misclassified_examples.png" width="620" alt="Examples of misclassified test images" />

**Analysis:** The custom CNN reached a test accuracy of 78.6% on the 1,600 test images (400 per class). It recognized the no-tumor and pituitary classes best, with recall of 0.99 and 0.98, but it was much weaker on the two tumor types that are harder to tell apart: glioma (recall 0.67) and meningioma (recall 0.50). The lower precision for no tumor (0.73) and pituitary (0.75) shows that the model tends to over-predict these two classes, so a large share of glioma and meningioma scans were assigned to them: according to the confusion matrix, 59 gliomas and 85 meningiomas were predicted as no tumor and a further 29 gliomas and 102 meningiomas as pituitary, whereas only 45 gliomas were labeled meningioma and 14 meningiomas glioma. In a medical setting the most serious error is a tumor classified as no tumor, so this result shows the model is not yet reliable enough for real use. Possible reasons include the limited resolution (128 x 128), a small amount of training for a model built from scratch, and the fact that the dataset merges images from three different sources with different imaging characteristics and labeling quality, which makes the tumor classes harder to separate.

## 2.7 Conclusion

**Limitations:** the images come from a few public sources with varying quality and were reduced to 128 x 128 pixels, there is no clinical validation, and the model must not be used for real medical diagnosis.

**Future work:** train at higher resolution, fine-tune the pre-trained model, apply Grad-CAM to show which image regions drive each prediction, use ensembles, and validate on an independent external dataset.

# Overall Conclusion

This project applied two deep learning architectures to two different problems: a DNN for tabular churn prediction and a CNN for image-based brain tumor classification. Both models were built, trained and evaluated on held-out test data using appropriate metrics.

# References

1.  Kaggle. Telco Customer Churn (BlastChar). https://www.kaggle.com/datasets/blastchar/telco-customer-churn

2.  Kaggle. Brain Tumor MRI Dataset (Masoud Nickparvar). https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

3.  Kingma, D. P., and Ba, J. (2015). Adam: A Method for Stochastic Optimization. ICLR.

4.  Ioffe, S., and Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. ICML.

5.  Srivastava, N., et al. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting. Journal of Machine Learning Research, 15.

6.  Sandler, M., et al. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. CVPR.

7.  TensorFlow and Keras documentation. https://www.tensorflow.org/
