# Breast_Cancer_Detection_and_Classification
A deep learning framework using VGG-16-based Convolutional Neural Networks for breast cancer detection and classification from mammographic images with additional model comparisons.
# Breast Cancer Detection and Classification Using a CNN-Based Deep Learning Framework

## Abstract

Breast cancer is a major cause of mortality in women, necessitating reliable diagnostic tools for early detection. Convolutional Neural Networks (CNNs), particularly VGG-16, have emerged as effective solutions for identifying malignant tissues in mammographic images. 

VGG-16, with its deep architecture, extracts complex features such as texture, edges, and structural patterns that differentiate benign from malignant tissue. In this study, we employ a CNN-based framework to achieve high accuracy in breast cancer detection and classification, supported by data augmentation techniques to overcome dataset limitations.

**Model evaluation metrics—including accuracy, precision, recall, and F1-score—highlight the framework’s robustness, achieving 88.83% training accuracy and 87.7% validation accuracy.**

Additional machine learning classifiers, such as:
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Random Forest  
were also applied for performance comparison. The Random Forest model yielded 100% training accuracy but lower validation accuracy (88.07%), suggesting overfitting. AdaBoost achieved balanced results with 91.69% training accuracy and 88.5% validation accuracy.

**These results affirm CNN-based models, especially VGG-16, as transformative in breast cancer diagnostics, offering healthcare professionals an automated, accurate tool for timely interventions.**

---

##  Keywords

Breast cancer detection, CNN, VGG-16, mammographic images, classification accuracy, data augmentation, machine learning classifiers, diagnostic tool, feature extraction, early detection.

---

## Introduction

Breast cancer remains one of the leading causes of death among women globally. Early and accurate detection is critical in improving survival rates and patient outcomes. While traditional diagnostic methods rely on manual image assessment, advances in deep learning and artificial intelligence have enabled automated systems that offer greater precision and consistency.

This project implements a CNN-based deep learning framework using the VGG-16 architecture to classify mammographic images as benign or malignant. The approach also integrates additional classifiers like KNN, SVM, Random Forest, and AdaBoost for comparative analysis.

Key aspects include:
- Data preprocessing and augmentation
- CNN feature extraction
- Model training and validation
- Performance evaluation using accuracy, precision, recall, and F1-score metrics

---

## Files

- `Breast_Cancer_Detection_and_Classification (1).ipynb` — The main Jupyter notebook containing model development, training, evaluation, and results.

---

##  Technologies Used

- Python 3.9
- TensorFlow / Keras
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## 🚀 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/satishyedge2000/breast-cancer-detection-and-classification-cnn.git
