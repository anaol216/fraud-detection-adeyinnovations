
# **Fraud Detection for E-Commerce and Banking Transactions**

This project develops a complete **end-to-end fraud detection pipeline** for e-commerce and bank transactions. It includes **data preprocessing, exploratory data analysis (EDA), machine learning model building, and model explainability using SHAP**.

---

## **Table of Contents**

1. [Project Overview](#project-overview)
2. [Datasets](#datasets)
3. [Project Structure](#project-structure)
4. [Features](#features)
5. [Setup & Installation](#setup--installation)
6. [Usage](#usage)
7. [Results](#results)
8. [Model Explainability](#model-explainability)
9. [License](#license)

---

## **Project Overview**

Fraud detection is a critical business challenge in the financial technology sector. The project aims to:

* Detect fraudulent transactions in **highly imbalanced datasets**.
* Build **accurate and interpretable models**.
* Apply **geolocation analysis** and **behavioral feature engineering**.
* Explain model predictions using **SHAP (Shapley Additive Explanations)**.

---

## **Datasets**

1. **E-Commerce Transaction Data:** `Fraud_Data.csv`

   * Includes user signup/purchase details, IP address, device, source, and class labels.

2. **Bank Transaction Data:** `creditcard.csv`

   * PCA-transformed features with anonymized transaction details.

3. **Geolocation Data:** `IpAddress_to_Country.csv`

   * Maps IP ranges to countries.

> **Class Imbalance:** Fraudulent transactions form a small fraction of the dataset.

---

## **Project Structure**

```
fraud-detection-adey/
│
├── data/                          # Raw datasets
│   ├── creditcard.csv
│   ├── Fraud_Data.csv
│   ├── IpAddress_to_Country.csv
│
├── models/                        # Saved models and scalers
│   ├── scaler.pkl
│
├── notebooks/                     # Jupyter notebooks
│   ├── eda.ipynb                   # Task 1 - EDA & preprocessing
│   ├── task3_model_explainability.ipynb  # Task 3 - SHAP explainability
│
├── outputs/                       # Processed data & plots
│   └── processed_data/
│       ├── merged_fraud_data.csv
│       ├── X_train.csv
│       ├── y_train.csv
│       ├── X_test.csv
│       ├── y_test.csv
│
├── reports/                       # Project reports
│   └── interim_report-1.md
│
├── src/                           # Source scripts
│   ├── data_pipeline.py            # Task 1 - preprocessing pipeline
│   ├── preprocess.py               # Task 1 - transformations & SMOTE
│   ├── task2_model_training.py     # Task 2 - model training & evaluation
│   ├── task3_model_explainability.py (optional)
│
├── tests/                         # (Optional) Unit tests
│
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Containerization
├── .gitignore
└── README.md
```

---

## **Features**

1. **Data Preprocessing & Cleaning**

   * Missing values handling, data type corrections.
   * IP address → country mapping.
   * Feature engineering: `hour_of_day`, `day_of_week`, `time_since_signup`, transaction velocity.
   * Class imbalance handling (SMOTE).

2. **Model Building (Task 2)**

   * **Logistic Regression:** Baseline model.
   * **XGBoost:** Gradient boosting ensemble.
   * Evaluation metrics: **F1-Score, AUC-PR, Confusion Matrix**.

3. **Model Explainability (Task 3)**

   * SHAP summary plots for **global feature importance**.
   * SHAP force plots for **local explanations**.

4. **Dockerized** for reproducibility.

---

## **Setup & Installation**

### **1. Clone the repository**

```bash
git clone https://github.com/<anaol216>/fraud-detection-adey.git
cd fraud-detection-adey
```

### **2. Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate      # Linux / Mac
venv\Scripts\activate         # Windows
```

### **3. Install dependencies**

```bash
pip install -r requirements.txt
```

### **4. (Optional) Build Docker image**

```bash
docker build -t fraud-detection-adey .
```

---

## **Usage**

### **Task 1: Data Preprocessing & EDA**

Run the pipeline to process raw data:

```bash
python src/data_pipeline.py
python src/preprocess.py
```

Or open `notebooks/eda.ipynb` for interactive EDA.

### **Task 2: Model Training**

Train Logistic Regression and XGBoost:

```bash
python src/model_training.py
```

### **Task 3: Model Explainability**

Use the SHAP notebook:

```bash
jupyter notebook notebooks/model_explainability.ipynb
```

---

## **Results**

* **XGBoost outperformed Logistic Regression** across both datasets.
* Fraudulent transactions were better captured (higher recall & AUC-PR).
* Time-based features and IP country were the strongest predictors of fraud.

Example metrics:

| Dataset     | Model               | F1 Score | AUC-PR |
| ----------- | ------------------- | -------- | ------ |
| creditcard  | Logistic Regression | 0.65     | 0.75   |
| creditcard  | XGBoost             | 0.88     | 0.92   |
| Fraud\_Data | Logistic Regression | 0.68     | 0.78   |
| Fraud\_Data | XGBoost             | 0.91     | 0.95   |

---

## **Model Explainability**

We used **SHAP** for interpreting predictions:

* **Summary plots:** Show global feature importance.
* **Force plots:** Explain individual predictions.

Example:
![SHAP Summary Plot](outputs/shap_summary_plot.png)

---

## **License**

This project is part of the **10 Academy Week 8 Challenge** and is for educational purposes.

---
