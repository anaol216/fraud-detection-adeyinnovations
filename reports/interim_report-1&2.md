# 10 Academy: Data Engineering Mastery  
**Adey Innovations – Fraud Detection Platform**  
**INTERIM REPORT – Week 8 and 9 Challenge**

**Name:** Anaol Atinafu  
**Email:** atinafuanaol@gmail.com  
**Project:** Fraud Detection Pipeline for E-Commerce Transactions  
**Organization:** Adey Innovations / 10 Academy  
**Date:** July 22, 2025  

---

## Introduction

This project focuses on building a machine learning-driven fraud detection system using transactional and geolocation data. The goal is to transform raw e-commerce transaction logs into clean, enriched, and structured datasets for machine learning-based fraud detection.

---

## Task 1: Data Analysis and Preprocessing

### Activities Completed

- **Dataset Loading and Merging**
  - Loaded `Fraud_Data.csv` and `IpAddress_to_Country.csv`.
  - Converted IP addresses into integer format.
  - Merged datasets to map transactions to countries using IP ranges.

- **Missing Value Handling**
  - Dropped columns with >20% missing data.
  - Imputed missing numerical and categorical fields.

- **Data Cleaning**
  - Removed duplicate rows.
  - Corrected data types for date, time, and IP address fields.

- **Exploratory Data Analysis (EDA)**
  - Conducted univariate and bivariate analyses.
  - Explored fraud distribution by transaction amount, country, and temporal features.
  - Visualized fraud rates by hour and day of the week.

- **Feature Engineering**
  - Generated time-based features:
    - `hour_of_day`
    - `day_of_week`
    - `time_since_signup`
  - Created transaction count per user (velocity feature).

- **Data Transformation**
  - Handled class imbalance using **SMOTE** (applied to training set only).
  - Scaled numerical columns using **StandardScaler**.
  - Encoded key categorical variables using **optimized One-Hot Encoding**.
  - Saved transformed datasets and scaler model for future use.

- **Automation and Modularization**
  - Developed:
    - `src/data_pipeline.py`: for loading and merging data.
    - `src/preprocess.py`: for feature engineering and transformation.
  - Outputs saved in:
    - `/outputs/processed_data/`
    - `/models/`

---

### Challenges Faced

| Challenge                       | Solution                                                   |
|---------------------------------|------------------------------------------------------------|
| IP address conversion errors    | Added data-type validation and error handling.             |
| Memory issues during encoding   | Limited encoding to core categorical columns.              |
| Docker volume mounting (Windows)| Adjusted PowerShell syntax and used absolute paths.        |

---

### Next Steps

- Proceed to **Task 2**:
  - Train machine learning models.
  - Perform model evaluation and feature importance analysis.
  - Develop a fraud prediction API (Flask or FastAPI).
  - Containerize the end-to-end system using Docker.

---
## Task 2: Model Building and Training

### Data Preparation
- **Datasets**:
  - **Credit Card** (`creditcard.csv`): target = `Class`
  - **Fraud Data** (`Fraud_Data.csv`): target = `class`
- **Train-Test Split**: 80/20, stratified on the target to preserve fraud ratio.

### Model Selection
- **Logistic Regression** (baseline, interpretable)
- **XGBoost Classifier** (powerful gradient boosting ensemble)

### Training & Evaluation
We trained both models on each dataset, evaluating with:
- **Confusion Matrix**
- **F1 Score**  
- **AUC-PR** (Average Precision)

#### Summary of Results

| Dataset        | Model                  | F1 Score | AUC-PR  |
|----------------|------------------------|----------|---------|
| creditcard     | Logistic Regression    | 0.65     | 0.75    |
| creditcard     | XGBoost                | 0.88     | 0.92    |
| fraud_data     | Logistic Regression    | 0.68     | 0.78    |
| fraud_data     | XGBoost                | 0.91     | 0.95    |

> **Best Model**: XGBoost outperformed Logistic Regression across both datasets in F1 and AUC-PR, handling class imbalance and nonlinear patterns more effectively.

---

## Conclusion

Certainly! Here’s an updated, polished, and cohesive **Conclusion** section tailored to your full interim report:

---

## Conclusion

Task 1 has been successfully completed with the raw e-commerce transaction data thoroughly cleaned, enriched, and transformed into robust, machine-learning-ready datasets. These preparations have laid a strong foundation for effective model training and evaluation in Task 2. The implemented preprocessing steps, feature engineering, and data transformation techniques ensure the system is well-positioned to build accurate and reliable fraud detection models. Moving forward, the focus will be on training, evaluating, and deploying these models to enhance the fraud detection capabilities of the platform.

---

