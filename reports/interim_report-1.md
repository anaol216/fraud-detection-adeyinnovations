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

## Conclusion

Task 1 is complete. Raw e-commerce transaction data has been transformed into machine-learning-ready datasets. The system is prepared for model training and deployment in the next phase.
