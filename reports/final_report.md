# 10 Academy: Artificial Intelligence Mastery  
## Improved Detection of Fraud Cases for E-Commerce and Bank Transactions  
### Final Report – Week 8 Challenge  

**Name:** Anaol Atinafu  
**Email:** atinafuanaol@gmail.com  
**Project:** Improved Detection of Fraud Cases for E-Commerce and Bank Transactions  
**Organization:** 10 Academy  
**Date:** July 29, 2025  

---

## Introduction  

This project develops a complete **end-to-end fraud detection pipeline** for e-commerce and banking transactions. Fraudulent activities, though rare, pose serious financial risks. The solution leverages **data engineering and machine learning techniques** to handle highly imbalanced datasets, enrich transactions with behavioral and geolocation features, and provide transparent model predictions using **SHAP (Shapley Additive Explanations)**.  

The project is divided into **5 main tasks:**  

1. **Data Ingestion & Integration:** Load multiple transaction datasets into a unified schema.  
2. **Exploratory Data Analysis (EDA) & Feature Engineering:** Identify data quality issues, handle class imbalance, and engineer time and location-based features.  
3. **Model Training & Evaluation:** Build baseline (Logistic Regression) and advanced (XGBoost) models.  
4. **Model Explainability:** Apply SHAP for global and local feature importance.  
5. **Deployment & Documentation:** Prepare a professional GitHub repository with Docker support and reproducible workflows.  

---

## Technical Choices  

| **Component**        | **Final Decision**                                                       |
|----------------------|---------------------------------------------------------------------------|
| **Data Sources**     | `Fraud_Data.csv`, `creditcard.csv`, `IpAddress_to_Country.csv`           |
| **Database/Storage** | CSV-based processing (Pandas) → processed data in `outputs/`             |
| **Feature Engineering** | IP-to-country mapping, behavioral features, SMOTE for class imbalance   |
| **Modeling Frameworks** | scikit-learn (Logistic Regression), XGBoost (ensemble model)             |
| **Model Explainability** | SHAP (summary and force plots)                                         |
| **Deployment**       | Modular Python scripts + Jupyter notebooks, Docker-ready                 |
| **Documentation**    | Structured GitHub repository with README and notebooks                   |

---

## System Breakdown  

### Task 1: Data Ingestion & Integration  
- Loaded **two transaction datasets** and geolocation mapping.  
- Unified the schema with consistent column names and data types.  
- Validated missing values and anomalies before downstream processing.  

### Task 2: EDA & Feature Engineering  
- Identified **class imbalance** (fraud < 1% of total transactions).  
- Engineered features including:  
  - `hour_of_day`, `day_of_week`, `time_since_signup`  
  - Geolocation mapping using IP addresses  
  - Transaction velocity metrics  
- Applied **SMOTE** to balance training data.  
- Stored processed datasets in `outputs/processed_data/`.  

### Task 3: Model Training & Evaluation  
- **Baseline:** Logistic Regression for interpretability.  
- **Advanced:** XGBoost (gradient boosting) for improved performance.  
- Metrics used: **F1-Score**, **AUC-PR**, **Recall**, and **Confusion Matrix**.  
- **XGBoost consistently outperformed Logistic Regression on all metrics.**  

### Task 4: Model Explainability  
- Applied **SHAP** for global and local explanations:  
  - **SHAP Summary Plot:** Highlighted transaction amount, time of day, and IP country as top fraud predictors.  
  - **SHAP Force Plot:** Explained individual predictions for edge cases.  

### Task 5: Deployment & Documentation  
- Organized repository into modular folders: `src/`, `notebooks/`, `outputs/`, `models/`.  
- Built **Dockerfile** for containerized reproducibility.  
- README provides **full setup instructions** for analysts and engineers.  

---

## System Evaluation  

### Model Performance Summary  

| **Dataset**   | **Model**            | **F1-Score** | **AUC-PR** |
|---------------|----------------------|--------------|------------|
| creditcard    | Logistic Regression  | 0.65         | 0.75       |
| creditcard    | XGBoost              | **0.88**     | **0.92**   |
| Fraud_Data    | Logistic Regression  | 0.68         | 0.78       |
| Fraud_Data    | XGBoost              | **0.91**     | **0.95**   |

- **XGBoost achieved the highest recall and AUC-PR**, critical for fraud detection where missing fraudulent transactions is costly.  

### Explainability Highlights  
- SHAP plots confirmed **time-based and geolocation features** are key drivers of fraud predictions.  
- Local SHAP force plots help **analysts understand why specific transactions were flagged**.  

---

## User Interface and Monitoring  

- **Jupyter Notebooks**: `notebooks/eda.ipynb` and `notebooks/model_explainability.ipynb` enable interactive data exploration and visualization.  
- **Docker Support**: Ensures reproducible environment setup.  
- **Logging and Outputs**: Intermediate data and logs are available in `outputs/` for debugging.  

---

## Challenges and Solutions  

| **Challenge**                         | **Solution**                                                         |
|---------------------------------------|---------------------------------------------------------------------|
| Severe class imbalance (<1% fraud)    | Applied SMOTE and used recall/AUC-PR as primary metrics.             |
| Data leakage in feature engineering   | Enforced strict train/test split before applying SMOTE.              |
| SHAP computation overhead for XGBoost | Sampled representative test data to reduce computation time.         |
| Repository reproducibility            | Added Dockerfile and comprehensive README for environment setup.     |

---

## Key Learnings  

- **Feature engineering** significantly improves fraud detection in imbalanced datasets.  
- **SHAP explainability** builds trust with stakeholders and highlights model biases.  
- Ensemble methods like **XGBoost** perform substantially better than simple baselines.  
- A **clean repository structure** enables collaboration and easier deployment.  

---

## Future Improvements  

1. Integrate **real-time inference API** (FastAPI) for production use.  
2. Add **streaming data ingestion** (Kafka) for real-time fraud detection.  
3. Extend explainability using **LIME** in addition to SHAP.  
4. Develop a **Streamlit dashboard** for non-technical stakeholders.  
5. Schedule automated jobs with **Dagster or Airflow**.  

---

## Repository Overview  

| **File / Folder**                 | **Purpose**                                     |
|-----------------------------------|-------------------------------------------------|
| `data/`                           | Raw datasets                                    |
| `outputs/processed_data/`         | Processed datasets and engineered features      |
| `models/`                         | Trained models and scalers                      |
| `notebooks/eda.ipynb`             | EDA and preprocessing visualization             |
| `notebooks/model_explainability.ipynb` | SHAP-based global and local explainability   |
| `src/data_pipeline.py`            | Data ingestion and cleaning                     |
| `src/preprocess.py`               | Feature engineering and SMOTE handling          |
| `src/model_training.py`           | Model training and evaluation                   |
| `requirements.txt`                | Python dependencies                             |
| `Dockerfile`                      | Containerization for reproducibility            |
| `README.md`                       | Setup instructions and usage guide              |

**GitHub Link:**  
👉 [https://github.com/anaol216/fraud-detection-adey](https://github.com/anaol216/fraud-detection-adey)  

---

## Conclusion  

This project demonstrates how **structured data engineering** and **explainable machine learning** can turn raw transactional data into actionable fraud detection insights. By combining **advanced feature engineering**, **XGBoost ensemble modeling**, and **SHAP-based visualizations**, the pipeline delivers a **scalable, accurate, and interpretable solution** suitable for real-world e-commerce and banking systems.  
