# src/model_training.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    average_precision_score,
    PrecisionRecallDisplay,
)
import matplotlib.pyplot as plt
import os

# ── CONFIG ────────────────────────────────────────────────────────────────
DATA_DIR = "data"
OUTPUT_DIR = "outputs/processed_data"

# ── HELPERS ───────────────────────────────────────────────────────────────

def load_datasets():
    # Credit Card dataset
    cc = pd.read_csv(os.path.join(DATA_DIR, "creditcard.csv"))
    X_cc = cc.drop("Class", axis=1)
    y_cc = cc["Class"]

    # Fraud_Data dataset
    fd = pd.read_csv(os.path.join(DATA_DIR, "Fraud_Data.csv"))
    X_fd = fd.drop("class", axis=1)
    y_fd = fd["class"]

    return (X_cc, y_cc), (X_fd, y_fd)


def split_data(X, y):
    return train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )


def evaluate_and_plot(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\n{name} — Confusion Matrix")
    print(confusion_matrix(y_test, y_pred))
    print(f"\n{name} — Classification Report")
    print(classification_report(y_test, y_pred))
    print(f"{name} — F1 Score: {f1_score(y_test, y_pred):.4f}")
    print(f"{name} — AUC-PR: {average_precision_score(y_test, y_prob):.4f}")

    disp = PrecisionRecallDisplay.from_predictions(y_test, y_prob)
    disp.ax_.set_title(f"{name} Precision-Recall Curve")
    plt.show()


# ── MAIN ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 1. Load
    (X_cc, y_cc), (X_fd, y_fd) = load_datasets()

    # 2. Split
    Xcc_train, Xcc_test, ycc_train, ycc_test = split_data(X_cc, y_cc)
    Xfd_train, Xfd_test, yfd_train, yfd_test = split_data(X_fd, y_fd)

    # 3. Define models
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)

    # 4. Train & Evaluate on Credit Card
    print("=== CREDIT CARD FRAUD DETECTION ===")
    lr.fit(Xcc_train, ycc_train)
    evaluate_and_plot("LogisticRegression (CC)", lr, Xcc_test, ycc_test)

    xgb.fit(Xcc_train, ycc_train)
    evaluate_and_plot("XGBoost (CC)", xgb, Xcc_test, ycc_test)

    # 5. Train & Evaluate on Fraud_Data
    print("\n=== E-COMMERCE FRAUD DETECTION ===")
    lr.fit(Xfd_train, yfd_train)
    evaluate_and_plot("LogisticRegression (FD)", lr, Xfd_test, yfd_test)

    xgb.fit(Xfd_train, yfd_train)
    evaluate_and_plot("XGBoost (FD)", xgb, Xfd_test, yfd_test)
