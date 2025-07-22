import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

def preprocess_data(df):
    print("Loading merged data...")

    # Identify target column
    target_col = 'class' if 'class' in df.columns else 'is_fraud'
    print(f"Detected target column: {target_col}")

    print("Splitting features and target...")
    X = df.drop([target_col], axis=1)
    y = df[target_col]

    print("Identifying column types...")

    # Extract time-based features
    for time_col in ['signup_time', 'purchase_time']:
        if time_col in X.columns:
            X[f'{time_col}_hour'] = pd.to_datetime(X[time_col], errors='coerce').dt.hour

    # Drop high-cardinality or non-useful columns
    drop_cols = ['signup_time', 'purchase_time', 'user_id', 'device_id', 'ip_address']
    X = X.drop([col for col in drop_cols if col in X.columns], axis=1)

    # Categorical Columns
    cat_cols = [col for col in X.select_dtypes(include=['object']).columns if col not in drop_cols]
    print(f"Categorical Columns (for encoding): {cat_cols}")

    # One-Hot Encoding
    X_encoded = pd.get_dummies(X[cat_cols], drop_first=True)
    print(f"One-Hot Encoding complete. Shape: {X_encoded.shape}")

    # Numerical Columns
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    print(f"Numerical Columns: {num_cols}")

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X[num_cols]), columns=num_cols)

    # Combine scaled numerical and encoded categorical data
    X_final = pd.concat([X_scaled.reset_index(drop=True), X_encoded.reset_index(drop=True)], axis=1)

    joblib.dump(scaler, 'models/scaler.pkl')
    print("Scaler saved to models/scaler.pkl")

    # Train-test split
    print("Splitting into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, stratify=y, random_state=42)

    # SMOTE for class imbalance
    print("Applying SMOTE...")
    smote = SMOTE(random_state=42)
    smote_result = smote.fit_resample(X_train, y_train)
    if isinstance(smote_result, tuple) and len(smote_result) == 3:
        X_train_resampled, y_train_resampled, _ = smote_result
    else:
        X_train_resampled, y_train_resampled = smote_result
    print(f"SMOTE complete. Resampled train shape: {X_train_resampled.shape}")

    # Save datasets
    X_train_resampled.to_csv('outputs/processed_data/X_train.csv', index=False)
    y_train_resampled.to_csv('outputs/processed_data/y_train.csv', index=False)
    X_test.to_csv('outputs/processed_data/X_test.csv', index=False)
    y_test.to_csv('outputs/processed_data/y_test.csv', index=False)

    print("Processed datasets saved under outputs/processed_data/")

if __name__ == '__main__':
    df = pd.read_csv('outputs/processed_data/merged_fraud_data.csv')
    preprocess_data(df)
