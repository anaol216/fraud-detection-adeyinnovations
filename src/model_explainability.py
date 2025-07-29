import pandas as pd
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# Load dataset (choose best dataset; Fraud_Data.csv for this example)
df = pd.read_csv("data/Fraud_Data.csv")
X = df.drop("class", axis=1)
y = df["class"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# Train XGBoost (best model)
model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
model.fit(X_train, y_train)

# Initialize SHAP explainer
explainer = shap.Explainer(model, X_train)

# Compute SHAP values
shap_values = explainer(X_test)

# Global Feature Importance
print("Generating SHAP Summary Plot...")
shap.summary_plot(shap_values, X_test, show=False)
plt.savefig("outputs/shap_summary_plot.png", bbox_inches="tight")
plt.close()

# Local Explanation (force plot for a single instance)
sample_idx = 0  # pick first row in test set
# Use explainer.expected_value if available, otherwise use shap_values.base_values
expected_value = getattr(explainer, "expected_value", None)
if expected_value is None:
    expected_value = shap_values.base_values[sample_idx]
else:
    # If expected_value is an array (multi-class), select the correct class
    if hasattr(expected_value, "__len__") and not isinstance(expected_value, str):
        expected_value = expected_value[0]

shap.force_plot(
    expected_value, shap_values[sample_idx, :], X_test.iloc[sample_idx, :],
    matplotlib=True, show=False
)
plt.savefig("outputs/shap_force_plot_sample.png", bbox_inches="tight")
plt.close()

print("SHAP plots saved in outputs/")
