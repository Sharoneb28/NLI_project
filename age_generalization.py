# age_generalization.py
import pandas as pd
import joblib
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

print("🔹 Loading HuBERT features and model...")

# Load features and trained objects
df = pd.read_csv("hubert_features.csv")
model = joblib.load("hubert_model.pkl")
scaler = joblib.load("hubert_scaler.pkl")
encoder = joblib.load("hubert_label_encoder.pkl")

# Split features/labels
X = df.drop(columns=["label"])
y = df["label"]

# Simulate "adult" (train) and "child" (test) groups
# 80% = adults, 20% = children (unseen group)
X_adult, X_child, y_adult, y_child = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale using the original scaler
X_adult_scaled = scaler.transform(X_adult)
X_child_scaled = scaler.transform(X_child)

print("🧠 Evaluating generalization (trained on adults → tested on children)...")
y_pred_child = model.predict(X_child_scaled)

# Convert back to labels
y_pred_child_labels = encoder.inverse_transform(y_pred_child)
y_child_labels = y_child.values

# Report
acc = accuracy_score(y_child_labels, y_pred_child_labels)
print(f"\n✅ Cross-Age Generalization Accuracy: {acc:.2f}\n")
print(classification_report(y_child_labels, y_pred_child_labels))
