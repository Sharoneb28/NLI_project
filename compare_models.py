# compare_models.py
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

print("🔹 Loading models and data...")

# === MFCC Model ===
mfcc_model = joblib.load("accent_model.pkl")
mfcc_scaler = joblib.load("scaler.pkl")
mfcc_le = joblib.load("label_encoder.pkl")
mfcc_df = pd.read_csv("accent_features.csv")

# === HuBERT Model ===
hubert_model = joblib.load("hubert_model.pkl")
hubert_scaler = joblib.load("hubert_scaler.pkl")
hubert_le = joblib.load("hubert_label_encoder.pkl")
hubert_df = pd.read_csv("hubert_features.csv")

# -----------------------------
# Evaluate MFCC model
# -----------------------------
print("\n🎧 Evaluating MFCC model...")
X_mfcc = mfcc_df.drop(columns=["label"])
y_mfcc = mfcc_df["label"]
y_mfcc_encoded = mfcc_le.transform(y_mfcc)
X_mfcc_scaled = mfcc_scaler.transform(X_mfcc)
X_train_mfcc, X_test_mfcc, y_train_mfcc, y_test_mfcc = train_test_split(
    X_mfcc_scaled, y_mfcc_encoded, test_size=0.2, random_state=42, stratify=y_mfcc_encoded
)
y_pred_mfcc = mfcc_model.predict(X_test_mfcc)
acc_mfcc = accuracy_score(y_test_mfcc, y_pred_mfcc)
print(f"✅ MFCC Model Accuracy: {acc_mfcc:.2f}")
print(classification_report(y_test_mfcc, y_pred_mfcc, target_names=mfcc_le.classes_))

# -----------------------------
# Evaluate HuBERT model
# -----------------------------
print("\n🎧 Evaluating HuBERT model...")
X_hubert = hubert_df.drop(columns=["label"])
y_hubert = hubert_df["label"]
y_hubert_encoded = hubert_le.transform(y_hubert)
X_hubert_scaled = hubert_scaler.transform(X_hubert)
X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
    X_hubert_scaled, y_hubert_encoded, test_size=0.2, random_state=42, stratify=y_hubert_encoded
)
y_pred_h = hubert_model.predict(X_test_h)
acc_hubert = accuracy_score(y_test_h, y_pred_h)
print(f"✅ HuBERT Model Accuracy: {acc_hubert:.2f}")
print(classification_report(y_test_h, y_pred_h, target_names=hubert_le.classes_))

# -----------------------------
# Summary Comparison
# -----------------------------
print("\n📊 Model Comparison Summary:")
print(f"MFCC Model Accuracy : {acc_mfcc:.2f}")
print(f"HuBERT Model Accuracy: {acc_hubert:.2f}")

if acc_hubert > acc_mfcc:
    print("🏆 HuBERT model performs better overall.")
elif acc_hubert < acc_mfcc:
    print("🏆 MFCC model performs better overall.")
else:
    print("⚖️ Both models perform equally well.")

