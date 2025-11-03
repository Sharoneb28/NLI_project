# compare_age_generalization.py
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

print("🔹 Loading models and feature data...")

# --- Load MFCC model ---
mfcc_model = joblib.load("accent_model.pkl")
mfcc_scaler = joblib.load("scaler.pkl")
mfcc_encoder = joblib.load("label_encoder.pkl")
mfcc_df = pd.read_csv("accent_features.csv")

# --- Load HuBERT model ---
hubert_model = joblib.load("hubert_model.pkl")
hubert_scaler = joblib.load("hubert_scaler.pkl")
hubert_encoder = joblib.load("hubert_label_encoder.pkl")
hubert_df = pd.read_csv("hubert_features.csv")

print("✅ Data and models loaded successfully.\n")

# --- Simulate "Adult" vs "Child" split (even indices = adults, odd = children) ---
mfcc_train_df = mfcc_df.iloc[::2]   # adults
mfcc_test_df  = mfcc_df.iloc[1::2]  # children

hubert_train_df = hubert_df.iloc[::2]
hubert_test_df  = hubert_df.iloc[1::2]

# --- Prepare MFCC data ---
X_mfcc_test = mfcc_test_df.drop(columns=["label"])
y_mfcc_test = mfcc_encoder.transform(mfcc_test_df["label"])
X_mfcc_test_scaled = mfcc_scaler.transform(X_mfcc_test)

# --- Prepare HuBERT data ---
X_hubert_test = hubert_test_df.drop(columns=["label"])
y_hubert_test = hubert_encoder.transform(hubert_test_df["label"])
X_hubert_test_scaled = hubert_scaler.transform(X_hubert_test)

# --- Evaluate MFCC model ---
print("🎧 Evaluating MFCC model on 'child' set...")
y_pred_mfcc = mfcc_model.predict(X_mfcc_test_scaled)
acc_mfcc = accuracy_score(y_mfcc_test, y_pred_mfcc)
print(f"✅ MFCC Generalization Accuracy: {acc_mfcc:.2f}")
print(classification_report(y_mfcc_test, y_pred_mfcc, target_names=mfcc_encoder.classes_))

# --- Evaluate HuBERT model ---
print("\n🎙 Evaluating HuBERT model on 'child' set...")
y_pred_hubert = hubert_model.predict(X_hubert_test_scaled)
acc_hubert = accuracy_score(y_hubert_test, y_pred_hubert)
print(f"✅ HuBERT Generalization Accuracy: {acc_hubert:.2f}")
print(classification_report(y_hubert_test, y_pred_hubert, target_names=hubert_encoder.classes_))

# --- Compare ---
print("\n📊 Generalization Comparison Summary:")
print(f"MFCC Accuracy : {acc_mfcc:.2f}")
print(f"HuBERT Accuracy: {acc_hubert:.2f}")

if acc_hubert > acc_mfcc:
    print("🏆 HuBERT features generalize better across age groups.")
else:
    print("🏆 MFCC features generalize better across age groups.")
