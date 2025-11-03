# linguistic_level_analysis.py
import os
import pandas as pd
import joblib
import librosa
from sklearn.metrics import classification_report, accuracy_score

print("🔹 Loading trained MFCC and HuBERT models...")

# --- Load saved models, encoders, and scalers ---
mfcc_model = joblib.load("accent_model.pkl")
mfcc_scaler = joblib.load("scaler.pkl")
mfcc_encoder = joblib.load("label_encoder.pkl")

hubert_model = joblib.load("hubert_model.pkl")
hubert_scaler = joblib.load("hubert_scaler.pkl")
hubert_encoder = joblib.load("hubert_label_encoder.pkl")

# --- Load features and metadata ---
mfcc_df = pd.read_csv("accent_features.csv")
hubert_df = pd.read_csv("hubert_features.csv")
meta = pd.read_csv("indic_accent_metadata.csv")

print(f"✅ Loaded {len(meta)} metadata entries.")

# ------------------------------------------------------
# 1️⃣ Measure duration to classify word vs sentence level
# ------------------------------------------------------
durations = []
for path in meta["filepath"]:
    try:
        duration = librosa.get_duration(filename=path)
    except Exception:
        duration = 0
    durations.append(duration)

meta["duration"] = durations
meta["level"] = meta["duration"].apply(lambda d: "word" if d < 2.0 else "sentence")

# Merge into feature DataFrames
mfcc_df["level"] = meta["level"]
hubert_df["level"] = meta["level"]

print(meta["level"].value_counts())

# ------------------------------------------------------
# 2️⃣ Split data by level
# ------------------------------------------------------
word_mfcc = mfcc_df[mfcc_df["level"] == "word"]
sent_mfcc = mfcc_df[mfcc_df["level"] == "sentence"]

word_hubert = hubert_df[hubert_df["level"] == "word"]
sent_hubert = hubert_df[hubert_df["level"] == "sentence"]

print(f"🗣 Word samples: {len(word_mfcc)} | Sentence samples: {len(sent_mfcc)}")

# ------------------------------------------------------
# 3️⃣ Evaluation helper
# ------------------------------------------------------
def evaluate(df, model, scaler, encoder, name):
    if len(df) == 0:
        print(f"⚠️ No samples for {name}, skipping...")
        return 0.0
    X = df.drop(columns=["label", "level"], errors="ignore")
    y_true = df["label"]
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    y_pred_labels = encoder.inverse_transform(y_pred)
    acc = accuracy_score(y_true, y_pred_labels)
    print(f"\n✅ {name} Accuracy: {acc:.2f}")
    print(classification_report(y_true, y_pred_labels))
    return acc

# ------------------------------------------------------
# 4️⃣ Evaluate Word-Level vs Sentence-Level
# ------------------------------------------------------
print("\n🎙 Evaluating MFCC model...")
mfcc_word_acc = evaluate(word_mfcc, mfcc_model, mfcc_scaler, mfcc_encoder, "MFCC (Word-Level)")
mfcc_sent_acc = evaluate(sent_mfcc, mfcc_model, mfcc_scaler, mfcc_encoder, "MFCC (Sentence-Level)")

print("\n🎙 Evaluating HuBERT model...")
hubert_word_acc = evaluate(word_hubert, hubert_model, hubert_scaler, hubert_encoder, "HuBERT (Word-Level)")
hubert_sent_acc = evaluate(sent_hubert, hubert_model, hubert_scaler, hubert_encoder, "HuBERT (Sentence-Level)")

# ------------------------------------------------------
# 5️⃣ Comparison Summary
# ------------------------------------------------------
print("\n📊 Comparison Summary:")
print(f"MFCC (Word-Level): {mfcc_word_acc:.2f}")
print(f"MFCC (Sentence-Level): {mfcc_sent_acc:.2f}")
print(f"HuBERT (Word-Level): {hubert_word_acc:.2f}")
print(f"HuBERT (Sentence-Level): {hubert_sent_acc:.2f}")

best = max(
    [("MFCC (Word)", mfcc_word_acc),
     ("MFCC (Sentence)", mfcc_sent_acc),
     ("HuBERT (Word)", hubert_word_acc),
     ("HuBERT (Sentence)", hubert_sent_acc)],
    key=lambda x: x[1],
)
print(f"\n🏆 Best performing setting: {best[0]} with accuracy {best[1]:.2f}")

